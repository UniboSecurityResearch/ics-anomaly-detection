"""Instance windows and per-feature errors of the DEFENDED detector.

A detector decision at timestep t uses an "instance": for the AE the row X[t]; for a
sequence model the history X[t-o-h : t-o] and the observed target X[t] (h = history,
o = target offset; the repository's detectors use o = 0, LIN is one-step with h = 1).

DiffPure purifies the whole input of the classifier. Here the purified input is the
window of the most recent Lw rows ending at t,
    W_t = X[t-Lw+1 .. t],   Lw = max(context, h + o + 1)   (AE: Lw = context),
and the defended detector is the unchanged detector applied to P(W_t):
    AE:        e_t = (f(P(W_t)[-1]) - P(W_t)[-1])^2
    sequence:  e_t = (f(P(W_t)[Lw-1-o-h : Lw-1-o]) - P(W_t)[-1])^2
so the observed target row is purified too (no un-purified path from the input to
the score). `context` gives the purifier temporal context even for the AE and LIN,
whose own input is one or two rows. Rows before the start of the series (only for
the first Lw-h-o-1 targets, which the detector itself can score) are replaced by the
first row, which leaves the rows used by the detector untouched.

With an identity purifier (t* = 0) these errors are exactly the repository's errors
(checked against main_train.py's stored threshold, window and F1 by
main_purification.py calibrate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

from adversarial.constants import ONE_STEP_MODELS, POINT_MODELS, SEQUENCE_MODELS


@dataclass(frozen=True)
class Geometry:
    model_type: str
    history: int        # 0 for the AE
    offset: int         # target offset o
    length: int         # Lw

    @property
    def first_target(self) -> int:
        """First timestep the repository scores in a series (AE 0, others h + o)."""
        return 0 if self.model_type in POINT_MODELS else self.history + self.offset

    @property
    def hist_slice(self):
        end = self.length - 1 - self.offset
        return end - self.history, end


def make_geometry(model_type: str, history: Optional[int], offset: int, context: int) -> Geometry:
    if model_type in POINT_MODELS:
        return Geometry(model_type, 0, 0, max(int(context), 1))
    if model_type not in SEQUENCE_MODELS:
        raise ValueError(f"unsupported model type {model_type}")
    if history is None or history < 1:
        raise ValueError("sequence models need history >= 1")
    if model_type in ONE_STEP_MODELS and history != 1:
        raise ValueError(f"{model_type} is one-step: history must be 1")
    need = int(history) + int(offset) + 1
    return Geometry(model_type, int(history), int(offset), max(int(context), need))


def window_index(targets: np.ndarray, geom: Geometry) -> np.ndarray:
    """(n, Lw) row indices of the windows ending at `targets` (clipped at row 0)."""
    targets = np.asarray(targets, dtype=np.int64)
    idx = targets[:, None] - geom.length + 1 + np.arange(geom.length, dtype=np.int64)[None, :]
    return np.maximum(idx, 0)


def detector_input(purified: tf.Tensor, geom: Geometry):
    """Split purified windows into (model input, observed target)."""
    observed = purified[:, -1, :]
    if geom.model_type in POINT_MODELS:
        return observed, observed
    a, b = geom.hist_slice
    hist = purified[:, a:b, :]
    if geom.model_type in ONE_STEP_MODELS:
        hist = hist[:, 0, :]
    return hist, observed


def errors_tf(model, purifier, windows: tf.Tensor, geom: Geometry, seed, gradient: str = "full"):
    """Differentiable per-feature squared errors (B, d) of the defended detector."""
    purified = purifier(windows, seed, gradient) if not purifier.identity else windows
    inputs, observed = detector_input(purified, geom)
    pred = model(inputs, training=False)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = tf.reshape(tf.cast(pred, tf.float32), tf.shape(observed))
    return tf.square(pred - observed)


class Scorer:
    """Batched evaluation of defended per-feature errors on a numpy series.

    Identity purifier: numpy float64 windows and Keras predict(), exactly like the
    repository (float32 prediction minus float64 target). Otherwise the windows are
    purified in float32 by a compiled TensorFlow function, then scored the same way.
    """

    def __init__(self, keras_model, purifier, geom: Geometry, batch: int = 1024):
        self.model = keras_model
        self.purifier = purifier
        self.geom = geom
        self.batch = int(batch)
        self._purify = None
        if not purifier.identity:
            p = purifier

            @tf.function(reduce_retracing=True)
            def purify(w, seed):
                return p(w, seed, "full")

            self._purify = purify

    def errors(self, series: np.ndarray, targets: np.ndarray, seed_base: int) -> np.ndarray:
        from .purifier import seed_pair
        series = np.asarray(series)
        targets = np.asarray(targets, dtype=np.int64)
        out = np.empty((len(targets), series.shape[1]), dtype=np.float64)
        for k, start in enumerate(range(0, len(targets), self.batch)):
            tb = targets[start:start + self.batch]
            win = series[window_index(tb, self.geom)]
            if self._purify is not None:
                n = len(tb)
                pad = self.batch - n  # fixed batch shape: one trace only
                w32 = win.astype(np.float32)
                if pad:
                    w32 = np.concatenate([w32, np.repeat(w32[-1:], pad, axis=0)])
                w32 = self._purify(tf.constant(w32), seed_pair(seed_base, k)).numpy()[:n]
                win = w32.astype(np.float64)
            observed = win[:, -1, :]
            if self.geom.model_type in POINT_MODELS:
                inputs = observed
            else:
                a, b = self.geom.hist_slice
                inputs = win[:, a:b, :]
                if self.geom.model_type in ONE_STEP_MODELS:
                    inputs = inputs[:, 0, :]
            pred = self.model.predict(inputs, batch_size=4096, verbose=0)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = np.asarray(pred).reshape(len(tb), -1)
            out[start:start + len(tb)] = (pred - observed) ** 2
        return out

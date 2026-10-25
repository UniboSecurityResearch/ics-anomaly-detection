"""Instance windows and per-feature errors of the DEFENDED detector.

A detector decision at timestep t uses an "instance": for the AE the row X[t]; for a
sequence model the history X[t-o-h : t-o] and the observed target X[t] (h = history,
o = target offset; the repository's detectors use o = 0, LIN is one-step with h = 1).

DiffPure purifies the whole input of the classifier. Here the purified input is the
window of Lw consecutive rows that contains every row the detector uses at t,
    W_t = X[t+d-Lw+1 .. t+d],   Lw = max(context, h + o + 1 + d)   (AE: max(context, 1 + d)),
and the defended detector is the unchanged detector applied to P(W_t). The scored
row t sits at position p = Lw-1-d of the window:
    AE:        e_t = (f(P(W_t)[p]) - P(W_t)[p])^2
    sequence:  e_t = (f(P(W_t)[p-o-h : p-o]) - P(W_t)[p])^2
so the observed target row is purified too (no un-purified path from the input to
the score).

d >= 0 is the purifier's LOOK-AHEAD (seconds of detection delay on SWaT). A causal
purifier (d = 0) puts the scored row at the right edge of the window, where the
denoiser has no right-hand context and is markedly worse than inside the window
(measured by purification/check_denoiser.py); d is chosen there by a fixed rule on
benign validation rows. The repository's own decision rule already uses future
scores (np.convolve(..., 'same') over w timesteps, i.e. up to w//2 steps ahead), so a
look-ahead of a few rows keeps the same kind of non-causal decision.

Rows outside the series are replaced by the first / last row (only near the two
ends of a series); this leaves the rows used by the detector untouched.

With an identity purifier (t* = 0) these errors are exactly the repository's errors,
for any d (checked against main_train.py's stored threshold, window and F1 by
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
    lookahead: int = 0  # d

    @property
    def first_target(self) -> int:
        """First timestep the repository scores in a series (AE 0, others h + o)."""
        return 0 if self.model_type in POINT_MODELS else self.history + self.offset

    @property
    def target_pos(self) -> int:
        """Position of the scored row inside the window."""
        return self.length - 1 - self.lookahead

    @property
    def hist_slice(self):
        end = self.target_pos - self.offset
        return end - self.history, end


def make_geometry(model_type: str, history: Optional[int], offset: int, context: int,
                  lookahead: int = 0) -> Geometry:
    d = int(lookahead)
    if d < 0:
        raise ValueError("lookahead must be >= 0")
    if model_type in POINT_MODELS:
        return Geometry(model_type, 0, 0, max(int(context), 1 + d), d)
    if model_type not in SEQUENCE_MODELS:
        raise ValueError(f"unsupported model type {model_type}")
    if history is None or history < 1:
        raise ValueError("sequence models need history >= 1")
    if model_type in ONE_STEP_MODELS and history != 1:
        raise ValueError(f"{model_type} is one-step: history must be 1")
    need = int(history) + int(offset) + 1 + d
    return Geometry(model_type, int(history), int(offset), max(int(context), need), d)


def window_index(targets: np.ndarray, geom: Geometry, n_rows: int) -> np.ndarray:
    """(n, Lw) row indices of the windows of `targets` in a series of n_rows rows
    (clipped to [0, n_rows - 1])."""
    targets = np.asarray(targets, dtype=np.int64)
    start = targets + geom.lookahead - geom.length + 1
    idx = start[:, None] + np.arange(geom.length, dtype=np.int64)[None, :]
    return np.clip(idx, 0, int(n_rows) - 1)


def detector_input(purified, geom: Geometry):
    """Split purified windows into (model input, observed target)."""
    observed = purified[:, geom.target_pos, :]
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

            @tf.function  # fixed shapes (batches are padded): one trace
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
            win = series[window_index(tb, self.geom, len(series))]
            if self._purify is not None:
                n = len(tb)
                pad = self.batch - n  # fixed batch shape: one trace only
                w32 = win.astype(np.float32)
                if pad:
                    w32 = np.concatenate([w32, np.repeat(w32[-1:], pad, axis=0)])
                w32 = self._purify(tf.constant(w32), seed_pair(seed_base, k)).numpy()[:n]
                win = w32.astype(np.float64)
            inputs, observed = detector_input(win, self.geom)
            pred = self.model.predict(inputs, batch_size=4096, verbose=0)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = np.asarray(pred).reshape(len(tb), -1)
            out[start:start + len(tb)] = (pred - observed) ** 2
        return out

"""Run-time state passed to every attack.

`ModelAdapter` wraps whatever prediction interface was loaded (a differentiable
tf.keras.Model for white-box attacks, or any object exposing predict()/callable for
black-box). `AttackContext` bundles the fully-prepared inputs, budget, detector
configuration and precomputed clean-series detection so each attack only implements
its optimisation, not the plumbing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, List, Optional, Set

import numpy as np
import tensorflow as tf


@dataclass
class ModelAdapter:
    """Common interface for Keras white-box and generic black-box predictors."""

    predictor: Any
    keras_model: Optional[tf.keras.Model]
    source_description: str

    def predict_numpy(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        if isinstance(self.predictor, tf.keras.Model):
            output = self.predictor.predict(x, batch_size=batch_size, verbose=0)
        elif hasattr(self.predictor, "predict"):
            try:
                output = self.predictor.predict(x, batch_size=batch_size)
            except TypeError:
                output = self.predictor.predict(x)
        elif callable(self.predictor):
            output = self.predictor(x)
        else:
            raise RuntimeError("Loaded model object does not expose a prediction interface.")

        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise RuntimeError(
                    "Model returned multiple outputs. Supply a wrapper whose predict() "
                    "returns only the reconstructed/predicted state."
                )
            output = output[0]
        if tf.is_tensor(output):
            output = output.numpy()
        return np.asarray(output, dtype=np.float32)


@dataclass
class AttackContext:
    args: argparse.Namespace
    adapter: ModelAdapter
    x_test: np.ndarray
    labels: np.ndarray
    sensor_cols: List[str]
    scaler: Any
    target_indices: np.ndarray
    modification_mask: np.ndarray
    epsilon: np.ndarray
    thresholds: Optional[np.ndarray]
    lower_domain: Optional[np.ndarray]
    upper_domain: Optional[np.ndarray]
    train_scaled: Optional[np.ndarray]
    kl_reference: Optional[np.ndarray]
    instance_threshold: Optional[float]
    instance_score_kind: str
    detection_window: int
    instance_threshold_source: str
    detection_percentile: Optional[float]
    first_valid_index: int
    clean_detect_point: Optional[np.ndarray]
    clean_detect_window: Optional[np.ndarray]
    protected: Set[int]

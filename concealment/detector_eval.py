"""Detector evaluation for concealment experiments (reuses the adversarial package).

The victim detector is used ONLY here, to measure whether concealment evades it. theta
is either the value printed by main_eval.py (--instance-threshold) or reproduced on the
validation split; the consecutive window replicates each model's cached_detect.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np

from adversarial.context import ModelAdapter
from adversarial.detector import (
    _normalize_percentile_fraction,
    repo_cached_detect,
    reproduce_validation_theta,
)
from adversarial.errors import instance_detection_scores, model_errors_numpy
from adversarial.targets import valid_target_bounds


class DetectorEvaluator:
    def __init__(
        self,
        adapter: ModelAdapter,
        scaler: Any,
        sensor_cols: List[str],
        model_type: str,
        history: Optional[int],
        target_offset: int,
        instance_score: str = "mean_mse",
        prediction_batch_size: int = 4096,
    ) -> None:
        self.adapter = adapter
        self.scaler = scaler
        self.sensor_cols = sensor_cols
        self.model_type = model_type
        self.history = history
        self.target_offset = target_offset
        self.instance_score = instance_score
        self.batch = prediction_batch_size

    def _args(self, dataset: str) -> SimpleNamespace:
        return SimpleNamespace(
            dataset=dataset,
            model_type=self.model_type,
            history=self.history,
            target_offset=self.target_offset,
            prediction_batch_size=self.batch,
            instance_score=self.instance_score,
        )

    def resolve_theta(
        self,
        dataset: str,
        train_scaled: np.ndarray,
        instance_threshold: Optional[float],
        percentile: Optional[float],
    ) -> Tuple[float, str, Optional[float]]:
        if instance_threshold is not None:
            if instance_threshold <= 0:
                raise ValueError("--instance-threshold must be positive.")
            return float(instance_threshold), "cli-value", None
        if percentile is None:
            raise ValueError(
                "Provide --instance-threshold or --instance-threshold-percentile to "
                "obtain theta for the detector."
            )
        frac = _normalize_percentile_fraction(percentile, "--instance-threshold-percentile")
        theta, _ = reproduce_validation_theta(
            self._args(dataset), self.adapter, self.scaler, self.sensor_cols, frac, train_scaled
        )
        return theta, "validation-percentile", frac

    def detect_series(
        self, x_series: np.ndarray, theta: float, window: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Point and window detection over the WHOLE valid series. Returns
        (point_full, window_full, first_index) so callers can index by absolute
        timestep as ``flags[t - first_index]``."""
        first, last = valid_target_bounds(
            self.model_type, len(x_series), self.history, self.target_offset
        )
        indices = np.arange(first, last, dtype=np.int64)
        errors = model_errors_numpy(
            self.adapter, x_series, self.model_type, indices,
            self.history, self.target_offset, self.batch,
        )
        inst = instance_detection_scores(errors, self.instance_score)
        point_full = inst > float(theta)
        window_full = repo_cached_detect(inst, theta, window, self.model_type)
        return point_full, window_full, first

    def detect_at(
        self, x_series: np.ndarray, target_indices: np.ndarray, theta: float, window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Window+point detection restricted to ``target_indices`` (computed on the
        whole series, then indexed)."""
        point_full, window_full, first = self.detect_series(x_series, theta, window)
        pos = np.asarray(target_indices, dtype=np.int64) - first
        return point_full[pos], window_full[pos]

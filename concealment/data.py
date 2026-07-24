"""Data plumbing for concealment experiments, reusing the adversarial package.

Enforces the leakage boundary: the attacker's benign set is the standardized TRAINING
data only; the anomalous targets come from the test set; the detector model is used
only later, for evaluation.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from data_loader import load_test_data
from adversarial.context import ModelAdapter
from adversarial.io_utils import load_model_adapter, load_scaler, load_train_scaled
from adversarial.targets import infer_attack_labels, valid_target_bounds


def load_detector_and_test(
    dataset: str, model_path: str
) -> Tuple[ModelAdapter, Any, np.ndarray, np.ndarray, List[str]]:
    """Load the (victim) detector model, the fitted scaler and the standardized test
    series with its labels. The detector is used ONLY for evaluation."""
    adapter = load_model_adapter(model_path)
    scaler = load_scaler(dataset)
    x_test, labels, sensor_cols = load_test_data(dataset, scaler=scaler, verbose=True)
    x_test = np.asarray(x_test, dtype=np.float64)
    labels = np.asarray(labels)
    sensor_cols = [str(c) for c in sensor_cols]
    return adapter, scaler, x_test, labels, sensor_cols


def load_attacker_normal(dataset: str, scaler: Any, sensor_cols: List[str]) -> np.ndarray:
    """Attacker-observable NORMAL set = standardized training data (benign). This is the
    ONLY data the attacker AE / replay buffer may use."""
    x_train = load_train_scaled(dataset, scaler, sensor_cols)
    return np.asarray(x_train, dtype=np.float64)


def subsample_normal(
    normal: np.ndarray,
    size: int = 0,
    fraction: Optional[float] = None,
    random: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Restrict how much benign data the attacker observes.

    The amount of eavesdropped normal data is an experimental variable: the more benign
    data the attacker AE / replay buffer sees, the better it maps a malicious input onto
    the benign manifold. Default (size<=0 and fraction is None) uses ALL training data.
    ``random=False`` keeps a contiguous prefix (a realistic single eavesdropping window);
    ``random=True`` samples a seeded subset.
    """
    n = len(normal)
    count = n
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("--attacker-train-fraction must be in (0, 1].")
        count = max(1, int(round(fraction * n)))
    elif size and size > 0:
        count = min(int(size), n)
    if count >= n:
        return normal
    if random:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=count, replace=False))
        return normal[idx]
    return normal[:count]


def select_anomalous_targets(
    dataset: str,
    labels: np.ndarray,
    n_rows: int,
    model_type: str,
    history: Optional[int],
    target_offset: int,
    max_targets: int = 0,
) -> np.ndarray:
    """Indices of anomalous test timesteps (ATT_FLAG>0), inside the detector's valid
    target range. These are the samples the concealment attack rewrites."""
    first, last = valid_target_bounds(model_type, n_rows, history, target_offset)
    attack_mask = infer_attack_labels(dataset, labels)
    indices = np.arange(first, last, dtype=np.int64)
    indices = indices[attack_mask[indices]]
    if indices.size == 0:
        raise ValueError("No anomalous target timesteps found in the test set.")
    if max_targets and max_targets > 0 and indices.size > max_targets:
        indices = indices[:max_targets]
    return indices


def assert_no_leakage(normal_data: np.ndarray, test_data: np.ndarray) -> None:
    """Cheap structural guard: the attacker normal set must not be (a slice of) the
    test set. Full identity/statistics leakage is prevented by construction (we only
    ever pass training-derived data to fit), this catches accidental wiring mistakes."""
    if normal_data is test_data:
        raise AssertionError("Attacker normal set IS the test set: data leakage.")
    if normal_data.shape == test_data.shape and np.array_equal(normal_data, test_data):
        raise AssertionError("Attacker normal set equals the test set: data leakage.")

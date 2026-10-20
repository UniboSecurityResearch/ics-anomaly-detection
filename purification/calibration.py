"""Threshold/window calibration of a (defended) detector, replicating main_train.py.

The purifier changes the error distribution, so the defended detector needs its own
threshold theta and window w. They are chosen by EXACTLY the procedure that chose
the undefended ones (main_train.hyperparameter_search):
  * validation errors on the chronological validation part of the training file
    (AE: last 20% of the rows, train_test_split(shuffle=False); sequence models:
    utils.train_val_history_idx_split; LIN: the same indices predicting idx + 1);
  * theta = quantile of the validation instance scores (mean over features);
  * for every (percentile, window) of the grid, detection on the tuning part of the
    test file (utils.custom_train_test_split), metric (F1) with the FPR cap,
    tie-break on FPR; Yhat[window-1:] and normalize_array_length as in main_train;
  * the chosen configuration is scored once on the final part of the test file.
With the identity purifier this must give back the theta, window and final F1
stored in the model .json; `check_reproduction` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import metrics
import utils
from adversarial.constants import ONE_STEP_MODELS, POINT_MODELS
from adversarial.detector import repo_cached_detect
from .instances import Geometry


@dataclass
class Splits:
    """Series and target timesteps of the three parts used by the calibration.
    Targets index into the given series; labels are aligned with the targets."""
    x_train: np.ndarray
    val_targets: np.ndarray
    x_test: np.ndarray
    tune_targets: np.ndarray
    tune_labels: np.ndarray
    final_targets: np.ndarray
    final_labels: np.ndarray
    tune_offset: int
    final_offset: int
    tune_len: int
    final_len: int


def _locate(full: np.ndarray, part: np.ndarray) -> int:
    n = len(part)
    for off in (0, len(full) - n):
        if np.array_equal(full[off:off + n], part):
            return off
    raise RuntimeError("cannot locate a test split inside the test file")


def build_splits(dataset: str, x_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                 geom: Geometry, test_split: float = 0.7) -> Splits:
    n = len(x_train)
    if geom.model_type in POINT_MODELS:
        from sklearn.model_selection import train_test_split
        _, x_val, _, _ = train_test_split(x_train, x_train, test_size=0.2, random_state=42, shuffle=False)
        val_targets = np.arange(n - len(x_val), n, dtype=np.int64)
        assert np.array_equal(x_train[val_targets], x_val)
    else:
        _, val_idxs = utils.train_val_history_idx_split(x_train, geom.history, shuffle=False)
        val_targets = val_idxs + 1 if geom.model_type in ONE_STEP_MODELS else val_idxs
        val_targets = val_targets.astype(np.int64)
    y = np.asarray(y_test).astype(int)
    xt_val, xt_test, yt_val, yt_test = utils.custom_train_test_split(dataset, x_test, y, test_size=test_split, shuffle=False)
    tune_off = _locate(x_test, xt_val)
    final_off = _locate(x_test, xt_test)
    first = geom.first_target
    tune_local = np.arange(first, len(xt_val), dtype=np.int64)
    final_local = np.arange(first, len(xt_test), dtype=np.int64)
    return Splits(x_train=x_train, val_targets=val_targets, x_test=x_test,
                  tune_targets=tune_off + tune_local, tune_labels=yt_val[tune_local],
                  final_targets=final_off + final_local, final_labels=yt_test[final_local],
                  tune_offset=tune_off, final_offset=final_off,
                  tune_len=len(xt_val), final_len=len(xt_test))


def _detect_eval(scores: np.ndarray, labels: np.ndarray, theta: float, window: int, model_type: str):
    """main_train: Yhat = cached_detect(...); Yhat[window-1:]; normalize_array_length."""
    yhat = repo_cached_detect(scores, theta, window, model_type, "same").astype(int)
    yhat = yhat[window - 1:]
    return utils.normalize_array_length(yhat, np.asarray(labels).astype(int))


def detection_metrics(scores, labels, theta, window, model_type) -> Dict[str, float]:
    yhat, y = _detect_eval(scores, labels, theta, window, model_type)
    return dict(f1=float(metrics.f1_score(yhat, y)), precision=float(metrics.precision(yhat, y)),
                recall=float(metrics.recall(yhat, y)), fpr=float(metrics.false_positive_rate(yhat, y)))


def grid_search(val_scores, tune_scores, tune_labels, final_scores, final_labels,
                model_type: str, percentiles: Sequence[float], windows: Sequence[int],
                metric: str = "F1", max_fpr: Optional[float] = 0.05) -> dict:
    """Line-by-line replica of main_train.hyperparameter_search's selection."""
    metric_func = metrics.get(metric)
    negative = metric == "false_positive_rate"
    best_metric = 1 if negative else -1000
    best_fpr = np.inf
    found = False
    best_p, best_w = 0, 0
    for p in percentiles:
        theta = np.quantile(val_scores, p)
        for w in windows:
            yhat, y = _detect_eval(tune_scores, tune_labels, theta, w, model_type)
            value = metric_func(yhat, y)
            fpr = metrics.false_positive_rate(yhat, y)
            if max_fpr is not None and fpr > max_fpr:
                continue
            better = value < best_metric if negative else value > best_metric
            if found and np.isclose(value, best_metric) and fpr < best_fpr:
                better = True
            if better:
                best_metric, best_p, best_w, best_fpr, found = value, p, w, fpr, True
    if not found:
        raise RuntimeError(f"No detection configuration satisfies max_fpr={max_fpr}")
    theta = float(np.quantile(val_scores, best_p))
    final = detection_metrics(final_scores, final_labels, theta, best_w, model_type)
    tune = detection_metrics(tune_scores, tune_labels, theta, best_w, model_type)
    return dict(percentile=float(best_p), window=int(best_w), theta=theta,
                tune_selection_value=float(best_metric), tune_fpr=float(best_fpr),
                tune=tune, final=final)


def check_reproduction(result: dict, stored: dict, f1_tol: float = 1e-3, theta_rtol: float = 1e-4) -> Tuple[bool, str]:
    ok = (result["window"] == stored.get("best_window")
          and np.isclose(result["percentile"], stored.get("threshold_percentile", np.nan))
          and np.isclose(result["theta"], stored.get("best_theta"), rtol=theta_rtol, atol=0)
          and abs(result["final"]["f1"] - stored.get("final_f1", np.nan)) <= f1_tol)
    msg = (f"stored: percentile={stored.get('threshold_percentile')} window={stored.get('best_window')} "
           f"theta={stored.get('best_theta'):.8g} F1={stored.get('final_f1')}\n"
           f"again : percentile={result['percentile']} window={result['window']} "
           f"theta={result['theta']:.8g} F1={result['final']['f1']:.6f}")
    return bool(ok), msg

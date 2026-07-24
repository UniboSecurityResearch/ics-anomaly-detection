"""Metrics for a concealment experiment.

Given the original anomalous targets and their concealed counterparts, evaluate the
victim detector before/after and quantify the perturbation. Attack success is measured
only on targets that were originally detected, and an attack that changed a declared
non-controllable feature is NOT counted successful (§13).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import AttackResult
from .detector_eval import DetectorEvaluator


def perturbation_norms(perturbation: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """L0/L1/L2/Linf on the controllable cells (mask == 1)."""
    mask = np.asarray(mask).astype(bool)
    delta = np.asarray(perturbation, dtype=np.float64)
    controllable = delta[:, mask] if np.any(mask) else delta[:, :0]
    active = np.abs(controllable) > 1e-9
    flat = controllable.reshape(-1)
    n_cells = controllable.size or 1
    n_rows = len(delta)
    return {
        "l0": int(np.sum(active)),
        "l0_fraction": float(np.sum(active) / n_cells),
        "l1": float(np.sum(np.abs(flat))),
        "l2": float(np.linalg.norm(flat, ord=2)),
        "linf": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "changed_features": int(np.sum(np.any(active, axis=0))),
        "changed_rows": int(np.sum(np.any(active, axis=1))),
        "n_controlled": int(np.sum(mask)),
        "controlled_fraction": float(np.sum(mask) / len(mask)),
        "mean_changed_features_per_row": float(np.mean(np.sum(active, axis=1))) if n_rows else 0.0,
    }


def constraint_violation(result: AttackResult, atol: float = 1e-6) -> float:
    """Max absolute change on NON-controllable features (should be 0)."""
    mask = result.feature_mask
    if not np.any(~mask):
        return 0.0
    return float(np.max(np.abs(result.adversarial[:, ~mask] - result.original[:, ~mask])))


def compute_metrics(
    *,
    config_name: str,
    attack: str,
    constraint: str,
    result: AttackResult,
    x_test: np.ndarray,
    target_indices: np.ndarray,
    evaluator: DetectorEvaluator,
    theta: float,
    window: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one comparison-table row for a single concealment configuration."""
    # Build the full concealed series (only anomalous targets are rewritten).
    adv_series = np.asarray(x_test, dtype=np.float64).copy()
    adv_series[target_indices] = result.adversarial

    _, det_before = evaluator.detect_at(x_test, target_indices, theta, window)
    _, det_after = evaluator.detect_at(adv_series, target_indices, theta, window)
    det_before = np.asarray(det_before, dtype=bool)
    det_after = np.asarray(det_after, dtype=bool)

    n_targets = len(target_indices)
    detected_before = int(np.sum(det_before))
    # ASR over targets originally detected as anomalies.
    evaded = int(np.sum(det_before & ~det_after))
    asr = float(evaded / detected_before) if detected_before else math.nan
    # recall over the anomalous targets (all targets are anomalous by construction).
    recall_before = float(np.mean(det_before)) if n_targets else math.nan
    recall_after = float(np.mean(det_after)) if n_targets else math.nan

    violation = constraint_violation(result)
    norms = perturbation_norms(result.perturbation, result.feature_mask)

    meta = result.metadata or {}
    train_time = float(meta.get("train_time_s", 0.0))
    runtime = float(meta.get("transform_time_s", 0.0))

    row: Dict[str, Any] = {
        "config": config_name,
        "attack": attack,
        "constraint": constraint,
        "targets": n_targets,
        "detected_before": detected_before,
        "evaded": evaded,
        "attack_success_rate": asr,
        "recall_before": recall_before,
        "recall_after": recall_after,
        "constraint_violation": violation,
        "valid": violation <= 1e-6,
        **norms,
        "attacker_train_time_s": train_time,
        "gen_time_per_sample_ms": float(1000.0 * runtime / n_targets) if n_targets else math.nan,
        "theta": float(theta),
        "detection_window": int(window),
    }
    if extra:
        row.update(extra)
    return row


def to_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "config", "attack", "constraint", "targets", "detected_before", "evaded",
        "attack_success_rate", "recall_before", "recall_after", "valid",
        "constraint_violation", "n_controlled", "controlled_fraction",
        "changed_features", "l0", "l1", "l2", "linf",
        "attacker_train_samples", "attacker_train_time_s", "gen_time_per_sample_ms",
    ]
    df = pd.DataFrame(rows)
    ordered = [c for c in columns if c in df.columns] + [c for c in df.columns if c not in columns]
    return df[ordered]

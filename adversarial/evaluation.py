"""Evaluation, output writing and the mandatory sanity check.

`evaluate_series` reports the optimisation score (--score) on the targets.
`save_run` writes per-target/summary CSVs and the config JSON, deciding
attack-success purely from the scalar theta+window detector (never any_j).
`run_sanity_check` reproduces main_eval.py's detector on the full test series.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .context import AttackContext
from .detector import (
    repo_cached_detect,
    series_instance_scores,
    windowed_target_detection,
)
from .errors import detector_score_numpy, model_errors_numpy
from .targets import infer_attack_labels, valid_target_bounds


def evaluate_series(ctx: AttackContext, x_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    errors = model_errors_numpy(
        ctx.adapter,
        x_series,
        ctx.args.model_type,
        ctx.target_indices,
        ctx.args.history,
        ctx.args.target_offset,
        ctx.args.prediction_batch_size,
    )
    scores = detector_score_numpy(
        errors,
        ctx.args.score,
        ctx.thresholds,
        ctx.args.margin_beta,
    )
    return errors, scores


def perturbation_statistics(delta: np.ndarray, modification_mask: np.ndarray) -> Dict[str, float]:
    active = np.abs(delta) > 1e-9
    allowed_values = np.abs(delta[modification_mask > 0])
    if allowed_values.size == 0:
        allowed_values = np.array([0.0], dtype=np.float32)
    per_modified_row = np.any(active, axis=1)
    per_modified_feature = np.any(active, axis=0)
    return {
        "linf_scaled": float(np.max(np.abs(delta))),
        "l2_scaled": float(np.linalg.norm(delta.reshape(-1), ord=2)),
        "mean_abs_delta_scaled": float(np.mean(allowed_values)),
        "changed_cells": int(np.sum(active)),
        "changed_rows": int(np.sum(per_modified_row)),
        "changed_features": int(np.sum(per_modified_feature)),
    }


def save_run(
    ctx: AttackContext,
    attack_name: str,
    goal: str,
    x_adv: np.ndarray,
    metadata: Dict[str, Any],
    errors_before: np.ndarray,
    scores_before: np.ndarray,
) -> Dict[str, Any]:
    args = ctx.args
    run_dir = Path(args.output_dir) / attack_name / goal
    run_dir.mkdir(parents=True, exist_ok=True)

    errors_after, scores_after = evaluate_series(ctx, x_adv)
    delta = x_adv - ctx.x_test

    # Operative detector: scalar theta+window computed on the WHOLE series and then
    # restricted to the selected targets. The clean-series decision is precomputed
    # once in build_context; the adversarial one is computed here.
    detect_before = ctx.clean_detect_point
    detect_before_window = ctx.clean_detect_window
    detect_after, detect_after_window = windowed_target_detection(ctx, x_adv)

    if detect_before_window is not None and detect_after_window is not None:
        if goal == "evasion":
            # Evasion = detected before AND no longer detected after.
            success = detect_before_window & ~detect_after_window
            eligible = detect_before_window
        else:
            # False alarm = not detected before AND detected after.
            success = ~detect_before_window & detect_after_window
            eligible = ~detect_before_window
    else:
        success = None
        eligible = None

    metrics_data: Dict[str, Any] = {
        "target_index": ctx.target_indices,
        "label": ctx.labels[ctx.target_indices],
        "score_before": scores_before,
        "score_after": scores_after,
        "score_change": scores_after - scores_before,
    }
    if detect_before is not None:
        metrics_data.update(
            {
                "detected_before_point": detect_before,
                "detected_after_point": detect_after,
                "detected_before_window": detect_before_window,
                "detected_after_window": detect_after_window,
                "eligible": eligible,
                "attack_success": success,
            }
        )
    pd.DataFrame(metrics_data).to_csv(run_dir / "attack_scores.csv", index=False)

    np.save(run_dir / "target_indices.npy", ctx.target_indices)
    np.save(run_dir / "feature_errors_before.npy", errors_before)
    np.save(run_dir / "feature_errors_after.npy", errors_after)
    if args.save_series in {"full", "delta"}:
        np.save(run_dir / "delta_scaled.npy", delta)
    if args.save_series == "full":
        np.save(run_dir / "x_test_adversarial_scaled.npy", x_adv)
    if args.save_raw_csv:
        raw_adv = ctx.scaler.inverse_transform(x_adv)
        pd.DataFrame(raw_adv, columns=ctx.sensor_cols).to_csv(
            run_dir / "x_test_adversarial_raw.csv", index=False
        )

    stats = perturbation_statistics(delta, ctx.modification_mask)
    summary: Dict[str, Any] = {
        "attack": attack_name,
        "goal": goal,
        "targets": int(len(ctx.target_indices)),
        "mean_score_before": float(np.mean(scores_before)),
        "mean_score_after": float(np.mean(scores_after)),
        "mean_score_change": float(np.mean(scores_after - scores_before)),
        **stats,
        **metadata,
    }
    if success is not None and eligible is not None:
        eligible_count = int(np.sum(eligible))
        success_count = int(np.sum(success))
        summary.update(
            {
                "eligible_targets": eligible_count,
                "successful_targets": success_count,
                "attack_success_rate": (
                    float(success_count / eligible_count) if eligible_count else math.nan
                ),
                "detection_rate_before": float(np.mean(detect_before_window)),
                "detection_rate_after": float(np.mean(detect_after_window)),
            }
        )
    else:
        summary.update(
            {
                "eligible_targets": None,
                "successful_targets": None,
                "attack_success_rate": None,
                "detection_rate_before": None,
                "detection_rate_after": None,
            }
        )

    # Detector provenance recorded in BOTH summary.csv and attack_config.json.
    summary.update(
        {
            "instance_threshold": ctx.instance_threshold,
            "instance_threshold_source": ctx.instance_threshold_source,
            "detection_percentile": ctx.detection_percentile,
            "detection_window": ctx.detection_window,
            "instance_score_kind": ctx.instance_score_kind,
            "optimization_score": args.score,
        }
    )

    config = vars(args).copy()
    config.update(
        {
            "attack_executed": attack_name,
            "goal_executed": goal,
            "model_source": ctx.adapter.source_description,
            "sensor_cols": ctx.sensor_cols,
            "epsilon_scaled": ctx.epsilon.tolist(),
            "thresholds": None if ctx.thresholds is None else ctx.thresholds.tolist(),
            "detector": {
                "instance_threshold": ctx.instance_threshold,
                "instance_threshold_source": ctx.instance_threshold_source,
                "detection_percentile": ctx.detection_percentile,
                "detection_window": ctx.detection_window,
                "instance_score_kind": ctx.instance_score_kind,
                "note": (
                    "Operative detector = scalar theta on the mean per-feature MSE "
                    "plus a consecutive-window (repo cached_detect); NOT the "
                    "per-feature any_j rule. Optimization loss (--score) is separate."
                ),
            },
            "summary": summary,
        }
    )
    with (run_dir / "attack_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    print(
        f"[{attack_name}/{goal}] mean score {summary['mean_score_before']:.8f} -> "
        f"{summary['mean_score_after']:.8f}; L_inf={summary['linf_scaled']:.6f}; "
        f"saved to {run_dir.resolve()}"
    )
    return summary


def run_sanity_check(ctx: AttackContext) -> None:
    """Mandatory pre-attack check: reproduce main_eval.py's scalar detector on the full
    test series and report benign/attack detection rates so they can be compared against
    the numbers printed by main_eval.py. If they diverge, the benchmark is not reliable.
    """
    args = ctx.args
    print("\n=== Sanity check: reproducing main_eval.py's scalar detector ===")
    if ctx.instance_threshold is None:
        raise ValueError(
            "Sanity check requires a theta. Pass --instance-threshold-percentile 0.95 "
            "(reproduces main_eval.py on the validation split) and/or the exact "
            "--instance-threshold printed by main_eval.py."
        )
    theta = float(ctx.instance_threshold)
    window = ctx.detection_window
    print(f"theta                 = {theta:.8f}   (source: {ctx.instance_threshold_source})")
    print(f"percentile            = {ctx.detection_percentile}")
    print(f"detection window      = {window}")
    print(f"instance aggregation  = {ctx.instance_score_kind} (repo default: mean)")
    print(f"target-offset         = {args.target_offset}")

    first, last = valid_target_bounds(
        args.model_type, len(ctx.x_test), args.history, args.target_offset
    )
    all_indices = np.arange(first, last, dtype=np.int64)
    inst = series_instance_scores(ctx, ctx.x_test, all_indices)
    point = inst > theta
    windowed = repo_cached_detect(inst, theta, window, args.model_type)

    attack_mask_full = infer_attack_labels(args.dataset, ctx.labels)
    y_true_full = np.asarray(attack_mask_full[all_indices], dtype=bool)

    def _rates(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float, float, float]:
        benign = ~truth
        det_attack = float(np.mean(pred[truth])) if np.any(truth) else math.nan
        det_benign = float(np.mean(pred[benign])) if np.any(benign) else math.nan
        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & benign))
        fn = int(np.sum(~pred & truth))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return det_attack, det_benign, precision, recall, f1

    # main_eval.py:52 drops the first window-1 samples before scoring. For 'same' models
    # that shift is not yet applied here, so drop it to mirror main_eval exactly; GRU's
    # 'full'+backfill path already had Yhat[window-1:] applied inside repo_cached_detect.
    window_drop = 0 if (window <= 1 or args.model_type == "GRU") else window - 1
    print(f"evaluated positions   = {len(all_indices)} "
          f"(attack={int(np.sum(y_true_full))}, benign={int(np.sum(~y_true_full))})")
    rows = [("point  (w=1)", point, 0), (f"window (w={window})", windowed, window_drop)]
    for label, pred, drop in rows:
        det_a, det_b, precision, recall, f1 = _rates(pred[drop:], y_true_full[drop:])
        print(
            f"[{label}] detection: attack={det_a:.4f} benign_FP={det_b:.4f} | "
            f"precision={precision:.4f} recall={recall:.4f} F1~={f1:.4f}"
        )
    print(
        "Compare F1/recall above with main_eval.py's printed value for the SAME "
        "percentile+window. A match (within tolerance) confirms the detector is "
        "reproduced; otherwise stop and investigate before benchmarking."
    )

#!/usr/bin/env python3
"""Entrypoint for the modular adversarial-attack suite.

Run from the repository root (so `data_loader`, `models/`, `data/` resolve):

    python main_adversarial.py --dataset BATADAL --model-type AE \
        --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
        --instance-threshold-percentile 0.95 --detection-window 3 --sanity-check

The heavy lifting lives in the `adversarial` package; this file only orchestrates:
parse args -> build context -> (sanity check | run selected attacks) -> aggregate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import tensorflow  # noqa: F401  (imported for the friendly error only)
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "TensorFlow is required. Activate the virtual environment used by the "
        "ics-anomaly-detection repository and install its requirements.txt."
    ) from exc

try:
    import data_loader  # noqa: F401  (imported for the friendly error only)
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "Run this script from the root of pwwl/ics-anomaly-detection, where "
        "data_loader.py is available."
    ) from exc

import numpy as np
import pandas as pd

from adversarial.attacks import ATTACKS
from adversarial.build import build_context
from adversarial.cli import parse_args
from adversarial.evaluation import evaluate_series, run_sanity_check, save_run


def select_attacks(args, thresholds) -> List[str]:
    """Threshold-gate the requested attacks, preserving the monolith's messaging."""
    requested = list(ATTACKS) if args.attack == "all" else [args.attack]
    result: List[str] = []
    for name in requested:
        if ATTACKS[name].requires_thresholds and thresholds is None:
            if args.attack == "all":
                print(f"Skipping {name}: thresholds are unavailable.")
                continue
            raise ValueError(f"{name} requires --thresholds or --threshold-percentile.")
        result.append(name)
    return result


def main() -> None:
    args = parse_args()
    ctx = build_context(args)

    if args.sanity_check:
        run_sanity_check(ctx)
        return

    attacks = select_attacks(args, ctx.thresholds)
    goals = ["evasion", "false_alarm"] if args.goal == "both" else [args.goal]

    whitebox = [name for name in attacks if ATTACKS[name].requires_gradients]
    if whitebox and ctx.adapter.keras_model is None:
        if args.attack != "all":
            raise RuntimeError(
                "The selected attack is white-box but no tf.keras.Model was found in "
                "the loaded artifact. Use corrshift or supply the Keras submodel."
            )
        attacks = [name for name in attacks if not ATTACKS[name].requires_gradients]
        print(f"Skipping non-differentiable white-box attacks: {', '.join(whitebox)}")

    print("\n=== Experiment configuration ===")
    print(f"model source:       {ctx.adapter.source_description}")
    print(f"model type:         {args.model_type}")
    print(f"test shape:         {ctx.x_test.shape}")
    print(f"selected targets:   {len(ctx.target_indices)}")
    print(f"modifiable cells:   {int(np.sum(ctx.modification_mask))}")
    print(f"protected features: {sorted(ctx.protected)}")
    print(f"attacks:            {attacks}")
    print(f"goals:              {goals}")
    print(f"opt. loss (--score):{args.score}")
    print(f"thresholds:         {'yes' if ctx.thresholds is not None else 'no'}")
    if ctx.instance_threshold is not None:
        print(
            f"instance detector:  {ctx.instance_score_kind} > {ctx.instance_threshold:.8f} "
            f"(window {ctx.detection_window}, source {ctx.instance_threshold_source})"
        )
        if ctx.clean_detect_window is not None:
            print(
                f"clean detection:    {float(np.mean(ctx.clean_detect_window)):.4f} of the "
                f"selected targets detected before attack (window)"
            )
    else:
        print(
            "instance detector:  NONE -> attack-success/detection metrics are DISABLED. "
            "Pass --instance-threshold (from main_eval.py) or "
            "--instance-threshold-percentile to enable the ESORICS scalar detector."
        )

    errors_before, scores_before = evaluate_series(ctx, ctx.x_test)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    # Save the exact clean/original series used in this experiment.
    if args.save_series == "full":
        np.save(output_root / "x_test_original_scaled.npy", ctx.x_test)

    if args.save_raw_csv:
        raw_original = ctx.scaler.inverse_transform(ctx.x_test)
        pd.DataFrame(
            raw_original,
            columns=ctx.sensor_cols
        ).to_csv(
            output_root / "x_test_original_raw.csv",
            index=False
        )
    if ctx.thresholds is not None:
        np.save(output_root / "thresholds_used.npy", ctx.thresholds)
    if ctx.kl_reference is not None:
        np.save(output_root / "kl_reference_used.npy", ctx.kl_reference)
    np.save(output_root / "target_indices.npy", ctx.target_indices)

    summaries: List[Dict[str, Any]] = []
    for name in attacks:
        attack = ATTACKS[name]()
        for goal in goals:
            print(f"\n=== Running {name} / {goal} ===")
            x_adv, metadata = attack.run(ctx, goal)
            summaries.append(
                save_run(ctx, name, goal, x_adv, metadata, errors_before, scores_before)
            )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_root / "summary.csv", index=False)
    print("\n=== Aggregate summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary_df)
    print(f"\nSaved aggregate summary to {(output_root / 'summary.csv').resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

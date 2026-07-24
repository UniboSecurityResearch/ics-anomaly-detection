#!/usr/bin/env python3
"""Entrypoint for the concealment / evasion attack suite (run from the repo root).

Examples
--------
Sanity: reproduce the detector and run one unconstrained autoencoder attack:

    python main_concealment.py --dataset BATADAL --model-type AE \
        --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
        --instance-threshold-percentile 0.95 --detection-window 3 \
        --attack autoencoder --constraint unconstrained

Full comparison table (the 6 required configurations, sweeping |C|):

    python main_concealment.py --dataset BATADAL --model-type AE \
        --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
        --instance-threshold-percentile 0.95 --detection-window 3 \
        --compare --controlled-k-list 1 2 5 10
"""

from __future__ import annotations

import sys

try:
    import tensorflow  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required. Activate the ics-anomaly-detection virtual environment."
    ) from exc

try:
    import data_loader  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Run this script from the root of pwwl/ics-anomaly-detection."
    ) from exc

from concealment.cli import parse_args
from concealment.config import apply_config, load_config
from concealment.runner import run_compare, run_single


def main() -> None:
    args = parse_args()
    if args.config:
        apply_config(args, load_config(args.config))
    if args.compare:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

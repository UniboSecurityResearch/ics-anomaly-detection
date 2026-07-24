"""Command-line interface for the concealment suite."""

from __future__ import annotations

import argparse

POINT_MODELS = ("AE", "DNN")
SEQUENCE_MODELS = ("CNN", "GRU", "LSTM")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Concealment / evasion attacks (generic replay + black-box attacker "
            "autoencoder) against reconstruction-based ICS anomaly detectors."
        ),
    )

    g = p.add_argument_group("repository, model and data")
    g.add_argument("--dataset", required=True,
                   choices=["BATADAL", "SWAT", "SWAT-CLEAN", "WADI", "WADI-CLEAN"])
    g.add_argument("--model-type", required=True, choices=sorted(POINT_MODELS + SEQUENCE_MODELS))
    g.add_argument("--model-path", required=True, help="Victim detector (Keras .h5/SavedModel).")
    g.add_argument("--history", type=int, default=None, help="History for CNN/GRU/LSTM.")
    g.add_argument("--target-offset", type=int, choices=[0, 1], default=1)
    g.add_argument("--prediction-batch-size", type=int, default=4096)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output-dir", default="outputs/concealment")
    g.add_argument("--max-targets", type=int, default=0,
                   help="Cap on anomalous targets rewritten; <=0 = all.")
    g.add_argument("--attacker-train-size", type=int, default=0,
                   help="How many benign samples the attacker observes (AE/replay buffer); "
                        "<=0 = all training data. More data => better concealment.")
    g.add_argument("--attacker-train-fraction", type=float, default=None,
                   help="Fraction of benign training data observed, in (0,1]. Overrides size.")
    g.add_argument("--attacker-train-random", action="store_true",
                   help="Sample the attacker's benign data randomly (seeded) instead of a "
                        "contiguous prefix.")

    d = p.add_argument_group("victim detector (evaluation only)")
    d.add_argument("--instance-threshold", type=float, default=None,
                   help="Scalar theta from main_eval.py (preferred).")
    d.add_argument("--instance-threshold-percentile", type=float, default=None,
                   help="Reproduce theta on the validation split (0.95 or 95).")
    d.add_argument("--detection-window", type=int, default=1)
    d.add_argument("--instance-score", choices=["mean_mse", "max_mse"], default="mean_mse")

    a = p.add_argument_group("attack")
    a.add_argument("--attack", choices=["replay", "autoencoder"], default="autoencoder")
    a.add_argument("--constraint",
                   choices=["unconstrained", "partially_constrained", "fully_constrained"],
                   default="unconstrained")
    a.add_argument("--compare", action="store_true",
                   help="Run the 6 required configurations and emit a comparison table.")
    a.add_argument("--config", default=None, help="Optional YAML/JSON attack config.")

    c = p.add_argument_group("controlled feature subset C")
    c.add_argument("--controlled-features", default=None,
                   help="Comma-separated feature names or 0-based indices.")
    c.add_argument("--controlled-percentage", type=float, default=None,
                   help="Fraction of features that are controllable, in (0,1].")
    c.add_argument("--controlled-k", type=int, default=None, help="Number of controllable features.")
    c.add_argument("--controlled-k-list", type=int, nargs="+", default=None,
                   help="--compare only: sweep several |C| for the constrained configs.")
    c.add_argument("--controlled-file", default=None,
                   help="File with controllable feature names/indices (JSON list or one per line).")
    c.add_argument("--controlled-random", action="store_true",
                   help="Random (seeded) subset instead of the first-k columns.")

    r = p.add_argument_group("replay attack")
    r.add_argument("--replay-strategy",
                   choices=["fixed", "cyclic", "random", "nearest_context", "offline_oracle"],
                   default="cyclic")
    r.add_argument("--replay-start", type=int, default=0)
    r.add_argument("--replay-length", type=int, default=None)
    r.add_argument("--allow-offline-oracle", action="store_true")

    e = p.add_argument_group("attacker autoencoder")
    e.add_argument("--ae-hidden", default=None, help="Comma-separated encoder layer sizes.")
    e.add_argument("--ae-layers", type=int, default=4)
    e.add_argument("--ae-compression", type=float, default=2.0)
    e.add_argument("--ae-latent", type=int, default=None)
    e.add_argument("--ae-activation", default="relu")
    e.add_argument("--ae-lr", type=float, default=1e-3)
    e.add_argument("--ae-batch-size", type=int, default=256)
    e.add_argument("--ae-epochs", type=int, default=100)
    e.add_argument("--ae-patience", type=int, default=5)
    e.add_argument("--ae-val-split", type=float, default=0.1)
    e.add_argument("--ae-loss", default="mse")
    e.add_argument("--ae-device", choices=["auto", "cpu", "gpu"], default="auto")
    e.add_argument("--ae-verbose", type=int, default=0)

    pp = p.add_argument_group("domain post-processing")
    pp.add_argument("--postprocess", action="store_true",
                    help="Project controllable features onto the training domain.")
    pp.add_argument("--discrete-max-unique", type=int, default=10,
                    help="Features with <= this many distinct training values are discrete.")
    pp.add_argument("--no-clip", action="store_true",
                    help="Do not clip continuous features to the training range.")

    args = p.parse_args(argv)
    if args.model_type in SEQUENCE_MODELS and (args.history is None or args.history <= 0):
        p.error("--history must be a positive integer for CNN/GRU/LSTM")
    if args.detection_window <= 0:
        p.error("--detection-window must be positive")
    return args

"""Command-line interface. Identical flags/semantics to the former monolith so
existing commands keep working; `--attack` choices come from the attack registry.
"""

from __future__ import annotations

import argparse

from .attacks import ALL_ATTACKS
from .constants import POINT_MODELS, SEQUENCE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "White-box and black-box adversarial attacks against reconstruction/"
            "prediction-based ICS anomaly detectors."
        ),
    )

    io = parser.add_argument_group("repository, model and data")
    io.add_argument(
        "--dataset",
        required=True,
        choices=["BATADAL", "SWAT", "SWAT-CLEAN", "WADI", "WADI-CLEAN"],
    )
    io.add_argument(
        "--model-type",
        required=True,
        choices=sorted(POINT_MODELS | SEQUENCE_MODELS),
    )
    io.add_argument(
        "--model-path",
        required=True,
        help="Keras .h5/.keras/SavedModel path or a pickle exposing predict().",
    )
    io.add_argument(
        "--history",
        type=int,
        default=None,
        help="History length for CNN/GRU/LSTM; ignored for AE/DNN.",
    )
    io.add_argument(
        "--target-offset",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Repository-compatible alignment: history X[i-h:i] and target X[i+1]. "
            "Use 0 only for a locally trained history X[i-h:i] -> X[i] model."
        ),
    )
    io.add_argument("--output-dir", default="outputs/adversarial_suite")
    io.add_argument("--seed", type=int, default=42)
    io.add_argument(
        "--prediction-batch-size",
        type=int,
        default=4096,
        help="Batch size for model evaluation and CorrShift queries.",
    )

    selection = parser.add_argument_group("target selection")
    mutually_exclusive = selection.add_mutually_exclusive_group()
    mutually_exclusive.add_argument(
        "--selection",
        choices=["all", "attack", "normal"],
        default="all",
        help="Select target timesteps using the repository test labels.",
    )
    mutually_exclusive.add_argument(
        "--start", type=int, default=None, help="First target timestep, inclusive."
    )
    selection.add_argument(
        "--end", type=int, default=None, help="Last target timestep, exclusive."
    )
    selection.add_argument(
        "--max-targets",
        type=int,
        default=2048,
        help="Maximum selected target timesteps; <=0 disables the limit.",
    )

    attack = parser.add_argument_group("attack configuration")
    attack.add_argument(
        "--attack",
        choices=ALL_ATTACKS + ["all"],
        default="pgd_mse",
    )
    attack.add_argument(
        "--goal",
        choices=["evasion", "false_alarm", "both"],
        default="evasion",
    )
    attack.add_argument(
        "--scope",
        choices=["target", "history", "both"],
        default="both",
        help="Values that may be modified. Point models always use target.",
    )
    attack.add_argument(
        "--score",
        choices=["mean_mse", "max_mse", "threshold_max", "threshold_smooth"],
        default="mean_mse",
        help=(
            "DIFFERENTIABLE OPTIMIZATION LOSS for fgsm_mse/pgd_mse/pgd_topk/corrshift. "
            "This is NOT the operative detector: attack success and detection rates are "
            "ALWAYS decided by the scalar instance-level theta+window detector "
            "(see --instance-threshold*). 'mean_mse' is the loss aligned with the "
            "repository detector (errors.mean(axis=1); main_eval.py:298-299). "
            "'threshold_max'/'threshold_smooth' encode the legacy per-feature "
            "any_j(error_j>theta_j) rule and are kept ONLY as a research loss."
        ),
    )
    attack.add_argument(
        "--margin-beta",
        type=float,
        default=20.0,
        help="Smooth-max beta for threshold_smooth and margin-based attacks.",
    )

    budget = parser.add_argument_group("perturbation constraints")
    budget.add_argument(
        "--epsilon",
        type=float,
        default=0.10,
        help="L-infinity budget in standardized feature units.",
    )
    budget.add_argument(
        "--epsilon-raw-file",
        default=None,
        help=(
            "Optional .npy/.csv/.json vector of per-feature budgets in raw physical "
            "units. Converted through StandardScaler.scale_ and overrides --epsilon."
        ),
    )
    budget.add_argument("--alpha", type=float, default=0.01, help="PGD step size.")
    budget.add_argument("--iterations", type=int, default=20)
    budget.add_argument("--random-start", action="store_true")
    budget.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="pgd_topk: modified features per timestep and iteration.",
    )
    budget.add_argument(
        "--protected-cols",
        default="",
        help="Comma-separated feature names or zero-based feature indices.",
    )
    budget.add_argument(
        "--clip-train-range",
        action="store_true",
        help="Clip each feature to the standardized benign-training min/max.",
    )

    threshold = parser.add_argument_group("threshold and decision boundary")
    threshold.add_argument(
        "--thresholds",
        default=None,
        help=".npy/.csv/.json vector with one squared-error threshold per feature.",
    )
    threshold.add_argument(
        "--threshold-percentile",
        type=float,
        default=None,
        help=(
            "Estimate per-feature thresholds from benign training errors. Accepts "
            "0.995 or 99.5. Used ONLY as a differentiable loss for pgd_margin/pgd_cw; "
            "it is not the operative detector."
        ),
    )
    threshold.add_argument(
        "--reference-max-targets",
        type=int,
        default=10000,
        help="Maximum benign-training targets used for threshold/KL estimation.",
    )
    threshold.add_argument(
        "--detection-window",
        type=int,
        default=1,
        help=(
            "Consecutive-timestep window `w` of the scalar detector, exactly as each "
            "model's cached_detect(): a timestep is flagged only if `w` consecutive "
            "instance scores exceed theta. Applied on the WHOLE series then restricted "
            "to selected targets. Pass the `window` tuned jointly with theta by main_eval.py."
        ),
    )
    threshold.add_argument(
        "--instance-threshold",
        type=float,
        default=None,
        help=(
            "Scalar threshold theta on the aggregated per-instance score, as PRINTED by "
            "the repository's main_eval.py on the validation set (main_eval.py:176). This "
            "reproduces the ESORICS instance-level detector and is the PREFERRED way to "
            "set theta. Takes precedence over --instance-threshold-percentile; if both "
            "are given, the two values are compared and a divergence beyond "
            "--sanity-tolerance is reported."
        ),
    )
    threshold.add_argument(
        "--instance-threshold-percentile",
        type=float,
        default=None,
        help=(
            "Recompute theta internally as np.quantile(validation_instance_errors, p), "
            "reproducing main_eval.py on the SAME validation split (AE: "
            "train_test_split(Xfull,test_size=0.2,random_state=42,shuffle=True); "
            "sequence: the arange(history,len-1) index split). Accepts 0.95 or 95. "
            "See --instance-threshold-ref to select the benign reference set."
        ),
    )
    threshold.add_argument(
        "--instance-threshold-ref",
        choices=["validation", "training"],
        default="validation",
        help=(
            "Benign reference set for --instance-threshold-percentile. 'validation' "
            "(default) reproduces main_eval.py's held-out 20%% validation split exactly. "
            "'training' uses a uniform sample of the whole training set (legacy behaviour, "
            "runs slightly strict and does NOT match main_eval.py)."
        ),
    )
    threshold.add_argument(
        "--instance-score",
        choices=["mean_mse", "max_mse"],
        default="mean_mse",
        help=(
            "Aggregation of per-feature squared errors into the per-instance score used "
            "for the detection decision AND for theta reproduction. The repository "
            "detector uses the MEAN (main_eval.py:298-299); 'max_mse' is non-standard."
        ),
    )

    sanity = parser.add_argument_group("sanity check / reproducibility")
    sanity.add_argument(
        "--sanity-check",
        action="store_true",
        help=(
            "Reproduce main_eval.py's scalar detector and print theta plus the benign/"
            "attack detection rate on the full test series, then EXIT without attacking. "
            "Run this first: if the numbers do not match main_eval.py the benchmark is "
            "not trustworthy. Requires --instance-threshold-percentile (and/or "
            "--instance-threshold)."
        ),
    )
    sanity.add_argument(
        "--sanity-tolerance",
        type=float,
        default=0.05,
        help=(
            "Relative tolerance for the theta divergence check between "
            "--instance-threshold and the validation-reproduced theta."
        ),
    )

    cw = parser.add_argument_group("CW-style margin attack")
    cw.add_argument(
        "--cw-confidence",
        type=float,
        default=0.0,
        help="Required margin beyond the detector boundary.",
    )
    cw.add_argument(
        "--cw-l2-weight",
        type=float,
        default=0.0,
        help="Optional L2 regularization weight inside the projected CW-style loss.",
    )

    kl = parser.add_argument_group("experimental residual-KL attack")
    kl.add_argument(
        "--kl-reference",
        default="auto",
        help=(
            "'auto' estimates a benign residual profile from training data; otherwise "
            "provide a .npy/.csv/.json positive vector with one value per feature."
        ),
    )
    kl.add_argument("--kl-temperature", type=float, default=1.0)
    kl.add_argument(
        "--kl-score-weight",
        type=float,
        default=1.0,
        help="Weight of detector-score alignment added to the KL objective.",
    )

    corr = parser.add_argument_group("CorrShift black-box attack")
    corr.add_argument(
        "--corrshift-rounds",
        type=int,
        default=1,
        help="Greedy correlation-guided search rounds.",
    )
    corr.add_argument(
        "--corrshift-step",
        type=float,
        default=None,
        help="Standardized step per round; defaults to --alpha.",
    )
    corr.add_argument(
        "--corrshift-top-k",
        type=int,
        default=0,
        help="Keep only k strongest anchor correlations; 0 keeps all features.",
    )
    corr.add_argument(
        "--corrshift-min-abs-corr",
        type=float,
        default=0.0,
        help="Ignore correlations with smaller absolute value.",
    )
    corr.add_argument(
        "--corrshift-range-fraction",
        type=float,
        default=None,
        help=(
            "Optional original-style feature budget: fraction * benign standardized "
            "feature range. Overrides the regular epsilon only for CorrShift."
        ),
    )

    output = parser.add_argument_group("saved outputs")
    output.add_argument(
        "--save-series",
        choices=["full", "delta", "none"],
        default="full",
        help="Whether to save the complete adversarial series, only delta, or neither.",
    )
    output.add_argument(
        "--save-raw-csv",
        action="store_true",
        help="Also save the complete adversarial series in physical units as CSV.",
    )

    args = parser.parse_args()
    if args.start is not None and args.end is None:
        parser.error("--start requires --end")
    if args.end is not None and args.start is None:
        parser.error("--end requires --start")
    if args.model_type in SEQUENCE_MODELS and (args.history is None or args.history <= 0):
        parser.error("--history must be a positive integer for CNN/GRU/LSTM")
    if args.epsilon < 0 or args.alpha < 0:
        parser.error("--epsilon and --alpha must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.margin_beta <= 0 or args.kl_temperature <= 0:
        parser.error("--margin-beta and --kl-temperature must be positive")
    if args.detection_window <= 0:
        parser.error("--detection-window must be positive")
    return args

"""The operative ESORICS detector and its calibration.

This module owns everything about the *decision* (never the attack loss):
  * per-feature threshold vectors used only as differentiable losses
    (`resolve_thresholds`, `resolve_kl_reference`);
  * the scalar instance-level threshold theta reproduced from main_eval.py's
    validation split (`reproduce_validation_theta`, `resolve_instance_threshold`);
  * the consecutive-window detector `cached_detect`, reproduced per model
    (`repo_cached_detect`), applied on the whole series then restricted to targets
    (`windowed_target_detection`).
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from .constants import POINT_MODELS, SEQUENCE_MODELS
from .context import AttackContext, ModelAdapter
from .errors import instance_detection_scores, model_errors_numpy
from .io_utils import load_train_scaled, load_vector
from .targets import valid_target_bounds


# ---------------------------------------------------------------------------
# Benign reference errors and per-feature thresholds (differentiable-loss only)
# ---------------------------------------------------------------------------


def sample_reference_indices(
    model_type: str,
    n_rows: int,
    history: Optional[int],
    target_offset: int,
    max_targets: int,
) -> np.ndarray:
    first, last = valid_target_bounds(model_type, n_rows, history, target_offset)
    indices = np.arange(first, last, dtype=np.int32)
    if max_targets > 0 and len(indices) > max_targets:
        # Uniform sampling covers the whole benign process rather than only the start.
        positions = np.linspace(0, len(indices) - 1, max_targets, dtype=np.int64)
        indices = indices[positions]
    return indices


def estimate_reference_errors(
    adapter: ModelAdapter,
    x_train: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    indices = sample_reference_indices(
        args.model_type,
        len(x_train),
        args.history,
        args.target_offset,
        args.reference_max_targets,
    )
    return model_errors_numpy(
        adapter,
        x_train,
        args.model_type,
        indices,
        args.history,
        args.target_offset,
        args.prediction_batch_size,
    )


def resolve_thresholds(
    args: argparse.Namespace,
    n_features: int,
    adapter: ModelAdapter,
    train_scaled: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if args.thresholds:
        thresholds = load_vector(args.thresholds, n_features, "threshold vector")
        if np.any(thresholds <= 0):
            raise ValueError("All squared-error thresholds must be positive.")
        return thresholds
    if args.threshold_percentile is None:
        return None
    if train_scaled is None:
        raise RuntimeError("Training data are required to estimate thresholds.")
    percentile = args.threshold_percentile
    if percentile <= 1.0:
        percentile *= 100.0
    if not 0.0 < percentile < 100.0:
        raise ValueError("--threshold-percentile must be in (0,1) or (0,100).")
    errors = estimate_reference_errors(adapter, train_scaled, args)
    thresholds = np.percentile(errors, percentile, axis=0).astype(np.float32)
    thresholds = np.maximum(thresholds, 1e-12)
    print(
        f"Estimated per-feature thresholds at percentile {percentile:.4f} "
        f"from {len(errors)} benign training targets."
    )
    return thresholds


# ---------------------------------------------------------------------------
# Scalar instance-level theta reproduced from main_eval.py
# ---------------------------------------------------------------------------


def _sequence_errors_by_lead(
    adapter: ModelAdapter,
    x_full: np.ndarray,
    leads: np.ndarray,
    history: int,
    batch_size: int,
    target_offset: int = 1,
) -> np.ndarray:
    """Squared per-feature errors for a sequence model. With target_offset=1 this
    replicates utils.reconstruction_errors_by_idxs (utils.py:61-90) exactly (input
    X[lead-history:lead], target X[lead+1]); a different offset keeps the theta error
    definition consistent with the detection path (target X[lead+offset]). Kept in
    float64 to match main_eval.py's float64 training matrix."""
    leads = np.asarray(leads, dtype=np.int64)
    errors = np.empty((len(leads), x_full.shape[1]), dtype=np.float64)
    for start in range(0, len(leads), batch_size):
        chunk = leads[start : start + batch_size]
        inputs = np.stack([x_full[lead - history : lead] for lead in chunk])
        targets = x_full[chunk + target_offset]
        preds = np.asarray(
            adapter.predict_numpy(inputs, batch_size=batch_size)
        ).reshape(len(chunk), -1)
        errors[start : start + len(chunk)] = np.square(preds - targets)
    return errors


def reproduce_validation_theta(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    scaler: Any,
    sensor_cols: Sequence[str],
    percentile_fraction: float,
    x_full: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Recompute theta EXACTLY as main_eval.py does, on the same benign validation
    split of the training data.

    Reproduces (main_eval.py):
      * AE / point models (main_eval.py:293-299):
          _, Xval, _, _ = train_test_split(Xfull, Xfull, test_size=0.2,
                                           random_state=42, shuffle=True)
          val_errors = (predict(Xval) - Xval) ** 2
          theta = np.quantile(val_errors.mean(axis=1), percentile)
      * Sequence models (main_eval.py:287-289):
          all_idxs = np.arange(history, len(Xfull) - 1)
          _, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2,
                                               random_state=42, shuffle=True)
          val_errors = reconstruction_errors_by_idxs(model, Xfull, val_idxs, history)
          theta = np.quantile(val_errors.mean(axis=1), percentile)

    Xfull is the FULL, order-preserved, standardized training matrix. load_train_scaled
    applies the SAME fitted StandardScaler that main_eval.py refits on the identical
    training data, so the two Xfull matrices coincide (data_loader.py:83-93).
    """
    from sklearn.model_selection import train_test_split  # repository dependency

    if x_full is None:
        x_full = load_train_scaled(args.dataset, scaler, sensor_cols)
    # Keep float64 (main_eval.py uses the float64 fit_transform matrix); Keras still
    # computes predictions in float32, so this only sharpens the error/quantile math.
    x_full = np.asarray(x_full, dtype=np.float64)

    if args.model_type in POINT_MODELS:
        _, x_val, _, _ = train_test_split(
            x_full, x_full, test_size=0.2, random_state=42, shuffle=True
        )
        preds = np.asarray(
            adapter.predict_numpy(x_val, batch_size=args.prediction_batch_size)
        ).reshape(len(x_val), -1)
        val_errors = np.square(preds - x_val)
    else:
        history = int(args.history)
        all_idxs = np.arange(history, len(x_full) - 1)
        _, val_idxs, _, _ = train_test_split(
            all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True
        )
        val_errors = _sequence_errors_by_lead(
            adapter,
            x_full,
            val_idxs,
            history,
            args.prediction_batch_size,
            args.target_offset,
        )

    val_instance = instance_detection_scores(val_errors, args.instance_score)
    theta = float(np.quantile(val_instance, percentile_fraction))
    return theta, val_instance


def _normalize_percentile_fraction(value: float, name: str) -> float:
    fraction = value / 100.0 if value > 1.0 else value
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"{name} must be in (0,1) or (0,100).")
    return fraction


def resolve_instance_threshold(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    scaler: Any,
    sensor_cols: Sequence[str],
    train_scaled: Optional[np.ndarray],
) -> Tuple[Optional[float], str, Optional[float]]:
    """Resolve the scalar instance-level threshold theta of the ESORICS detector.

    Returns (theta, source, percentile_fraction). Priority:
      1. --instance-threshold: the value PRINTED by main_eval.py (preferred).
      2. --instance-threshold-percentile with --instance-threshold-ref validation:
         theta recomputed on the exact main_eval.py validation split.
      3. ...with --instance-threshold-ref training: legacy training-sample estimate.
    When both a CLI value and a percentile are supplied, the two thetas are compared
    and a relative divergence beyond --sanity-tolerance is reported.
    """
    cli_theta: Optional[float] = None
    if args.instance_threshold is not None:
        if args.instance_threshold <= 0:
            raise ValueError("--instance-threshold must be positive.")
        cli_theta = float(args.instance_threshold)

    percentile_fraction: Optional[float] = None
    reproduced_theta: Optional[float] = None
    if args.instance_threshold_percentile is not None:
        percentile_fraction = _normalize_percentile_fraction(
            args.instance_threshold_percentile, "--instance-threshold-percentile"
        )
        if args.instance_threshold_ref == "validation":
            if args.model_type in SEQUENCE_MODELS and args.target_offset != 1:
                print(
                    "WARNING: --instance-threshold-ref validation reproduces main_eval.py, "
                    "which is defined only for the repository's +1 alignment, but "
                    f"--target-offset is {args.target_offset}. Theta is computed with your "
                    "offset for consistency with detection, but will NOT match main_eval.py."
                )
            reproduced_theta, _ = reproduce_validation_theta(
                args, adapter, scaler, sensor_cols, percentile_fraction, train_scaled
            )
            print(
                f"Reproduced validation theta = {reproduced_theta:.8f} at percentile "
                f"{percentile_fraction:.5f} ({args.instance_score}); mirrors "
                f"main_eval.py's held-out validation split."
            )
        else:
            if train_scaled is None:
                raise RuntimeError(
                    "Benign reference data are required to estimate the instance threshold."
                )
            errors = estimate_reference_errors(adapter, train_scaled, args)
            scores = instance_detection_scores(errors, args.instance_score)
            reproduced_theta = float(np.quantile(scores, percentile_fraction))
            print(
                f"Estimated instance theta = {reproduced_theta:.8f} at percentile "
                f"{percentile_fraction:.5f} ({args.instance_score}) from {len(scores)} "
                f"TRAINING instances. NOTE: does not reproduce main_eval.py; pass "
                f"--instance-threshold-ref validation for the faithful value."
            )

    if cli_theta is not None and reproduced_theta is not None:
        denom = max(abs(reproduced_theta), 1e-12)
        rel = abs(cli_theta - reproduced_theta) / denom
        detail = (
            f"--instance-threshold={cli_theta:.8f} vs reproduced={reproduced_theta:.8f} "
            f"(relative {rel:.3%}, tolerance {args.sanity_tolerance:.3%})"
        )
        if rel > args.sanity_tolerance:
            print(f"WARNING: theta divergence {detail}. The supplied theta may not match "
                  f"this model / validation split / percentile / aggregation.")
        else:
            print(f"OK: theta agreement {detail}.")

    if cli_theta is not None:
        source = (
            "cli-value(checked-vs-validation-percentile)"
            if reproduced_theta is not None and args.instance_threshold_ref == "validation"
            else "cli-value"
        )
        return cli_theta, source, percentile_fraction
    if reproduced_theta is not None:
        source = f"{args.instance_threshold_ref}-percentile"
        return reproduced_theta, source, percentile_fraction
    return None, "none", None


def resolve_kl_reference(
    args: argparse.Namespace,
    n_features: int,
    adapter: ModelAdapter,
    train_scaled: Optional[np.ndarray],
    thresholds: Optional[np.ndarray],
    requested_attacks: Iterable[str],
) -> Optional[np.ndarray]:
    if "pgd_kl" not in set(requested_attacks):
        return None

    if args.kl_reference != "auto":
        values = load_vector(args.kl_reference, n_features, "KL reference")
    else:
        if train_scaled is None:
            raise RuntimeError("Training data are required for --kl-reference auto.")
        errors = estimate_reference_errors(adapter, train_scaled, args)
        if thresholds is not None:
            values = np.mean(errors / np.maximum(thresholds[None, :], 1e-12), axis=0)
        else:
            values = np.mean(errors, axis=0)
        print(f"Estimated benign KL residual profile from {len(errors)} training targets.")

    values = np.maximum(np.asarray(values, dtype=np.float32), 1e-12)
    # Same transformation used by residual_distribution_tf, then normalize.
    transformed = np.log1p(values) / args.kl_temperature
    transformed -= np.max(transformed)
    probs = np.exp(transformed)
    probs /= np.sum(probs)
    return probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Consecutive-window detector (per-model cached_detect) applied on the series
# ---------------------------------------------------------------------------


def repo_cached_detect(
    instance_scores: np.ndarray, theta: float, window: int, model_type: str = "AE"
) -> np.ndarray:
    """Replicate <model>.cached_detect, returning a boolean array aligned 1:1 with
    ``instance_scores``. Point-wise detection is ``instance_score > theta``.

    For ``window>1`` the repository detectors do NOT all agree:

    * AE / CNN / LSTM / DNN (detector/autoencoder.py:160-170, cnn.py:242-254,
      lstm.py:245-266 -- its backfill is commented out, dnn.py:235-247): centered
      ``np.convolve(det, ones(w), 'same') // w`` -- a timestep is flagged only if its
      full ``w``-long neighbourhood is above theta;
    * GRU (detector/gru.py:241-262): ``'full'`` convolution PLUS a backfill loop
      (``fill[idx-w:idx]=1`` for every flagged idx), then main_eval.py aligns it back
      with ``Yhat[window-1:]`` (main_eval.py:52). GRU therefore flags a much larger
      (trailing + backfilled) region than the 'same' models.

    We reproduce each exactly so the benchmark's detection/attack-success numbers match
    main_eval.py per model. This MUST be applied on the whole series and only afterwards
    restricted to the selected targets, or the window counter resets spuriously.
    """
    det = np.asarray(instance_scores, dtype=np.float64) > float(theta)
    if window <= 1:
        return det.astype(bool)
    if model_type == "GRU":
        # detector/gru.py:250-260 verbatim: 'full' + backfill, then main_eval.py:52
        # truncation Yhat[window-1:] realigns it 1:1 with the instance positions.
        conv = np.convolve(det.astype(np.float64), np.ones(window), "full") // window
        filled = conv.copy()
        for idx in np.flatnonzero(conv):
            filled[idx - window : idx] = 1
        return filled[window - 1 :].astype(bool)
    conv = np.convolve(det.astype(np.float64), np.ones(window), "same") // window
    return conv.astype(bool)


def series_instance_scores(
    ctx: AttackContext, x_series: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    errors = model_errors_numpy(
        ctx.adapter,
        x_series,
        ctx.args.model_type,
        indices,
        ctx.args.history,
        ctx.args.target_offset,
        ctx.args.prediction_batch_size,
    )
    return instance_detection_scores(errors, ctx.instance_score_kind)


def windowed_target_detection(
    ctx: AttackContext, x_series: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run the scalar theta+window detector over the ENTIRE valid series, then restrict
    to the selected targets. Returns (point_flags, window_flags) aligned to
    ctx.target_indices, or (None, None) if no theta is available."""
    if ctx.instance_threshold is None:
        return None, None
    first, last = valid_target_bounds(
        ctx.args.model_type, len(x_series), ctx.args.history, ctx.args.target_offset
    )
    all_indices = np.arange(first, last, dtype=np.int64)
    inst = series_instance_scores(ctx, x_series, all_indices)
    point_full = inst > float(ctx.instance_threshold)
    window_full = repo_cached_detect(
        inst, ctx.instance_threshold, ctx.detection_window, ctx.args.model_type
    )
    positions = ctx.target_indices.astype(np.int64) - first
    return point_full[positions], window_full[positions]


def hard_detection(
    instance_scores: np.ndarray, instance_threshold: Optional[float]
) -> Optional[np.ndarray]:
    """Point-wise instance-level detection: a single scalar threshold on the
    aggregated per-instance score. The consecutive-window requirement is applied
    separately by repo_cached_detect() on the full series."""
    if instance_threshold is None:
        return None
    return np.asarray(instance_scores, dtype=np.float64) > float(instance_threshold)

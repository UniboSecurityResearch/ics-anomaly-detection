"""Assemble a fully-prepared AttackContext from parsed CLI args (ex prepare_context).

Loads model/scaler/test data, selects targets, builds the modification mask and the
budget, resolves thresholds/KL-reference/theta, and precomputes the clean-series
detector decision once so every attack/goal reuses it.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf

from data_loader import load_test_data
from .attacks import ATTACKS
from .constants import POINT_MODELS
from .context import AttackContext
from .detector import (
    resolve_instance_threshold,
    resolve_kl_reference,
    resolve_thresholds,
    windowed_target_detection,
)
from .io_utils import load_epsilon, load_model_adapter, load_scaler, load_train_scaled
from .targets import (
    build_modification_mask,
    parse_protected_cols,
    select_target_indices,
    valid_target_bounds,
)


def build_context(args: argparse.Namespace) -> AttackContext:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if not tf.executing_eagerly():
        raise RuntimeError("TensorFlow eager execution is required.")

    adapter = load_model_adapter(args.model_path)
    scaler = load_scaler(args.dataset)
    x_test, labels, sensor_cols = load_test_data(
        args.dataset, scaler=scaler, verbose=True
    )
    x_test = np.asarray(x_test, dtype=np.float32)
    labels = np.asarray(labels)
    sensor_cols = [str(x) for x in sensor_cols]
    n_rows, n_features = x_test.shape

    if args.model_type in POINT_MODELS:
        args.history = None
        args.scope = "target"

    if args.model_type == "DNN":
        print(
            "WARNING: this suite treats DNN as a point reconstruction model "
            "(error = (model(X[t]) - X[t])^2). The pwwl repository instead trains DNN as "
            "a 1-step forecaster and main_eval.py scores it through the SEQUENCE branch "
            "(input X[lead-1], target X[lead+1]; detector/dnn.py, main_eval.py:287-289). "
            "The two error definitions differ, so theta may NOT reproduce main_eval.py. "
            "RUN --sanity-check FIRST: if the detection rates diverge, your DNN must be "
            "attacked as a sequence model (this suite currently supports CNN/GRU/LSTM for "
            "that). AE/CNN/GRU/LSTM are unaffected."
        )

    target_indices = select_target_indices(args, labels, n_rows)
    protected = parse_protected_cols(args.protected_cols, sensor_cols)
    modification_mask = build_modification_mask(
        args.model_type,
        n_rows,
        n_features,
        target_indices,
        args.history,
        args.target_offset,
        args.scope,
        protected,
    )
    if not np.any(modification_mask):
        raise ValueError("Modification mask is empty; no cell can be attacked.")

    epsilon = load_epsilon(args, n_features, scaler)
    if protected:
        epsilon[list(protected)] = 0.0

    requested_attacks = list(ATTACKS) if args.attack == "all" else [args.attack]
    need_train = (
        args.clip_train_range
        or args.threshold_percentile is not None
        or args.instance_threshold_percentile is not None
        or args.sanity_check
        or any(ATTACKS[a].requires_train_data for a in requested_attacks)
        or ("pgd_kl" in requested_attacks and args.kl_reference == "auto")
    )
    train_scaled = load_train_scaled(args.dataset, scaler, sensor_cols) if need_train else None

    lower_domain = None
    upper_domain = None
    if args.clip_train_range:
        assert train_scaled is not None
        lower_domain = np.min(train_scaled, axis=0).astype(np.float32)
        upper_domain = np.max(train_scaled, axis=0).astype(np.float32)

    thresholds = resolve_thresholds(args, n_features, adapter, train_scaled)
    if args.score in {"threshold_max", "threshold_smooth"} and thresholds is None:
        raise ValueError(f"--score {args.score} requires thresholds.")

    kl_reference = resolve_kl_reference(
        args, n_features, adapter, train_scaled, thresholds, requested_attacks
    )

    instance_threshold, instance_threshold_source, detection_percentile = (
        resolve_instance_threshold(args, adapter, scaler, sensor_cols, train_scaled)
    )

    first_valid, _ = valid_target_bounds(
        args.model_type, n_rows, args.history, args.target_offset
    )

    context = AttackContext(
        args=args,
        adapter=adapter,
        x_test=x_test,
        labels=labels,
        sensor_cols=sensor_cols,
        scaler=scaler,
        target_indices=target_indices,
        modification_mask=modification_mask,
        epsilon=epsilon,
        thresholds=thresholds,
        lower_domain=lower_domain,
        upper_domain=upper_domain,
        train_scaled=train_scaled,
        kl_reference=kl_reference,
        instance_threshold=instance_threshold,
        instance_score_kind=args.instance_score,
        detection_window=int(args.detection_window),
        instance_threshold_source=instance_threshold_source,
        detection_percentile=detection_percentile,
        first_valid_index=int(first_valid),
        clean_detect_point=None,
        clean_detect_window=None,
        protected=protected,
    )

    # Precompute the clean-series detector decision ONCE (identical for every
    # attack/goal): the scalar theta+window detector over the whole series,
    # restricted to the selected targets.
    if instance_threshold is not None:
        context.clean_detect_point, context.clean_detect_window = (
            windowed_target_detection(context, x_test)
        )
    return context

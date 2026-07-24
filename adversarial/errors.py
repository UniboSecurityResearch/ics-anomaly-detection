"""Model inputs, per-feature squared errors and (differentiable) detector scores.

Everything here is model-family aware via `model_type` and `target_offset`:
point models compare model(X[t]) with X[t]; sequence models compare
model(X[t-offset-h : t-offset]) with X[t]. The tf variants are used inside the
white-box gradient tape; the numpy variants for evaluation and black-box search.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import tensorflow as tf

from .constants import POINT_MODELS
from .context import ModelAdapter


def history_index_matrix_np(
    target_indices: np.ndarray, history: int, target_offset: int
) -> np.ndarray:
    lead = target_indices.astype(np.int64) - target_offset
    starts = lead[:, None] - history
    offsets = np.arange(history, dtype=np.int64)[None, :]
    return starts + offsets


def history_index_matrix_tf(
    target_indices: np.ndarray, history: int, target_offset: int
) -> tf.Tensor:
    lead = tf.convert_to_tensor(target_indices - target_offset, dtype=tf.int32)
    offsets = tf.range(history, dtype=tf.int32)[tf.newaxis, :]
    return lead[:, tf.newaxis] - history + offsets


def normalize_model_output_tf(output: Any) -> tf.Tensor:
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise RuntimeError("White-box model must return exactly one output tensor.")
        output = output[0]
    output = tf.cast(output, tf.float32)
    if output.shape.rank != 2:
        output = tf.reshape(output, [tf.shape(output)[0], -1])
    return output


def model_errors_tf(
    model: tf.keras.Model,
    x_series: tf.Tensor,
    model_type: str,
    target_indices: np.ndarray,
    history: Optional[int],
    target_offset: int,
) -> tf.Tensor:
    target_idx = tf.convert_to_tensor(target_indices, dtype=tf.int32)
    observed = tf.cast(tf.gather(x_series, target_idx), tf.float32)
    if model_type in POINT_MODELS:
        predicted = model(observed, training=False)
    else:
        assert history is not None
        hist_idx = history_index_matrix_tf(target_indices, history, target_offset)
        histories = tf.gather(x_series, hist_idx)
        predicted = model(histories, training=False)
    predicted = normalize_model_output_tf(predicted)
    if observed.shape.rank != 2:
        observed = tf.reshape(observed, [tf.shape(observed)[0], -1])
    tf.debugging.assert_equal(
        tf.shape(predicted),
        tf.shape(observed),
        message="Model output and observed target must have identical shapes.",
    )
    return tf.square(predicted - observed)


def model_errors_numpy(
    adapter: ModelAdapter,
    x_series: np.ndarray,
    model_type: str,
    target_indices: np.ndarray,
    history: Optional[int],
    target_offset: int,
    batch_size: int,
) -> np.ndarray:
    all_errors: List[np.ndarray] = []
    for start in range(0, len(target_indices), batch_size):
        idx = target_indices[start : start + batch_size]
        observed = x_series[idx]
        if model_type in POINT_MODELS:
            model_input = observed
        else:
            assert history is not None
            hist_idx = history_index_matrix_np(idx, history, target_offset)
            model_input = x_series[hist_idx]
        predicted = adapter.predict_numpy(model_input, batch_size=batch_size)
        predicted = np.asarray(predicted, dtype=np.float32).reshape(len(idx), -1)
        observed = np.asarray(observed, dtype=np.float32).reshape(len(idx), -1)
        if predicted.shape != observed.shape:
            raise RuntimeError(
                f"Model output shape {predicted.shape} does not match target shape "
                f"{observed.shape}. Check --model-type, --history and --target-offset."
            )
        all_errors.append(np.square(predicted - observed))
    return np.concatenate(all_errors, axis=0)


def threshold_margin_tf(
    errors: tf.Tensor,
    thresholds: tf.Tensor,
    smooth: bool,
    beta: float,
) -> tf.Tensor:
    normalized = errors / tf.maximum(thresholds[tf.newaxis, :], 1e-12) - 1.0
    if smooth:
        return tf.reduce_logsumexp(beta * normalized, axis=1) / beta
    return tf.reduce_max(normalized, axis=1)


def detector_score_tf(
    errors: tf.Tensor,
    score_kind: str,
    thresholds: Optional[np.ndarray],
    beta: float,
) -> tf.Tensor:
    if score_kind == "mean_mse":
        return tf.reduce_mean(errors, axis=1)
    if score_kind == "max_mse":
        return tf.reduce_max(errors, axis=1)
    if thresholds is None:
        raise ValueError(f"{score_kind} requires thresholds.")
    threshold_tensor = tf.convert_to_tensor(thresholds, dtype=tf.float32)
    return threshold_margin_tf(
        errors,
        threshold_tensor,
        smooth=(score_kind == "threshold_smooth"),
        beta=beta,
    )


def detector_score_numpy(
    errors: np.ndarray,
    score_kind: str,
    thresholds: Optional[np.ndarray],
    beta: float,
) -> np.ndarray:
    if score_kind == "mean_mse":
        return np.mean(errors, axis=1)
    if score_kind == "max_mse":
        return np.max(errors, axis=1)
    if thresholds is None:
        raise ValueError(f"{score_kind} requires thresholds.")
    normalized = errors / np.maximum(thresholds[None, :], 1e-12) - 1.0
    if score_kind == "threshold_max":
        return np.max(normalized, axis=1)
    # Numerically stable smooth maximum.
    z = beta * normalized
    z_max = np.max(z, axis=1, keepdims=True)
    return ((z_max[:, 0] + np.log(np.sum(np.exp(z - z_max), axis=1))) / beta).astype(
        np.float32
    )


def instance_detection_scores(errors: np.ndarray, kind: str) -> np.ndarray:
    """Aggregate per-feature squared errors into a single per-instance score,
    replicating the ESORICS repository's instance-level detector."""
    if kind == "mean_mse":
        return np.mean(errors, axis=1)
    if kind == "max_mse":
        return np.max(errors, axis=1)
    raise ValueError(f"Unsupported instance score: {kind}")


def residual_distribution_tf(
    errors: tf.Tensor,
    thresholds: Optional[np.ndarray],
    temperature: float,
) -> tf.Tensor:
    if thresholds is not None:
        scaled = errors / tf.maximum(
            tf.convert_to_tensor(thresholds, tf.float32)[tf.newaxis, :], 1e-12
        )
    else:
        mean = tf.reduce_mean(errors, axis=1, keepdims=True)
        scaled = errors / tf.maximum(mean, 1e-12)
    # log1p limits extreme dominance while preserving ordering.
    return tf.nn.softmax(tf.math.log1p(tf.maximum(scaled, 0.0)) / temperature, axis=1)


def kl_divergence_tf(reference: tf.Tensor, candidate: tf.Tensor) -> tf.Tensor:
    reference = tf.clip_by_value(reference, 1e-8, 1.0)
    candidate = tf.clip_by_value(candidate, 1e-8, 1.0)
    return tf.reduce_sum(reference * tf.math.log(reference / candidate), axis=1)

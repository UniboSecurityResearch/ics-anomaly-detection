"""L-infinity / domain projection and sparse-gradient helpers shared by attacks."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


def project_numpy(
    x_candidate: np.ndarray,
    x_original: np.ndarray,
    epsilon: np.ndarray,
    modification_mask: np.ndarray,
    lower_domain: Optional[np.ndarray],
    upper_domain: Optional[np.ndarray],
) -> np.ndarray:
    lower = x_original - epsilon[None, :]
    upper = x_original + epsilon[None, :]
    if lower_domain is not None:
        lower = np.maximum(lower, lower_domain[None, :])
    if upper_domain is not None:
        upper = np.minimum(upper, upper_domain[None, :])
    clipped = np.clip(x_candidate, lower, upper)
    return x_original + (clipped - x_original) * modification_mask


def build_tf_bounds(
    x0: tf.Tensor,
    epsilon: np.ndarray,
    lower_domain: Optional[np.ndarray],
    upper_domain: Optional[np.ndarray],
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    eps = tf.convert_to_tensor(epsilon, dtype=tf.float32)[tf.newaxis, :]
    lower = x0 - eps
    upper = x0 + eps
    if lower_domain is not None:
        lower = tf.maximum(
            lower, tf.convert_to_tensor(lower_domain, tf.float32)[tf.newaxis, :]
        )
    if upper_domain is not None:
        upper = tf.minimum(
            upper, tf.convert_to_tensor(upper_domain, tf.float32)[tf.newaxis, :]
        )
    return eps, lower, upper


def keep_top_k_per_row(gradient: tf.Tensor, modification_mask: tf.Tensor, k: int) -> tf.Tensor:
    if k <= 0:
        return tf.zeros_like(gradient)
    n_features = gradient.shape[-1]
    if n_features is None:
        raise RuntimeError("Static feature dimension is required for pgd_topk.")
    k = min(k, int(n_features))
    masked_abs = tf.where(
        modification_mask > 0,
        tf.abs(gradient),
        tf.fill(tf.shape(gradient), tf.constant(-1e30, gradient.dtype)),
    )
    top_indices = tf.math.top_k(masked_abs, k=k, sorted=False).indices
    sparse_mask = tf.reduce_max(
        tf.one_hot(top_indices, depth=int(n_features), dtype=gradient.dtype), axis=1
    )
    # Rows with no modifiable feature have only -1e30 entries; explicitly zero them.
    row_allowed = tf.cast(
        tf.reduce_any(modification_mask > 0, axis=1, keepdims=True), gradient.dtype
    )
    return gradient * sparse_mask * modification_mask * row_allowed

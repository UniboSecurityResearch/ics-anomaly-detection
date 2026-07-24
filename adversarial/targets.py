"""Label inference, target-timestep selection and modification masks.

These functions turn the repository test labels and CLI selection options into the
concrete set of timesteps an attack may target and the boolean mask of cells it may
modify. They are model-family aware (point vs sequence) but attack-agnostic.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Set, Tuple

import numpy as np

from .constants import POINT_MODELS


def infer_attack_labels(dataset: str, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    text = np.char.lower(labels.astype(str))
    if dataset.startswith("SWAT"):
        return np.char.find(text, "attack") >= 0

    numeric: Optional[np.ndarray]
    try:
        numeric = labels.astype(float)
    except (TypeError, ValueError):
        numeric = None

    if dataset == "BATADAL" and numeric is not None:
        return numeric > 0
    if dataset.startswith("WADI") and numeric is not None:
        unique = set(np.unique(numeric).tolist())
        if unique.issubset({-1.0, 1.0}):
            return numeric < 0
        return numeric > 0
    return (
        (np.char.find(text, "attack") >= 0)
        | (np.char.find(text, "anomal") >= 0)
        | (text == "true")
    )


def parse_protected_cols(spec: str, sensor_cols: Sequence[str]) -> Set[int]:
    protected: Set[int] = set()
    if not spec.strip():
        return protected
    name_to_idx = {str(name): i for i, name in enumerate(sensor_cols)}
    for token in (x.strip() for x in spec.split(",") if x.strip()):
        if token in name_to_idx:
            protected.add(name_to_idx[token])
            continue
        try:
            idx = int(token)
        except ValueError as exc:
            raise ValueError(f"Unknown protected feature: {token}") from exc
        if idx < 0 or idx >= len(sensor_cols):
            raise ValueError(f"Protected feature index out of range: {idx}")
        protected.add(idx)
    return protected


def valid_target_bounds(
    model_type: str, n_rows: int, history: Optional[int], target_offset: int
) -> Tuple[int, int]:
    if model_type in POINT_MODELS:
        return 0, n_rows
    assert history is not None
    return history + target_offset, n_rows


def select_target_indices(
    args: argparse.Namespace, labels: np.ndarray, n_rows: int
) -> np.ndarray:
    first, last = valid_target_bounds(
        args.model_type, n_rows, args.history, args.target_offset
    )
    if args.start is not None:
        start = max(first, args.start)
        end = min(last, args.end)
        if start >= end:
            raise ValueError(f"Empty target interval after clipping: [{start}, {end}).")
        indices = np.arange(start, end, dtype=np.int32)
    else:
        indices = np.arange(first, last, dtype=np.int32)
        if args.selection != "all":
            attack_mask = infer_attack_labels(args.dataset, labels)
            desired = attack_mask if args.selection == "attack" else ~attack_mask
            indices = indices[desired[indices]]
    if indices.size == 0:
        raise ValueError("No target timesteps matched the requested selection.")
    if args.max_targets > 0 and indices.size > args.max_targets:
        indices = indices[: args.max_targets]
    return indices


def build_modification_mask(
    model_type: str,
    n_rows: int,
    n_features: int,
    target_indices: np.ndarray,
    history: Optional[int],
    target_offset: int,
    scope: str,
    protected: Set[int],
) -> np.ndarray:
    mask = np.zeros((n_rows, n_features), dtype=np.float32)
    allowed = np.ones(n_features, dtype=np.float32)
    if protected:
        allowed[list(protected)] = 0.0

    if model_type in POINT_MODELS:
        mask[target_indices] = allowed
        return mask

    assert history is not None
    if scope in {"target", "both"}:
        mask[target_indices] = allowed
    if scope in {"history", "both"}:
        for target_idx in target_indices:
            lead = int(target_idx) - target_offset
            mask[lead - history : lead] = allowed
    return mask

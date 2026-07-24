"""Selection of the controllable feature subset C and its binary mask m.

Supports explicit names, explicit indices, a fraction, a fixed count k (reproducible
random subset) and loading from a config. The mask is a boolean vector of length
n_features with True on controllable features.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np


def mask_from_indices(indices: Sequence[int], n_features: int) -> np.ndarray:
    mask = np.zeros(n_features, dtype=bool)
    idx = np.asarray(list(indices), dtype=int)
    if idx.size and (idx.min() < 0 or idx.max() >= n_features):
        raise ValueError(f"Controlled feature index out of range [0, {n_features}).")
    mask[idx] = True
    return mask


def resolve_controlled_indices(
    sensor_cols: Sequence[str],
    *,
    names: Optional[Sequence[str]] = None,
    indices: Optional[Sequence[int]] = None,
    fraction: Optional[float] = None,
    k: Optional[int] = None,
    random: bool = False,
    seed: int = 42,
) -> List[int]:
    """Return the sorted list of controllable feature indices.

    Exactly one selector should be provided (names | indices | fraction | k). When
    ``random`` is False and a fraction/k is given, the first features (by column
    order) are taken for reproducibility; with ``random`` a seeded RNG samples them.
    """
    n = len(sensor_cols)
    name_to_idx = {str(c): i for i, c in enumerate(sensor_cols)}

    if names:
        out = []
        for token in names:
            token = str(token).strip()
            if token in name_to_idx:
                out.append(name_to_idx[token])
            else:
                try:
                    idx = int(token)
                except ValueError as exc:
                    raise ValueError(f"Unknown controlled feature: {token}") from exc
                if not 0 <= idx < n:
                    raise ValueError(f"Controlled feature index out of range: {idx}")
                out.append(idx)
        return sorted(set(out))

    if indices is not None:
        return sorted(set(int(i) for i in indices))

    count: Optional[int] = None
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        count = max(1, int(round(fraction * n)))
    elif k is not None:
        if not 0 < k <= n:
            raise ValueError(f"k must be in (0, {n}].")
        count = int(k)

    if count is None:
        raise ValueError(
            "Provide one of: names, indices, fraction, k (or use unconstrained)."
        )

    if random:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(n, size=count, replace=False).tolist())
    return list(range(count))


def full_mask(n_features: int) -> np.ndarray:
    """Unconstrained mask m = 1 (all features controllable)."""
    return np.ones(n_features, dtype=bool)


def describe_mask(mask: np.ndarray, sensor_cols: Sequence[str]) -> str:
    idx = np.flatnonzero(np.asarray(mask).astype(bool))
    names = [str(sensor_cols[i]) for i in idx]
    return f"|C|={len(idx)} / {len(mask)} : {names}"

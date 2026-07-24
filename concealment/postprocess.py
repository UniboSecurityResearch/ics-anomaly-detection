"""Domain post-processing of the adversarial vector.

Everything is expressed in the detector's STANDARDIZED space and every statistic is
derived from the attacker's benign TRAINING data (or explicit config), NEVER from the
test set (Erba et al. §; prompt §10). Only controllable features are ever modified;
non-controllable features are left byte-for-byte identical to the anomalous input.

  * discrete/binary/categorical features (few distinct values in training) are
    projected onto the nearest allowed (standardized) value;
  * continuous features are optionally clipped to the training [min, max].
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


class DomainConstraints:
    def __init__(
        self,
        allowed_values: Dict[int, np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        clip_continuous: bool = True,
    ) -> None:
        self.allowed_values = allowed_values  # feature index -> sorted 1D array
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.clip_continuous = clip_continuous

    @classmethod
    def from_training(
        cls,
        train_scaled: np.ndarray,
        discrete_max_unique: int = 10,
        clip_continuous: bool = True,
        explicit_allowed: Optional[Dict[int, Sequence[float]]] = None,
        explicit_bounds: Optional[Dict[int, Sequence[float]]] = None,
    ) -> "DomainConstraints":
        """Derive constraints from benign training data (standardized).

        A feature with <= ``discrete_max_unique`` distinct training values (e.g. the
        BATADAL STATUS_* binaries) is treated as discrete; others as continuous.
        Explicit overrides take precedence.
        """
        x = np.asarray(train_scaled, dtype=np.float64)
        n_features = x.shape[1]
        allowed: Dict[int, np.ndarray] = {}
        lower = np.min(x, axis=0)
        upper = np.max(x, axis=0)

        for j in range(n_features):
            uniques = np.unique(x[:, j])
            if len(uniques) <= discrete_max_unique:
                allowed[j] = uniques

        if explicit_allowed:
            for j, vals in explicit_allowed.items():
                allowed[int(j)] = np.sort(np.asarray(vals, dtype=np.float64))
        if explicit_bounds:
            for j, (lo, hi) in explicit_bounds.items():
                lower[int(j)] = float(lo)
                upper[int(j)] = float(hi)

        return cls(allowed, lower, upper, clip_continuous=clip_continuous)

    def discrete_features(self) -> List[int]:
        return sorted(self.allowed_values)

    def apply(
        self, x_adv: np.ndarray, x_anom: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Project controllable features to the allowed domain. Non-controllable
        features are restored from ``x_anom`` to guarantee they stay untouched."""
        mask = np.asarray(mask).astype(bool)
        out = np.asarray(x_adv, dtype=np.float64).copy()
        controllable = np.flatnonzero(mask)

        for j in controllable:
            if j in self.allowed_values:
                values = self.allowed_values[j]           # (V,)
                # nearest allowed value per row
                diffs = np.abs(out[:, j][:, None] - values[None, :])
                out[:, j] = values[np.argmin(diffs, axis=1)]
            elif self.clip_continuous:
                out[:, j] = np.clip(out[:, j], self.lower[j], self.upper[j])

        # Hard guarantee: untouched features are exactly the anomalous originals.
        non_controllable = ~mask
        out[:, non_controllable] = np.asarray(x_anom, dtype=np.float64)[:, non_controllable]
        return out

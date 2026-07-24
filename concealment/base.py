"""Common interface for concealment / evasion attacks.

Threat model (Erba et al., "Constrained Concealment Attacks against
Reconstruction-Based Anomaly Detectors in ICS"): the attack receives samples that
already come from a compromised (anomalous) process, ``x_anom``, and must produce
``x_adv`` to feed the detector in their place. It never synthesizes the physical
attack and never touches the detector's weights/gradients/threshold/scores.

Every attack implements ``fit`` (on attacker-observable NORMAL data only),
``transform`` (on anomalous samples) and returns an ``AttackResult`` bundling the
original vector, the manipulated vector, the perturbation, the controllable-feature
mask and metadata -- so replay and autoencoder attacks are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AttackResult:
    """Result of a concealment attack.

    All arrays are in the detector's standardized feature space and share the same
    shape ``(n_samples, n_features)`` (one row per attacked timestep).
    """

    original: np.ndarray          # x_anom : the true compromised vectors
    adversarial: np.ndarray       # x_adv  : what is fed to the detector
    perturbation: np.ndarray      # x_adv - x_anom
    feature_mask: np.ndarray      # (n_features,) 1 = controllable, 0 = untouched
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.original = np.asarray(self.original, dtype=np.float64)
        self.adversarial = np.asarray(self.adversarial, dtype=np.float64)
        self.perturbation = np.asarray(self.perturbation, dtype=np.float64)
        self.feature_mask = np.asarray(self.feature_mask).astype(bool)


def combine_masked(
    controllable_output: np.ndarray, x_anom: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """x_adv = m ⊙ output + (1-m) ⊙ x_anom.

    Untouched features (mask == 0) are copied EXACTLY from the anomalous input.
    """
    mask = np.asarray(mask).astype(bool)
    x_anom = np.asarray(x_anom, dtype=np.float64)
    out = np.asarray(controllable_output, dtype=np.float64)
    x_adv = x_anom.copy()
    x_adv[:, mask] = out[:, mask]
    return x_adv


def assert_mask_preserved(
    x_adv: np.ndarray, x_anom: np.ndarray, mask: np.ndarray, atol: float = 1e-6
) -> None:
    """Fail loudly if any NON-controllable feature was modified. This is the core
    constraint of partially/fully constrained attacks."""
    mask = np.asarray(mask).astype(bool)
    if not np.any(~mask):
        return
    max_leak = float(np.max(np.abs(x_adv[:, ~mask] - x_anom[:, ~mask])))
    if max_leak > atol:
        raise AssertionError(
            f"Constraint violated: non-controllable features changed by up to "
            f"{max_leak:.3e} (> {atol:.1e}). Untouched features must equal the "
            f"original anomalous values."
        )


class ConcealmentAttack(ABC):
    """Base class. Subclasses are interchangeable in the experimental pipeline."""

    name: str = "base"

    def fit(self, normal_data: np.ndarray, **kwargs) -> "ConcealmentAttack":
        """Learn from attacker-observable NORMAL data only. Default: no-op."""
        return self

    @abstractmethod
    def transform(
        self,
        anomalous_data: np.ndarray,
        feature_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> AttackResult:
        """Conceal the anomalous samples, respecting ``feature_mask`` (1 = controllable).
        ``feature_mask=None`` means fully unconstrained (all features controllable)."""

    def fit_transform(
        self,
        normal_data: np.ndarray,
        anomalous_data: np.ndarray,
        feature_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> AttackResult:
        self.fit(normal_data, feature_mask=feature_mask, **kwargs)
        return self.transform(anomalous_data, feature_mask=feature_mask, **kwargs)

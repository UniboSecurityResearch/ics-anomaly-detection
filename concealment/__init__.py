"""Concealment / evasion attacks against reconstruction-based ICS anomaly detectors.

Methodology: Erba et al., "Constrained Concealment Attacks against Reconstruction-Based
Anomaly Detectors in Industrial Control Systems", and "On the Practical Realization of
Evasion Attacks for Industrial Control Systems".

The attacks operate on already-anomalous samples and hand the detector a manipulated
version. They are strictly black-box (no detector weights/gradients/threshold/scores).
Two attacks share the `ConcealmentAttack` interface:
  * `ReplayAttack`               -- generic replay baseline;
  * `BlackBoxAutoencoderAttack`  -- attacker-owned autoencoder (unconstrained /
    partially_constrained / fully_constrained).

Run it with `python main_concealment.py ...` from the repository root.
"""

from __future__ import annotations

from .base import AttackResult, ConcealmentAttack, assert_mask_preserved, combine_masked
from .replay import ReplayAttack

__all__ = [
    "AttackResult",
    "ConcealmentAttack",
    "ReplayAttack",
    "assert_mask_preserved",
    "combine_masked",
]

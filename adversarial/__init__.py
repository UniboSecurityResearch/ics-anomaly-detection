"""Modular adversarial-attack suite for the pwwl/ics-anomaly-detection detectors.

Layout (mirrors the detector/ package):
  base.py            Attack / WhiteBoxAttack base classes
  attacks/           one module per attack + the ATTACKS registry
  context.py         ModelAdapter + AttackContext (run-time state)
  io_utils.py        model / scaler / training-data / vector loading
  targets.py         labels, target selection, modification masks
  errors.py          per-feature errors + (differentiable) detector scores
  projection.py      L-inf/domain projection + top-k gradient sparsification
  detector.py        scalar theta+window detector, threshold/KL calibration
  evaluation.py      evaluate_series, save_run, run_sanity_check
  build.py           build_context (assembles everything from CLI args)
  cli.py             parse_args

Run it with `python main_adversarial.py ...` from the repository root.
"""

from __future__ import annotations

from .context import AttackContext, ModelAdapter
from .base import Attack, WhiteBoxAttack
from .attacks import ALL_ATTACKS, ATTACKS, WHITEBOX_ATTACKS

__all__ = [
    "AttackContext",
    "ModelAdapter",
    "Attack",
    "WhiteBoxAttack",
    "ATTACKS",
    "ALL_ATTACKS",
    "WHITEBOX_ATTACKS",
]

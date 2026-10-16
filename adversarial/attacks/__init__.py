"""Attack registry. Add a new attack by dropping a module here that subclasses
`adversarial.base.Attack` (or WhiteBoxAttack) and appending its class below --
no if/elif dispatch anywhere else needs editing.
"""

from __future__ import annotations

from typing import Dict, List, Set, Type

from ..base import Attack
from .fgsm_mse import FgsmMse
from .pgd_mse import PgdMse
from .pgd_topk import PgdTopk
from .pgd_hinge import PgdHinge
from .pgd_margin import PgdMargin
from .pgd_cw import PgdCw
from .pgd_kl import PgdKl
from .corrshift import CorrShift
from .corrshift_feva import CorrShiftFeva

# Insertion order defines the canonical order used by --attack all.
_REGISTERED = [FgsmMse, PgdMse, PgdTopk, PgdHinge, PgdMargin, PgdCw, PgdKl, CorrShift,
               CorrShiftFeva]

ATTACKS: Dict[str, Type[Attack]] = {cls.name: cls for cls in _REGISTERED}
# --attack all keeps its original set: attacks marked in_all = False run only by name.
ALL_ATTACKS: List[str] = [name for name, cls in ATTACKS.items() if getattr(cls, "in_all", True)]
ATTACK_CHOICES: List[str] = list(ATTACKS)
WHITEBOX_ATTACKS: Set[str] = {
    name for name, cls in ATTACKS.items() if cls.requires_gradients
}

__all__ = ["ATTACKS", "ALL_ATTACKS", "ATTACK_CHOICES", "WHITEBOX_ATTACKS"]

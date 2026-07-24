"""FGSM on the detector score: a single full-epsilon step, no random start.

Reuses PgdMse's objective; only the loop schedule differs (1 iteration, step = eps).
"""

from __future__ import annotations

from .pgd_mse import PgdMse


class FgsmMse(PgdMse):
    name = "fgsm_mse"

    def iterations(self, args) -> int:
        return 1

    def use_random_start(self, args) -> bool:
        return False

    def step_multiplier(self, args, eps):
        return eps

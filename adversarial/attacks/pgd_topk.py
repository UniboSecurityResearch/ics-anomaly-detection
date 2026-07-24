"""Sparse PGD: keep only the top-k gradient features per modified timestep.

Reuses PgdMse's objective and loop; only the gradient post-processing changes.
"""

from __future__ import annotations

from .pgd_mse import PgdMse
from ..context import AttackContext
from ..projection import keep_top_k_per_row


class PgdTopk(PgdMse):
    name = "pgd_topk"

    def process_gradient(self, ctx: AttackContext, gradient, mask):
        return keep_top_k_per_row(gradient, mask, ctx.args.top_k)

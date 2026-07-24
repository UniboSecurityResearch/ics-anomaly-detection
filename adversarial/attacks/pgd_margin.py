"""PGD on the smooth per-feature threshold margin (logit-style)."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from ..base import WhiteBoxAttack
from ..context import AttackContext
from ..errors import threshold_margin_tf


class PgdMargin(WhiteBoxAttack):
    name = "pgd_margin"
    requires_thresholds = True

    def objective(self, ctx: AttackContext, goal, errors, x_adv, x0) -> Tuple[tf.Tensor, tf.Tensor]:
        if ctx.thresholds is None:
            raise ValueError("pgd_margin requires thresholds.")
        margin = threshold_margin_tf(
            errors,
            tf.convert_to_tensor(ctx.thresholds, tf.float32),
            smooth=True,
            beta=ctx.args.margin_beta,
        )
        loss = tf.reduce_mean(margin)
        return (loss if goal == "evasion" else -loss), margin

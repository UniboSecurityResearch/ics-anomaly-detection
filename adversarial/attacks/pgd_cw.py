"""Projected PGD with a CW-style hinge on the detector margin."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from ..base import WhiteBoxAttack
from ..context import AttackContext
from ..errors import threshold_margin_tf


class PgdCw(WhiteBoxAttack):
    name = "pgd_cw"
    requires_thresholds = True

    def objective(self, ctx: AttackContext, goal, errors, x_adv, x0) -> Tuple[tf.Tensor, tf.Tensor]:
        args = ctx.args
        if ctx.thresholds is None:
            raise ValueError("pgd_cw requires thresholds.")
        margin = threshold_margin_tf(
            errors,
            tf.convert_to_tensor(ctx.thresholds, tf.float32),
            smooth=True,
            beta=args.margin_beta,
        )
        if goal == "evasion":
            hinge = tf.nn.relu(margin + args.cw_confidence)
        else:
            hinge = tf.nn.relu(args.cw_confidence - margin)
        loss = tf.reduce_mean(hinge)
        if args.cw_l2_weight > 0:
            loss += args.cw_l2_weight * tf.reduce_mean(tf.square(x_adv - x0))
        return loss, margin

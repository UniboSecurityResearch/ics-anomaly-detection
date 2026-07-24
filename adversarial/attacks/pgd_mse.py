"""PGD on the detector score (--score). The MSE-family white-box baseline."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from ..base import WhiteBoxAttack
from ..context import AttackContext
from ..errors import detector_score_tf


class PgdMse(WhiteBoxAttack):
    name = "pgd_mse"

    def objective(self, ctx: AttackContext, goal, errors, x_adv, x0) -> Tuple[tf.Tensor, tf.Tensor]:
        args = ctx.args
        report_score = detector_score_tf(errors, args.score, ctx.thresholds, args.margin_beta)
        loss = tf.reduce_mean(report_score)
        return (loss if goal == "evasion" else -loss), report_score

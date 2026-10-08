"""Threshold-aware PGD: a per-target hinge on the operative detector score.

pgd_mse minimises the MEAN score over all targets, so it keeps spending budget on
targets that are already below theta and on targets it can never bring below it.
This attack optimises the quantity the detector actually thresholds: for evasion

    loss = mean_t relu( s_t - margin * theta ),   s_t = mean_j error_{t,j}

which is zero for every target already under the (slightly tightened) threshold,
so the gradient only comes from targets that are still detected. False alarm uses
the mirrored hinge relu(theta / margin - s_t).
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from ..base import WhiteBoxAttack
from ..context import AttackContext
from ..errors import detector_score_tf


class PgdHinge(WhiteBoxAttack):
    name = "pgd_hinge"

    def objective(self, ctx: AttackContext, goal, errors, x_adv, x0) -> Tuple[tf.Tensor, tf.Tensor]:
        if ctx.instance_threshold is None:
            raise ValueError(
                "pgd_hinge requires the detector threshold "
                "(--instance-threshold or --instance-threshold-percentile)."
            )
        theta = tf.constant(float(ctx.instance_threshold), tf.float32)
        margin = float(getattr(ctx.args, "hinge_margin", 0.9))
        score = detector_score_tf(errors, ctx.instance_score_kind, None, 0.0)
        if goal == "evasion":
            hinge = tf.nn.relu(score - margin * theta)
        else:
            hinge = tf.nn.relu(theta / margin - score)
        # Normalise by theta so the loss scale does not depend on the model.
        return tf.reduce_mean(hinge) / theta, score

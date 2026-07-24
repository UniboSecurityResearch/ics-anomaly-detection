"""Experimental PGD on the residual-error distribution (KL) plus detector score."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from ..base import WhiteBoxAttack
from ..context import AttackContext
from ..errors import detector_score_tf, kl_divergence_tf, residual_distribution_tf


class PgdKl(WhiteBoxAttack):
    name = "pgd_kl"

    def objective(self, ctx: AttackContext, goal, errors, x_adv, x0) -> Tuple[tf.Tensor, tf.Tensor]:
        args = ctx.args
        if ctx.kl_reference is None:
            raise ValueError("pgd_kl requires a benign KL reference profile.")
        candidate = residual_distribution_tf(errors, ctx.thresholds, args.kl_temperature)
        reference = tf.convert_to_tensor(ctx.kl_reference, tf.float32)[tf.newaxis, :]
        reference = tf.repeat(reference, tf.shape(candidate)[0], axis=0)
        kl = kl_divergence_tf(reference, candidate)
        # A distribution-only KL objective can preserve a high total error. Add the
        # actual detector score so evasion/false-alarm remains operationally meaningful.
        score_kind = "threshold_smooth" if ctx.thresholds is not None else "mean_mse"
        score = detector_score_tf(errors, score_kind, ctx.thresholds, args.margin_beta)
        combined = kl + args.kl_score_weight * score
        loss = tf.reduce_mean(combined)
        return (loss if goal == "evasion" else -loss), combined

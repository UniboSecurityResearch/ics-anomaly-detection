"""Attack base classes, analogous to detector/detector.py's ICSDetector.

`Attack` is the minimal contract (`run(ctx, goal) -> (x_adv, metadata)`).
`WhiteBoxAttack` implements the shared PGD/FGSM loop once; each concrete white-box
attack only overrides `objective` (and optionally `process_gradient`,
`iterations`, `use_random_start`, `step_multiplier`) -- exactly as each detector
subclass only overrides `create_model`/`train` while reusing `cached_detect`.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from .context import AttackContext
from .errors import model_errors_tf
from .projection import build_tf_bounds


class Attack(ABC):
    """Common contract for every attack.

    Class attributes let the orchestrator decide what an attack needs without
    running it:
      requires_thresholds  -> per-feature threshold vector must be available;
      requires_gradients   -> needs a differentiable tf.keras.Model (white-box);
      requires_train_data  -> needs the standardized training set at build time.
    """

    name: str = "base"
    requires_thresholds: bool = False
    requires_gradients: bool = True
    requires_train_data: bool = False

    @abstractmethod
    def run(self, ctx: AttackContext, goal: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return (adversarial_series, metadata). The series has the same shape as
        ctx.x_test; only cells allowed by ctx.modification_mask may differ."""


class WhiteBoxAttack(Attack):
    """Shared projected-gradient loop. Faithful port of the original
    run_whitebox_attack: identical iteration count, random-start rule, sign step,
    projection and logging, so results match the monolithic script bit-for-bit."""

    requires_gradients = True

    # --- hooks a concrete attack may override -----------------------------------
    def objective(
        self,
        ctx: AttackContext,
        goal: str,
        errors: tf.Tensor,
        x_adv: tf.Tensor,
        x0: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Return (scalar loss to MINIMIZE, per-sample reporting score)."""
        raise NotImplementedError

    def process_gradient(self, ctx: AttackContext, gradient: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        return gradient

    def iterations(self, args) -> int:
        return int(args.iterations)

    def use_random_start(self, args) -> bool:
        return bool(args.random_start)

    def step_multiplier(self, args, eps: tf.Tensor):
        return args.alpha

    # --- the loop ---------------------------------------------------------------
    def run(self, ctx: AttackContext, goal: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        if ctx.adapter.keras_model is None:
            raise RuntimeError(
                f"{self.name} requires a differentiable tf.keras.Model. The loaded "
                "artifact is usable only for black-box attacks such as CorrShift."
            )

        args = ctx.args
        x0 = tf.convert_to_tensor(ctx.x_test, dtype=tf.float32)
        mask = tf.convert_to_tensor(ctx.modification_mask, dtype=tf.float32)
        eps, lower, upper = build_tf_bounds(
            x0, ctx.epsilon, ctx.lower_domain, ctx.upper_domain
        )

        x_adv = tf.identity(x0)
        if self.use_random_start(args):
            noise = tf.random.uniform(tf.shape(x_adv), -1.0, 1.0) * eps
            x_adv = x_adv + noise * mask
            x_adv = tf.clip_by_value(x_adv, lower, upper)
            x_adv = x0 + (x_adv - x0) * mask

        n_iterations = self.iterations(args)
        start_time = time.time()
        last_loss = math.nan

        for iteration in range(n_iterations):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                errors = model_errors_tf(
                    ctx.adapter.keras_model,
                    x_adv,
                    args.model_type,
                    ctx.target_indices,
                    args.history,
                    args.target_offset,
                )
                loss, report_score = self.objective(ctx, goal, errors, x_adv, x0)

            gradient = tape.gradient(loss, x_adv)
            if gradient is None:
                raise RuntimeError("Gradient is None; the model is not differentiable end-to-end.")
            gradient = gradient * mask
            gradient = self.process_gradient(ctx, gradient, mask)

            step = self.step_multiplier(args, eps) * tf.sign(gradient)
            # All objectives return a loss to MINIMIZE.
            x_adv = x_adv - step
            x_adv = tf.clip_by_value(x_adv, lower, upper)
            x_adv = x0 + (x_adv - x0) * mask
            last_loss = float(loss.numpy())

            if iteration == 0 or iteration == n_iterations - 1 or (iteration + 1) % 5 == 0:
                print(
                    f"[{self.name}/{goal}] iteration {iteration + 1:>3d}/"
                    f"{n_iterations:<3d} loss={last_loss:.8f} "
                    f"report={float(tf.reduce_mean(report_score).numpy()):.8f}"
                )

        metadata = {
            "iterations_executed": n_iterations,
            "final_optimization_loss": last_loss,
            "runtime_seconds": time.time() - start_time,
            "query_count": None,
        }
        return x_adv.numpy(), metadata

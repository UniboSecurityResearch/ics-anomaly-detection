"""DiffPure-style purification of time-series windows: x -> x_t* -> x0_hat.

`Purifier(denoiser, t_star, n_steps, sampler)` maps a batch of windows (B, L, d) to
purified windows of the same shape. With t_star = 0 it is the identity (no defense),
so the same code path gives the undefended detector.

Randomness is stateless and explicit: every call takes an int64 seed pair, and the
noise of the forward step and of each reverse step is derived from it. This makes
every evaluation draw and every EOT sample reproducible, and keeps the attacker's
randomness separate from the randomness used to evaluate the defense.

Gradient modes (what an attacker back-propagates through):
  "full" -- exact gradient of the whole stochastic process (reparameterisation of
            every Gaussian draw): Lee & Kim's recommended direct back-propagation;
  "bpda" -- forward pass is the purification, backward pass is the identity
            (Athalye et al. 2018), the attack Lee & Kim compare against (Table 3).
A "surrogate" process (Lee & Kim, Sec. 3.2) is simply another Purifier with the same
t_star and fewer denoising steps, used by the attacker only.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf

from . import schedule as S


class Purifier:
    def __init__(self, denoiser, t_star: int, n_steps: int, sampler: str = "ddpm",
                 T: int = S.T_DEFAULT):
        if t_star < 0 or t_star > T:
            raise ValueError("t_star must be in [0, T]")
        if t_star > 0 and denoiser is None:
            raise ValueError("a denoiser is required for t_star > 0")
        if sampler not in ("ddpm", "ddim"):
            raise ValueError(sampler)
        self.denoiser = denoiser
        self.t_star = int(t_star)
        self.sampler = sampler
        self.T = T
        self.steps = S.respaced_timesteps(self.t_star, n_steps)
        self.coeffs = S.step_coefficients(self.steps, T) if self.t_star > 0 else []
        self.n_steps = len(self.coeffs)

    @property
    def identity(self) -> bool:
        return self.t_star == 0

    def describe(self) -> dict:
        return dict(t_star=self.t_star, n_denoising_steps=self.n_steps, sampler=self.sampler,
                    timesteps=self.steps, T=self.T)

    def _eps(self, x, t_vec):
        return self.denoiser(x, t_vec, training=False)

    def __call__(self, x: tf.Tensor, seed: tf.Tensor, gradient: str = "full") -> tf.Tensor:
        """x: (B, L, d) float32; seed: int64 tensor of shape [2]."""
        if self.identity:
            return x
        seed = tf.cast(seed, tf.int64)
        shape = tf.shape(x)

        def noise(i: int) -> tf.Tensor:
            return tf.random.stateless_normal(
                shape, seed=seed + tf.constant([0, 1_000_003], tf.int64) * (i + 1), dtype=x.dtype)

        x_t = S.diffuse(x, self.t_star, noise(0), self.T)
        noises = [noise(i + 1) for i in range(len(self.coeffs))]
        x0 = S.reverse(x_t, self._eps, self.coeffs, noises, self.sampler)
        if gradient == "full":
            return x0
        if gradient == "bpda":
            return x + tf.stop_gradient(x0 - x)
        raise ValueError(f"unknown gradient mode {gradient}")


def seed_pair(*parts: int) -> tf.Tensor:
    """Deterministic int64 seed pair from integers (e.g. draw id, batch id)."""
    a, b = 0x9E3779B1, 0
    for p in parts:
        b = (b * 1_000_003 + int(p) + 1) % (2 ** 62)
        a = (a * 6_364_136_223 + int(p)) % (2 ** 62)
    return tf.constant([a, b], tf.int64)

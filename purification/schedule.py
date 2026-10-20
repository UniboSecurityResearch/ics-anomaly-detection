"""DDPM noise schedule and the two processes used for purification.

Notation (Ho et al. 2020; Lee & Kim, ICCV 2023, Sec. 2.2):
  beta_t       linear from 1e-4 to 0.02, t = 1..T, T = 1000;
  alpha_bar_t  prod_{s<=t} (1 - beta_s), with alpha_bar_0 = 1 (no noise).

This is the discretisation, with N = T = 1000 steps, of the VP-SDE that DiffPure
uses (Nie et al. 2022, Sec. 2: beta(t) = beta_min + (beta_max - beta_min) t with
beta_min = 0.1, beta_max = 20, so beta_t = beta(t/T) / T runs from 1e-4 to 0.02).
A diffusion timestep t* in DiffPure's [0, 1] corresponds to t* x T here
(DiffPure's t* = 0.1 is 100 forward steps).

Forward (DiffPure Eq. 3):    x_t* = sqrt(ab_t*) x + sqrt(1 - ab_t*) z.
Reverse: ancestral DDPM sampling on a sub-sequence of timesteps
  t* = s_K > s_{K-1} > ... > s_1 > s_0 = 0
("respacing", Nichol & Dhariwal 2021; the "number of denoising steps" of Lee & Kim).
With K = t* this is the exact DDPM reverse chain, i.e. the Euler-Maruyama-type
discretisation of DiffPure's reverse SDE (Eq. 2/4). Each step uses the posterior
q(x_{s_{i-1}} | x_{s_i}, x0_hat) of the respaced chain:
  b_i   = 1 - ab_{s_i} / ab_{s_{i-1}}
  mean  = sqrt(ab_{s_{i-1}}) b_i / (1 - ab_{s_i}) * x0_hat
          + sqrt(1 - b_i) (1 - ab_{s_{i-1}}) / (1 - ab_{s_i}) * x
  var   = b_i (1 - ab_{s_{i-1}}) / (1 - ab_{s_i})      (zero at the last step)
  x0_hat = (x - sqrt(1 - ab_{s_i}) eps_theta(x, s_i)) / sqrt(ab_{s_i}).
DDIM (eta = 0, Song et al. 2021) is available for comparison (Lee & Kim, Fig. 1-2):
  x <- sqrt(ab_{s_{i-1}}) x0_hat + sqrt(1 - ab_{s_{i-1}}) eps_theta.

Everything is written with TensorFlow ops so that gradients flow from the purified
output back to the input through the forward noising (reparameterisation) and every
denoising step: this is the "direct back-propagation of the full defense process"
recommended by Lee & Kim (Sec. 3.2, Recommendation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import tensorflow as tf

T_DEFAULT = 1000
BETA_START = 1e-4
BETA_END = 0.02


def linear_betas(T: int = T_DEFAULT, beta_start: float = BETA_START,
                 beta_end: float = BETA_END) -> np.ndarray:
    """beta_1..beta_T (float64), returned with a leading 0 so index t is beta_t."""
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    return np.concatenate([[0.0], betas])


def alpha_bars(T: int = T_DEFAULT) -> np.ndarray:
    """alpha_bar_0..alpha_bar_T (float64); alpha_bar_0 = 1."""
    betas = linear_betas(T)
    return np.cumprod(1.0 - betas)


def respaced_timesteps(t_star: int, n_steps: int) -> List[int]:
    """Decreasing timesteps [s_K = t_star, ..., s_1, s_0 = 0], K <= t_star distinct
    steps, evenly spread (rounded) over (0, t_star]."""
    t_star = int(t_star)
    if t_star <= 0:
        return [0]
    k = int(min(max(n_steps, 1), t_star))
    grid = np.round(np.linspace(t_star, 0, k + 1)).astype(int)
    steps = sorted(set(grid.tolist()), reverse=True)
    if steps[0] != t_star or steps[-1] != 0:
        raise AssertionError("respacing must start at t_star and end at 0")
    return steps


@dataclass(frozen=True)
class StepCoefficients:
    t: int            # network timestep s_i
    prev: int         # s_{i-1}
    sqrt_ab: float
    sqrt_one_minus_ab: float
    coef_x0: float
    coef_xt: float
    sigma: float
    sqrt_ab_prev: float
    sqrt_one_minus_ab_prev: float


def step_coefficients(steps: Sequence[int], T: int = T_DEFAULT) -> List[StepCoefficients]:
    ab = alpha_bars(T)
    out = []
    for t, p in zip(steps[:-1], steps[1:]):
        ab_t, ab_p = ab[t], ab[p]
        b = 1.0 - ab_t / ab_p
        out.append(StepCoefficients(
            t=int(t), prev=int(p),
            sqrt_ab=float(np.sqrt(ab_t)),
            sqrt_one_minus_ab=float(np.sqrt(1.0 - ab_t)),
            coef_x0=float(np.sqrt(ab_p) * b / (1.0 - ab_t)),
            coef_xt=float(np.sqrt(1.0 - b) * (1.0 - ab_p) / (1.0 - ab_t)),
            sigma=float(np.sqrt(max(b * (1.0 - ab_p) / (1.0 - ab_t), 0.0))),
            sqrt_ab_prev=float(np.sqrt(ab_p)),
            sqrt_one_minus_ab_prev=float(np.sqrt(1.0 - ab_p)),
        ))
    return out


EpsFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def diffuse(x: tf.Tensor, t_star: int, noise: tf.Tensor, T: int = T_DEFAULT) -> tf.Tensor:
    """x_t* = sqrt(ab) x + sqrt(1-ab) noise (DiffPure Eq. 3)."""
    if t_star <= 0:
        return x
    ab = alpha_bars(T)[int(t_star)]
    return tf.cast(np.sqrt(ab), x.dtype) * x + tf.cast(np.sqrt(1.0 - ab), x.dtype) * noise


def reverse(x_t: tf.Tensor, eps_fn: EpsFn, coeffs: Sequence[StepCoefficients],
            noises: Optional[Sequence[tf.Tensor]], sampler: str = "ddpm") -> tf.Tensor:
    """Run the respaced reverse chain from x_t (at coeffs[0].t) down to 0.

    noises[i] is the Gaussian noise of step i (unused at the last step and by DDIM).
    """
    x = x_t
    batch = tf.shape(x)[0]
    for i, c in enumerate(coeffs):
        t_vec = tf.fill([batch], tf.constant(float(c.t), tf.float32))
        eps = eps_fn(x, t_vec)
        x0_hat = (x - c.sqrt_one_minus_ab * eps) / c.sqrt_ab
        if sampler == "ddpm":
            x = c.coef_x0 * x0_hat + c.coef_xt * x
            if c.prev > 0:
                x = x + c.sigma * noises[i]
        elif sampler == "ddim":
            x = c.sqrt_ab_prev * x0_hat + c.sqrt_one_minus_ab_prev * eps
        else:
            raise ValueError(f"unknown sampler {sampler}")
    return x


def gaussian_eps_fn(mu0: float, var0: float, T: int = T_DEFAULT) -> EpsFn:
    """Exact noise predictor for data x ~ N(mu0, var0 I) (DiffPure App. B.5):
    p_t = N(sqrt(ab) mu0, ab var0 + 1 - ab), score = -(x - mu_t)/var_t and
    eps = -sqrt(1-ab) * score. Used only by the tests."""
    ab_table = tf.constant(alpha_bars(T), tf.float64)

    def eps_fn(x, t_vec):
        ab = tf.cast(tf.gather(ab_table, tf.cast(t_vec, tf.int32)), x.dtype)
        shape = tf.concat([tf.shape(ab), tf.ones([tf.rank(x) - 1], tf.int32)], 0)
        ab = tf.reshape(ab, shape)
        mu_t = tf.sqrt(ab) * mu0
        var_t = ab * var0 + 1.0 - ab
        return tf.sqrt(1.0 - ab) * (x - mu_t) / var_t

    return eps_fn

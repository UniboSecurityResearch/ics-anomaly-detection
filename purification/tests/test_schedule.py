"""Analytic checks of the forward/reverse processes (DiffPure, Appendix B.5).

For Gaussian data x ~ N(mu0, var0) the exact noise predictor is known, and so is the
posterior p(x0 | x_t) = N(m, v) with (DiffPure Eq. 30/31)
    m = ((1-ab) mu0 + var0 sqrt(ab) x_t) / (1 - ab + var0 ab)
    v = (1-ab) var0 / (1 - ab + var0 ab)
    d m / d x_t = var0 sqrt(ab) / (1 - ab + var0 ab).
With the exact predictor, the full DDPM chain (K = t*) must sample this posterior up
to discretisation error, and back-propagation through the sampled chain must give the
analytic derivative. Respaced chains (K < t*) are approximations; their error is
printed, not asserted, except for a loose bound.

Run from the repository root:  python -m purification.tests.test_schedule
"""

import numpy as np
import tensorflow as tf

from purification import schedule as S


def posterior(mu0, var0, ab, x_t):
    den = 1.0 - ab + var0 * ab
    m = ((1.0 - ab) * mu0 + var0 * np.sqrt(ab) * x_t) / den
    v = (1.0 - ab) * var0 / den
    phi = var0 * np.sqrt(ab) / den
    return m, v, phi


def run_chain(x_t, mu0, var0, t_star, k, sampler, seed):
    eps_fn = S.gaussian_eps_fn(mu0, var0)
    coeffs = S.step_coefficients(S.respaced_timesteps(t_star, k))
    gen = tf.random.Generator.from_seed(seed)
    noises = [gen.normal(tf.shape(x_t), dtype=tf.float64) for _ in coeffs]
    with tf.GradientTape() as tape:
        tape.watch(x_t)
        x0 = S.reverse(x_t, eps_fn, coeffs, noises, sampler)
        total = tf.reduce_sum(x0)
    grad = tape.gradient(total, x_t)
    return x0.numpy(), grad.numpy()


def test_respacing():
    assert S.respaced_timesteps(0, 5) == [0]
    for t_star in (1, 7, 50, 100, 300, 1000):
        for k in (1, 5, 10, 20, 1000):
            steps = S.respaced_timesteps(t_star, k)
            assert steps[0] == t_star and steps[-1] == 0
            assert all(a > b for a, b in zip(steps, steps[1:]))
            assert len(steps) - 1 == min(k, t_star)
    ab = S.alpha_bars()
    assert ab[0] == 1.0 and np.all(np.diff(ab) < 0)
    # DiffPure correspondence: VP-SDE alpha(t) = exp(-int_0^t beta(s) ds)
    # with beta(s) = 0.1 + 19.9 s, at t = 0.1 (t* = 100 steps).
    t = 0.1
    alpha_sde = np.exp(-(0.1 * t + 0.5 * 19.9 * t ** 2))
    print(f"alpha_bar_100 = {ab[100]:.5f}  vs VP-SDE alpha(0.1) = {alpha_sde:.5f}")
    assert abs(ab[100] - alpha_sde) < 2e-3


def chain_moments(mu0, var0, t_star, k, x_val):
    """Mean, variance and derivative that the implemented chain must produce with the
    exact (linear) predictor, computed in closed form step by step:
    x_prev = A x + B + sigma z  =>  propagate (slope, offset, variance)."""
    ab_all = S.alpha_bars()
    slope, offset, var = 1.0, 0.0, 0.0  # x_current = slope*x_t + offset + noise(var)
    for c in S.step_coefficients(S.respaced_timesteps(t_star, k)):
        ab = ab_all[c.t]
        cvec = np.sqrt(1 - ab) / (ab * var0 + 1 - ab)
        # eps = cvec (x - sqrt(ab) mu0);  x0_hat = (x - sqrt(1-ab) eps) / sqrt(ab)
        a_x0 = (1 - np.sqrt(1 - ab) * cvec) / np.sqrt(ab)
        b_x0 = np.sqrt(1 - ab) * cvec * np.sqrt(ab) * mu0 / np.sqrt(ab)
        a = c.coef_x0 * a_x0 + c.coef_xt
        b = c.coef_x0 * b_x0
        sigma2 = c.sigma ** 2 if c.prev > 0 else 0.0
        slope, offset, var = a * slope, a * offset + b, a * a * var + sigma2
    return slope * x_val + offset, var, slope


def test_posterior_and_gradient():
    """1) The implementation is exact: Monte Carlo moments equal the closed-form
    moments of the implemented chain. 2) The chain approximates the true posterior:
    the mean and the derivative are matched; the variance is not exactly, because
    DDPM's fixed per-step variance (beta tilde) is exact only for low-variance data
    (Ho et al. 2020, Sec. 3.2; Bao et al. 2022). That gap is printed, not asserted."""
    n = 50_000  # (n x 1) chains; the tape keeps every step, so memory grows with t*
    for var0 in (0.05, 0.5, 1.0):
        for t_star in (20, 100, 300):
            ab = S.alpha_bars()[t_star]
            for x_val in (-2.0, 0.0, 1.5):
                x_t = tf.constant(np.full((n, 1), x_val), tf.float64)
                m, v, phi = posterior(0.3, var0, ab, x_val)
                cm, cv, cg = chain_moments(0.3, var0, t_star, t_star, x_val)
                x0, grad = run_chain(x_t, 0.3, var0, t_star, t_star, "ddpm", 1)
                m_hat, v_hat, g_hat = x0.mean(), x0.var(), grad.mean()
                se = np.sqrt(cv / n)
                print(f"var0={var0:<4} t*={t_star:<3} x_t={x_val:+.1f}: "
                      f"MC mean {m_hat:+.4f} chain {cm:+.4f} true {m:+.4f} | "
                      f"MC var {v_hat:.5f} chain {cv:.5f} true {v:.5f} | "
                      f"grad {g_hat:.5f} chain {cg:.5f} true {phi:.5f}")
                # (1) implementation exactness
                assert abs(m_hat - cm) < 6 * se + 1e-9, "chain mean"
                assert abs(v_hat - cv) / cv < 0.02, "chain variance"
                assert np.allclose(grad, cg, rtol=1e-9), "per-sample derivative"
                # (2) approximation of the true posterior
                assert abs(cm - m) < 0.01 * max(1.0, abs(m)), "posterior mean"
                assert abs(cg - phi) / phi < 0.01, "posterior derivative"


def test_respaced_and_ddim():
    """Respaced DDPM (K < t*) keeps the posterior mean and derivative but loses
    variance (fewer noise injections); this is the price of fewer denoising steps.
    DDIM (eta = 0) is deterministic and follows the probability-flow ODE, whose exact
    map for Gaussian data is x0 = mu0 + sqrt(var0 / var_t) (x_t - sqrt(ab) mu0):
    it is NOT the posterior mean, and it injects no randomness (the reason DDIM-type
    samplers are weaker defenses, DiffPure Table 6 / Lee & Kim Fig. 1)."""
    n = 50_000
    mu0, var0, t_star, x_val = 0.3, 0.5, 100, 1.0
    ab = S.alpha_bars()[t_star]
    m, v, phi = posterior(mu0, var0, ab, x_val)
    var_t = ab * var0 + 1 - ab
    ode_slope = np.sqrt(var0 / var_t)
    ode_map = mu0 + ode_slope * (x_val - np.sqrt(ab) * mu0)
    x_t = tf.constant(np.full((n, 1), x_val), tf.float64)
    for sampler in ("ddpm", "ddim"):
        for k in (5, 10, 20, 100):
            x0, grad = run_chain(x_t, mu0, var0, t_star, k, sampler, 2)
            print(f"{sampler} K={k:<3}: mean {x0.mean():+.4f} (posterior {m:+.4f}, ODE {ode_map:+.4f}), "
                  f"var {x0.var():.4f} (posterior {v:.4f}), grad {grad.mean():.4f} "
                  f"(posterior {phi:.4f}, ODE {ode_slope:.4f})")
            if sampler == "ddpm":
                cm, cv, cg = chain_moments(mu0, var0, t_star, k, x_val)
                assert abs(x0.mean() - cm) < 6 * np.sqrt(cv / n) + 1e-9
                assert abs(cm - m) < 0.01 and abs(cg - phi) / phi < 0.01
            else:
                assert x0.var() < 1e-20  # deterministic given x_t
                if k == 100:
                    assert abs(x0.mean() - ode_map) < 5e-3 and abs(grad.mean() - ode_slope) < 5e-3


def test_purification_shrinkage():
    """End-to-end purification x -> x_t* -> x0_hat with the exact predictor:
    E[x0_hat | x] = ((1-ab) mu0 + var0 ab x) / (1 - ab + var0 ab): the input is pulled
    toward the data mean, more strongly for larger t*. This is the mechanism that
    removes small perturbations and, for an anomaly detector, can also shrink real
    anomalies."""
    n = 100_000
    mu0, var0 = 0.0, 1.0
    for t_star in (20, 100, 200):
        ab = S.alpha_bars()[t_star]
        x = tf.constant(np.full((n, 1), 3.0), tf.float64)
        z = tf.random.Generator.from_seed(3).normal(tf.shape(x), dtype=tf.float64)
        x_t = S.diffuse(x, t_star, z)
        x0, _ = run_chain(x_t, mu0, var0, t_star, t_star, "ddpm", 4)
        expected = ((1 - ab) * mu0 + var0 * ab * 3.0) / (1 - ab + var0 * ab)
        print(f"t*={t_star}: E[x0_hat | x=3] = {x0.mean():.4f} (analytic {expected:.4f})")
        assert abs(x0.mean() - expected) < 0.01


if __name__ == "__main__":
    tf.random.set_seed(0)
    test_respacing()
    test_posterior_and_gradient()
    test_respaced_and_ddim()
    test_purification_shrinkage()
    print("ALL SCHEDULE TESTS PASSED")

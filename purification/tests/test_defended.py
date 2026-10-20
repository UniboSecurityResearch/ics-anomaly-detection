"""Consistency checks of the defended detector and of its gradients.

1. Identity purifier: errors and gradients of purification.instances.errors_tf equal
   those of the existing attack suite (adversarial.errors.model_errors_tf) on the
   same series and targets, for a point model and a sequence model.
2. Through a (random) denoiser: the tape gradient of the purified score along a random
   direction equals a central finite difference with the same purification noise
   (the purifier is a deterministic function of input and seed).
3. BPDA: the backward pass is the identity, i.e. the gradient equals the detector's
   gradient evaluated at the purified windows.

    python -m purification.tests.test_defended  (from the repository root)
"""

import numpy as np
import tensorflow as tf

from adversarial.errors import model_errors_tf
from purification.denoiser import UNet1D
from purification.instances import errors_tf, make_geometry, window_index
from purification.purifier import Purifier, seed_pair


def toy_models(d):
    ae = tf.keras.Sequential([tf.keras.layers.Dense(8, activation="tanh", input_shape=(d,)),
                              tf.keras.layers.Dense(d)])
    seq = tf.keras.Sequential([tf.keras.layers.Conv1D(8, 3, activation="tanh", input_shape=(12, d)),
                               tf.keras.layers.Flatten(), tf.keras.layers.Dense(d)])
    return ae, seq


def test_identity_matches_attack_suite():
    rng = np.random.default_rng(0)
    d, n = 5, 400
    series = tf.constant(rng.standard_normal((n, d)), tf.float32)
    ae, seq = toy_models(d)
    targets = np.arange(100, 300, 7)
    ident = Purifier(None, 0, 0)
    for model, mtype, hist in ((ae, "AE", None), (seq, "CNN", 12)):
        geom = make_geometry(mtype, hist, 0, 64)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(series)
            mine = errors_tf(model, ident, tf.gather(series, window_index(targets, geom)), geom, seed_pair(0))
            ref = model_errors_tf(model, series, mtype, targets, hist, 0)
            l1, l2 = tf.reduce_sum(mine), tf.reduce_sum(ref)
        g1 = tf.convert_to_tensor(tape.gradient(l1, series)).numpy()
        g2 = tf.convert_to_tensor(tape.gradient(l2, series)).numpy()
        assert np.allclose(mine.numpy(), ref.numpy(), rtol=1e-6, atol=1e-7), mtype
        assert np.allclose(g1, g2, rtol=1e-5, atol=1e-6), mtype
        print(f"{mtype}: identity errors and gradients equal the attack suite's "
              f"(max |dg| {np.abs(g1 - g2).max():.2e})")


def test_gradient_through_purifier():
    rng = np.random.default_rng(1)
    d = 5
    net = UNet1D(d, base_channels=16, seed=3)
    for v in net.variables_list:  # zero-initialised output layers -> make it non-trivial
        v.assign(v + 0.05 * tf.random.stateless_normal(v.shape, seed=[1, 2]))
    ae, seq = toy_models(d)
    for model, mtype, hist in ((ae, "AE", None), (seq, "CNN", 12)):
        geom = make_geometry(mtype, hist, 0, 16)
        x = tf.constant(rng.standard_normal((6, geom.length, d)), tf.float32)
        direction = tf.constant(rng.standard_normal(x.shape), tf.float32)
        for t_star, k in ((30, 5), (100, 10)):
            p = Purifier(net, t_star, k)
            seed = seed_pair(7)
            f = lambda z: tf.reduce_sum(errors_tf(model, p, z, geom, seed))
            with tf.GradientTape() as tape:
                tape.watch(x)
                val = f(x)
            g = tape.gradient(val, x)
            analytic = float(tf.reduce_sum(g * direction))
            h = 1e-2
            numeric = float((f(x + h * direction) - f(x - h * direction)) / (2 * h))
            rel = abs(analytic - numeric) / max(abs(numeric), 1e-6)
            print(f"{mtype} t*={t_star} K={k}: directional derivative tape {analytic:.6f} "
                  f"finite difference {numeric:.6f} (rel {rel:.2e})")
            assert rel < 2e-2
            # BPDA: identity backward pass
            with tf.GradientTape() as tape:
                tape.watch(x)
                vb = tf.reduce_sum(errors_tf(model, p, x, geom, seed, "bpda"))
            gb = tape.gradient(vb, x)
            purified = p(x, seed, "full")
            with tf.GradientTape() as tape:
                tape.watch(purified)
                vp = tf.reduce_sum(errors_tf(model, Purifier(None, 0, 0), purified, geom, seed))
            gp = tape.gradient(vp, purified)
            assert np.allclose(vb.numpy(), vp.numpy(), rtol=1e-5)
            assert np.allclose(gb.numpy(), gp.numpy(), rtol=1e-4, atol=1e-6)


def test_seed_determinism():
    d = 4
    net = UNet1D(d, base_channels=16, seed=0)
    p = Purifier(net, 50, 5)
    x = tf.random.stateless_normal([3, 20, d], seed=[0, 1])
    a = p(x, seed_pair(1, 2)).numpy()
    b = p(x, seed_pair(1, 2)).numpy()
    c = p(x, seed_pair(1, 3)).numpy()
    assert np.array_equal(a, b) and not np.allclose(a, c)
    print("purifier: same seed -> same output, different seed -> different output")


if __name__ == "__main__":
    test_identity_matches_attack_suite()
    test_gradient_through_purifier()
    test_seed_determinism()
    print("ALL DEFENDED-DETECTOR TESTS PASSED")

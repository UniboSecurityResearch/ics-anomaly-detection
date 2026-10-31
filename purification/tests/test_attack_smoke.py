"""End-to-end smoke test of `main_purification.py attack` on synthetic data (no SWaT,
no trained models): the loaders are replaced by a toy AE, a toy denoiser and a
synthetic test series with two attacks. Checks that every arm used by phase 2 runs to
the end and writes summary.json and a consistent per_target.json, that the log loss
and the noise ablation run, and that --require-reproduction refuses a detector whose
t* = 0 calibration did not reproduce it.

    python -m purification.tests.test_attack_smoke  (from the repository root)
"""

import argparse
import json
import os
import tempfile

import numpy as np
import tensorflow as tf

import purification.attack as A
from purification.calibration import Splits
from purification.denoiser import UNet1D
from purification.purifier import Purifier

D, N_TRAIN, N_TEST, TUNE = 6, 600, 900, 450
STORED = dict(best_window=5, threshold_percentile=0.99, best_theta=0.5, final_f1=0.8)


def t0_record(path, **change):
    sel = dict(window=5, percentile=0.99, theta=0.5, final=dict(f1=0.8))
    for k, v in change.items():
        if k == "f1":
            sel["final"]["f1"] = v
        else:
            sel[k] = v
    with open(path, "w") as fd:
        json.dump(dict(model_dir="toy", purifier=dict(t_star=0), selection=sel, reproduces_stored=False), fd)
    return path


def fake_world():
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((N_TRAIN, D)) * 0.3
    x_test = rng.standard_normal((N_TEST, D)) * 0.3
    y = np.zeros(N_TEST, int)
    for a, b in ((100, 160), (300, 340), (600, 680)):
        y[a:b] = 1
        x_test[a:b, :3] += 2.0          # sensors
        x_test[a:b, 5] += 3.0           # an "actuator" (protected)
    ae = tf.keras.Sequential([tf.keras.layers.Dense(3, activation="tanh", input_shape=(D,)),
                              tf.keras.layers.Dense(D)])
    ae.compile("adam", "mse")
    ae.fit(x_train, x_train, epochs=30, verbose=0)
    net = UNet1D(D, base_channels=16, seed=3)
    cols = [f"S{i}" for i in range(5)] + ["MV101"]
    return x_train, x_test, y, cols, ae, net


def install(world, tmp):
    x_train, x_test, y, cols, ae, net = world
    A.load_data = lambda ds: (x_train, x_test, y, cols)
    A.load_detector = lambda d: (ae, "AE", None, {}, "AE-toy")
    A.actuator_indices = lambda c: [5]
    A.load_params = lambda d: STORED

    def splits(ds, xtr, xte, yte, geom):
        first = geom.first_target
        tl = np.arange(first, TUNE)
        fl = np.arange(first, N_TEST - TUNE)
        return Splits(x_train=xtr, val_targets=np.arange(500, 600), x_test=xte,
                      tune_targets=tl, tune_labels=yte[tl], final_targets=TUNE + fl,
                      final_labels=yte[TUNE + fl], tune_offset=0, final_offset=TUNE,
                      tune_len=TUNE, final_len=N_TEST - TUNE)
    A.build_splits = splits
    A.load_purifier = lambda diff, t, k, s: (Purifier(None if t == 0 else net, t, k, s), None)


def defense(tmp, t_star, sampler="ddpm"):
    path = os.path.join(tmp, f"defense_{t_star}_{sampler}.json")
    with open(path, "w") as fd:
        json.dump(dict(model_dir="toy", model="AE-toy", model_type="AE", history=None, target_offset=0,
                       context=16, lookahead=2, diffusion="toy", t_star=t_star, steps=4, sampler=sampler,
                       theta=0.5, window=5, percentile=0.99, calibration="x"), fd)
    return path


def run(tmp, name, **kw):
    ap = argparse.ArgumentParser()
    A.add_attack_args(ap)
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dataset", default="SWAT-CLEAN")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-dir", default="toy")
    argv = []
    for k, v in kw.items():
        k = "--" + k.replace("_", "-")
        if v is True:
            argv.append(k)
        else:
            argv += [k, str(v)]
    out = os.path.join(tmp, name)
    args = ap.parse_args(argv + ["--out", out])
    return A.run_attack(args), out


def check_per_target(out):
    with open(os.path.join(out, "per_target.json")) as fd:
        p = json.load(fd)
    with open(os.path.join(out, "summary.json")) as fd:
        s = json.load(fd)
    n = len(p["targets"])
    assert n == s["n_targets"] and len(p["eligible"]) == n and len(p["adv_flag_rate"]) == n
    assert len(p["scored"]) == len(p["clean_score"]) == len(p["adv_score"]) == len(p["modifiable"])
    assert abs(np.mean(p["adv_flag_rate"]) - s["reported"]["flagged_mean"]) < 1e-3
    assert abs(np.mean(p["clean_flag_rate"]) - s["clean"]["flagged_mean"]) < 1e-3
    assert sum(p["modifiable"]) <= s["attacker"]["modifiable_rows"]
    return p, s


def check_objective_terms():
    """The margin objective: values and gradients on both sides of the cap."""
    theta, k = 0.5, 0.10
    s = tf.constant([0.05, 0.4, 0.5, 0.54, 0.56, 5.0], tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(s)
        fa = A.objective_terms(s, "margin", "false_alarm", theta, k)
        ev = A.objective_terms(s, "margin", "evasion", theta, k)
    z = np.log(s.numpy() / theta)
    assert np.allclose(fa.numpy(), np.minimum(z, np.log(1.1)))
    assert np.allclose(ev.numpy(), np.maximum(z, np.log(0.9)))
    g_fa, g_ev = tape.gradient(fa, s).numpy(), tape.gradient(ev, s).numpy()
    # false alarm: gradient 1/s below theta (1 + k), none past it
    assert np.allclose(g_fa[:4], 1 / s.numpy()[:4]) and np.all(g_fa[4:] == 0)
    # evasion: no gradient at or below theta (1 - k) = 0.45, 1/s above it
    assert np.all(g_ev[:2] == 0) and np.allclose(g_ev[2:], 1 / s.numpy()[2:])
    assert np.allclose(A.objective_terms(s, "mean", "evasion", theta).numpy(), s.numpy())
    assert np.allclose(A.objective_terms(s, "log", "evasion", theta).numpy(), np.log(s.numpy() + 1e-12))
    for bad in (0.0, 1.0, -0.1):
        try:
            A.objective_terms(s, "margin", "evasion", theta, bad)
        except ValueError:
            continue
        raise AssertionError(f"margin {bad} must be refused")
    print("objective_terms checks passed")


def main():
    tf.random.set_seed(0)
    check_objective_terms()
    world = fake_world()
    keep = os.environ.get("SMOKE_OUT")  # keep the outputs (e.g. to try purification_per_target.py)
    with tempfile.TemporaryDirectory() as tmp:
        if keep:
            os.makedirs(keep, exist_ok=True)
            tmp = keep
        install(world, tmp)
        # the guard is recomputed from the record's selection (the saved boolean is ignored):
        # theta within 2e-3 relative passes (phase 2: up to 8.5e-4 from GPU numerics),
        # anything else refuses
        rec_ok = t0_record(os.path.join(tmp, "t0_ok.json"), theta=0.5 * (1 + 8.5e-4), f1=0.80001)
        bad = dict(theta=t0_record(os.path.join(tmp, "t0_theta.json"), theta=0.5 * (1 + 5e-3)),
                   window=t0_record(os.path.join(tmp, "t0_w.json"), window=10),
                   f1=t0_record(os.path.join(tmp, "t0_f1.json"), f1=0.79))
        common = dict(split="final", max_targets=60, epsilon_range_fraction=0.3, protect_actuators=True,
                      iterations=6, eval_draws=2, trace_every=3, attack_batch=16, require_reproduction=rec_ok)
        for why, rec in bad.items():
            status, _ = run(tmp, "refused_" + why, defense=defense(tmp, 0), goal="evasion", eot=1,
                            **dict(common, require_reproduction=rec))
            assert status == 3, f"the attack must refuse a detector not reproduced ({why})"
        # one block of 32 rows from the middle of every run: the toy final split has one
        # attack (rows 150-229 of the split) and two normal stretches
        status, out = run(tmp, "blocks_ev", defense=defense(tmp, 0), goal="evasion", eot=1,
                          **dict(common, block_length=32))
        p, s = check_per_target(out)
        assert status == 0 and p["targets"] == list(range(174, 206)) and s["attacker"]["n_blocks"] == 1
        status, out = run(tmp, "blocks_fa", defense=defense(tmp, 0), goal="false_alarm", eot=1,
                          **dict(common, block_length=32))
        p, s = check_per_target(out)
        assert status == 0 and len(p["targets"]) == 64 and s["attacker"]["n_blocks"] == 2
        print("guard and block-length checks passed")
        try:
            run(tmp, "blind_margin", defense=defense(tmp, 20), goal="false_alarm", gradient="blind",
                loss="margin", **common)
        except SystemExit:
            pass
        else:
            raise AssertionError("--loss margin with --gradient blind must be refused")
        cases = [
            ("undef_ev", dict(defense=defense(tmp, 0), goal="evasion", eot=1)),
            ("undef_fa", dict(defense=defense(tmp, 0), goal="false_alarm", eot=1)),
            ("full_ev", dict(defense=defense(tmp, 20), goal="evasion", eot=2)),
            ("full_ev_log", dict(defense=defense(tmp, 20), goal="evasion", eot=2, loss="log")),
            ("full_fa_R2", dict(defense=defense(tmp, 20), goal="false_alarm", eot=2, restarts=2)),
            ("full_fa_mrg", dict(defense=defense(tmp, 20), goal="false_alarm", eot=2, loss="margin")),
            ("undef_ev_mrg", dict(defense=defense(tmp, 0), goal="evasion", eot=1, loss="margin", margin=0.2)),
            ("surrogate", dict(defense=defense(tmp, 20), goal="evasion", eot=2, gradient="surrogate", attack_steps=2)),
            ("bpda", dict(defense=defense(tmp, 20), goal="false_alarm", eot=2, gradient="bpda")),
            ("blind", dict(defense=defense(tmp, 20), goal="evasion", gradient="blind")),
            ("noise_ev", dict(defense=defense(tmp, 20, "noise"), goal="evasion", eot=2)),
            ("window_ev", dict(defense=defense(tmp, 20), goal="evasion", eot=2, modifiable="window")),
            ("free_noRS", dict(defense=defense(tmp, 0), goal="evasion", eot=1, no_random_start=True)),
        ]
        for name, kw in cases:
            args = dict(common, **kw)
            if name == "free_noRS":
                args.pop("protect_actuators")
            status, out = run(tmp, name, **args)
            assert status == 0, name
            p, s = check_per_target(out)
            if s["attacker"]["loss"] == "margin":
                assert s["attacker"]["margin"] == args.get("margin", 0.5), name
            print(f"{name:<12} loss={s['attacker']['loss']:<4} sampler={s['defense']['sampler']:<5} "
                  f"clean {s['clean']['flagged_mean']:.3f} -> {s['reported']['flagged_mean']:.3f} "
                  f"modifiable {s['attacker']['modifiable_rows']}")
    print("ATTACK SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

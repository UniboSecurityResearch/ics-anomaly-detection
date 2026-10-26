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


def main():
    tf.random.set_seed(0)
    world = fake_world()
    keep = os.environ.get("SMOKE_OUT")  # keep the outputs (e.g. to try purification_per_target.py)
    with tempfile.TemporaryDirectory() as tmp:
        if keep:
            os.makedirs(keep, exist_ok=True)
            tmp = keep
        install(world, tmp)
        rec_ok = os.path.join(tmp, "t0_ok.json")
        rec_bad = os.path.join(tmp, "t0_bad.json")
        json.dump(dict(purifier=dict(t_star=0), reproduces_stored=True), open(rec_ok, "w"))
        json.dump(dict(purifier=dict(t_star=0), reproduces_stored=False), open(rec_bad, "w"))
        common = dict(split="final", max_targets=60, epsilon_range_fraction=0.3, protect_actuators=True,
                      iterations=6, eval_draws=2, trace_every=3, attack_batch=16, require_reproduction=rec_ok)
        status, _ = run(tmp, "refused", defense=defense(tmp, 0), goal="evasion", eot=1,
                        **dict(common, require_reproduction=rec_bad))
        assert status == 3, "the attack must refuse a detector that was not reproduced"
        cases = [
            ("undef_ev", dict(defense=defense(tmp, 0), goal="evasion", eot=1)),
            ("undef_fa", dict(defense=defense(tmp, 0), goal="false_alarm", eot=1)),
            ("full_ev", dict(defense=defense(tmp, 20), goal="evasion", eot=2)),
            ("full_ev_log", dict(defense=defense(tmp, 20), goal="evasion", eot=2, loss="log")),
            ("full_fa_R2", dict(defense=defense(tmp, 20), goal="false_alarm", eot=2, restarts=2)),
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
            print(f"{name:<12} loss={s['attacker']['loss']:<4} sampler={s['defense']['sampler']:<5} "
                  f"clean {s['clean']['flagged_mean']:.3f} -> {s['reported']['flagged_mean']:.3f} "
                  f"modifiable {s['attacker']['modifiable_rows']}")
    print("ATTACK SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

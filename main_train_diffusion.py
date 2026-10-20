#!/usr/bin/env python3
"""Train the diffusion model (noise predictor) used for adversarial purification.

The model is trained on BENIGN data only (the training file), as the pre-trained
diffusion models of DiffPure / Lee & Kim are trained on clean data only.

Data and split (no leakage into the detectors' calibration):
  * the training file is standardised with the repository's saved scaler
    (models/<dataset>_scaler.pkl, never rewritten);
  * rows [0, split) train the diffusion model, rows [split, n) validate it, with
    split = int((n - 1) * 0.8): the start of the chronological validation part of
    the sequence detectors (utils.train_val_history_idx_split) and not after the start
    of the AE validation rows (train_test_split(test_size=0.2, shuffle=False)). So no
    row that calibrates a detector's threshold is ever seen in diffusion training.

Objective: the DDPM "simple" loss E ||eps - eps_theta(sqrt(ab_t) x + sqrt(1-ab_t) eps, t)||^2,
t ~ U{1..T}, on random windows of --window consecutive rows (Lee & Kim Eq. 6).
Adam with linear warm-up, global-norm clipping and an EMA of the weights; the EMA
weights with the lowest validation loss on t <= --val-tmax (the noise levels used for
purification) are saved.

Validation report (also written to the output folder):
  * loss per noise band on fixed validation windows/noise, against the loss of the
    best predictor that ignores all structure between rows and features (data treated
    as N(0, I), eps_hat = sqrt(1 - ab_t) x_t); the ratio shows how much temporal and
    cross-feature structure the model has learnt;
  * the same loss at the window lengths the detectors will feed (length
    generalisation of the fully convolutional network).

    python3 main_train_diffusion.py SWAT-CLEAN --out models/diffusion_swat
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from adversarial.io_utils import load_scaler, load_train_scaled
from data_loader import load_train_data
from purification import schedule as S
from purification.denoiser import UNet1D, save_denoiser

BANDS = [(1, 20), (21, 50), (51, 100), (101, 200), (201, 300), (301, 1000)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True, help="output folder (created; must not exist)")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--channel-mults", type=int, nargs="+", default=[1, 2, 2])
    ap.add_argument("--n-res", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--val-windows", type=int, default=4096)
    ap.add_argument("--val-tmax", type=int, default=300)
    ap.add_argument("--check-lengths", type=int, nargs="*", default=[64, 101])
    ap.add_argument("--expect-data", nargs=3, metavar=("TRAIN", "TEST", "LABELS"), default=None,
                    help="SHA-256 digests (purification.common.data_digest) the data must have; exit 5 otherwise")
    ap.add_argument("--gate-ratio", type=float, default=0.5)
    ap.add_argument("--gate-length", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def chronological_split(n_rows: int) -> int:
    return int((n_rows - 1) * 0.8)


def fixed_validation_set(x_val: np.ndarray, length: int, n: int, t_max: int, seed: int):
    """Evenly spaced windows; half with t ~ U{1..t_max}, half with t ~ U{1..T}; fixed noise."""
    rng = np.random.default_rng(seed)
    starts = np.linspace(0, len(x_val) - length, n).astype(np.int64)
    windows = np.stack([x_val[s:s + length] for s in starts]).astype(np.float32)
    t = np.concatenate([rng.integers(1, t_max + 1, n // 2), rng.integers(1, S.T_DEFAULT + 1, n - n // 2)])
    noise = rng.standard_normal(windows.shape).astype(np.float32)
    return windows, t.astype(np.int64), noise


def evaluate(model, windows, t, noise, ab, batch=512):
    """Per-window loss of the model and of the structure-free N(0, I) predictor."""
    losses, base = [], []
    for i in range(0, len(windows), batch):
        x = windows[i:i + batch]
        tt = t[i:i + batch]
        z = noise[i:i + batch]
        a = ab[tt][:, None, None].astype(np.float32)
        x_t = np.sqrt(a) * x + np.sqrt(1 - a) * z
        pred = model(tf.constant(x_t), tf.constant(tt, tf.float32), training=False).numpy()
        losses.append(np.mean((pred - z) ** 2, axis=(1, 2)))
        base.append(np.mean((np.sqrt(1 - a) * x_t - z) ** 2, axis=(1, 2)))
    return np.concatenate(losses), np.concatenate(base)


def band_report(losses, base, t):
    rows = []
    for lo, hi in BANDS:
        sel = (t >= lo) & (t <= hi)
        if np.any(sel):
            rows.append(dict(band=f"{lo}-{hi}", n=int(sel.sum()), loss=float(losses[sel].mean()),
                             baseline=float(base[sel].mean()),
                             ratio=float(losses[sel].mean() / base[sel].mean())))
    return rows


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; choose a new folder (nothing is overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    print("TensorFlow", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    if args.expect_data:
        from purification.common import data_digest
        digest = data_digest(args.dataset)
        expected = dict(zip(("train", "test", "labels"), args.expect_data))
        for k in expected:
            print(f"data digest {k}: {digest[k]} ({'OK' if digest[k] == expected[k] else 'MISMATCH'})", flush=True)
        if digest != expected:
            print("ERROR: the data differ from the expected ones", flush=True)
            sys.exit(5)

    scaler = load_scaler(args.dataset)
    _, sensor_cols = load_train_data(args.dataset, no_transform=True)  # writes nothing
    sensor_cols = [str(c) for c in sensor_cols]
    x_full = load_train_scaled(args.dataset, scaler, sensor_cols)
    n_rows, n_feat = x_full.shape
    split = chronological_split(n_rows)
    x_tr, x_val = x_full[:split], x_full[split:]
    print(f"{args.dataset}: {n_rows} rows x {n_feat} features; diffusion train rows [0, {split}), "
          f"validation rows [{split}, {n_rows})", flush=True)
    if len(x_tr) < args.window or len(x_val) < max([args.window] + list(args.check_lengths)):
        raise SystemExit("not enough rows for the requested window")

    model = UNet1D(n_feat, base_channels=args.base_channels, channel_mults=tuple(args.channel_mults),
                   n_res=args.n_res, dropout=args.dropout, seed=args.seed)
    variables = model.variables_list
    ema = [tf.Variable(v.numpy(), trainable=False) for v in variables]
    m_state = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
    v_state = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
    step_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    print(f"denoiser parameters: {model.n_parameters():,}", flush=True)

    ab = S.alpha_bars()
    ab_tf = tf.constant(ab, tf.float32)
    x_tr_tf = tf.constant(x_tr, tf.float32)
    offsets = tf.range(args.window, dtype=tf.int64)
    max_start = len(x_tr) - args.window + 1
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    @tf.function
    def train_step():
        starts = tf.random.uniform([args.batch], 0, max_start, dtype=tf.int64)
        x0 = tf.gather(x_tr_tf, starts[:, None] + offsets[None, :])
        t = tf.random.uniform([args.batch], 1, S.T_DEFAULT + 1, dtype=tf.int64)
        a = tf.gather(ab_tf, t)[:, None, None]
        noise = tf.random.normal(tf.shape(x0))
        x_t = tf.sqrt(a) * x0 + tf.sqrt(1.0 - a) * noise
        with tf.GradientTape() as tape:
            pred = model(x_t, tf.cast(t, tf.float32), training=True)
            loss = tf.reduce_mean(tf.square(pred - noise))
        grads = tape.gradient(loss, variables)
        grads, gnorm = tf.clip_by_global_norm(grads, args.grad_clip)
        step_var.assign_add(1)
        step = tf.cast(step_var, tf.float32)
        lr = args.lr * tf.minimum(1.0, step / float(max(args.warmup, 1)))
        for var, g, m, v, e in zip(variables, grads, m_state, v_state, ema):
            m.assign(beta1 * m + (1 - beta1) * g)
            v.assign(beta2 * v + (1 - beta2) * tf.square(g))
            m_hat = m / (1 - tf.pow(beta1, step))
            v_hat = v / (1 - tf.pow(beta2, step))
            var.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + eps_adam))
            e.assign(args.ema * e + (1 - args.ema) * var)
        return loss, gnorm

    val_w, val_t, val_z = fixed_validation_set(x_val, args.window, args.val_windows, args.val_tmax, 123)
    sel = val_t <= args.val_tmax

    def with_ema(fn):
        backup = [v.numpy() for v in variables]
        for v, e in zip(variables, ema):
            v.assign(e)
        try:
            return fn()
        finally:
            for v, b in zip(variables, backup):
                v.assign(b)

    log_path = out / "train_log.csv"
    log = open(log_path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "train_loss", "grad_norm", "val_loss_tmax", "val_baseline_tmax", "val_loss_all", "seconds"])
    best = dict(loss=math.inf, step=0)
    best_weights = None
    started = time.time()
    running = []
    for it in range(1, args.steps + 1):
        loss, gnorm = train_step()
        running.append(float(loss))
        if not math.isfinite(running[-1]):
            raise SystemExit(f"non-finite training loss at step {it}")
        if it % args.eval_every == 0 or it == args.steps:
            losses, base = with_ema(lambda: evaluate(model, val_w, val_t, val_z, ab))
            vl, vb, va = float(losses[sel].mean()), float(base[sel].mean()), float(losses.mean())
            writer.writerow([it, np.mean(running), float(gnorm), vl, vb, va, time.time() - started])
            log.flush()
            print(f"step {it:>7d}  train {np.mean(running):.5f}  val(t<={args.val_tmax}) {vl:.5f} "
                  f"(N(0,I) baseline {vb:.5f})  val(all t) {va:.5f}  {time.time() - started:7.0f}s", flush=True)
            running = []
            if vl < best["loss"]:
                best = dict(loss=vl, step=it)
                best_weights = [e.numpy() for e in ema]
    log.close()

    for v, w in zip(variables, best_weights):
        v.assign(w)
    losses, base = evaluate(model, val_w, val_t, val_z, ab)
    bands = band_report(losses, base, val_t)
    print("validation by noise band (best EMA weights):")
    for r in bands:
        print(f"  t {r['band']:>8}: loss {r['loss']:.5f}  baseline {r['baseline']:.5f}  ratio {r['ratio']:.3f}")
    lengths = []
    for L in args.check_lengths:
        w, t, z = fixed_validation_set(x_val, L, args.val_windows, args.val_tmax, 321)
        lo, ba = evaluate(model, w, t, z, ab)
        s = t <= args.val_tmax
        w128, t128, z128 = fixed_validation_set(x_val, args.window, args.val_windows, args.val_tmax, 321)
        lo128, _ = evaluate(model, w128, t128, z128, ab)
        lengths.append(dict(length=int(L), loss_tmax=float(lo[s].mean()), baseline_tmax=float(ba[s].mean()),
                            loss_tmax_train_length=float(lo128[t128 <= args.val_tmax].mean())))
        print(f"  length {L:>4}: loss(t<={args.val_tmax}) {lo[s].mean():.5f} "
              f"(baseline {ba[s].mean():.5f}; at length {args.window}: {lo128[t128 <= args.val_tmax].mean():.5f})")
    meta = dict(dataset=args.dataset, sensor_cols=sensor_cols, n_rows=int(n_rows), split=int(split),
                train_rows=[0, int(split)], validation_rows=[int(split), int(n_rows)],
                schedule=dict(T=S.T_DEFAULT, beta_start=S.BETA_START, beta_end=S.BETA_END),
                training={k: v for k, v in vars(args).items() if k not in ("out",)},
                best_step=int(best["step"]), best_val_loss_tmax=float(best["loss"]),
                validation_bands=bands, length_check=lengths, git_commit=git_commit(),
                tensorflow=tf.__version__, runtime_seconds=time.time() - started)
    # Sanity gate (not a quality criterion): the model must have learnt structure
    # (validation loss on t <= val_tmax well below the structure-free predictor) and
    # must work at the window lengths the detectors use.
    ratio = best["loss"] / float(base[sel].mean())
    worst_len = max([r["loss_tmax"] / r["loss_tmax_train_length"] for r in lengths] or [1.0])
    gate_ok = ratio < args.gate_ratio and worst_len < args.gate_length
    meta["gate"] = dict(ratio_tmax=ratio, max_ratio=args.gate_ratio,
                        worst_length_ratio=worst_len, max_length_ratio=args.gate_length, passed=gate_ok)
    save_denoiser(str(out), model, model.get_weights(), meta)
    print(f"saved {out}/weights.npz and config.json (best step {best['step']})", flush=True)
    print(f"GATE: loss/baseline on t<={args.val_tmax} = {ratio:.3f} (must be < {args.gate_ratio}); "
          f"worst length ratio {worst_len:.3f} (must be < {args.gate_length}): "
          f"{'PASSED' if gate_ok else 'FAILED'}", flush=True)
    if not gate_ok:
        sys.exit(4)


if __name__ == "__main__":
    main()

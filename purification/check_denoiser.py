#!/usr/bin/env python3
"""Quality check of a trained noise predictor and choice of the purifier look-ahead.

Run as a module from the repository root:
    python3 -m purification.check_denoiser --diffusion models/diffusion_swat_v1 \
        --out outputs/purification/phase1_v2/denoiser_check.json

On benign VALIDATION rows of the training file (the rows the diffusion model did not
train on, never test data), at the training window length L and noise levels
t <= --tmax (those used for purification):

  * eps-prediction loss per window position, against the structure-free predictor
    (data treated as N(0, I)), and the error of the one-step estimate of the clean
    window x0_hat = (x_t - sqrt(1-ab) eps_theta) / sqrt(ab), per position, for the
    noise bands 1-50, 51-100, 101-300;
  * look-ahead rule (fixed before any defense result is seen): d = the smallest
    d in 0..--max-lookahead such that, in EVERY band, the x0 error at window row
    L-1-d is at most --tol (1.20) times the smallest x0 error reachable with any
    look-ahead in 0..--max-lookahead (i.e. within 20% of the best the scored row can
    get from right-hand context). Relative to the best reachable rather than to the
    interior mean, because near the right edge the error levels off slightly above
    the interior mean, and a threshold on that plateau would pick d by noise.
    The tolerance (20%) is above the Monte Carlo noise of the profile with the
    default number of windows (checked on two seeds: same d);
  * gate: overall loss ratio < --gate-ratio (0.5). Exit code 4 otherwise, so that
    the jobs depending on it are cancelled. The relative error at the chosen d with
    respect to the interior is reported.

Why: the scored row of a causal purifier is the last row of its window, where the
denoiser has no right-hand context. Measured on SWAT-CLEAN it is several times worse
there than inside the window, and one to three rows of context remove most of the
gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import schedule as S
from .common import check_data_digests, git_state, load_data
from .denoiser import load_denoiser

BANDS = [(1, 50), (51, 100), (101, 300)]
EDGE = 16


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--diffusion", required=True)
    ap.add_argument("--dataset", default=None, help="default: the dataset of the diffusion model")
    ap.add_argument("--out", required=True, help="json report (must not exist)")
    ap.add_argument("--windows", type=int, default=16384)
    ap.add_argument("--tmax", type=int, default=300)
    ap.add_argument("--max-lookahead", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1.20)
    ap.add_argument("--gate-ratio", type=float, default=0.5)
    ap.add_argument("--expect-data", nargs=3, metavar=("TRAIN", "TEST", "LABELS"), default=None)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (nothing is overwritten)")
    out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_denoiser(args.diffusion)
    dataset = args.dataset or cfg["dataset"]
    if args.expect_data:
        check_data_digests(dataset, args.expect_data)
    x_train, _, _, cols = load_data(dataset)
    if cols != cfg["sensor_cols"]:
        raise SystemExit("feature list differs from the one the diffusion model was trained on")
    split = int(cfg["split"])
    x_val = x_train[split:].astype(np.float32)
    L = int(cfg["training"]["window"])
    print(f"{args.diffusion}: window {L}, validation rows [{split}, {len(x_train)}), "
          f"{args.windows} windows, t <= {args.tmax}", flush=True)

    rng = np.random.default_rng(args.seed)
    starts = rng.integers(0, len(x_val) - L + 1, args.windows)
    t_all = rng.integers(1, args.tmax + 1, args.windows)
    ab = S.alpha_bars()
    eps_sum = np.zeros((len(BANDS), L))
    x0_sum = np.zeros((len(BANDS), L))
    base_sum = 0.0
    count = np.zeros(len(BANDS))
    loss_sum = 0.0
    for i in range(0, args.windows, args.batch):
        st, t = starts[i:i + args.batch], t_all[i:i + args.batch]
        x0 = np.stack([x_val[s:s + L] for s in st])
        z = rng.standard_normal(x0.shape).astype(np.float32)
        a = ab[t][:, None, None].astype(np.float32)
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * z
        pred = model(tf.constant(xt), tf.constant(t, tf.float32), training=False).numpy()
        x0_hat = (xt - np.sqrt(1 - a) * pred) / np.sqrt(a)
        e_eps = ((pred - z) ** 2).mean(axis=2)          # (B, L)
        e_x0 = ((x0_hat - x0) ** 2).mean(axis=2)
        loss_sum += float(e_eps.mean(axis=1).sum())
        base_sum += float(((np.sqrt(1 - a) * xt - z) ** 2).mean(axis=(1, 2)).sum())
        for b, (lo, hi) in enumerate(BANDS):
            sel = (t >= lo) & (t <= hi)
            eps_sum[b] += e_eps[sel].sum(axis=0)
            x0_sum[b] += e_x0[sel].sum(axis=0)
            count[b] += sel.sum()
    eps_prof = eps_sum / count[:, None]
    x0_prof = x0_sum / count[:, None]
    ratio = loss_sum / base_sum
    interior = x0_prof[:, EDGE:L - EDGE].mean(axis=1)

    positions = [L - 1 - d for d in range(args.max_lookahead + 1)]
    best = x0_prof[:, positions].min(axis=1)                 # per band
    chosen = None
    rows = []
    for d, pos in enumerate(positions):
        rel_best = x0_prof[:, pos] / best
        rel_int = x0_prof[:, pos] / interior
        ok = bool(np.all(rel_best <= args.tol))
        rows.append(dict(lookahead=d, x0_error=x0_prof[:, pos].tolist(),
                         relative_to_best=rel_best.tolist(), relative_to_interior=rel_int.tolist(),
                         eps_loss=eps_prof[:, pos].tolist(), within_tolerance=ok))
        print(f"look-ahead {d}: x0 error / best reachable " +
              "  ".join(f"t {lo}-{hi}: {r:5.2f}" for (lo, hi), r in zip(BANDS, rel_best)) +
              " | / interior " + " ".join(f"{r:5.2f}" for r in rel_int) +
              ("  <- chosen" if ok and chosen is None else ""), flush=True)
        if ok and chosen is None:
            chosen = d
    passed = ratio < args.gate_ratio and chosen is not None  # chosen always exists
    print(f"interior x0 error by band: " + "  ".join(f"t {lo}-{hi}: {v:.5f}" for (lo, hi), v in zip(BANDS, interior)))
    print(f"GATE: loss/baseline on t<={args.tmax} = {ratio:.4f} (must be < {args.gate_ratio}); "
          f"look-ahead rule (tol {args.tol}, max {args.max_lookahead}): "
          f"{chosen if chosen is not None else 'none'} -> {'PASSED' if passed else 'FAILED'}", flush=True)
    report = dict(diffusion=args.diffusion, dataset=dataset, window=L, validation_rows=[split, len(x_train)],
                  windows=args.windows, tmax=args.tmax, bands=BANDS, loss_ratio_tmax=ratio,
                  interior_x0_error=interior.tolist(), edge=EDGE, tol=args.tol,
                  max_lookahead=args.max_lookahead, profile_last_rows=rows,
                  x0_error_profile=x0_prof.tolist(), eps_loss_profile=eps_prof.tolist(),
                  chosen_lookahead=chosen, passed=bool(passed), git=git_state())
    with open(out, "w") as fd:
        json.dump(report, fd, indent=1, default=float)
    return 0 if passed else 4


if __name__ == "__main__":
    sys.exit(main())

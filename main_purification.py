#!/usr/bin/env python3
"""Diffusion-based adversarial purification (DiffPure) for the ICS detectors.

Two sub-commands, run from the repository root:

  calibrate  For each diffusion timestep t*, purify every instance window, re-choose
             the threshold and window of the DEFENDED detector with main_train.py's
             own procedure, and score it on the final part of the test file over
             several independent purification draws. t* = 0 (no defense) must
             reproduce the values stored in the model .json, otherwise the command
             exits with code 3 and nothing downstream is trusted.

  attack     PGD + EOT against the defended detector (purification/attack.py): full
             gradients through the whole purification, a surrogate process, BPDA, or a
             defense-blind attacker; evasion or false alarm.

    python3 main_purification.py calibrate --dataset SWAT-CLEAN \
        --model-dir models/swatchrono_ae_s7 --diffusion models/diffusion_swat \
        --t-star 0 20 50 100 150 200 300 --steps 10 --out outputs/purification/ae_s7/calib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from purification.calibration import build_splits, check_reproduction, detection_metrics, grid_search
from purification.common import GRIDS, git_state, load_data, load_detector, load_purifier
from purification.instances import Scorer, make_geometry


def add_common(ap):
    ap.add_argument("--dataset", default="SWAT-CLEAN")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--diffusion", default=None, help="folder written by main_train_diffusion.py")
    ap.add_argument("--steps", type=int, default=10, help="denoising steps K (respaced DDPM)")
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--context", type=int, default=64, help="minimum purified window length")
    ap.add_argument("--target-offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cal = sub.add_parser("calibrate")
    add_common(cal)
    cal.add_argument("--t-star", type=int, nargs="+", required=True)
    cal.add_argument("--grid", choices=sorted(GRIDS), default="train")
    cal.add_argument("--metric", default=None, help="default: the model's selection metric")
    cal.add_argument("--max-fpr", type=float, default=None, help="default: the model's FPR cap")
    cal.add_argument("--draws", type=int, default=5, help="independent purification draws on the final split")
    att = sub.add_parser("attack")
    add_common(att)
    from purification.attack import add_attack_args
    add_attack_args(att)
    return ap.parse_args()


def calibrate(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(args.seed)
    model, model_type, history, params, name = load_detector(args.model_dir)
    geom = make_geometry(model_type, history, args.target_offset, args.context)
    x_train, x_test, y_test, cols = load_data(args.dataset)
    splits = build_splits(args.dataset, x_train, x_test, y_test, geom)
    percentiles, windows = GRIDS[args.grid]
    metric = args.metric or (params.get("threshold_selection_metric") or ["F1"])[0]
    max_fpr = args.max_fpr if args.max_fpr is not None else params.get("threshold_max_fpr", 0.05)
    print(f"{name}: {model_type} history={history} window length Lw={geom.length}; "
          f"validation {len(splits.val_targets)}, tuning {len(splits.tune_targets)} "
          f"(offset {splits.tune_offset}), final {len(splits.final_targets)} (offset {splits.final_offset}) "
          f"targets; grid={args.grid} metric={metric} max_fpr={max_fpr}", flush=True)
    stored_theta = params.get("best_theta")
    stored_window = params.get("best_window")
    summary = []
    status = 0
    for t_star in args.t_star:
        started = time.time()
        purifier, diff_cfg = load_purifier(args.diffusion, t_star, args.steps, args.sampler)
        scorer = Scorer(model, purifier, geom, args.batch)
        tag = f"tstar{t_star:04d}_k{purifier.n_steps:03d}_{args.sampler}"
        seed0 = args.seed * 1000 + t_star

        def scores(series, targets, draw, part):
            return scorer.errors(series, targets, seed_base=(seed0 * 100 + draw) * 10 + part).mean(axis=1)

        val = scores(splits.x_train, splits.val_targets, 0, 0)
        tune = scores(splits.x_test, splits.tune_targets, 0, 1)
        final0 = scores(splits.x_test, splits.final_targets, 0, 2)
        res = grid_search(val, tune, splits.tune_labels, final0, splits.final_labels, model_type,
                          percentiles, windows, metric, max_fpr)
        record = dict(model=name, model_dir=args.model_dir, model_type=model_type, history=history,
                      geometry=dict(length=geom.length, offset=geom.offset, context=args.context),
                      purifier=purifier.describe(), diffusion=args.diffusion,
                      diffusion_best_step=(diff_cfg or {}).get("best_step"),
                      grid=args.grid, metric=metric, max_fpr=max_fpr, selection=res,
                      git=git_state())
        if t_star == 0:
            ok, msg = check_reproduction(res, params)
            print(msg, flush=True)
            print(f"t*=0 reproduces the stored detector: {'YES' if ok else 'NO'}", flush=True)
            record["reproduces_stored"] = ok
            if not ok:
                status = 3
        # fresh, independent draws on the final split (the selection used draw 0)
        draws, fixed = [], []
        n_draws = 1 if purifier.identity else args.draws
        for d in range(1, n_draws + 1):
            s = final0 if purifier.identity else scores(splits.x_test, splits.final_targets, d, 2)
            draws.append(detection_metrics(s, splits.final_labels, res["theta"], res["window"], model_type))
            fixed.append(detection_metrics(s, splits.final_labels, stored_theta, stored_window, model_type))
        agg = lambda rows, k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
        record["final_draws"] = draws
        record["final_draws_stored_threshold"] = fixed
        record["runtime_seconds"] = time.time() - started
        with open(out / f"{tag}.json", "w") as fd:
            json.dump(record, fd, indent=1, default=float)
        defense = dict(model_dir=args.model_dir, model=name, model_type=model_type, history=history,
                       target_offset=args.target_offset, context=args.context, diffusion=args.diffusion,
                       t_star=t_star, steps=args.steps, sampler=args.sampler,
                       theta=res["theta"], window=res["window"], percentile=res["percentile"],
                       calibration=str(out / f"{tag}.json"))
        with open(out / f"defense_{tag}.json", "w") as fd:
            json.dump(defense, fd, indent=1)
        f1m, f1s = agg(draws, "f1")
        rm, rs = agg(draws, "recall")
        fm, fs = agg(draws, "fpr")
        f1o, _ = agg(fixed, "f1")
        row = dict(t_star=t_star, steps=purifier.n_steps, percentile=res["percentile"], window=res["window"],
                   theta=res["theta"], tune_f1=res["tune"]["f1"], final_f1=f1m, final_f1_std=f1s,
                   final_recall=rm, final_recall_std=rs, final_fpr=fm, final_fpr_std=fs,
                   final_f1_stored_threshold=f1o, seconds=record["runtime_seconds"])
        summary.append(row)
        print(f"t*={t_star:>4} K={purifier.n_steps:>3}: pct={res['percentile']} w={res['window']} "
              f"theta={res['theta']:.5g} | tune F1 {res['tune']['f1']:.4f} | final F1 {f1m:.4f}+-{f1s:.4f} "
              f"recall {rm:.4f} FPR {fm:.4f} | with the undefended theta/window: F1 {f1o:.4f} "
              f"| {record['runtime_seconds']:.0f}s", flush=True)
    import csv
    with open(out / "summary.csv", "w", newline="") as fd:
        w = csv.DictWriter(fd, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    return status


def main() -> int:
    args = parse_args()
    if args.cmd == "calibrate":
        return calibrate(args)
    from purification.attack import run_attack
    return run_attack(args)


if __name__ == "__main__":
    sys.exit(main())

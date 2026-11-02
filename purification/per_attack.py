"""Detection of every WHOLE attack of the final split, without an attacker.

The attacks of phase 2 are measured on a central block of 64 rows of every attack, and
row-level F1 is dominated by one attack (doc. 28: 77% of the attack rows of the final
split). This writes, for each defense and each multiple of its theta:

  per_attack.csv     one row per attack run: length, share of its rows flagged, share of
                     the draws in which at least one of its rows is flagged, delay of the
                     first flagged row (median over those draws);
  final_metrics.csv  F1, precision, recall, FPR on the whole final split, only for the
                     undefended detector (t* = 0), whose scores are computed on every row.

With a purifier only the attack runs of at most --max-run rows are scored (a run longer
than that is not reported: doc. 28 alone would be 3 times all the others).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from adversarial.detector import repo_cached_detect
from .attack import DefendedDetector, kept_channels
from .calibration import build_splits, detection_metrics
from .common import load_data, load_detector, load_purifier
from .instances import Scorer, make_geometry


def add_per_attack_args(ap):
    ap.add_argument("--defense", nargs="+", required=True, help="defense_*.json written by calibrate")
    ap.add_argument("--theta-factors", type=float, nargs="+", default=[1.0],
                    help="multiples of each defense's theta to evaluate (same window)")
    ap.add_argument("--eval-draws", type=int, default=5)
    ap.add_argument("--max-run", type=int, default=10000,
                    help="with a purifier, attack runs longer than this are not scored")


def attack_runs(labels, first: int):
    """Contiguous attack runs as local indices of the split chunk; labels start at `first`."""
    idx = first + np.flatnonzero(np.asarray(labels).astype(int) == 1)
    if len(idx) == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)


def run_stats(flags_by_draw, run):
    """Share of rows flagged, share of draws with a flagged row, median delay; flags_by_draw
    (draws, n) are the decisions on the rows of the run."""
    flags = np.asarray(flags_by_draw, dtype=bool)
    hit = flags.any(axis=1)
    delays = [int(np.argmax(f)) for f in flags[hit]]
    return dict(attack_start=int(run[0]), attack_rows=len(run), flagged_share=float(flags.mean()),
                detected_share=float(hit.mean()), delay_median=float(np.median(delays)) if delays else "")


def run_per_attack(args) -> int:
    out = Path(args.out)
    if (out / "per_attack.csv").exists():
        raise SystemExit(f"{out}/per_attack.csv exists; use a new --out (nothing is overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    model, model_type, history, params, name = load_detector(args.model_dir)
    x_train, x_test, y_test, cols = load_data(args.dataset)
    per_attack, final = [], []
    for path in args.defense:
        with open(path) as fd:
            dfn = json.load(fd)
        if dfn["model_dir"] != args.model_dir:
            raise SystemExit(f"{path}: defense of {dfn['model_dir']}, not of {args.model_dir}")
        geom = make_geometry(model_type, history, int(dfn["target_offset"]), int(dfn["context"]),
                             int(dfn.get("lookahead", 0)))
        splits = build_splits(args.dataset, x_train, x_test, y_test, geom)
        off, n, labels = splits.final_offset, splits.final_len, splits.final_labels
        first = geom.first_target
        purifier, _ = load_purifier(dfn.get("diffusion"), int(dfn["t_star"]), int(dfn["steps"]), dfn["sampler"],
                                    kept_channels(dfn, cols))
        scorer = Scorer(model, purifier, geom, args.batch)
        theta, window = float(dfn["theta"]), int(dfn["window"])
        runs = attack_runs(labels, first)
        if purifier.identity:
            positions = np.arange(first, n)
            draws = 1
        else:
            runs = [r for r in runs if len(r) <= args.max_run]
            positions = DefendedDetector(scorer, theta, window, model_type, off, n, first).near(np.concatenate(runs))
            draws = args.eval_draws
        scores = np.zeros((draws, n - first))
        for d in range(draws):
            scores[d, positions - first] = scorer.errors(x_test, off + positions, 40_000_000 + 1000 * d).mean(axis=1)
        base = dict(model=name, t_star=int(dfn["t_star"]), sampler=dfn["sampler"],
                    keep_actuators=bool(dfn.get("keep_actuators")), window=window, draws=draws)
        for k in args.theta_factors:
            th = theta * k
            det = np.stack([repo_cached_detect(s, th, window, model_type, "same") for s in scores])
            for r in runs:
                per_attack.append(dict(base, theta_factor=k, theta=th, chunk_offset=off,
                                       **run_stats(det[:, r - first], r)))
            if purifier.identity:
                m = detection_metrics(scores[0], labels, th, window, model_type)
                final.append(dict(base, theta_factor=k, theta=th, **m))
                stored = params.get("final_f1")
                if k == 1.0 and stored is not None and abs(m["f1"] - stored) > 1e-3:
                    raise SystemExit(f"t*=0 F1 {m['f1']:.6f} differs from the stored {stored:.6f}")
        print(f"{path}: {len(runs)} attacks scored", flush=True)
    for fname, rows in (("per_attack.csv", per_attack), ("final_metrics.csv", final)):
        if rows:
            with open(out / fname, "w", newline="") as fd:
                w = csv.DictWriter(fd, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return 0

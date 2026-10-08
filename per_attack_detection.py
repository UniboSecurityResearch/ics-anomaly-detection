#!/usr/bin/env python3
"""Per-attack detection of trained SWAT-CLEAN detectors on the whole test file.

For every model directory given on the command line, this recomputes the
operative detector exactly as the attack code does (mean squared error per
timestep, threshold and window from the model's .json, the model's own window
rule) over the WHOLE test series, then reports for each labelled attack:
  - length (timesteps)
  - fraction of its timesteps flagged (window detection)
  - whether it was detected at all
  - delay from attack start to first flagged timestep
and, per model, the number of benign false-alarm runs.

Output: one CSV per model in --out-dir plus a printed summary.

Run from the repository root:
  python3 per_attack_detection.py --out-dir outputs/per_attack models/swatclean_cnn ...
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from data_loader import load_test_data
from adversarial.constants import POINT_MODELS
from adversarial.detector import repo_cached_detect
from adversarial.errors import model_errors_numpy, instance_detection_scores
from adversarial.io_utils import load_model_adapter, load_scaler
from adversarial.targets import infer_attack_labels, valid_target_bounds


def segments(mask):
    padded = np.r_[False, mask.astype(bool), False]
    diff = np.diff(padded.astype(int))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def analyse(model_dir, dataset, out_dir, batch_size):
    jsons = sorted(glob.glob(os.path.join(model_dir, "*.json")))
    if not jsons:
        print(f"SKIP {model_dir}: no .json")
        return None
    params = json.load(open(jsons[0]))
    name = os.path.basename(jsons[0])[:-5]
    model_type = name.split("-")[0]
    theta = float(params["best_theta"])
    window = int(params["best_window"])
    history = None if model_type in POINT_MODELS else int(params["history"])
    target_offset = 0

    adapter = load_model_adapter(os.path.join(model_dir, name + ".h5"))
    scaler = load_scaler(dataset)
    x_test, labels, _ = load_test_data(dataset, scaler=scaler)
    x_test = np.asarray(x_test, dtype=np.float32)
    attack = np.asarray(infer_attack_labels(dataset, np.asarray(labels)), dtype=bool)

    first, last = valid_target_bounds(model_type, len(x_test), history, target_offset)
    idx = np.arange(first, last, dtype=np.int64)
    errors = model_errors_numpy(adapter, x_test, model_type, idx, history, target_offset, batch_size)
    scores = instance_detection_scores(errors, "mean_mse")
    flagged = np.zeros(len(x_test), dtype=bool)
    flagged[idx] = repo_cached_detect(scores, theta, window, model_type)

    rows = []
    for k, (s, e) in enumerate(segments(attack)):
        f = flagged[s:e]
        hits = np.flatnonzero(f)
        rows.append({
            "attack": k + 1, "start": int(s), "end": int(e), "length": int(e - s),
            "flagged_fraction": float(f.mean()),
            "detected": bool(hits.size > 0),
            "delay": int(hits[0]) if hits.size else -1,
        })
    fa_runs = sum(1 for s, e in segments(flagged & ~attack))
    benign_fpr = float(flagged[~attack][first:].mean()) if np.any(~attack) else float("nan")
    point_recall = float(flagged[attack].mean())

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(model_dir))
    with open(os.path.join(out_dir, f"{tag}.csv"), "w") as fd:
        fd.write("attack,start,end,length,flagged_fraction,detected,delay\n")
        for r in rows:
            fd.write("{attack},{start},{end},{length},{flagged_fraction:.4f},{detected},{delay}\n".format(**r))

    detected = sum(r["detected"] for r in rows)
    print(f"\n=== {tag} ({name}) theta={theta:.6g} window={window}")
    print(f"timestep recall={point_recall:.3f}  attacks detected={detected}/{len(rows)}  "
          f"false-alarm runs={fa_runs}  benign FPR={benign_fpr:.4f}")
    print("attack  length  flagged  delay")
    for r in rows:
        print(f"{r['attack']:>6}  {r['length']:>6}  {r['flagged_fraction']:>7.2f}  "
              f"{r['delay'] if r['detected'] else 'missed':>5}")
    return tag, detected, len(rows), point_recall, fa_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dirs", nargs="+")
    ap.add_argument("--dataset", default="SWAT-CLEAN")
    ap.add_argument("--out-dir", default="outputs/per_attack")
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()
    summary = [s for s in (analyse(d, args.dataset, args.out_dir, args.batch_size)
                           for d in args.model_dirs) if s]
    print("\n=== SUMMARY")
    print("model                     attacks_detected  timestep_recall  false_alarm_runs")
    for tag, det, n, rec, fa in summary:
        print(f"{tag:<25} {det:>3}/{n:<12} {rec:>15.3f} {fa:>17}")


if __name__ == "__main__":
    sys.exit(main())

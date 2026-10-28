#!/usr/bin/env python3
"""Per-target analysis of purification attacks (read-only; needs per_target.json,
written by main_purification.py attack since the phase-2 commit).

    python3 purification_per_target.py RUN [RUN_B] [--boot 2000]

RUN is an attack output folder (with per_target.json and summary.json).
* the decision geometry: the share of targets whose whole detection window lies inside
  the target blocks (the most an attacker who can only change target rows can flag in
  a false-alarm attack, when the rows around the blocks stay below theta);
* the result per target block (one block per attack in the split), clean and after
  the attack, with a 95% bootstrap interval over blocks for the share flagged;
* why targets stay flagged after an evasion attack: how many of the timesteps in their
  window are modifiable and how many crossed theta;
* with RUN_B (same targets, e.g. the undefended and the defended arm): the paired
  difference of the share flagged (B - A) with a 95% bootstrap interval over blocks.
"""

import argparse
import json
import os
import sys

import numpy as np


def load(run):
    with open(os.path.join(run, "per_target.json")) as fd:
        p = json.load(fd)
    with open(os.path.join(run, "summary.json")) as fd:
        s = json.load(fd)
    for k in ("targets", "eligible", "clean_flag_rate", "adv_flag_rate", "scored", "scored_label",
              "modifiable", "clean_score", "adv_score"):
        p[k] = np.asarray(p[k])
    p["goal"] = s["goal"]
    p["blocks"] = np.concatenate([[0], np.cumsum(np.diff(p["targets"]) > 1)])
    return p


def window_members(p):
    """For each target, the positions (into `scored`) of the timesteps its centered
    window uses: np.convolve(det, ones(w), 'same') at i covers i-w//2 .. i+(w-1)//2... as
    numpy aligns it (checked numerically below)."""
    w = int(p["window"])
    n = int(p["scored"].max()) + w + 2
    probe = np.zeros(n)
    probe[n // 2] = 1
    cover = np.flatnonzero(np.convolve(probe, np.ones(w), "same") > 0) - n // 2   # i with window ∋ n//2
    offsets = -cover  # window of i contains i + offsets
    pos = {int(v): k for k, v in enumerate(p["scored"])}
    return [[pos.get(int(t) + int(o)) for o in offsets] for t in p["targets"]]


def boot_ci(values_by_block, n_boot, rng):
    blocks = list(values_by_block)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        v = np.concatenate([values_by_block[blocks[i]] for i in pick])
        stats.append(v.mean())
    return np.percentile(stats, [2.5, 97.5])


def describe(run, p, n_boot, rng):
    t = p["targets"]
    w = int(p["window"])
    inside = np.zeros(len(t), bool)
    tset = set(t.tolist())
    mem = window_members(p)
    for i, ti in enumerate(t):
        inside[i] = all((ti + o) in tset for o in range(-(w // 2), w - w // 2))
    print(f"== {run}\n  goal {p['goal']}, theta {p['theta']:.5g}, window {w}, {len(t)} targets in "
          f"{p['blocks'].max() + 1} blocks; window entirely inside the target blocks: {inside.mean():.4f}")
    clean = p["clean_flag_rate"] > 0.5
    adv = p["adv_flag_rate"] > 0.5
    print(f"  flagged (majority of draws): clean {clean.mean():.4f}, after the attack {adv.mean():.4f}"
          f" (mean rates {p['clean_flag_rate'].mean():.4f} -> {p['adv_flag_rate'].mean():.4f})")
    ub = np.unique(p["blocks"])
    ep_c = np.mean([clean[p["blocks"] == b].any() for b in ub])
    ep_a = np.mean([adv[p["blocks"] == b].any() for b in ub])
    print(f"  ceiling of the share flagged (targets whose window lies inside their block): {inside.mean():.4f}")
    print(f"  blocks with at least one target flagged (episode level): clean {ep_c:.4f}, after the attack {ep_a:.4f} "
          f"({len(ub)} blocks)")
    byb = {b: p["adv_flag_rate"][p["blocks"] == b] for b in ub}
    lo, hi = boot_ci(byb, n_boot, rng)
    print(f"  share flagged after the attack: {p['adv_flag_rate'].mean():.4f}, 95% interval over blocks [{lo:.4f}, {hi:.4f}]")
    print("  block  first_target  n   clean  attacked")
    for b in np.unique(p["blocks"]):
        m = p["blocks"] == b
        print(f"  {b:>5}  {t[m][0]:>12}  {m.sum():<3} {p['clean_flag_rate'][m].mean():.3f}  {p['adv_flag_rate'][m].mean():.3f}")
    if p["goal"] == "evasion":
        above_c = p["clean_score"] > p["theta"]
        above_a = p["adv_score"] > p["theta"]
        mod = p["modifiable"] > 0
        crossed = above_c & ~above_a
        print(f"  scored timesteps: {len(above_c)}, modifiable {mod.sum()}, above theta clean {above_c.sum()}, "
              f"after {above_a.sum()}; pushed below theta: {crossed.sum()} (of which modifiable {(crossed & mod).sum()})")
        still = np.flatnonzero(adv)
        n_mod = [sum(1 for k in mem[i] if k is not None and mod[k]) for i in still]
        print(f"  targets still flagged: {len(still)}; of them with the window entirely inside the "
              f"blocks {inside[still].mean() if len(still) else float('nan'):.3f}; modifiable timesteps in "
              f"their window: median {np.median(n_mod) if n_mod else float('nan')} of {w}")
        ratio = p["adv_score"][mod] / p["theta"]
        if len(ratio):
            print(f"  modifiable timesteps, score / theta after the attack: quartiles "
                  f"{np.round(np.percentile(ratio, [25, 50, 75]), 2).tolist()}, below 1: {(ratio < 1).mean():.3f}")
    return byb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    ps = [load(r) for r in args.runs[:2]]
    byb = [describe(r, p, args.boot, rng) for r, p in zip(args.runs, ps)]
    if len(ps) == 2:
        a, b = ps
        if not np.array_equal(a["targets"], b["targets"]):
            print("the two runs do not share the targets: no paired difference")
            return 1
        diff = {k: byb[1][k] - byb[0][k] for k in byb[0]}
        lo, hi = boot_ci(diff, args.boot, rng)
        d = b["adv_flag_rate"].mean() - a["adv_flag_rate"].mean()
        print(f"== paired difference (B - A) of the share flagged after the attack: {d:+.4f}, "
              f"95% interval over blocks [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

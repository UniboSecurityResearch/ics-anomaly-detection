#!/usr/bin/env python3
"""Collect the purification results into tables (CSV + text), read-only.

    python3 purification_report.py outputs/purification/phase1 [more roots]

For every calibrate summary.csv: the clean-detection table (t*, threshold, window,
F1 / recall / FPR on the final split, mean +- std over purification draws).
For every attack summary.json: one row per (model, t*, K, attacker, goal, split,
budget): clean and adversarial flagged rate, detection rate on E and ASR (evasion),
mean +- std over evaluation draws, iterations, EOT, runtime.
"""

import argparse
import csv
import glob
import json
import os
import sys


def attack_rows(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
        with open(path) as fd:
            s = json.load(fd)
        if "adversarial" not in s:
            continue
        r = s["reported"]
        a = s["attacker"]
        rows.append(dict(
            run=os.path.relpath(os.path.dirname(path), root), model=s["model"], split=s["split"],
            goal=s["goal"], t_star=s["defense"]["t_star"], steps=s["defense"]["purifier"]["n_denoising_steps"],
            sampler=s["defense"]["sampler"], theta=s["defense"]["theta"], window=s["defense"]["window"],
            attacker=a["gradient"], attacker_steps=a["purifier"]["n_denoising_steps"], eot=a["eot"],
            iterations=a["iterations"], restarts=a["restarts"], alpha_rel=a["alpha_relative"],
            eps_range=a["epsilon_range_fraction"], protected=a["protect_actuators"],
            modifiable=a.get("modifiable", "window" if a["gradient"] not in ("blind",) and s["defense"]["t_star"] > 0 else "detector"),
            modifiable_rows=a.get("modifiable_rows", ""), random_start=a.get("random_start", True),
            targets=s["n_targets"], eligible=s["n_eligible"], draws=s["eval_draws"],
            clean_flagged=s["clean"]["flagged_mean"], clean_flagged_std=s["clean"]["flagged_std"],
            adv_flagged=r["flagged_mean"], adv_flagged_std=r["flagged_std"],
            dr_E=r.get("dr_E_mean", ""), dr_E_std=r.get("dr_E_std", ""), asr=r.get("asr_mean", ""),
            point=s["reported_point"], linf=r["linf_scaled"], changed_cells=r["changed_cells"],
            minutes=round(s["runtime_seconds"] / 60, 1),
            stopped_early=bool(s.get("stopped_early"))))
    return rows


def calib_rows(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "summary.csv"), recursive=True)):
        with open(path) as fd:
            for r in csv.DictReader(fd):
                if "final_f1" in r:
                    r["run"] = os.path.relpath(os.path.dirname(path), root)
                    rows.append(r)
    return rows


def write(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fd:
        w = csv.DictWriter(fd, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    args = ap.parse_args()
    for root in args.roots:
        cal = calib_rows(root)
        att = attack_rows(root)
        write(cal, os.path.join(root, "table_calibration.csv"))
        write(att, os.path.join(root, "table_attacks.csv"))
        print(f"== {root}: clean detection with purification (final split)")
        for r in cal:
            print(f"  {r['run']:<28} t*={r['t_star']:>4} K={r['steps']:>3} pct={r['percentile']:<7} "
                  f"w={r['window']:>3} | tune F1 {fmt(r['tune_f1'])} | final F1 {fmt(r['final_f1'])}"
                  f"+-{fmt(r['final_f1_std'])} recall {fmt(r['final_recall'])} FPR {fmt(r['final_fpr'])}")
        print(f"== {root}: attacks")
        for r in sorted(att, key=lambda r: (r["model"], r["goal"], r["modifiable"], r["eps_range"], r["t_star"],
                                            r["attacker"], r["alpha_rel"], r["iterations"], r["eot"])):
            extra = (f" DR_E {fmt(r['dr_E'])}+-{fmt(r['dr_E_std'])} ASR {fmt(r['asr'])}"
                     if r["goal"] == "evasion" else "")
            print(f"  {r['model'][:22]:<22} {r['goal']:<11} eps={r['eps_range']:<5} rows={r['modifiable'][:3]} "
                  f"a={r['alpha_rel']:<5} it={r['iterations']:<3} n={r['targets']:<4} "
                  f"{'' if r['protected'] else 'actFREE '}{'' if r['random_start'] else 'noRS '}t*={r['t_star']:>4} "
                  f"K={r['steps']:>3} {r['attacker']:<9}(K={r['attacker_steps']},EOT={r['eot']}) "
                  f"flagged {fmt(r['clean_flagged'])} -> {fmt(r['adv_flagged'])}+-{fmt(r['adv_flagged_std'])}"
                  f"{extra} [{r['split']}, {r['minutes']} min{', STOPPED EARLY' if r['stopped_early'] else ''}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

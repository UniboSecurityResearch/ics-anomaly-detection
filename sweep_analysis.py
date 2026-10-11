#!/usr/bin/env python3
"""Summarise the evasion sweep (outputs/sweep_v2). Standard library only, so it
runs on the login node:

    python3 sweep_analysis.py                       # all runs found
    python3 sweep_analysis.py --out outputs/sweep_v2/analysis

Layout read:  <root>/<run>/<budget>/<attack>/<name>/evasion/
                  checkpoint_metrics.json  iteration_metrics.csv  attack_scores.csv
with <run> = swatchrono_<model>_s<seed>.

Writes three files in --out and prints the by-model table:
  per_run.csv   one row per run, budget and attack
  by_model.csv  mean and standard deviation over the training seeds
  evaded.csv    which targets evade, as index ranges, and which other attack of
                the same run and budget evades exactly the same targets

Definitions. E = targets detected on clean data (eligible). Evasions = targets
of E no longer detected. ASR = evasions / |E|. Clean DR = |E| / targets.
"asr" is the checkpoint with most evasions (best_asr), "asr_minloss" the
checkpoint with lowest loss (best_loss). The standard deviation is the sample
one (n - 1); it is empty with a single seed.
Flags: last_it = the best_asr point is the last iteration (attack not
converged, the value is a lower bound); dr_up = detection after the attack is
higher than on clean data (the attack made things worse); incomplete = no
result files yet.
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

RUN_RE = re.compile(r"^(?P<prefix>.+)_(?P<model>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
BUDGET_ORDER = ["range0.02_protected", "range0.05_protected", "range0.10_protected",
                "range0.20_protected", "range0.10_unprotected", "std0.10_protected"]
BEST = "BEST"  # strongest attack per run and budget


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def read_scores(path):
    """Eligible and evaded target indices from attack_scores.csv."""
    eligible, evaded, total = [], [], 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            index = int(float(row["target_index"]))
            if truthy(row.get("eligible", "")):
                eligible.append(index)
            if truthy(row.get("attack_success", "")):
                evaded.append(index)
    return total, eligible, evaded


def last_iteration(path):
    last = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != "trajectory":
                continue
            value = int(row["iteration"])
            last = value if last is None else max(last, value)
    return last


def ranges(indices):
    """[3,4,5,9] -> '3-5 9' and the number of ranges."""
    out, start, prev = [], None, None
    for value in sorted(indices):
        if start is None:
            start = prev = value
        elif value == prev + 1:
            prev = value
        else:
            out.append((start, prev))
            start = prev = value
    if start is not None:
        out.append((start, prev))
    text = " ".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in out)
    return text, len(out)


def collect(root):
    rows, evaded_sets = [], {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        model, seed = match.group("model").upper(), int(match.group("seed"))
        for budget_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            for attack_dir in sorted(p for p in budget_dir.iterdir() if p.is_dir()):
                row = dict(run=run_dir.name, model=model, seed=seed,
                           budget=budget_dir.name, attack=attack_dir.name)
                scores = sorted(attack_dir.glob("*/evasion/attack_scores.csv"))
                checkpoint = sorted(attack_dir.glob("*/evasion/checkpoint_metrics.json"))
                if not scores:
                    row["flags"] = "incomplete"
                    rows.append(row)
                    continue
                total, eligible, evaded = read_scores(scores[0])
                row.update(targets=total, eligible=len(eligible),
                           clean_dr=len(eligible) / total if total else "",
                           evasions=len(evaded),
                           asr=len(evaded) / len(eligible) if eligible else "",
                           dr_after=(len(eligible) - len(evaded)) / total if total else "")
                flags = []
                if checkpoint:
                    best = json.load(checkpoint[0].open())
                    a, b = best["best_asr"], best["best_loss"]
                    trace = checkpoint[0].with_name("iteration_metrics.csv")
                    n_it = last_iteration(trace) if trace.exists() else None
                    row.update(evasions=a["evasions"], asr=a["asr"],
                               dr_after=a["detection_rate"], iteration=a["iteration"],
                               restart=a["restart"], iterations_run=n_it,
                               evasions_minloss=b["evasions"], asr_minloss=b["asr"],
                               dr_after_minloss=b["detection_rate"],
                               iteration_minloss=b["iteration"], loss_min=b["loss"])
                    if a["evasions"] != len(evaded):
                        flags.append("scores_mismatch")
                    if n_it and n_it > 1 and a["iteration"] == n_it:  # not for one-step FGSM
                        flags.append("last_it")
                    if b["detection_rate"] > row["clean_dr"] + 1e-12:
                        flags.append("dr_up_minloss")
                if row["dr_after"] != "" and row["dr_after"] > row["clean_dr"] + 1e-12:
                    flags.append("dr_up")
                row["flags"] = " ".join(flags)
                rows.append(row)
                evaded_sets[(run_dir.name, budget_dir.name, attack_dir.name)] = frozenset(evaded)
    return rows, evaded_sets


def mean_std(values):
    values = [v for v in values if v != "" and v is not None]
    if not values:
        return "", ""
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else "")


def aggregate(rows):
    done = [r for r in rows if r.get("flags") != "incomplete"]
    groups = defaultdict(list)
    for r in done:
        groups[(r["model"], r["budget"], r["attack"])].append(r)
    # Strongest attack: per run and budget, the attack with most evasions.
    per_run = defaultdict(list)
    for r in done:
        per_run[(r["model"], r["budget"], r["seed"])].append(r)
    for (model, budget, _), items in per_run.items():
        top = max(items, key=lambda r: r["evasions"])
        groups[(model, budget, BEST)].append(dict(top, best_attack=top["attack"]))
    out = []
    for (model, budget, attack), items in groups.items():
        asr_m, asr_s = mean_std([r["asr"] for r in items])
        loss_m, loss_s = mean_std([r.get("asr_minloss", "") for r in items])
        out.append(dict(
            model=model, budget=budget, attack=attack, n_seeds=len(items),
            seeds=" ".join(str(s) for s in sorted(r["seed"] for r in items)),
            clean_dr_mean=mean_std([r["clean_dr"] for r in items])[0],
            eligible_mean=mean_std([r["eligible"] for r in items])[0],
            evasions_mean=mean_std([r["evasions"] for r in items])[0],
            asr_mean=asr_m, asr_std=asr_s,
            asr_minloss_mean=loss_m, asr_minloss_std=loss_s,
            dr_after_mean=mean_std([r["dr_after"] for r in items])[0],
            n_last_it=sum("last_it" in r["flags"].split() for r in items),
            n_dr_up=sum("dr_up" in r["flags"].split() for r in items),
            best_attacks=" ".join(sorted(r["best_attack"] for r in items)) if attack == BEST else ""))
    order = {b: i for i, b in enumerate(BUDGET_ORDER)}
    out.sort(key=lambda r: (r["model"], order.get(r["budget"], 99), r["budget"],
                            r["attack"] == BEST, r["attack"]))
    return out


def evaded_table(rows, evaded_sets):
    out, first_with = [], {}
    for r in rows:
        key = (r["run"], r["budget"], r["attack"])
        if key not in evaded_sets:
            continue
        current = evaded_sets[key]
        text, n_ranges = ranges(current)
        same = first_with.setdefault((r["run"], current), "%s/%s" % (r["budget"], r["attack"]))
        out.append(dict(run=r["run"], budget=r["budget"], attack=r["attack"],
                        evaded=len(current), n_ranges=n_ranges,
                        same_set_as="" if same == "%s/%s" % (r["budget"], r["attack"]) or not current else same,
                        ranges=text))
    return out


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("%.6g" % v if isinstance(v, float) else v)
                             for k, v in row.items()})


def fmt(mean, std, n):
    if mean == "":
        return "-"
    return "%.3f" % mean if std == "" else "%.3f +- %.3f" % (mean, std)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="outputs/sweep_v2")
    parser.add_argument("--out", default=None, help="default: <root>/analysis")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows, evaded_sets = collect(root)
    table = aggregate(rows)
    write_csv(out / "per_run.csv", rows,
              ["run", "model", "seed", "budget", "attack", "targets", "eligible", "clean_dr",
               "evasions", "asr", "dr_after", "iteration", "restart", "iterations_run",
               "evasions_minloss", "asr_minloss", "dr_after_minloss", "iteration_minloss",
               "loss_min", "flags"])
    write_csv(out / "by_model.csv", table,
              ["model", "budget", "attack", "n_seeds", "seeds", "clean_dr_mean",
               "eligible_mean", "evasions_mean", "asr_mean", "asr_std", "asr_minloss_mean",
               "asr_minloss_std", "dr_after_mean", "n_last_it", "n_dr_up", "best_attacks"])
    write_csv(out / "evaded.csv", evaded_table(rows, evaded_sets),
              ["run", "budget", "attack", "evaded", "n_ranges", "same_set_as", "ranges"])

    line = "%-5s %-22s %-17s %-7s %8s %9s  %-15s %-15s %8s  %s"
    print(line % ("model", "budget", "attack", "seeds", "clean DR", "evasions",
                  "ASR", "ASR min-loss", "DR after", "notes"))
    for r in table:
        notes = []
        if r["n_last_it"]:
            notes.append("last_it x%d" % r["n_last_it"])
        if r["n_dr_up"]:
            notes.append("dr_up x%d" % r["n_dr_up"])
        if r["best_attacks"]:
            notes.append("from: " + r["best_attacks"])
        print(line % (r["model"], r["budget"], r["attack"], r["n_seeds"],
                      "%.3f" % r["clean_dr_mean"], "%.1f" % r["evasions_mean"],
                      fmt(r["asr_mean"], r["asr_std"], r["n_seeds"]),
                      fmt(r["asr_minloss_mean"], r["asr_minloss_std"], r["n_seeds"]),
                      "%.3f" % r["dr_after_mean"], "; ".join(notes)))
    incomplete = [r for r in rows if r.get("flags") == "incomplete"]
    mismatch = [r for r in rows if "scores_mismatch" in r.get("flags", "").split()]
    print("\nruns read: %d   incomplete: %d   scores_mismatch: %d"
          % (len(rows) - len(incomplete), len(incomplete), len(mismatch)))
    for r in incomplete + mismatch:
        print("  %s  %s/%s/%s" % (r["flags"], r["run"], r["budget"], r["attack"]))
    print("written: %s/{per_run,by_model,evaded}.csv" % out)


if __name__ == "__main__":
    main()

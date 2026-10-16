#!/usr/bin/env python3
"""Summarise the evasion sweep (outputs/sweep_v2). Standard library only, so it
runs on the login node:

    python3 sweep_analysis.py                       # all runs found
    python3 sweep_analysis.py --out outputs/sweep_v2/analysis

Layout read:  <root>/<run>/<budget>/<attack>/<name>/evasion/
                  checkpoint_metrics.json  iteration_metrics.csv  attack_scores.csv
with <run> = swatchrono_<model>_s<seed>.

Writes three files in --out and prints the by-model tables, including how many
whole attacks are hidden (a run of consecutive target indices = one attack):
  per_run.csv   one row per run, budget and attack
  by_model.csv  mean and standard deviation over the training seeds
  evaded.csv    which targets evade, as index ranges, and which other attack of
                the same run and budget evades exactly the same targets

Definitions. E = targets detected on clean data (eligible). Evasions = targets
of E no longer detected. ASR = evasions / |E|. Clean DR = |E| / targets.
"asr" is the checkpoint with most evasions (best_asr), "asr_minloss" the
checkpoint with lowest loss (best_loss). The standard deviation is the sample
one (n - 1); it is empty with a single seed.
"Final split only": the first 30% of the SWaT test file (rows before
--final-start, default 134975 of 449919) is the part used in training to choose
theta and the window, and it holds 14 of the 32 attacks. The *_final columns
and the second printed table count only the targets after that row, i.e. the
18 attacks the detector was not tuned on. They refer to the same checkpoint
as "asr" (most evasions over all targets).
Without checkpoint_metrics.json (CorrShift) "DR after" is the share of all targets
still detected after the attack (detected_after_window), so new alarms on missed
targets count; only files without that column fall back to (|E| - evasions) / targets.
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
BUDGET_ORDER = ["range0.01_protected", "range0.02_protected", "range0.05_protected", "range0.10_protected",
                "range0.20_protected", "range0.10_unprotected", "std0.10_protected"]
BEST = "BEST"  # strongest attack per run and budget


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def read_scores(path):
    """Eligible and evaded target indices from attack_scores.csv, plus all target
    indices and those still detected after the attack (None if not saved)."""
    eligible, evaded, targets, after = [], [], [], []
    has_after = True
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(float(row["target_index"]))
            targets.append(index)
            if truthy(row.get("eligible", "")):
                eligible.append(index)
            if truthy(row.get("attack_success", "")):
                evaded.append(index)
            if "detected_after_window" not in row:
                has_after = False
            elif truthy(row["detected_after_window"]):
                after.append(index)
    return len(targets), eligible, evaded, targets, (after if has_after else None)


def final_part(row, final_start, targets, eligible, evaded, after):
    """Same counts restricted to targets at or after row final_start."""
    n = sum(t >= final_start for t in targets)
    elig = sum(t >= final_start for t in eligible)
    evad = sum(t >= final_start for t in evaded)
    row.update(targets_final=n, eligible_final=elig, evasions_final=evad,
               clean_dr_final=elig / n if n else "",
               asr_final=evad / elig if elig else "",
               dr_after_final="")
    if after is not None and n:
        row["dr_after_final"] = sum(t >= final_start for t in after) / n


def attack_counts(row, targets, eligible, after, final_start):
    """Whole attacks. Targets are one contiguous block in the middle of each labelled
    attack, so each run of consecutive target indices is one attack. An attack is
    detected if at least one of its targets is flagged (the original repository's
    per-attack rule); it is hidden if it was detected before the attack and none of
    its targets is flagged after. Needs detected_after_window."""
    if after is None or not targets:
        return
    order = sorted(targets)
    blocks, start, prev = [], order[0], order[0]
    for t in order[1:]:
        if t != prev + 1:
            blocks.append((start, prev))
            start = t
        prev = t
    blocks.append((start, prev))
    before_set, after_set = set(eligible), set(after)
    counts = dict(attacks=0, attacks_before=0, attacks_after=0, attacks_hidden=0,
                  attacks_final=0, attacks_before_final=0, attacks_hidden_final=0)
    for a, b in blocks:
        det_before = any(t in before_set for t in range(a, b + 1))
        det_after = any(t in after_set for t in range(a, b + 1))
        final = a >= final_start
        counts["attacks"] += 1
        counts["attacks_before"] += det_before
        counts["attacks_after"] += det_after
        counts["attacks_hidden"] += det_before and not det_after
        counts["attacks_final"] += final
        counts["attacks_before_final"] += final and det_before
        counts["attacks_hidden_final"] += final and det_before and not det_after
    row.update(counts)


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


def collect(root, final_start):
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
                total, eligible, evaded, targets, after = read_scores(scores[0])
                final_part(row, final_start, targets, eligible, evaded, after)
                attack_counts(row, targets, eligible, after, final_start)
                row.update(targets=total, eligible=len(eligible),
                           clean_dr=len(eligible) / total if total else "",
                           evasions=len(evaded),
                           asr=len(evaded) / len(eligible) if eligible else "",
                           dr_after=((len(after) if after is not None
                                      else len(eligible) - len(evaded)) / total
                                     if total else ""))
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
        fin_m, fin_s = mean_std([r["asr_final"] for r in items])
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
            clean_dr_final_mean=mean_std([r["clean_dr_final"] for r in items])[0],
            eligible_final_mean=mean_std([r["eligible_final"] for r in items])[0],
            evasions_final_mean=mean_std([r["evasions_final"] for r in items])[0],
            asr_final_mean=fin_m, asr_final_std=fin_s,
            dr_after_final_mean=mean_std([r["dr_after_final"] for r in items])[0],
            attacks_before_mean=mean_std([r.get("attacks_before", "") for r in items])[0],
            attacks_hidden_mean=mean_std([r.get("attacks_hidden", "") for r in items])[0],
            attacks_before_final_mean=mean_std([r.get("attacks_before_final", "") for r in items])[0],
            attacks_hidden_final_mean=mean_std([r.get("attacks_hidden_final", "") for r in items])[0],
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
    parser.add_argument("--final-start", type=int, default=134975,
                        help="first test row of the final split (rows before it were "
                             "used to tune theta and window); default is SWaT, "
                             "449919 - ceil(0.7 * 449919)")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows, evaded_sets = collect(root, args.final_start)
    table = aggregate(rows)
    write_csv(out / "per_run.csv", rows,
              ["run", "model", "seed", "budget", "attack", "targets", "eligible", "clean_dr",
               "evasions", "asr", "dr_after", "iteration", "restart", "iterations_run",
               "evasions_minloss", "asr_minloss", "dr_after_minloss", "iteration_minloss",
               "loss_min", "targets_final", "eligible_final", "clean_dr_final",
               "evasions_final", "asr_final", "dr_after_final", "attacks", "attacks_before",
               "attacks_after", "attacks_hidden", "attacks_final", "attacks_before_final",
               "attacks_hidden_final", "flags"])
    write_csv(out / "by_model.csv", table,
              ["model", "budget", "attack", "n_seeds", "seeds", "clean_dr_mean",
               "eligible_mean", "evasions_mean", "asr_mean", "asr_std", "asr_minloss_mean",
               "asr_minloss_std", "dr_after_mean", "clean_dr_final_mean",
               "eligible_final_mean", "evasions_final_mean", "asr_final_mean",
               "asr_final_std", "dr_after_final_mean", "attacks_before_mean",
               "attacks_hidden_mean", "attacks_before_final_mean",
               "attacks_hidden_final_mean", "n_last_it", "n_dr_up", "best_attacks"])
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

    def num(value, pattern):
        return "-" if value == "" else pattern % value

    print("\nFinal split only (targets at or after test row %d: attacks not used to "
          "choose theta and window)" % args.final_start)
    line2 = "%-5s %-22s %-17s %-7s %8s %9s  %-15s %8s"
    print(line2 % ("model", "budget", "attack", "seeds", "clean DR", "evasions",
                   "ASR", "DR after"))
    for r in table:
        print(line2 % (r["model"], r["budget"], r["attack"], r["n_seeds"],
                       num(r["clean_dr_final_mean"], "%.3f"),
                       num(r["evasions_final_mean"], "%.1f"),
                       fmt(r["asr_final_mean"], r["asr_final_std"], r["n_seeds"]),
                       num(r["dr_after_final_mean"], "%.3f")))
    print("\nWhole attacks hidden (detected before, no target flagged after; "
          "mean over seeds; all 32 attacks and the 18 of the final split)")
    line3 = "%-5s %-22s %-17s %-7s %16s %16s"
    print(line3 % ("model", "budget", "attack", "seeds", "hidden/32", "hidden/18"))
    for r in table:
        if r["attack"] != BEST:
            continue
        print(line3 % (r["model"], r["budget"], r["attack"], r["n_seeds"],
                       num(r["attacks_hidden_mean"], "%.1f") + " of " +
                       num(r["attacks_before_mean"], "%.1f"),
                       num(r["attacks_hidden_final_mean"], "%.1f") + " of " +
                       num(r["attacks_before_final_mean"], "%.1f")))
    incomplete = [r for r in rows if r.get("flags") == "incomplete"]
    mismatch = [r for r in rows if "scores_mismatch" in r.get("flags", "").split()]
    print("\nruns read: %d   incomplete: %d   scores_mismatch: %d"
          % (len(rows) - len(incomplete), len(incomplete), len(mismatch)))
    for r in incomplete + mismatch:
        print("  %s  %s/%s/%s" % (r["flags"], r["run"], r["budget"], r["attack"]))
    print("written: %s/{per_run,by_model,evaded}.csv" % out)


if __name__ == "__main__":
    main()

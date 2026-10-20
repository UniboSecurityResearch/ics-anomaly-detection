"""PGD + EOT attacks against the purified (defended) detector.

Follows the evaluation recommended by Lee & Kim (ICCV 2023, Sec. 3.2):
  * PGD with Expectation Over Transformation (EOT) rather than AutoAttack;
  * gradients by DIRECT back-propagation through the whole defense (forward noising
    with the reparameterisation trick, every denoising step, the detector) -- no
    adjoint method; a SURROGATE process (same t*, fewer denoising steps) for the
    attacker as an alternative; BPDA only as a weaker reference (their Table 3);
  * stochastic defense evaluated with fresh randomness, over several draws.

Threat model and budget are those of the undefended benchmark (swat_sweep.sbatch):
L-infinity box of `--epsilon-range-fraction` of each feature's training range,
actuators protected on request, only the rows that feed the loss may change (the
purified windows of the loss targets, which for the defended detector are the
Lw-row windows), sign-gradient steps of `--alpha-relative` x epsilon, random start.

Goals:
  evasion      targets = attack-labelled timesteps of the chosen test split
               (contiguous blocks over every attack, adversarial.targets.sample_segment_blocks);
               loss = mean instance score over E, the targets the defended detector
               flags on clean data in the majority of the evaluation draws;
               metrics: detection rate (DR) on all targets and on E (ASR = 1 - DR_E).
  false_alarm  targets = normal timesteps (same sampling); loss = minus the mean
               score; metric: fraction of targets flagged (false-alarm rate).

Gradient modes (--gradient):
  full       the attacker differentiates the exact defense (same t*, K, sampler);
  surrogate  same t*, --attack-steps denoising steps (Lee & Kim's surrogate process);
  bpda       defense forward pass, identity backward pass;
  blind      the attacker ignores the defense (PGD on the undefended detector) and the
             result is evaluated on the defended one (non-adaptive / transfer).
With the identity defense (t* = 0) this is plain PGD on the undefended detector,
the reference the defense is compared with, computed by the same code.

Detection always uses the full theta + window detector of the (defended) detector
on the chunk of the test split, as in calibration (decisions at a target depend only
on scores within the detection window around it, so scores are recomputed there).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from adversarial.detector import repo_cached_detect
from adversarial.targets import actuator_indices, sample_segment_blocks
from .calibration import build_splits
from .common import git_state, load_data, load_detector, load_purifier
from .instances import Scorer, errors_tf, make_geometry, window_index
from .purifier import Purifier, seed_pair


def add_attack_args(ap):
    ap.add_argument("--defense", required=True, help="defense_*.json written by calibrate")
    ap.add_argument("--goal", choices=["evasion", "false_alarm"], required=True)
    ap.add_argument("--gradient", choices=["full", "surrogate", "bpda", "blind"], default="full")
    ap.add_argument("--attack-steps", type=int, default=None, help="denoising steps of the surrogate")
    ap.add_argument("--split", choices=["tune", "final"], default="final")
    ap.add_argument("--max-targets", type=int, default=2048)
    ap.add_argument("--epsilon-range-fraction", type=float, default=0.10)
    ap.add_argument("--protect-actuators", action="store_true")
    ap.add_argument("--alpha-relative", type=float, default=0.10)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--eot", type=int, default=20, help="EOT samples per iteration (1 if deterministic)")
    ap.add_argument("--eval-draws", type=int, default=10)
    ap.add_argument("--trace-every", type=int, default=10)
    ap.add_argument("--attack-batch", type=int, default=256)
    ap.add_argument("--max-minutes", type=float, default=600.0,
                    help="stop the PGD loop after this wall time and evaluate the point reached "
                         "(the job limit is 12 h); recorded as stopped_early")


class DefendedDetector:
    """theta + window decision of the defended detector on one test-split chunk."""

    def __init__(self, scorer: Scorer, theta, window, model_type, chunk_offset, chunk_len, first):
        self.scorer, self.theta, self.window = scorer, float(theta), int(window)
        self.model_type = model_type
        self.offset, self.len, self.first = int(chunk_offset), int(chunk_len), int(first)

    def near(self, local_targets: np.ndarray) -> np.ndarray:
        margin = 2 * self.window + 2
        keep = np.zeros(self.len, dtype=bool)
        for t in local_targets:
            keep[max(t - margin, self.first):min(t + margin + 1, self.len)] = True
        return np.flatnonzero(keep)

    def decide(self, series, local_targets, near_local, seed_base, full=False):
        """Window decisions at local_targets. Scores outside `near_local` are not
        needed (they cannot reach a target through the window) and are left at 0."""
        positions = np.arange(self.first, self.len) if full else near_local
        scores = np.zeros(self.len - self.first)
        errs = self.scorer.errors(series, self.offset + positions, seed_base)
        scores[positions - self.first] = errs.mean(axis=1)
        det = repo_cached_detect(scores, self.theta, self.window, self.model_type, "same")
        return det[local_targets - self.first].astype(bool)


def run_attack(args) -> int:
    started = time.time()
    with open(args.defense) as fd:
        dfn = json.load(fd)
    out = Path(args.out)
    if (out / "summary.json").exists():
        raise SystemExit(f"{out}/summary.json exists; use a new --out (nothing is overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    model, model_type, history, params, name = load_detector(dfn["model_dir"])
    geom = make_geometry(model_type, history, int(dfn["target_offset"]), int(dfn["context"]))
    x_train, x_test, y_test, cols = load_data(args.dataset)
    splits = build_splits(args.dataset, x_train, x_test, y_test, geom)
    if args.split == "tune":
        c_off, c_len, labels_all = splits.tune_offset, splits.tune_len, splits.tune_labels
    else:
        c_off, c_len, labels_all = splits.final_offset, splits.final_len, splits.final_labels
    first = geom.first_target
    local_all = np.arange(first, c_len, dtype=np.int64)  # labels_all is aligned with these

    # ---- defense, attacker's view of it ------------------------------------
    p_def, _ = load_purifier(dfn.get("diffusion"), int(dfn["t_star"]), int(dfn["steps"]), dfn["sampler"])
    stochastic = not p_def.identity
    if args.gradient in ("full", "bpda"):
        p_att, g_mode = p_def, ("bpda" if args.gradient == "bpda" else "full")
    elif args.gradient == "surrogate":
        if not args.attack_steps:
            raise SystemExit("--gradient surrogate needs --attack-steps")
        p_att, g_mode = Purifier(p_def.denoiser, p_def.t_star, args.attack_steps, dfn["sampler"]), "full"
    else:
        p_att, g_mode = Purifier(None, 0, 0), "full"
    att_stochastic = not p_att.identity
    eot = args.eot if att_stochastic else 1
    draws = args.eval_draws if stochastic else 1
    scorer = Scorer(model, p_def, geom, args.batch)
    det = DefendedDetector(scorer, dfn["theta"], dfn["window"], model_type, c_off, c_len, first)

    # ---- targets --------------------------------------------------------------
    want = 1 if args.goal == "evasion" else 0
    cand = local_all[np.asarray(labels_all).astype(int) == want]
    if len(cand) == 0:
        raise SystemExit("no target of the requested kind in this split")
    targets = sample_segment_blocks(cand, args.max_targets) if 0 < args.max_targets < len(cand) else cand
    near = det.near(targets)
    print(f"{name} | defense t*={p_def.t_star} K={p_def.n_steps} {dfn['sampler']} theta={det.theta:.6g} "
          f"w={det.window} | attacker: {args.gradient} (t*={p_att.t_star}, K={p_att.n_steps}, EOT={eot}) | "
          f"{args.goal} on the {args.split} split: {len(targets)} targets, {len(near)} scored timesteps",
          flush=True)

    if p_def.identity:  # the near-only decisions must equal the full-chunk detector
        a = det.decide(x_test, targets, near, 0)
        b = det.decide(x_test, targets, near, 0, full=True)
        if not np.array_equal(a, b):
            raise RuntimeError("near-only detection does not reproduce the full detector")

    clean = np.stack([det.decide(x_test, targets, near, 10_000_000 + d) for d in range(draws)])
    if args.goal == "evasion":
        eligible = clean.mean(axis=0) > 0.5
        loss_targets = targets[eligible]
        if len(loss_targets) == 0:
            # a valid (and bad) outcome for the defense: report it instead of failing
            with open(out / "summary.json", "w") as fd:
                json.dump(dict(model=name, goal=args.goal, split=args.split, defense=dfn,
                               n_targets=int(len(targets)), n_eligible=0, eval_draws=draws,
                               clean=dict(flagged_mean=float(clean.mean())),
                               note="the defended detector flags no target on clean data"), fd, indent=1)
            print("RESULT: the defended detector flags no target on clean data: nothing to evade")
            return 0
    else:
        eligible = np.ones(len(targets), dtype=bool)
        loss_targets = targets
    print(f"clean: flagged {clean.mean():.4f} of the targets (per draw: "
          f"{np.round(clean.mean(axis=1), 4).tolist()}); E = {len(loss_targets)}", flush=True)

    # ---- budget and modifiable cells --------------------------------------------
    feat_range = x_train.max(axis=0) - x_train.min(axis=0)
    eps = (args.epsilon_range_fraction * feat_range).astype(np.float32)
    protected = actuator_indices(cols) if args.protect_actuators else []
    if protected:
        eps[protected] = 0.0
    widx_global = window_index(c_off + loss_targets, geom)          # (n, Lw)
    rows = np.unique(widx_global)
    pos = {r: i for i, r in enumerate(rows)}
    widx = np.vectorize(pos.get)(widx_global).astype(np.int64)      # into the compact rows
    x0c = x_test[rows].astype(np.float32)
    # Rows that can change: those that feed the ATTACKER's loss. Through a purifier
    # that is the whole Lw-row window; without one (undefended / blind attacker) only
    # the detector's own input rows (history + target, AE: target), as in the
    # undefended benchmark. Other rows would only receive random-start noise.
    if p_att.identity:
        a, b = (geom.length - 1, geom.length) if model_type == "AE" else geom.hist_slice
        used = np.unique(np.concatenate([widx[:, a:b].ravel(), widx[:, -1]]))
        row_mask = np.zeros(len(rows), np.float32)
        row_mask[used] = 1.0
    else:
        row_mask = np.ones(len(rows), np.float32)
    eps_tf = tf.constant(eps[None, :])
    feat_mask = tf.constant(row_mask[:, None] * (eps > 0).astype(np.float32)[None, :])
    alpha = tf.constant((args.alpha_relative * eps)[None, :])
    sign = 1.0 if args.goal == "evasion" else -1.0   # we MINIMISE sign * mean score
    print(f"budget: {args.epsilon_range_fraction} x training range, {len(protected)} protected "
          f"features; {int(row_mask.sum())} modifiable rows", flush=True)

    # ---- EOT loss / gradient ------------------------------------------------------
    B = int(args.attack_batch)
    n_loss = len(loss_targets)
    n_pad = (-n_loss) % B
    widx_p = np.concatenate([widx, np.repeat(widx[-1:], n_pad, axis=0)]) if n_pad else widx
    weight = np.concatenate([np.full(n_loss, 1.0 / n_loss), np.zeros(n_pad)]).astype(np.float32)
    batches = [(tf.constant(widx_p[i:i + B]), tf.constant(weight[i:i + B])) for i in range(0, len(widx_p), B)]

    @tf.function  # fixed shapes (batches are padded): one trace
    def batch_loss_grad(xc, wi, wt, seed):
        with tf.GradientTape() as tape:
            tape.watch(xc)
            windows = tf.gather(xc, wi)
            errs = errors_tf(model, p_att, windows, geom, seed, g_mode)
            loss = tf.reduce_sum(tf.reduce_mean(errs, axis=1) * wt) * sign
        return loss, tf.convert_to_tensor(tape.gradient(loss, xc))

    def eot_loss_grad(xc, restart, it):
        total, grad = 0.0, tf.zeros_like(xc)
        for m in range(eot):
            for b, (wi, wt) in enumerate(batches):
                l, g = batch_loss_grad(xc, wi, wt, seed_pair(1, args.seed, restart, it, m, b))
                total += float(l)
                grad += g
        return total / eot, grad / eot

    def series_of(xc: np.ndarray) -> np.ndarray:
        s = x_test.copy()
        s[rows] = xc
        return s

    def metrics_of(detected: np.ndarray) -> dict:
        d = dict(flagged=float(detected.mean()))
        if args.goal == "evasion":
            d["dr_E"] = float(detected[eligible].mean())
            d["asr"] = 1.0 - d["dr_E"]
        return d

    # ---- PGD ------------------------------------------------------------------
    trace_path = out / "iterations.csv"
    fields = ["restart", "iteration", "loss", "trace_flagged", "trace_dr_E", "seconds"]
    best = dict(loss=np.inf)
    stopped_early = None
    with open(trace_path, "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        for restart in range(args.restarts):
            rng = tf.random.Generator.from_seed(args.seed * 7919 + restart)
            delta = (rng.uniform(tf.shape(x0c), -1.0, 1.0) * eps_tf * feat_mask)
            x0 = tf.constant(x0c)
            if stopped_early is not None:
                break
            for it in range(args.iterations + 1):
                xc = x0 + delta
                loss, grad = eot_loss_grad(xc, restart, it)
                row = dict(restart=restart, iteration=it, loss=sign * loss, seconds=time.time() - started)
                if it % args.trace_every == 0 or it == args.iterations:
                    detected = det.decide(series_of(xc.numpy()), targets, near, 30_000_000 + it)
                    m = metrics_of(detected)
                    row["trace_flagged"] = m["flagged"]
                    row["trace_dr_E"] = m.get("dr_E", "")
                    print(f"[{args.gradient}/{args.goal}] restart {restart} it {it:>4}: "
                          f"mean score {sign * loss:.6f} flagged {m['flagged']:.4f}"
                          + (f" DR_E {m['dr_E']:.4f}" if "dr_E" in m else "")
                          + f"  {time.time() - started:.0f}s", flush=True)
                writer.writerow(row)
                fd.flush()
                if loss < best["loss"]:
                    best = dict(loss=loss, restart=restart, iteration=it, xc=xc.numpy())
                out_of_time = (time.time() - started) / 60.0 > args.max_minutes
                if it == args.iterations or out_of_time:
                    final_point = dict(loss=loss, restart=restart, iteration=it, xc=xc.numpy())
                    if out_of_time and it < args.iterations:
                        stopped_early = dict(restart=restart, iteration=it)
                        print(f"WARNING: wall-time limit reached at restart {restart} iteration {it}", flush=True)
                    break
                g = grad * feat_mask
                delta = tf.clip_by_value(delta - alpha * tf.sign(g), -eps_tf, eps_tf) * feat_mask

    # ---- final evaluation with fresh randomness -----------------------------------
    results = {}
    for key, point in (("final_iterate", final_point), ("best_loss", best)):
        series = series_of(point["xc"])
        dets = np.stack([det.decide(series, targets, near, 20_000_000 + 1000 * d + (key == "best_loss"))
                         for d in range(draws)])
        per_draw = [metrics_of(x) for x in dets]
        delta_np = point["xc"] - x0c
        if np.any(np.abs(delta_np) > eps[None, :] + 1e-5):
            raise RuntimeError("perturbation exceeds the budget")
        results[key] = dict(restart=point["restart"], iteration=point["iteration"],
                            mean_score=float(sign * point["loss"]), per_draw=per_draw,
                            flagged_mean=float(np.mean([p["flagged"] for p in per_draw])),
                            flagged_std=float(np.std([p["flagged"] for p in per_draw])),
                            linf_scaled=float(np.abs(delta_np).max()),
                            l2_scaled=float(np.sqrt((delta_np ** 2).sum(axis=1)).mean()),
                            changed_cells=int((np.abs(delta_np) > 0).sum()))
        if args.goal == "evasion":
            results[key]["dr_E_mean"] = float(np.mean([p["dr_E"] for p in per_draw]))
            results[key]["dr_E_std"] = float(np.std([p["dr_E"] for p in per_draw]))
            results[key]["asr_mean"] = 1.0 - results[key]["dr_E_mean"]
        np.savez_compressed(out / f"{key}_changed_rows.npz", rows=rows, x_clean_scaled=x0c,
                            x_adv_scaled=point["xc"], feature_names=np.array(cols))
    # the stronger of the two points for the attacker (lower detection / more alarms)
    # (evasion: lower detection rate on E; false alarm: more targets flagged)
    if args.goal == "evasion":
        pick = min(results, key=lambda k: results[k]["dr_E_mean"])
    else:
        pick = max(results, key=lambda k: results[k]["flagged_mean"])
    clean_draws = [metrics_of(c) for c in clean]
    summary = dict(
        model=name, model_dir=dfn["model_dir"], model_type=model_type, dataset=args.dataset,
        defense=dict(dfn, purifier=p_def.describe()), goal=args.goal, split=args.split,
        attacker=dict(gradient=args.gradient, purifier=p_att.describe(), eot=eot,
                      iterations=args.iterations, restarts=args.restarts,
                      alpha_relative=args.alpha_relative,
                      epsilon_range_fraction=args.epsilon_range_fraction,
                      protect_actuators=bool(args.protect_actuators), seed=args.seed),
        n_targets=int(len(targets)), n_eligible=int(eligible.sum()), eval_draws=draws,
        stopped_early=stopped_early,
        clean=dict(per_draw=clean_draws, flagged_mean=float(clean.mean()),
                   flagged_std=float(clean.mean(axis=1).std())),
        adversarial=results, reported_point=pick, reported=results[pick],
        runtime_seconds=time.time() - started, git=git_state())
    with open(out / "summary.json", "w") as fd:
        json.dump(summary, fd, indent=1, default=float)
    np.save(out / "targets_local.npy", targets)
    r = results[pick]
    print(f"RESULT {name} t*={p_def.t_star} {args.gradient} {args.goal}: clean flagged "
          f"{clean.mean():.4f} -> adversarial {r['flagged_mean']:.4f}+-{r['flagged_std']:.4f}"
          + (f", DR on E {r['dr_E_mean']:.4f} (ASR {r['asr_mean']:.4f})" if args.goal == "evasion" else "")
          + f" [{pick}], {time.time() - started:.0f}s", flush=True)
    return 0

"""PGD on frozen clean-detected attack targets, retaining all evaluation targets.

Opt in with --attack pgd_mse --goal evasion --eligible-protocol.
Use --selection attack --max-targets 0 to retain all attack-labelled targets.
No structured perturbation: this is the original masked signal-space PGD.
Loss remains a point-score surrogate; metrics use the full temporal detector.
Both full checkpoints are saved even when --save-series is none/delta.
Use a separate output directory for each experiment: filenames are reused.
"""
from dataclasses import replace
from pathlib import Path
import csv
import json
import time

import numpy as np
import tensorflow as tf

from .detector import repo_cached_detect, series_instance_scores, windowed_target_detection
from .evaluation import load_changed_rows, save_changed_rows
from .errors import model_errors_tf
from .projection import build_tf_bounds
from .protocol_metrics import freeze_eligible, iteration_metrics
from .targets import infer_attack_labels, valid_target_bounds


def run_protocol(attack, ctx, goal):
    args = ctx.args
    if not getattr(attack, "requires_gradients", False) or goal != "evasion":
        raise ValueError("Eligible protocol supports white-box evasion attacks only")
    n_iterations = int(attack.iterations(args))
    use_random_start = bool(attack.use_random_start(args))
    if ctx.adapter.keras_model is None or ctx.instance_threshold is None:
        raise ValueError("A differentiable Keras model and instance threshold are required")
    eligible = freeze_eligible(ctx.target_indices,
                              infer_attack_labels(args.dataset, ctx.labels),
                              ctx.clean_detect_window)
    # Never replace the original evaluation targets or modification mask.
    loss_ctx = replace(ctx, target_indices=ctx.target_indices[eligible].copy())
    x0 = tf.convert_to_tensor(ctx.x_test, tf.float32)
    mask = tf.convert_to_tensor(ctx.modification_mask, tf.float32)
    eps, lower, upper = build_tf_bounds(x0, ctx.epsilon, ctx.lower_domain, ctx.upper_domain)
    active = ctx.modification_mask > 0
    if not np.all(np.isfinite(ctx.x_test)) or not np.all(np.isfinite(ctx.epsilon)) or np.any(ctx.epsilon < 0):
        raise ValueError("Input and epsilon must be finite; epsilon must be nonnegative")
    if np.any((ctx.x_test < lower.numpy())[active]) or np.any((ctx.x_test > upper.numpy())[active]):
        raise ValueError("Clean modifiable values are outside domain bounds; revise bounds or disable clipping")
    restarts = int(getattr(args, "restarts", 1))
    if restarts < 1 or n_iterations < 1 or (restarts > 1 and not use_random_start):
        raise ValueError("Require positive iterations/restarts and random-start for multiple restarts")
    run_dir = Path(args.output_dir) / attack.name / goal
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "eligible_indices.npy", loss_ctx.target_indices)
    np.save(run_dir / "eligible_mask.npy", eligible)
    np.save(run_dir / "evaluation_target_indices.npy", ctx.target_indices)
    best = {}
    started = time.time()

    # Fast per-iteration detection. The window decision at a target depends only on
    # point detections within +-margin of it, and those only on the scores there, so
    # we recompute scores only inside that neighbourhood and reuse the clean scores
    # elsewhere. This gives exactly the target-level result of
    # windowed_target_detection while predicting ~10k windows instead of the whole
    # series. Verified against the full computation on the clean series below.
    first, last = valid_target_bounds(args.model_type, len(ctx.x_test), args.history,
                                      args.target_offset)
    all_indices = np.arange(first, last, dtype=np.int64)
    clean_scores = series_instance_scores(ctx, ctx.x_test, all_indices)
    margin = 2 * int(ctx.detection_window) + 2
    near = np.zeros(len(all_indices), dtype=bool)
    for t in ctx.target_indices.astype(np.int64) - first:
        near[max(t - margin, 0):t + margin + 1] = True
    near_positions = np.flatnonzero(near)
    near_indices = all_indices[near_positions]
    target_positions = ctx.target_indices.astype(np.int64) - first

    def fast_target_detection(candidate):
        scores = clean_scores.copy()
        scores[near_positions] = series_instance_scores(ctx, candidate, near_indices)
        window = repo_cached_detect(scores, ctx.instance_threshold,
                                    ctx.detection_window, args.model_type)
        return window[target_positions]

    _, full_clean = windowed_target_detection(ctx, ctx.x_test)
    if not np.array_equal(fast_target_detection(ctx.x_test), full_clean):
        raise RuntimeError("Fast detection does not reproduce the full detector on the clean series")

    def save_checkpoint(directory, key, candidate):
        # Full series only when asked; otherwise just the modified rows (original and
        # adversarial, scaled and raw), which is what explainability needs.
        if args.save_series == "full":
            np.save(directory / (key + "_scaled.npy"), candidate)
        save_changed_rows(directory / (key + "_changed_rows.npz"), ctx, candidate)

    def load_checkpoint(directory, key):
        full = directory / (key + "_scaled.npy")
        if full.exists():
            return np.load(full)
        return load_changed_rows(directory / (key + "_changed_rows.npz"), ctx.x_test)

    def objective(series):
        errors = model_errors_tf(ctx.adapter.keras_model, series, args.model_type,
                                 loss_ctx.target_indices, args.history, args.target_offset)
        return attack.objective(loss_ctx, goal, errors, series, x0)[0]

    def record(series, loss, restart, iteration, phase):
        loss = float(loss)
        if not np.isfinite(loss):
            raise ValueError("Non-finite optimization loss")
        candidate = series.numpy()
        delta = candidate - ctx.x_test
        if np.any(np.abs(delta) > ctx.epsilon[None, :] + 1e-5) or np.any(delta[~active] != 0):
            raise RuntimeError("Perturbation violates budget or modification mask")
        detected = fast_target_detection(candidate)
        stats = iteration_metrics(eligible, detected)
        row = dict(restart=restart, seed=args.seed + max(restart, 0),
                   iteration=iteration, phase=phase, loss=loss, **stats)
        writer.writerow(row)
        trace_file.flush()
        for key in ("best_loss", "best_asr"):
            old = best.get(key)
            improved = old is None
            if old is not None:
                improved = (loss < old["loss"] if key == "best_loss" else
                            (stats["evasions"], -loss) > (old["evasions"], -old["loss"]))
            if improved:
                best[key] = row.copy()
                save_checkpoint(run_dir, key, candidate)
        with (run_dir / "checkpoint_metrics.json").open("w") as handle:
            json.dump(best, handle, indent=2)
        print(f"[pgd_mse] restart={restart} iteration={iteration} {phase} loss={loss:.8f} "
              f"evasions={stats['evasions']}/{stats['eligible_targets']} "
              f"ASR={stats['asr']:.4f} DR={stats['detection_rate']:.4f}", flush=True)

    fields = ["restart", "seed", "iteration", "phase", "loss", "evasions", "asr",
              "detection_rate", "detected_targets", "eligible_targets", "targets"]
    with (run_dir / "iteration_metrics.csv").open("w", newline="") as trace_file:
        writer = csv.DictWriter(trace_file, fieldnames=fields)
        writer.writeheader()
        initial_loss = float(objective(x0).numpy())
        # Clean is a candidate too. Equal ASR prefers lower loss; exact ties keep first.
        record(x0, initial_loss, -1, 0, "clean")
        for restart in range(restarts):
            tf.random.set_seed(args.seed + restart)
            series = tf.identity(x0)
            if use_random_start:
                series = x0 + tf.random.uniform(tf.shape(x0), -1., 1.) * eps * mask
                series = x0 + (tf.clip_by_value(series, lower, upper) - x0) * mask
            for iteration in range(n_iterations + 1):
                with tf.GradientTape() as tape:
                    tape.watch(series)
                    loss = objective(series)
                final_loss = float(loss.numpy())
                record(series, final_loss, restart, iteration, "trajectory")
                if iteration == n_iterations:
                    break
                gradient = tape.gradient(loss, series)
                if gradient is None:
                    raise RuntimeError("Missing gradient")
                gradient = tf.convert_to_tensor(gradient) * mask
                tf.debugging.assert_all_finite(gradient, "Non-finite gradient")
                gradient = attack.process_gradient(ctx, gradient, mask)
                step = attack.step_multiplier(args, eps)
                updated = tf.clip_by_value(series - step * tf.sign(gradient), lower, upper)
                series = x0 + (updated - x0) * mask

    returned = getattr(args, "return_point", "best_asr")
    metadata = dict(iterations_executed=n_iterations * restarts, restarts=restarts,
                    initial_optimization_loss=initial_loss, final_optimization_loss=final_loss,
                    best_loss=best["best_loss"]["loss"], best_iteration=best["best_loss"]["iteration"],
                    best_loss_restart=best["best_loss"]["restart"],
                    best_asr=best["best_asr"]["asr"], best_asr_iteration=best["best_asr"]["iteration"],
                    best_asr_restart=best["best_asr"]["restart"], returned_point=returned,
                    eligible_protocol=True, runtime_seconds=time.time() - started,
                    query_count=None, checkpoints=best)
    return load_checkpoint(run_dir, returned), metadata

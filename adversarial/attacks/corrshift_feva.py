"""CorrShift as described in FEVA-ICS (Ghosh et al., CPSS '26, Section 4.2 and
Algorithm 1), adapted to the evasion protocol of this repository.

What the paper does, and what is kept here:
  * the series is cut into batches of consecutive time steps (60 on SWaT);
  * every batch gets its OWN constant shift, the same for all its rows;
  * for each perturbable candidate feature c, the shift moves every perturbable
    feature j by its full budget eps_j, in the direction of sign(corr(c, j)) (+ when
    the correlation is >= 0 or undefined, - when it is negative); c itself moves +;
  * the detector is queried once per candidate (black-box, decisions only) and the
    candidate that changes most decisions with respect to the true label is kept;
  * one pass over the batches, no iterations, no gradients;
  * correlations are computed on the data being attacked (--feva-corr-source
    attacked, default) or, optionally, on benign training data (train).

What differs, and why:
  1. Budget. eps_j is the run's own budget (--epsilon-range-fraction alpha of the
     benign TRAINING range, as for PGD), not alpha * (max|X| - min|X|) of the
     attacked data, so CorrShift and PGD are compared at the same epsilon.
  2. The candidate feature moves by eps_c once. Algorithm 1 (lines 15-23) and the
     published notebook add eps_c twice (once as candidate, once because
     corr(c, c) = 1), i.e. 2 * eps_c, which breaks the paper's own L-inf budget.
  3. The best candidate is chosen per batch. In the published notebook
     (framework.ipynb, AdversarialExampleGenerator.apply_perturbation) max_diff and
     best_adv_examples are initialised once, before the loop over batches, so a
     batch whose candidates do not beat the best count seen so far receives a copy
     of an earlier batch's perturbed data. Here the search restarts for every batch.
  4. Correlation indexing. The notebook indexes the full correlation matrix with
     positions in the list of non-protected features; here the correlation of the
     actual features is used.
  5. Success counts evasion only. The paper counts every prediction that differs
     from the label, normal rows included. Here only attack-labelled targets are
     evaluated (all targets are attack rows): the objective is the number of
     targets NOT detected near the batch, so new alarms on missed targets count
     against the attacker.
  6. Rows shifted. As in the eligible protocol, only cells that feed the scores of
     the clean-detected targets E are shifted (E rows, plus their history for
     sequence models), not every row of the batch.
  7. A batch is changed only if its best candidate strictly improves on leaving it
     unchanged (the notebook always keeps its best candidate, even with no gain).
  8. Batches are processed in time order on the current adversarial series, so a
     batch sees the changes already accepted before it (sequence models read up to
     `history` earlier rows). The notebook queries every batch in isolation.

Exactness: a target's window decision depends only on scores within +-w of it and
a row only changes scores up to history + target_offset steps later, so each query
recomputes scores only there and runs the window rule on a padded slice. All
candidates of a batch, and the unchanged batch, are scored in one model call so the
comparison uses identical arithmetic. The tracked decisions are checked against the
full detector at the end; the reported metrics always come from the full detector
(save_run).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..base import Attack
from ..constants import POINT_MODELS
from ..context import AttackContext
from ..detector import repo_cached_detect, series_instance_scores, windowed_target_detection
from ..errors import instance_detection_scores, model_errors_numpy
from ..projection import project_numpy
from ..protocol_metrics import freeze_eligible
from ..targets import build_modification_mask, infer_attack_labels, valid_target_bounds


def feva_correlation(data: np.ndarray, perturbable: np.ndarray) -> np.ndarray:
    """Full-size correlation matrix; entries are computed among the perturbable
    features only (others stay 0). Undefined correlations (constant features) are 0,
    which Algorithm 1 treats like a non-negative correlation."""
    n_features = data.shape[1]
    corr = np.zeros((n_features, n_features), dtype=np.float64)
    if len(perturbable) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            sub = np.corrcoef(np.asarray(data, dtype=np.float64)[:, perturbable], rowvar=False)
        corr[np.ix_(perturbable, perturbable)] = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
    corr[perturbable, perturbable] = 1.0
    return corr


def feva_direction(corr: np.ndarray, candidate: int, perturbable: np.ndarray) -> np.ndarray:
    """Sign vector of Algorithm 1 for one candidate: +1 for the candidate and for
    features with correlation >= 0, -1 for negative correlation, 0 elsewhere."""
    direction = np.zeros(corr.shape[0], dtype=np.float32)
    row = corr[candidate, perturbable]
    direction[perturbable] = np.where(row < 0, -1.0, 1.0)
    direction[candidate] = 1.0
    return direction


def split_blocks(rows: np.ndarray, size: int) -> List[Tuple[int, int]]:
    """Consecutive blocks of at most `size` rows inside each contiguous run of rows
    (inclusive bounds)."""
    rows = np.asarray(rows, dtype=np.int64)
    if size < 1:
        raise ValueError("--feva-batch must be positive")
    if len(rows) == 0:
        return []
    blocks: List[Tuple[int, int]] = []
    for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
        for start in range(0, len(run), size):
            chunk = run[start:start + size]
            blocks.append((int(chunk[0]), int(chunk[-1])))
    return blocks


class CorrShiftFeva(Attack):
    name = "corrshift_feva"
    requires_gradients = False
    requires_train_data = True
    in_all = False  # not part of --attack all (the original-protocol jobs)

    def run(self, ctx: AttackContext, goal: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        args = ctx.args
        if goal != "evasion":
            raise ValueError("corrshift_feva supports --goal evasion only")
        if ctx.instance_threshold is None or ctx.clean_detect_window is None:
            raise ValueError("corrshift_feva needs an instance threshold and window detector")
        started = time.time()
        n_rows, n_features = ctx.x_test.shape
        gru_rule = getattr(args, "gru_window_rule", "same")
        theta = float(ctx.instance_threshold)
        window = int(ctx.detection_window)
        point_model = args.model_type in POINT_MODELS
        history = 0 if point_model else int(args.history)
        offset = 0 if point_model else int(args.target_offset)
        lag = 0 if point_model else history + offset  # row r changes scores at r..r+lag

        eligible = freeze_eligible(ctx.target_indices,
                                   infer_attack_labels(args.dataset, ctx.labels),
                                   ctx.clean_detect_window)
        e_targets = ctx.target_indices[eligible]
        mask = build_modification_mask(args.model_type, n_rows, n_features, e_targets,
                                       args.history, args.target_offset, args.scope,
                                       ctx.protected) * ctx.modification_mask
        active_rows = np.flatnonzero(mask.sum(axis=1) > 0)
        perturbable = np.flatnonzero(mask.max(axis=0) > 0)
        if len(perturbable) == 0:
            raise ValueError("corrshift_feva has no perturbable feature")
        eps = np.asarray(ctx.epsilon, dtype=np.float32)

        source = getattr(args, "feva_corr_source", "attacked")
        if source == "train":
            if ctx.train_scaled is None:
                raise ValueError("--feva-corr-source train needs the training data")
            corr = feva_correlation(ctx.train_scaled, perturbable)
        else:
            corr = feva_correlation(ctx.x_test[active_rows], perturbable)
        directions = {int(c): feva_direction(corr, int(c), perturbable) for c in perturbable}

        first, last = valid_target_bounds(args.model_type, n_rows, args.history, args.target_offset)
        all_indices = np.arange(first, last, dtype=np.int64)
        scores = series_instance_scores(ctx, ctx.x_test, all_indices)
        detected = repo_cached_detect(scores, theta, window, args.model_type, gru_rule)
        target_sorted = np.sort(ctx.target_indices.astype(np.int64))
        target_positions = ctx.target_indices.astype(np.int64) - first
        if not np.array_equal(detected[target_positions], np.asarray(ctx.clean_detect_window, bool)):
            raise RuntimeError("corrshift_feva: clean detection differs from build_context")

        blocks = split_blocks(active_rows, int(getattr(args, "feva_batch", 60)))
        pad = 2 * window + 2
        x_adv = ctx.x_test.copy()
        batch_size = int(args.prediction_batch_size)
        queries = 0
        accepted = 0
        max_clean_diff = 0.0
        log_rows: List[Dict[str, Any]] = []
        print(f"[corrshift_feva] {len(blocks)} batches of <= {getattr(args, 'feva_batch', 60)} rows, "
              f"{len(active_rows)} rows, {len(perturbable)} candidate features, "
              f"correlations from {source}", flush=True)

        for block_number, (a, b) in enumerate(blocks):
            t_lo = max(a, first)
            t_hi = min(b + lag, last - 1)
            if t_lo > t_hi:
                continue
            affected = np.arange(t_lo, t_hi + 1, dtype=np.int64)
            base = t_lo - history - offset if not point_model else t_lo
            segment_clean = x_adv[base:t_hi + 1]
            seg_len = t_hi + 1 - base
            local_targets = affected - base

            block_mask = mask[a:b + 1]
            candidates = [None] + [int(c) for c in perturbable]
            segments = []
            for c in candidates:
                seg = segment_clean.copy()
                if c is not None:
                    shifted = ctx.x_test[a:b + 1] + block_mask * (eps * directions[c])[None, :]
                    seg[a - base:b - base + 1] = project_numpy(
                        shifted, ctx.x_test[a:b + 1], eps, block_mask,
                        ctx.lower_domain, ctx.upper_domain)
                segments.append(seg)
            stacked = np.concatenate(segments, axis=0)
            index = np.concatenate([k * seg_len + local_targets for k in range(len(candidates))])
            errors = model_errors_numpy(ctx.adapter, stacked, args.model_type, index,
                                        history if not point_model else None, offset, batch_size)
            all_scores = instance_detection_scores(errors, ctx.instance_score_kind).reshape(
                len(candidates), len(affected))
            queries += len(candidates) - 1
            max_clean_diff = max(max_clean_diff, float(np.max(np.abs(
                all_scores[0] - scores[t_lo - first:t_hi - first + 1]))))

            s_lo = max(t_lo - pad, first)
            s_hi = min(t_hi + pad, last - 1)
            r_lo = max(t_lo - window - 1, first)
            r_hi = min(t_hi + window + 1, last - 1)
            lo_k = np.searchsorted(target_sorted, r_lo, side="left")
            hi_k = np.searchsorted(target_sorted, r_hi, side="right")
            region_targets = target_sorted[lo_k:hi_k]
            counts = []
            decisions = []
            for k in range(len(candidates)):
                local = scores[s_lo - first:s_hi - first + 1].copy()
                local[t_lo - s_lo:t_hi - s_lo + 1] = all_scores[k]
                dec = repo_cached_detect(local, theta, window, args.model_type, gru_rule)
                decisions.append(dec)
                counts.append(int(np.sum(dec[region_targets - s_lo])))
            best_k = int(np.argmin(counts))  # first minimum, like Algorithm 1's strict >
            gain = counts[0] - counts[best_k]
            take = best_k != 0 and gain > 0
            if take:
                c = candidates[best_k]
                x_adv[base:t_hi + 1] = segments[best_k]
                scores[t_lo - first:t_hi - first + 1] = all_scores[best_k]
                detected[r_lo - first:r_hi - first + 1] = decisions[best_k][r_lo - s_lo:r_hi - s_lo + 1]
                accepted += 1
            log_rows.append(dict(
                batch=block_number, first_row=a, last_row=b,
                rows_shifted=int((block_mask.sum(axis=1) > 0).sum()),
                targets_in_reach=int(len(region_targets)),
                detected_unchanged=counts[0], detected_best=counts[best_k],
                best_feature=(ctx.sensor_cols[candidates[best_k]] if best_k else ""),
                accepted=int(take)))

        delta = x_adv - ctx.x_test
        if np.any(np.abs(delta) > eps[None, :] + 1e-5) or np.any(delta[mask <= 0] != 0):
            raise RuntimeError("corrshift_feva: perturbation violates budget or mask")
        _, full_after = windowed_target_detection(ctx, x_adv)
        mismatches = int(np.sum(full_after != detected[target_positions]))
        if mismatches:
            print(f"[corrshift_feva] WARNING: {mismatches} target decisions differ between "
                  f"the tracked and the full detector (rounding in batched predictions); "
                  f"reported metrics use the full detector.", flush=True)
        evasions = int(np.sum(eligible & ~full_after))

        run_dir = Path(args.output_dir) / self.name / goal
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "feva_batches.csv").open("w", newline="") as handle:
            fields = ["batch", "first_row", "last_row", "rows_shifted", "targets_in_reach",
                      "detected_unchanged", "detected_best", "best_feature", "accepted"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"[corrshift_feva] batches {len(blocks)}, changed {accepted}, queries {queries}, "
              f"evasions {evasions}/{int(eligible.sum())}, max |score diff| on clean "
              f"{max_clean_diff:.2e}", flush=True)
        metadata = {
            "iterations_executed": accepted,
            "feva_batches": len(blocks),
            "feva_batches_changed": accepted,
            "feva_batch_size": int(getattr(args, "feva_batch", 60)),
            "feva_corr_source": source,
            "feva_tracked_mismatches": mismatches,
            "feva_max_clean_score_diff": max_clean_diff,
            "query_count": queries,
            "runtime_seconds": time.time() - started,
        }
        return x_adv, metadata

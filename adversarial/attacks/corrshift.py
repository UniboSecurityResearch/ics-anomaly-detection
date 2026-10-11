"""CorrShift: gradient-free, correlation-guided score search (black-box)."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from ..base import Attack
from ..context import AttackContext
from ..errors import detector_score_numpy, model_errors_numpy
from ..projection import project_numpy
from ..protocol_metrics import freeze_eligible
from ..targets import build_modification_mask, infer_attack_labels


def corrshift_direction(
    corr_matrix: np.ndarray,
    anchor: int,
    protected: Set[int],
    top_k: int,
    min_abs_corr: float,
) -> np.ndarray:
    correlations = np.asarray(corr_matrix[anchor], dtype=np.float32)
    direction = np.sign(correlations)
    direction[np.abs(correlations) < min_abs_corr] = 0.0
    direction[anchor] = 1.0
    if protected:
        direction[list(protected)] = 0.0

    if top_k > 0:
        available = np.flatnonzero(direction != 0)
        if len(available) > top_k:
            ranked = available[np.argsort(np.abs(correlations[available]))[::-1]]
            keep = set(ranked[:top_k].tolist())
            keep.add(anchor)
            sparse = np.zeros_like(direction)
            for idx in keep:
                sparse[idx] = direction[idx]
            direction = sparse
    return direction


def mean_blackbox_score(ctx: AttackContext, x_series: np.ndarray) -> float:
    errors = model_errors_numpy(
        ctx.adapter,
        x_series,
        ctx.args.model_type,
        ctx.target_indices,
        ctx.args.history,
        ctx.args.target_offset,
        ctx.args.prediction_batch_size,
    )
    scores = detector_score_numpy(
        errors,
        ctx.args.score,
        ctx.thresholds,
        ctx.args.margin_beta,
    )
    return float(np.mean(scores))


class CorrShift(Attack):
    name = "corrshift"
    requires_gradients = False
    requires_train_data = True

    def run(self, ctx: AttackContext, goal: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        args = ctx.args
        protected = ctx.protected
        if ctx.train_scaled is None:
            correlation_data = ctx.x_test
            print("CorrShift warning: training data unavailable; using test data correlations.")
        else:
            correlation_data = ctx.train_scaled

        corr_matrix = np.corrcoef(correlation_data, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        epsilon = ctx.epsilon.copy()
        if args.corrshift_range_fraction is not None:
            if args.corrshift_range_fraction < 0:
                raise ValueError("--corrshift-range-fraction must be non-negative.")
            feature_range = np.max(correlation_data, axis=0) - np.min(correlation_data, axis=0)
            epsilon = args.corrshift_range_fraction * feature_range.astype(np.float32)

        relative = getattr(args, "corrshift_step_relative", None)
        if relative is not None:
            # Same convention as PGD --alpha-relative: each feature moves r * eps_j per
            # accepted step, so all features reach their bound after ~1/r steps.
            if relative <= 0:
                raise ValueError("--corrshift-step-relative must be positive.")
            step_vector = (relative * epsilon).astype(np.float32)
        else:
            step_scalar = args.alpha if args.corrshift_step is None else args.corrshift_step
            if step_scalar < 0:
                raise ValueError("CorrShift step must be non-negative.")
            step_vector = np.minimum(epsilon, step_scalar).astype(np.float32)

        # --corrshift-eligible: same evasion protocol as the white-box attacks. The
        # score is averaged over the clean-detected targets E only, and only the rows
        # feeding those scores are shifted. Otherwise the shift is applied to every
        # target row, including targets the detector already misses, where it can
        # only create new detections.
        score_ctx = ctx
        mask = ctx.modification_mask
        if getattr(args, "corrshift_eligible", False) and goal == "evasion":
            eligible = freeze_eligible(ctx.target_indices,
                                       infer_attack_labels(args.dataset, ctx.labels),
                                       ctx.clean_detect_window)
            score_ctx = replace(ctx, target_indices=ctx.target_indices[eligible].copy())
            mask = build_modification_mask(args.model_type, ctx.x_test.shape[0],
                                           ctx.x_test.shape[1], score_ctx.target_indices,
                                           args.history, args.target_offset, args.scope,
                                           protected) * ctx.modification_mask
            print(f"[corrshift] eligible protocol: score on {len(score_ctx.target_indices)} "
                  f"targets, shifting {int((mask.sum(axis=1) > 0).sum())} rows", flush=True)

        modifiable_features = np.flatnonzero(np.any(mask > 0, axis=0))
        if len(modifiable_features) == 0:
            raise ValueError("CorrShift has no modifiable features.")

        x_adv = ctx.x_test.copy()
        best_score = mean_blackbox_score(score_ctx, x_adv)
        query_count = 1
        start_time = time.time()
        rounds_completed = 0

        for round_idx in range(args.corrshift_rounds):
            round_best = best_score
            round_candidate: Optional[np.ndarray] = None
            round_description: Optional[Tuple[int, int]] = None

            for anchor in modifiable_features:
                direction = corrshift_direction(
                    corr_matrix,
                    int(anchor),
                    protected,
                    args.corrshift_top_k,
                    args.corrshift_min_abs_corr,
                )
                if not np.any(direction):
                    continue
                for orientation in (-1, 1):
                    delta_row = orientation * step_vector * direction
                    candidate = x_adv + mask * delta_row[None, :]
                    candidate = project_numpy(
                        candidate,
                        ctx.x_test,
                        epsilon,
                        mask,
                        ctx.lower_domain,
                        ctx.upper_domain,
                    )
                    candidate_score = mean_blackbox_score(score_ctx, candidate)
                    query_count += 1

                    improved = (
                        candidate_score < round_best
                        if goal == "evasion"
                        else candidate_score > round_best
                    )
                    if improved:
                        round_best = candidate_score
                        round_candidate = candidate
                        round_description = (int(anchor), orientation)

            if round_candidate is None:
                print(
                    f"[corrshift/{goal}] round {round_idx + 1}: no improving candidate; stop."
                )
                break
            x_adv = round_candidate
            best_score = round_best
            rounds_completed += 1
            anchor, orientation = round_description or (-1, 0)
            print(
                f"[corrshift/{goal}] round {round_idx + 1}: score={best_score:.8f}, "
                f"anchor={anchor}, orientation={orientation:+d}, queries={query_count}"
            )

        metadata = {
            "iterations_executed": rounds_completed,
            "final_optimization_loss": best_score,
            "runtime_seconds": time.time() - start_time,
            "query_count": query_count,
            "corrshift_effective_epsilon": epsilon.tolist(),
        }
        return x_adv, metadata

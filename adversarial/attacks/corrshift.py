"""CorrShift: gradient-free, correlation-guided score search (black-box)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from ..base import Attack
from ..context import AttackContext
from ..errors import detector_score_numpy, model_errors_numpy
from ..projection import project_numpy


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

        step_scalar = args.alpha if args.corrshift_step is None else args.corrshift_step
        if step_scalar < 0:
            raise ValueError("CorrShift step must be non-negative.")
        step_vector = np.minimum(epsilon, step_scalar).astype(np.float32)

        modifiable_features = np.flatnonzero(np.any(ctx.modification_mask > 0, axis=0))
        if len(modifiable_features) == 0:
            raise ValueError("CorrShift has no modifiable features.")

        x_adv = ctx.x_test.copy()
        best_score = mean_blackbox_score(ctx, x_adv)
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
                    candidate = x_adv + ctx.modification_mask * delta_row[None, :]
                    candidate = project_numpy(
                        candidate,
                        ctx.x_test,
                        epsilon,
                        ctx.modification_mask,
                        ctx.lower_domain,
                        ctx.upper_domain,
                    )
                    candidate_score = mean_blackbox_score(ctx, candidate)
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

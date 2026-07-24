"""Orchestration: run one concealment configuration, or the full comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .autoencoder_attack import AutoencoderConfig, BlackBoxAutoencoderAttack
from .base import AttackResult, ConcealmentAttack
from .config import load_config
from .data import (
    assert_no_leakage,
    load_attacker_normal,
    load_detector_and_test,
    select_anomalous_targets,
    subsample_normal,
)
from .detector_eval import DetectorEvaluator
from .masks import full_mask, mask_from_indices, resolve_controlled_indices
from .metrics import compute_metrics, to_table
from .postprocess import DomainConstraints
from .replay import ReplayAttack


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ae_config(args) -> AutoencoderConfig:
    hidden = None
    if args.ae_hidden:
        hidden = [int(x) for x in str(args.ae_hidden).split(",") if x.strip()]
    return AutoencoderConfig(
        hidden_layers=hidden,
        layers=args.ae_layers,
        compression=args.ae_compression,
        latent_dim=args.ae_latent,
        activation=args.ae_activation,
        learning_rate=args.ae_lr,
        batch_size=args.ae_batch_size,
        epochs=args.ae_epochs,
        patience=args.ae_patience,
        validation_split=args.ae_val_split,
        loss=args.ae_loss,
        seed=args.seed,
        device=args.ae_device,
        verbose=args.ae_verbose,
    )


def _load_controlled_file(path: str) -> List[str]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return [str(x) for x in json.loads(text)]
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_controlled(args, sensor_cols) -> Optional[List[int]]:
    """Controllable indices from the CLI, or None for unconstrained (all features)."""
    names = None
    if args.controlled_file:
        names = _load_controlled_file(args.controlled_file)
    elif args.controlled_features:
        names = [t for t in str(args.controlled_features).split(",") if t.strip()]
    if names is not None:
        return resolve_controlled_indices(sensor_cols, names=names)
    if args.controlled_percentage is not None:
        return resolve_controlled_indices(
            sensor_cols, fraction=args.controlled_percentage,
            random=args.controlled_random, seed=args.seed)
    if args.controlled_k is not None:
        return resolve_controlled_indices(
            sensor_cols, k=args.controlled_k, random=args.controlled_random, seed=args.seed)
    return None


def _timed_transform(attack: ConcealmentAttack, x_anom, mask) -> Tuple[AttackResult, float]:
    start = time.time()
    result = attack.transform(x_anom, feature_mask=mask)
    elapsed = time.time() - start
    result.metadata["transform_time_s"] = elapsed
    return result, elapsed


def _postprocess(result: AttackResult, dc: Optional[DomainConstraints]) -> AttackResult:
    if dc is None:
        return result
    x_adv = dc.apply(result.adversarial, result.original, result.feature_mask)
    return AttackResult(
        original=result.original, adversarial=x_adv, perturbation=x_adv - result.original,
        feature_mask=result.feature_mask, metadata=dict(result.metadata),
    )


def _save(output_dir: Path, name: str, result: AttackResult, targets: np.ndarray, row: Dict[str, Any]) -> None:
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "target_indices.npy", targets)
    np.save(run_dir / "original.npy", result.original)
    np.save(run_dir / "adversarial.npy", result.adversarial)
    np.save(run_dir / "perturbation.npy", result.perturbation)
    np.save(run_dir / "feature_mask.npy", result.feature_mask)
    with (run_dir / "result.json").open("w", encoding="utf-8") as fh:
        json.dump({"metadata": result.metadata, "metrics": row}, fh, indent=2, default=str)


def _identity_result(x_anom: np.ndarray, n_features: int) -> AttackResult:
    mask = full_mask(n_features)
    zeros = np.zeros_like(x_anom)
    return AttackResult(x_anom, x_anom.copy(), zeros, mask, {"attack": "none", "transform_time_s": 0.0})


# --------------------------------------------------------------------------- #
# Context shared by single and compare runs
# --------------------------------------------------------------------------- #
class _Context:
    def __init__(self, args):
        self.args = args
        self.adapter, self.scaler, self.x_test, self.labels, self.sensor_cols = \
            load_detector_and_test(args.dataset, args.model_path)
        # Full benign training set: used ONLY to reproduce the victim detector's theta
        # (theta is the defender's threshold; it must NOT depend on the attacker's budget).
        self.detector_train = load_attacker_normal(args.dataset, self.scaler, self.sensor_cols)
        # Attacker-observable benign data (possibly subsampled): the ONLY data the attack
        # (AE / replay buffer / domain bounds) may use.
        self.normal = subsample_normal(
            self.detector_train, size=args.attacker_train_size,
            fraction=args.attacker_train_fraction,
            random=args.attacker_train_random, seed=args.seed)
        self.attacker_train_samples = int(len(self.normal))
        print(f"Attacker benign observation set: {self.attacker_train_samples} samples "
              f"(detector theta uses the full {len(self.detector_train)})")
        assert_no_leakage(self.normal, self.x_test)
        self.n_features = self.x_test.shape[1]
        self.targets = select_anomalous_targets(
            args.dataset, self.labels, len(self.x_test), args.model_type,
            args.history, args.target_offset, args.max_targets)
        self.x_anom = self.x_test[self.targets]
        self.evaluator = DetectorEvaluator(
            self.adapter, self.scaler, self.sensor_cols, args.model_type,
            args.history, args.target_offset, args.instance_score, args.prediction_batch_size)
        self.theta, self.theta_source, self.percentile = self.evaluator.resolve_theta(
            args.dataset, self.detector_train, args.instance_threshold,
            args.instance_threshold_percentile)
        self.window = int(args.detection_window)
        self.dc = None
        if args.postprocess:
            self.dc = DomainConstraints.from_training(
                self.normal, discrete_max_unique=args.discrete_max_unique,
                clip_continuous=not args.no_clip)
        self.output_dir = Path(args.output_dir)

    def metrics_row(self, name, attack_kind, constraint, result) -> Dict[str, Any]:
        return compute_metrics(
            config_name=name, attack=attack_kind, constraint=constraint, result=result,
            x_test=self.x_test, target_indices=self.targets, evaluator=self.evaluator,
            theta=self.theta, window=self.window,
            extra={"theta_source": self.theta_source, "percentile": self.percentile,
                   "attacker_train_samples": self.attacker_train_samples})

    def evaluate(self, name, attack_kind, constraint, result, save=True,
                 postprocess=True) -> Dict[str, Any]:
        # The no-attack baseline must be the untouched anomalous samples: never
        # domain-clip it, or the baseline silently becomes a naive clipping attack.
        if postprocess and attack_kind != "none":
            result = _postprocess(result, self.dc)
        # Record the config-level labels in the saved metadata (an attack object reused
        # across configs, e.g. the full-input AE, keeps its own construction constraint).
        result.metadata["config_name"] = name
        result.metadata["config_constraint"] = constraint
        row = self.metrics_row(name, attack_kind, constraint, result)
        if save:
            _save(self.output_dir, name, result, self.targets, row)
        print(f"[{name}] ASR={row['attack_success_rate']} recall {row['recall_before']:.3f}"
              f"->{row['recall_after']:.3f} valid={row['valid']} |C|={row['n_controlled']}")
        return row


def _build_mask(ctx: _Context, constraint: str, controlled: Optional[List[int]]) -> np.ndarray:
    if constraint == "unconstrained" or controlled is None:
        return full_mask(ctx.n_features)
    return mask_from_indices(controlled, ctx.n_features)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def run_single(args) -> List[Dict[str, Any]]:
    ctx = _Context(args)
    controlled = resolve_controlled(args, ctx.sensor_cols)
    if args.constraint != "unconstrained" and controlled is None:
        raise ValueError("A constrained attack requires a controlled feature subset "
                         "(--controlled-features / --controlled-k / --controlled-percentage).")
    mask = _build_mask(ctx, args.constraint, controlled)
    name = f"{args.attack}_{args.constraint}"

    if args.attack == "replay":
        attack: ConcealmentAttack = ReplayAttack(
            strategy=args.replay_strategy, replay_start=args.replay_start,
            replay_length=args.replay_length, seed=args.seed,
            allow_offline_oracle=args.allow_offline_oracle)
    else:
        attack = BlackBoxAutoencoderAttack(
            constraint=args.constraint,
            controlled_indices=controlled if args.constraint != "unconstrained" else None,
            config=_ae_config(args))

    attack.fit(ctx.normal)
    result, _ = _timed_transform(attack, ctx.x_anom, mask)
    row = ctx.evaluate(name, args.attack, args.constraint, result)
    to_table([row]).to_csv(ctx.output_dir / "comparison.csv", index=False)
    return [row]


def run_compare(args) -> List[Dict[str, Any]]:
    ctx = _Context(args)
    controlled = resolve_controlled(args, ctx.sensor_cols)
    k_list = args.controlled_k_list
    if k_list is None:
        if controlled is None:
            raise ValueError("--compare needs a controlled subset (or --controlled-k-list) "
                             "for the constrained configurations.")
        k_masks = [(len(controlled), mask_from_indices(controlled, ctx.n_features), controlled)]
    else:
        k_masks = []
        for k in k_list:
            idx = resolve_controlled_indices(ctx.sensor_cols, k=k,
                                             random=args.controlled_random, seed=args.seed)
            k_masks.append((k, mask_from_indices(idx, ctx.n_features), idx))

    rows: List[Dict[str, Any]] = []

    # 1. no attack (baseline)
    rows.append(ctx.evaluate("no_attack", "none", "none",
                             _identity_result(ctx.x_anom, ctx.n_features)))

    # 2. replay unconstrained
    replay_full = ReplayAttack(strategy=args.replay_strategy, replay_start=args.replay_start,
                               replay_length=args.replay_length, seed=args.seed,
                               allow_offline_oracle=args.allow_offline_oracle).fit(ctx.normal)
    res, _ = _timed_transform(replay_full, ctx.x_anom, full_mask(ctx.n_features))
    rows.append(ctx.evaluate("replay_unconstrained", "replay", "unconstrained", res))

    # 4. autoencoder unconstrained (input = all features); reused for partial masks
    ae_full = BlackBoxAutoencoderAttack(constraint="partially_constrained",
                                        controlled_indices=None, config=_ae_config(args)).fit(ctx.normal)
    res, _ = _timed_transform(ae_full, ctx.x_anom, full_mask(ctx.n_features))
    rows.append(ctx.evaluate("autoencoder_unconstrained", "autoencoder", "unconstrained", res))

    for k, mask, idx in k_masks:
        # 3. replay constrained
        res, _ = _timed_transform(replay_full, ctx.x_anom, mask)
        rows.append(ctx.evaluate(f"replay_constrained_k{k}", "replay", "partially_constrained", res))
        # 5. autoencoder partially constrained (reuse full-input model, mask on C)
        res, _ = _timed_transform(ae_full, ctx.x_anom, mask)
        rows.append(ctx.evaluate(f"autoencoder_partial_k{k}", "autoencoder", "partially_constrained", res))
        # 6. autoencoder fully constrained (model observes only C)
        ae_c = BlackBoxAutoencoderAttack(constraint="fully_constrained",
                                         controlled_indices=idx, config=_ae_config(args)).fit(ctx.normal)
        res, _ = _timed_transform(ae_c, ctx.x_anom, mask)
        rows.append(ctx.evaluate(f"autoencoder_fully_k{k}", "autoencoder", "fully_constrained", res))

    table = to_table(rows)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(ctx.output_dir / "comparison.csv", index=False)
    print("\n=== Comparison table ===")
    import pandas as pd
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))
    print(f"\nSaved comparison table to {(ctx.output_dir / 'comparison.csv').resolve()}")
    return rows

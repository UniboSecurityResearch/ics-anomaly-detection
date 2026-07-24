"""Generic replay attack (baseline).

Replaces anomalous observations with observations drawn from a NORMAL operating
interval (the attacker's benign buffer, built only from training / attacker-observable
normal data). For controllable feature set C with binary mask m:

    x_adv_t = m ⊙ x_replay_t + (1-m) ⊙ x_anom_t

so untouched features stay exactly equal to the anomalous originals.

Selection strategies: ``fixed`` (a specified normal block, repeated), ``cyclic``
(default; cycle the whole normal buffer), ``random`` (seeded), ``nearest_context``
(pick the normal row nearest on the NON-controlled context features). The replay never
uses future test samples unless ``offline_oracle`` is explicitly enabled (off by
default).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import AttackResult, ConcealmentAttack, assert_mask_preserved, combine_masked

_STRATEGIES = ("fixed", "cyclic", "random", "nearest_context", "offline_oracle")


class ReplayAttack(ConcealmentAttack):
    name = "replay"

    def __init__(
        self,
        strategy: str = "cyclic",
        replay_start: int = 0,
        replay_length: Optional[int] = None,
        seed: int = 42,
        allow_offline_oracle: bool = False,
        nearest_max_candidates: int = 5000,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown replay strategy {strategy!r}; choose from {_STRATEGIES}.")
        if strategy == "offline_oracle" and not allow_offline_oracle:
            raise ValueError(
                "strategy 'offline_oracle' is non-causal and disabled by default; pass "
                "allow_offline_oracle=True to opt in explicitly."
            )
        self.strategy = strategy
        self.replay_start = int(replay_start)
        self.replay_length = replay_length
        self.seed = int(seed)
        self.allow_offline_oracle = allow_offline_oracle
        self.nearest_max_candidates = int(nearest_max_candidates)
        self.normal_buffer: Optional[np.ndarray] = None
        self.oracle_buffer: Optional[np.ndarray] = None

    def fit(self, normal_data: np.ndarray, oracle_data: Optional[np.ndarray] = None, **kwargs):
        buf = np.asarray(normal_data, dtype=np.float64)
        if buf.ndim != 2 or len(buf) == 0:
            raise ValueError("Replay needs a non-empty 2D normal buffer.")
        self.normal_buffer = buf
        if oracle_data is not None:
            self.oracle_buffer = np.asarray(oracle_data, dtype=np.float64)
        return self

    # --- replay index selection -------------------------------------------------
    def _select_replay(self, x_anom: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.normal_buffer is not None, "Call fit() before transform()."
        buffer = self.normal_buffer
        n_attack = len(x_anom)
        n_buf = len(buffer)

        if self.strategy == "offline_oracle":
            if self.oracle_buffer is None:
                raise ValueError("offline_oracle selected but no oracle_data was provided to fit().")
            buffer = self.oracle_buffer
            n_buf = len(buffer)

        if self.strategy in ("fixed",):
            block_len = self.replay_length or n_attack
            block_len = min(block_len, n_buf)
            block_idx = (self.replay_start + np.arange(block_len)) % n_buf
            rows = block_idx[np.arange(n_attack) % block_len]
        elif self.strategy in ("cyclic", "offline_oracle"):
            rows = (self.replay_start + np.arange(n_attack)) % n_buf
        elif self.strategy == "random":
            rng = np.random.default_rng(self.seed)
            rows = rng.integers(0, n_buf, size=n_attack)
        elif self.strategy == "nearest_context":
            rows = self._nearest_context_rows(x_anom, mask, buffer)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(self.strategy)
        return buffer[rows]

    def _nearest_context_rows(
        self, x_anom: np.ndarray, mask: np.ndarray, buffer: np.ndarray
    ) -> np.ndarray:
        context = ~np.asarray(mask).astype(bool)  # non-controlled features
        if not np.any(context):
            print("ReplayAttack: nearest_context has no context features (mask is full); "
                  "falling back to cyclic.")
            return (self.replay_start + np.arange(len(x_anom))) % len(buffer)
        cand = buffer
        if len(buffer) > self.nearest_max_candidates:
            pos = np.linspace(0, len(buffer) - 1, self.nearest_max_candidates).astype(int)
            cand = buffer[pos]
        else:
            pos = np.arange(len(buffer))
        cand_ctx = cand[:, context]
        rows = np.empty(len(x_anom), dtype=int)
        for i in range(len(x_anom)):
            d = np.sum((cand_ctx - x_anom[i, context][None, :]) ** 2, axis=1)
            rows[i] = pos[int(np.argmin(d))]
        return rows

    def transform(self, anomalous_data, feature_mask=None, **kwargs) -> AttackResult:
        x_anom = np.asarray(anomalous_data, dtype=np.float64)
        n_features = x_anom.shape[1]
        mask = (
            np.ones(n_features, dtype=bool)
            if feature_mask is None
            else np.asarray(feature_mask).astype(bool)
        )
        if self.normal_buffer is not None and self.normal_buffer.shape[1] != n_features:
            raise ValueError("Normal buffer and anomalous data have different feature counts.")

        x_replay = self._select_replay(x_anom, mask)
        x_adv = combine_masked(x_replay, x_anom, mask)
        assert_mask_preserved(x_adv, x_anom, mask)

        metadata = {
            "attack": self.name,
            "strategy": self.strategy,
            "replay_start": self.replay_start,
            "replay_length": self.replay_length,
            "seed": self.seed,
            "n_controlled": int(np.sum(mask)),
        }
        return AttackResult(
            original=x_anom,
            adversarial=x_adv,
            perturbation=x_adv - x_anom,
            feature_mask=mask,
            metadata=metadata,
        )

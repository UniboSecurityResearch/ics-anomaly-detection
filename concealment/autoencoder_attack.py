"""Black-box learning-based concealment attack (attacker-owned autoencoder).

The attacker trains its OWN feed-forward autoencoder G_A on benign data it can
observe, minimizing reconstruction MSE. At attack time it reconstructs the anomalous
vector and substitutes the controllable features. It is strictly black-box w.r.t. the
detector: G_A never uses the detector's weights, gradients, threshold, anomaly scores
or labels, and never queries it iteratively (Erba et al.).

Three configurations:
  * unconstrained          : observe all features, replace all (m = 1);
  * partially_constrained  : observe all features (as context), replace only C;
  * fully_constrained      : observe ONLY C, replace only C  (G_A^C : R^|C| -> R^|C|).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import tensorflow as tf

from .base import AttackResult, ConcealmentAttack, assert_mask_preserved

CONSTRAINTS = ("unconstrained", "partially_constrained", "fully_constrained")


@dataclass
class AutoencoderConfig:
    hidden_layers: Optional[List[int]] = None   # explicit encoder layer sizes; overrides layers/compression
    layers: int = 4                              # number of encoder layers if hidden_layers is None
    compression: float = 2.0                     # bottleneck = input_dim / compression
    latent_dim: Optional[int] = None             # explicit bottleneck size
    activation: str = "relu"
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    patience: int = 5                            # early stopping; <=0 disables
    validation_split: float = 0.1
    loss: str = "mse"
    seed: int = 42
    device: str = "auto"                         # 'cpu' | 'gpu' | 'auto'
    verbose: int = 0

    def __post_init__(self):
        if self.hidden_layers is not None:
            self.hidden_layers = [int(x) for x in self.hidden_layers]


def build_autoencoder(input_dim: int, cfg: AutoencoderConfig) -> "tf.keras.Model":
    from tensorflow.keras import optimizers
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential

    if cfg.hidden_layers:
        enc = list(cfg.hidden_layers)
    else:
        bottleneck = max(1, int(round(input_dim / cfg.compression)))
        enc = [int(s) for s in np.linspace(input_dim, bottleneck, cfg.layers + 1).astype(int)[1:]]
    if cfg.latent_dim:
        enc[-1] = int(cfg.latent_dim)
    encoder_sizes = enc
    decoder_sizes = list(reversed(enc[:-1]))

    model = Sequential(name="attacker_autoencoder")
    model.add(Input(shape=(input_dim,)))
    for size in encoder_sizes:
        model.add(Dense(size, activation=cfg.activation))
    for size in decoder_sizes:
        model.add(Dense(size, activation=cfg.activation))
    model.add(Dense(input_dim))  # linear reconstruction output
    model.compile(optimizer=optimizers.Adam(cfg.learning_rate), loss=cfg.loss)
    return model


def _device_scope(device: str):
    device = (device or "auto").lower()
    if device == "cpu":
        return tf.device("/CPU:0")
    if device == "gpu":
        return tf.device("/GPU:0")
    return tf.device("/CPU:0") if not tf.config.list_physical_devices("GPU") else tf.device("/GPU:0")


class BlackBoxAutoencoderAttack(ConcealmentAttack):
    name = "autoencoder"

    def __init__(
        self,
        constraint: str = "unconstrained",
        controlled_indices: Optional[Sequence[int]] = None,
        config: Optional[AutoencoderConfig] = None,
    ) -> None:
        if constraint not in CONSTRAINTS:
            raise ValueError(f"constraint must be one of {CONSTRAINTS}.")
        self.constraint = constraint
        self.controlled_indices = None if controlled_indices is None else sorted(int(i) for i in controlled_indices)
        self.config = config or AutoencoderConfig()
        self.model: Optional[tf.keras.Model] = None
        self.input_columns: Optional[np.ndarray] = None  # columns fed to G_A
        self.train_time_s: float = 0.0
        self.epochs_run: int = 0
        self.n_train_samples: int = 0

    def _resolve_columns(self, n_features: int) -> np.ndarray:
        """Columns that G_A observes. Fully constrained sees only C; the others see all."""
        if self.constraint == "fully_constrained":
            if not self.controlled_indices:
                raise ValueError("fully_constrained requires controlled_indices.")
            return np.asarray(self.controlled_indices, dtype=int)
        return np.arange(n_features, dtype=int)

    def fit(self, normal_data: np.ndarray, **kwargs) -> "BlackBoxAutoencoderAttack":
        x = np.asarray(normal_data, dtype=np.float32)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Autoencoder attack needs a non-empty 2D normal set.")
        # Leakage guard: this must be benign attacker-observable data only.
        assert kwargs.get("_is_normal", True), "Attacker AE may be fit on NORMAL data only."

        self.input_columns = self._resolve_columns(x.shape[1])
        x_in = x[:, self.input_columns]
        self.n_train_samples = int(len(x_in))

        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)
        with _device_scope(self.config.device):
            self.model = build_autoencoder(x_in.shape[1], self.config)
            callbacks = []
            use_val = self.config.validation_split and self.config.validation_split > 0
            if self.config.patience and self.config.patience > 0 and use_val:
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=self.config.patience,
                        restore_best_weights=True,
                    )
                )
            start = time.time()
            history = self.model.fit(
                x_in, x_in,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=self.config.verbose,
            )
            self.train_time_s = time.time() - start
        self.epochs_run = len(history.history.get("loss", []))
        return self

    def transform(self, anomalous_data, feature_mask=None, **kwargs) -> AttackResult:
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")
        x_anom = np.asarray(anomalous_data, dtype=np.float64)
        n_features = x_anom.shape[1]

        if feature_mask is None:
            if self.constraint == "unconstrained":
                mask = np.ones(n_features, dtype=bool)
            elif self.controlled_indices is not None:
                mask = np.zeros(n_features, dtype=bool)
                mask[self.controlled_indices] = True
            else:
                raise ValueError("A feature_mask or controlled_indices is required.")
        else:
            mask = np.asarray(feature_mask).astype(bool)

        model_input = x_anom[:, self.input_columns].astype(np.float32)
        recon = np.asarray(
            self.model.predict(model_input, batch_size=self.config.batch_size, verbose=0),
            dtype=np.float64,
        )

        x_adv = x_anom.copy()
        if self.constraint == "fully_constrained":
            # recon columns correspond to input_columns (== controlled_indices)
            x_adv[:, self.input_columns] = recon
            # ensure only controllable columns changed (input_columns == C here)
        else:
            full_recon = recon  # (n, n_features)
            x_adv[:, mask] = full_recon[:, mask]

        assert_mask_preserved(x_adv, x_anom, mask)

        metadata = {
            "attack": self.name,
            "constraint": self.constraint,
            "n_controlled": int(np.sum(mask)),
            "input_dim": int(len(self.input_columns)),
            "n_train_samples": self.n_train_samples,
            "train_time_s": self.train_time_s,
            "epochs_run": self.epochs_run,
            "config": vars(self.config).copy(),
        }
        return AttackResult(
            original=x_anom,
            adversarial=x_adv,
            perturbation=x_adv - x_anom,
            feature_mask=mask,
            metadata=metadata,
        )

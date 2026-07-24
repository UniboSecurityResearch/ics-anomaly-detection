"""Loading of models, the fitted scaler, standardized training data and vectors.

The scaler and data paths are the repository's (``models/<ds>_scaler.pkl``,
``data/<ds>/...``), so the entrypoint must run from the repository root.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader import load_train_data
from .context import ModelAdapter


def _keras_candidates(obj: Any) -> Iterable[Any]:
    yield obj
    for attr in (
        "model",
        "event_detector",
        "detector",
        "predictor",
        "network",
        "keras_model",
    ):
        if hasattr(obj, attr):
            yield getattr(obj, attr)


def load_model_adapter(path: str) -> ModelAdapter:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    keras_error: Optional[Exception] = None
    try:
        keras_model = tf.keras.models.load_model(str(model_path), compile=False)
        keras_model.trainable = False
        return ModelAdapter(keras_model, keras_model, "tf.keras.models.load_model")
    except Exception as exc:  # custom/pickled repository objects are common
        keras_error = exc

    pickle_error: Optional[Exception] = None
    try:
        with model_path.open("rb") as handle:
            obj = pickle.load(handle)
    except Exception as exc:
        pickle_error = exc
        try:
            import joblib

            obj = joblib.load(model_path)
        except Exception as joblib_exc:
            raise RuntimeError(
                "Could not load the model as Keras, pickle or joblib. Pass the actual "
                "trained model artifact rather than a weights-only checkpoint. "
                f"Keras error: {keras_error!r}; pickle error: {pickle_error!r}; "
                f"joblib error: {joblib_exc!r}"
            ) from joblib_exc

    candidates = list(_keras_candidates(obj))
    keras_model = next((x for x in candidates if isinstance(x, tf.keras.Model)), None)
    predictor = keras_model
    if predictor is None:
        predictor = next(
            (
                x
                for x in candidates
                if hasattr(x, "predict") or callable(x)
            ),
            None,
        )
    if predictor is None:
        raise RuntimeError("Pickled object does not expose predict() or a Keras submodel.")
    if keras_model is not None:
        keras_model.trainable = False
    return ModelAdapter(predictor, keras_model, f"pickle:{type(obj).__name__}")


def load_scaler(dataset: str) -> Any:
    path = Path("models") / f"{dataset}_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train a repository model first so data_loader.py saves "
            "the fitted StandardScaler."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_train_scaled(dataset: str, scaler: Any, expected_cols: Sequence[str]) -> np.ndarray:
    """Load training data without refitting the repository scaler."""
    x_raw, train_cols = load_train_data(
        dataset, scaler=copy.deepcopy(scaler), no_transform=True, verbose=True
    )
    train_cols = [str(x) for x in train_cols]
    if train_cols != list(expected_cols):
        raise ValueError("Training and test feature order do not match.")
    return np.asarray(scaler.transform(np.asarray(x_raw)), dtype=np.float32)


def load_vector(path: str, expected_size: int, name: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            values = np.asarray(json.load(handle))
    else:
        values = pd.read_csv(path, header=None).values
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != expected_size:
        raise ValueError(f"{name} has {values.size} values; expected {expected_size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return values


def load_epsilon(args: argparse.Namespace, n_features: int, scaler: Any) -> np.ndarray:
    if args.epsilon_raw_file:
        raw = load_vector(args.epsilon_raw_file, n_features, "raw epsilon vector")
        if np.any(raw < 0):
            raise ValueError("Raw epsilon values must be non-negative.")
        if not hasattr(scaler, "scale_"):
            raise ValueError("Loaded scaler does not expose StandardScaler.scale_.")
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        scale = np.where(scale == 0.0, 1.0, scale)
        return raw / scale
    return np.full(n_features, args.epsilon, dtype=np.float32)

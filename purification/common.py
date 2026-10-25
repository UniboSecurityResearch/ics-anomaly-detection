"""Loading helpers shared by the purification entrypoints (nothing is ever written
to models/ or to the scaler file)."""

from __future__ import annotations

import glob
import json
import os
import subprocess
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from adversarial.io_utils import load_scaler
from data_loader import load_test_data, load_train_data

# Threshold grids. "train" is the grid of the SWAT-CLEAN training runs
# (swat_chrono_seeds.sbatch, retune_thresholds.TRAIN_*), "repo" the default grid of
# main_train.py, "wide" their union (retune_thresholds.WIDE_*).
GRIDS = {
    "train": ([0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999], [1, 3, 5, 10, 20, 30]),
    "repo": ([0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997,
              0.998, 0.999, 0.9995, 0.99995], [1, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
}
GRIDS["wide"] = (sorted(set(GRIDS["train"][0]) | set(GRIDS["repo"][0])),
                 sorted(set(GRIDS["train"][1]) | set(GRIDS["repo"][1])))


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return dict(commit=run("git", "rev-parse", "HEAD"),
                dirty=bool(run("git", "status", "--porcelain", "--untracked-files=no")))


def load_detector(model_dir: str):
    jsons = sorted(glob.glob(os.path.join(model_dir, "*.json")))
    jsons = [j for j in jsons if not os.path.basename(j).startswith(("defense", "calib"))]
    if len(jsons) != 1:
        raise SystemExit(f"{model_dir}: expected exactly one model .json, found {jsons}")
    name = os.path.basename(jsons[0])[:-5]
    h5 = os.path.join(model_dir, name + ".h5")
    with open(jsons[0]) as fd:
        params = json.load(fd)
    model = tf.keras.models.load_model(h5, compile=False)
    model.trainable = False
    model_type = name.split("-")[0]
    history = params.get("history")
    return model, model_type, (int(history) if history else None), params, name


def load_data(dataset: str):
    """Standardised train/test data with the SAVED scaler, in float64 like main_train
    (whose fit_transform on the same file gives the same matrix; retune_thresholds.py
    checks the saved scaler against a fresh fit)."""
    scaler = load_scaler(dataset)
    raw, cols = load_train_data(dataset, no_transform=True)  # writes nothing
    cols = [str(c) for c in cols]
    x_train = scaler.transform(np.asarray(raw)).astype(np.float64)
    x_test, y_test, test_cols = load_test_data(dataset, scaler=scaler)
    if [str(c) for c in test_cols] != cols:
        raise SystemExit("training and test columns differ")
    return x_train, np.asarray(x_test, dtype=np.float64), np.asarray(y_test), cols


def load_purifier(diffusion_dir: Optional[str], t_star: int, n_steps: int, sampler: str):
    from .purifier import Purifier
    if t_star == 0:
        return Purifier(None, 0, 0, sampler), None
    from .denoiser import load_denoiser
    if not diffusion_dir:
        raise SystemExit("--diffusion is required for t* > 0")
    model, config = load_denoiser(diffusion_dir)
    return Purifier(model, t_star, n_steps, sampler), config


def data_digest(dataset: str) -> dict:
    """SHA-256 of the raw (unscaled) feature matrices and labels as the repository
    loads them: identical digests mean identical data, whatever the CSV line endings."""
    import hashlib
    raw, _ = load_train_data(dataset, no_transform=True)
    xt, yt, _ = load_test_data(dataset, scaler=load_scaler(dataset), no_transform=True)
    h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()
    return dict(train=h(raw), test=h(xt), labels=h(np.asarray(yt).astype(float)))


def check_data_digests(dataset: str, expected) -> None:
    """Exit with code 5 unless the data have the expected digests (train, test, labels)."""
    import sys
    digest = data_digest(dataset)
    expected = dict(zip(("train", "test", "labels"), expected))
    for k in expected:
        print(f"data digest {k}: {digest[k]} ({'OK' if digest[k] == expected[k] else 'MISMATCH'})", flush=True)
    if digest != expected:
        print("ERROR: the data differ from the expected ones", flush=True)
        sys.exit(5)

"""Noise-prediction network eps_theta(x_t, t) for multivariate time-series windows.

A 1-D U-Net over time (features are channels), the 1-D analogue of the DDPM / ADM
U-Net used by DiffPure and Lee & Kim on images: residual blocks with GroupNorm and
SiLU, a sinusoidal timestep embedding added in every block, two downsamplings, skip
connections. It is fully convolutional, so it accepts any window length (left-padded
internally to a multiple of 4 by repeating the oldest row, which leaves the most
recent rows -- the ones the detectors score -- untouched by the padding).

Written with plain tf.Variable / tf.nn ops (no Keras layers), so the weights are
saved as an .npz independent of the Keras version on the cluster, and the network is
differentiable end-to-end inside a tf.GradientTape.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


def _init(rng: np.random.Generator, shape: Sequence[int], fan_in: int, scale: float = 1.0):
    limit = scale * math.sqrt(3.0 / max(fan_in, 1))  # variance-scaling uniform
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


class _Params:
    """Ordered registry of named variables."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.vars: Dict[str, tf.Variable] = {}

    def add(self, name: str, value: np.ndarray) -> tf.Variable:
        if name in self.vars:
            raise KeyError(name)
        var = tf.Variable(value, name=name.replace("/", "_"), trainable=True)
        self.vars[name] = var
        return var

    def conv(self, name: str, k: int, cin: int, cout: int, zero: bool = False):
        w = np.zeros((k, cin, cout), np.float32) if zero else _init(self.rng, (k, cin, cout), k * cin)
        return self.add(name + "/w", w), self.add(name + "/b", np.zeros(cout, np.float32))

    def dense(self, name: str, cin: int, cout: int):
        return (self.add(name + "/w", _init(self.rng, (cin, cout), cin)),
                self.add(name + "/b", np.zeros(cout, np.float32)))

    def norm(self, name: str, c: int):
        return (self.add(name + "/g", np.ones(c, np.float32)),
                self.add(name + "/b", np.zeros(c, np.float32)))


def _conv(x, wb, stride: int = 1):
    w, b = wb
    return tf.nn.conv1d(x, w, stride=stride, padding="SAME") + b


def _dense(x, wb):
    w, b = wb
    return tf.matmul(x, w) + b


def _group_norm(x, gb, groups: int, eps: float = 1e-5):
    g, b = gb
    shape = tf.shape(x)
    c = x.shape[-1]
    xg = tf.reshape(x, [shape[0], shape[1], groups, c // groups])
    mean, var = tf.nn.moments(xg, axes=[1, 3], keepdims=True)
    xg = (xg - mean) * tf.math.rsqrt(var + eps)
    return tf.reshape(xg, shape) * g + b


def timestep_embedding(t: tf.Tensor, dim: int, max_period: float = 10000.0) -> tf.Tensor:
    half = dim // 2
    freqs = tf.exp(-math.log(max_period) * tf.range(half, dtype=tf.float32) / half)
    args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
    return tf.concat([tf.cos(args), tf.sin(args)], axis=1)


class UNet1D(tf.Module):
    def __init__(self, n_features: int, base_channels: int = 64,
                 channel_mults: Tuple[int, ...] = (1, 2, 2), n_res: int = 2,
                 groups: int = 8, dropout: float = 0.1, seed: int = 0):
        super().__init__(name="unet1d")
        self.config = dict(n_features=int(n_features), base_channels=int(base_channels),
                           channel_mults=list(channel_mults), n_res=int(n_res),
                           groups=int(groups), dropout=float(dropout), seed=int(seed))
        self.groups = groups
        self.dropout = dropout
        self.n_res = n_res
        self.levels = len(channel_mults)
        p = _Params(seed)
        C = base_channels
        self.temb_dim = C
        self.t1 = p.dense("temb/0", C, 4 * C)
        self.t2 = p.dense("temb/1", 4 * C, 4 * C)
        self.inp = p.conv("in", 3, n_features, C)

        def res(name, cin, cout):
            blk = dict(n1=p.norm(name + "/n1", cin), c1=p.conv(name + "/c1", 3, cin, cout),
                       t=p.dense(name + "/t", 4 * C, cout), n2=p.norm(name + "/n2", cout),
                       c2=p.conv(name + "/c2", 3, cout, cout, zero=True))
            if cin != cout:
                blk["skip"] = p.conv(name + "/skip", 1, cin, cout)
            return blk

        self.down: List[List[dict]] = []
        self.downsample: List[Optional[tuple]] = []
        skip_channels = [C]
        ch = C
        for lvl, mult in enumerate(channel_mults):
            blocks = []
            for r in range(n_res):
                blocks.append(res(f"down{lvl}/res{r}", ch, C * mult))
                ch = C * mult
                skip_channels.append(ch)
            self.down.append(blocks)
            if lvl < self.levels - 1:
                self.downsample.append(p.conv(f"down{lvl}/ds", 3, ch, ch))
                skip_channels.append(ch)
            else:
                self.downsample.append(None)
        self.mid = [res("mid/res0", ch, ch), res("mid/res1", ch, ch)]
        self.up: List[List[dict]] = []
        self.upsample: List[Optional[tuple]] = []
        for lvl, mult in reversed(list(enumerate(channel_mults))):
            blocks = []
            for r in range(n_res + 1):
                blocks.append(res(f"up{lvl}/res{r}", ch + skip_channels.pop(), C * mult))
                ch = C * mult
            self.up.append(blocks)
            self.upsample.append(p.conv(f"up{lvl}/us", 3, ch, ch) if lvl > 0 else None)
        self.out_norm = p.norm("out/n", ch)
        self.out = p.conv("out", 3, ch, n_features, zero=True)
        self.params = p.vars
        self.multiple = 2 ** (self.levels - 1)

    # ------------------------------------------------------------------ forward
    def _res(self, blk, x, temb, training):
        h = tf.nn.silu(_group_norm(x, blk["n1"], self.groups))
        h = _conv(h, blk["c1"])
        h = h + _dense(temb, blk["t"])[:, None, :]
        h = tf.nn.silu(_group_norm(h, blk["n2"], self.groups))
        if training and self.dropout > 0:
            h = tf.nn.dropout(h, rate=self.dropout)
        h = _conv(h, blk["c2"])
        skip = _conv(x, blk["skip"]) if "skip" in blk else x
        return skip + h

    def __call__(self, x: tf.Tensor, t: tf.Tensor, training: bool = False) -> tf.Tensor:
        """x: (B, L, n_features) noisy window; t: (B,) diffusion timesteps (1..T)."""
        length = tf.shape(x)[1]
        pad = (-length) % self.multiple
        x_in = tf.concat([tf.repeat(x[:, :1, :], pad, axis=1), x], axis=1)
        temb = timestep_embedding(t, self.temb_dim)
        temb = _dense(tf.nn.silu(_dense(temb, self.t1)), self.t2)
        h = _conv(x_in, self.inp)
        skips = [h]
        for blocks, ds in zip(self.down, self.downsample):
            for blk in blocks:
                h = self._res(blk, h, temb, training)
                skips.append(h)
            if ds is not None:
                h = _conv(h, ds, stride=2)
                skips.append(h)
        for blk in self.mid:
            h = self._res(blk, h, temb, training)
        for blocks, us in zip(self.up, self.upsample):
            for blk in blocks:
                h = self._res(blk, tf.concat([h, skips.pop()], axis=-1), temb, training)
            if us is not None:
                h = _conv(tf.repeat(h, 2, axis=1), us)
        h = tf.nn.silu(_group_norm(h, self.out_norm, self.groups))
        out = _conv(h, self.out)
        return out[:, pad:, :]

    # ------------------------------------------------------------------ io
    @property
    def variables_list(self) -> List[tf.Variable]:
        return list(self.params.values())

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {k: v.numpy() for k, v in self.params.items()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        missing = set(self.params) - set(weights)
        extra = set(weights) - set(self.params)
        if missing or extra:
            raise ValueError(f"weight mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
        for k, v in self.params.items():
            if tuple(v.shape) != tuple(weights[k].shape):
                raise ValueError(f"shape mismatch for {k}: {v.shape} vs {weights[k].shape}")
            v.assign(weights[k])

    def n_parameters(self) -> int:
        return int(sum(np.prod(v.shape) for v in self.params.values()))


def save_denoiser(directory: str, model: UNet1D, weights: Dict[str, np.ndarray],
                  meta: dict) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "weights.npz", **{k.replace("/", "__"): v for k, v in weights.items()})
    with (path / "config.json").open("w") as fd:
        json.dump(dict(model=model.config, **meta), fd, indent=1)


def load_denoiser(directory: str) -> Tuple[UNet1D, dict]:
    path = Path(directory)
    with (path / "config.json").open() as fd:
        config = json.load(fd)
    cfg = dict(config["model"])
    cfg["channel_mults"] = tuple(cfg["channel_mults"])
    model = UNet1D(**cfg)
    data = np.load(path / "weights.npz")
    model.set_weights({k.replace("__", "/"): data[k] for k in data.files})
    return model, config

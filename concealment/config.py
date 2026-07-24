"""Optional YAML/JSON attack configuration.

Example (§11):

    attack:
      type: blackbox_autoencoder
      constraint: partially_constrained
      controlled_features:
        - T7
        - PU10
        - PU11

YAML support is optional (PyYAML); JSON always works. CLI flags override the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional dependency
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "PyYAML is required for YAML configs (`pip install pyyaml`) or use JSON."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


_TYPE_ALIASES = {
    "blackbox_autoencoder": "autoencoder",
    "autoencoder": "autoencoder",
    "replay": "replay",
    "generic_replay": "replay",
}


def apply_config(args, cfg: Dict[str, Any]):
    """Overlay a loaded config onto the parsed CLI namespace. CLI flags that were left
    at their default are overridden; the config never touches the detector settings."""
    attack = cfg.get("attack", cfg)
    if "type" in attack:
        args.attack = _TYPE_ALIASES.get(str(attack["type"]).lower(), str(attack["type"]))
    if "constraint" in attack:
        args.constraint = str(attack["constraint"])
    if attack.get("controlled_features"):
        args.controlled_features = ",".join(str(x) for x in attack["controlled_features"])
    if "controlled_k" in attack:
        args.controlled_k = int(attack["controlled_k"])
    if "controlled_percentage" in attack:
        args.controlled_percentage = float(attack["controlled_percentage"])

    ae = attack.get("autoencoder", cfg.get("autoencoder", {}))
    ae_map = {
        "hidden_layers": "ae_hidden", "layers": "ae_layers", "compression": "ae_compression",
        "latent_dim": "ae_latent", "activation": "ae_activation", "learning_rate": "ae_lr",
        "batch_size": "ae_batch_size", "epochs": "ae_epochs", "patience": "ae_patience",
        "validation_split": "ae_val_split", "loss": "ae_loss", "device": "ae_device",
    }
    for key, dest in ae_map.items():
        if key in ae:
            value = ae[key]
            if key == "hidden_layers" and isinstance(value, (list, tuple)):
                value = ",".join(str(v) for v in value)
            setattr(args, dest, value)

    replay = attack.get("replay", cfg.get("replay", {}))
    for key, dest in {"strategy": "replay_strategy", "start": "replay_start",
                      "length": "replay_length"}.items():
        if key in replay:
            setattr(args, dest, replay[key])
    return args

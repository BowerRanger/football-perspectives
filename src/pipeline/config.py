from pathlib import Path

import yaml

_DEFAULTS_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"


def load_config(path: Path | None = None) -> dict:
    with open(_DEFAULTS_PATH) as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v

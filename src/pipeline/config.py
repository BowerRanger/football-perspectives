import os
import re
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"

# Match ``${NAME}`` standalone references. Strings that aren't pure
# substitutions are left untouched (we don't interpolate inside a
# larger string — that's a separate concern and easy to add later).
_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def load_config(path: Path | None = None) -> dict:
    with open(_DEFAULTS_PATH) as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        _deep_merge(cfg, overrides)
    _substitute_env(cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _substitute_env(node: Any) -> None:
    """Recursively rewrite ``${VAR}`` strings to the env var's value.

    Used by ``hmr_world.batch`` to pick up Terraform outputs sourced
    into the shell before ``recon.py run``. Missing vars become empty
    strings so config validators can flag them with a helpful message.
    """
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if isinstance(value, str):
                node[key] = _resolve_string(value)
            else:
                _substitute_env(value)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            if isinstance(value, str):
                node[idx] = _resolve_string(value)
            else:
                _substitute_env(value)


def _resolve_string(value: str) -> str:
    match = _ENV_PATTERN.match(value)
    if not match:
        return value
    return os.environ.get(match.group(1), "")

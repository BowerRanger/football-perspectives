"""``shorts/<shot>_shorts.json`` sidecar: what the shorts stage decided.

Layout::

    {
      "version": 1, "shot": "gberch",
      "moments":   {strike, line_cross, impact, keeper_dive, buildup_start,
                    scorer_pid, keeper_pid, goal_end, sources{...}},   # effective (pins applied)
      "templates": {"<name>": {
          "template": name, "ok": bool,
          "slots":   [{id, chosen, status, pass, cut, rejected[{candidate, framing}], framing}],
          "passes":  [render pass specs],
          "framing": {pass_id: {ok, failures, metrics}},
          "edl":     compositor EDL (src filled in), "captions": [...],
          "outputs": {"mp4": rel path, "duration_s": float, ...}}},
      "operator": {                    # hand-authored; NEVER overwritten by a re-run
          "moments":    {"strike": 371, ...},              # frame pins over derived moments
          "captions":   {"<template>": [caption, ...]},    # replace a template's captions
          "templates":  ["matchday", ...] | null,          # which templates to build
          "candidates": {"<template>": {"<slot>": idx}}    # pin a slot's candidate
      }
    }

Operator input always wins: ``merge_regenerated`` carries ``operator`` over
untouched and ``effective_moments`` applies the frame pins.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

SIDECAR_VERSION = 1
MOMENT_FRAME_KEYS = ("strike", "line_cross", "impact", "keeper_dive", "buildup_start")
CAPTION_STYLES = ("title", "kicker", "sub", "chip", "chip_dark")
OPERATOR_KEYS = ("moments", "captions", "templates", "candidates")


def sidecar_path(output_dir: Path, shot: str) -> Path:
    return Path(output_dir) / "shorts" / f"{shot}_shorts.json"


def empty_operator() -> dict:
    return {"moments": {}, "captions": {}, "templates": None, "candidates": {}}


def empty_sidecar(shot: str) -> dict:
    return {"version": SIDECAR_VERSION, "shot": shot, "moments": {}, "templates": {},
            "operator": empty_operator()}


def _validate_caption(cap: object, where: str) -> None:
    if not isinstance(cap, Mapping) or not isinstance(cap.get("text"), str) or not cap["text"]:
        raise ValueError(f"{where}: caption needs a non-empty 'text'")
    if cap.get("style", "title") not in CAPTION_STYLES:
        raise ValueError(f"{where}: unknown caption style {cap.get('style')!r}; valid: {CAPTION_STYLES}")
    for k in ("start", "end"):
        if k in cap and cap[k] is not None and not isinstance(cap[k], (int, float)):
            raise ValueError(f"{where}: caption '{k}' must be seconds (number)")


def validate_operator(op: object) -> dict:
    """Validate + normalise an ``operator`` block (unknown keys rejected)."""
    if op is None:
        return empty_operator()
    if not isinstance(op, Mapping):
        raise ValueError("operator must be a mapping")
    extra = set(op) - set(OPERATOR_KEYS)
    if extra:
        raise ValueError(f"operator: unknown key(s) {sorted(extra)}; valid: {OPERATOR_KEYS}")
    out = empty_operator()
    for name, frame in (op.get("moments") or {}).items():
        if name not in MOMENT_FRAME_KEYS:
            raise ValueError(f"operator.moments: unknown moment {name!r}; valid: {MOMENT_FRAME_KEYS}")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"operator.moments.{name}: frame pin must be an int, got {frame!r}")
        out["moments"][name] = frame
    for tpl, caps in (op.get("captions") or {}).items():
        if not isinstance(caps, list):
            raise ValueError(f"operator.captions.{tpl}: must be a list")
        for i, c in enumerate(caps):
            _validate_caption(c, f"operator.captions.{tpl}[{i}]")
        out["captions"][tpl] = copy.deepcopy(caps)
    tpls = op.get("templates")
    if tpls is not None:
        if not isinstance(tpls, list) or not all(isinstance(t, str) for t in tpls):
            raise ValueError("operator.templates must be a list of template names or null")
        out["templates"] = list(tpls)
    for tpl, pins in (op.get("candidates") or {}).items():
        for slot, idx in (pins or {}).items():
            if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                raise ValueError(f"operator.candidates.{tpl}.{slot}: must be a non-negative int")
        out["candidates"][tpl] = dict(pins or {})
    return out


def validate_sidecar(data: object) -> dict:
    """Validate a sidecar dict; returns it normalised (operator block cleaned)."""
    if not isinstance(data, Mapping):
        raise ValueError("shorts sidecar must be an object")
    if data.get("version") != SIDECAR_VERSION:
        raise ValueError(f"unsupported shorts sidecar version {data.get('version')!r}")
    if not isinstance(data.get("shot"), str) or not data["shot"]:
        raise ValueError("shorts sidecar needs a 'shot'")
    if not isinstance(data.get("moments", {}), Mapping) or not isinstance(data.get("templates", {}), Mapping):
        raise ValueError("'moments' and 'templates' must be objects")
    out = dict(copy.deepcopy(dict(data)))
    out.setdefault("moments", {})
    out.setdefault("templates", {})
    out["operator"] = validate_operator(data.get("operator"))
    for name, entry in out["templates"].items():
        for key in ("template", "ok", "slots", "passes", "edl"):
            if key not in entry:
                raise ValueError(f"templates.{name}: missing '{key}'")
    return out


def effective_moments(derived: Mapping, operator: Mapping | None) -> dict:
    """Derived moments with the operator's frame pins applied (pins win)."""
    out = copy.deepcopy(dict(derived))
    sources = dict(out.get("sources") or {})
    for name, frame in ((operator or {}).get("moments") or {}).items():
        out[name] = int(frame)
        sources[name] = "operator_pin"
    out["sources"] = sources
    return out


def template_entry(resolved: Mapping, outputs: Mapping | None = None) -> dict:
    """Sidecar entry for one resolved template (``shorts_templates.resolve_template``)."""
    framing = {s["pass"]: s["framing"] for s in resolved["slots"]
               if s.get("pass") and s.get("framing") is not None}
    return {
        "template": resolved["template"], "ok": bool(resolved["ok"]),
        "slots": copy.deepcopy(resolved["slots"]), "passes": copy.deepcopy(resolved["passes"]),
        "framing": framing, "edl": copy.deepcopy(resolved["edl"]),
        "captions": copy.deepcopy(resolved["edl"].get("captions", [])),
        "outputs": dict(outputs or {}),
    }


def merge_regenerated(existing: Mapping | None, generated: Mapping) -> dict:
    """New sidecar content = ``generated`` with the ``existing`` operator block
    carried over verbatim (a re-run never loses caption/template/moment pins)."""
    out = copy.deepcopy(dict(generated))
    out["operator"] = validate_operator((existing or {}).get("operator"))
    return out


def load_sidecar(path: Path) -> dict | None:
    """Load + validate; ``None`` when absent. A corrupt file raises ValueError
    (the operator block inside it must not be silently discarded)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    return validate_sidecar(raw)


def save_sidecar(path: Path, data: Mapping) -> Path:
    """Validate then atomically write."""
    clean = validate_sidecar(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(clean, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path

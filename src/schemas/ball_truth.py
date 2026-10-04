"""Ball Studio truth document: schema, validator, atomic persistence.

The truth file is OPERATOR data (``<output>/ball_truth/<group>_ball_truth.json``):
a hand-authored 3-D ball track on a group's shared reference timeline.
Contract: ``docs/superpowers/specs/2026-10-04-ball-studio-api.md``.

The document is plain JSON; this module validates it and fills defaults so
every consumer sees one normalised shape. Validation collects ALL problems
(``errors`` is a list of ``{"path", "message"}``) rather than failing on the
first one, so the UI can show them together.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VERSION = 1

KEY_SOURCES = (
    "triangulated", "ray_ground", "ray_height", "ray_plane",
    "ray_depth", "player", "manual",
)
SEGMENT_KINDS = ("flight", "roll", "carried", "linear", "static")
EVENT_KINDS = (
    "touch", "bounce", "post", "crossbar", "net", "line_cross", "out",
    "keeper_save",
)
OUTCOMES = ("goal", "no_goal", "unknown")
MAGNUS_MODES = ("auto", "off")
PLANE_AXES = ("x", "y", "z")

_ID_RE = re.compile(r"[A-Za-z0-9_-]+")

_TOP_FIELDS = {
    "version", "group_id", "reference_shot", "fps", "shots", "outcome",
    "keys", "segments", "observations", "events", "meta",
}
_KEY_FIELDS = {
    "id", "frame", "xyz", "source", "constraint", "observations",
    "residual_px", "note",
}
_CONSTRAINT_FIELDS = {
    "height_m", "plane", "depth_m", "player_id", "bone", "offset",
}
_SEGMENT_FIELDS = {"from", "to", "kind", "params"}
_PARAM_FIELDS = {"drag", "cd", "magnus", "player_id", "bone"}
_EVENT_FIELDS = {"frame", "kind", "player_id", "bone", "note"}
_OBS_FIELDS = {"shot_id", "shot_frame", "uv"}
_META_FIELDS = {"authored_by", "updated_at", "notes", "status"}
META_STATUSES = ("draft", "reviewed")


class BallTruthError(ValueError):
    """Raised when a truth document fails validation."""

    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{e['path']}: {e['message']}" for e in errors))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_truth(
    group_id: str,
    reference_shot: str,
    fps: float,
    shots: Iterable[tuple[str, int]],
) -> dict[str, Any]:
    """An empty, valid skeleton for a group with no authored truth yet."""
    return {
        "version": VERSION,
        "group_id": group_id,
        "reference_shot": reference_shot,
        "fps": float(fps),
        "shots": [
            {"shot_id": sid, "frame_offset": int(off)} for sid, off in shots
        ],
        "outcome": "unknown",
        "keys": [],
        "segments": [],
        "observations": [],
        "events": [],
        "meta": {
            "authored_by": "operator", "updated_at": None, "notes": "",
            "status": "draft",
        },
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return (
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and math.isfinite(float(v))
    )


def _vec(v: Any, n: int) -> list[float] | None:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    if not all(_is_num(x) for x in v):
        return None
    return [float(x) for x in v]


class _Errs:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, path: str, message: str) -> None:
        self.items.append({"path": path, "message": message})

    def unknown(self, path: str, obj: dict, allowed: set[str]) -> None:
        for k in obj:
            if k not in allowed:
                self.add(f"{path}.{k}" if path else k, "unknown field")


def _obs(raw: Any, path: str, shot_ids: set[str] | None, errs: _Errs) -> dict | None:
    if not isinstance(raw, dict):
        errs.add(path, "must be an object")
        return None
    errs.unknown(path, raw, _OBS_FIELDS)
    sid = raw.get("shot_id")
    if not isinstance(sid, str) or not sid:
        errs.add(f"{path}.shot_id", "required string")
        return None
    if shot_ids is not None and sid not in shot_ids:
        errs.add(f"{path}.shot_id", f"{sid!r} is not a member of the group")
    if not _is_int(raw.get("shot_frame")):
        errs.add(f"{path}.shot_frame", "must be an integer")
        return None
    uv = _vec(raw.get("uv"), 2)
    if uv is None:
        errs.add(f"{path}.uv", "must be two finite numbers")
        return None
    return {"shot_id": sid, "shot_frame": int(raw["shot_frame"]), "uv": uv}


def _constraint(raw: Any, path: str, errs: _Errs) -> dict:
    out = {
        "height_m": None, "plane": None, "depth_m": None,
        "player_id": None, "bone": None, "offset": None,
    }
    if raw is None:
        return out
    if not isinstance(raw, dict):
        errs.add(path, "must be an object or null")
        return out
    errs.unknown(path, raw, _CONSTRAINT_FIELDS)
    for name in ("height_m", "depth_m"):
        v = raw.get(name)
        if v is not None:
            if not _is_num(v):
                errs.add(f"{path}.{name}", "must be a finite number or null")
            else:
                out[name] = float(v)
    plane = raw.get("plane")
    if plane is not None:
        if (
            not isinstance(plane, dict)
            or plane.get("axis") not in PLANE_AXES
            or not _is_num(plane.get("value"))
        ):
            errs.add(f"{path}.plane", 'must be {"axis": "x"|"y"|"z", "value": number}')
        else:
            out["plane"] = {"axis": plane["axis"], "value": float(plane["value"])}
    for name in ("player_id", "bone"):
        v = raw.get(name)
        if v is not None:
            if not isinstance(v, str) or not v:
                errs.add(f"{path}.{name}", "must be a non-empty string or null")
            else:
                out[name] = v
    off = raw.get("offset")
    if off is not None:
        vec = _vec(off, 3)
        if vec is None:
            errs.add(f"{path}.offset", "must be three finite numbers or null")
        else:
            out["offset"] = vec
    return out


def _check_key_source(key: dict, path: str, errs: _Errs) -> None:
    src = key["source"]
    c = key["constraint"]
    n_obs = len(key["observations"])
    if src == "triangulated":
        shots = {o["shot_id"] for o in key["observations"]}
        if n_obs < 2 or len(shots) < 2:
            errs.add(f"{path}.observations", "triangulated key needs observations from >= 2 shots")
    elif src in ("ray_ground", "ray_height", "ray_plane", "ray_depth"):
        if n_obs < 1:
            errs.add(f"{path}.observations", f"{src} key needs an observation")
        if src == "ray_height" and c["height_m"] is None:
            errs.add(f"{path}.constraint.height_m", "ray_height needs height_m")
        if src == "ray_plane" and c["plane"] is None:
            errs.add(f"{path}.constraint.plane", "ray_plane needs plane")
        if src == "ray_depth" and c["depth_m"] is None:
            errs.add(f"{path}.constraint.depth_m", "ray_depth needs depth_m")
    elif src == "player":
        if not c["player_id"] or not c["bone"]:
            errs.add(f"{path}.constraint", "player key needs player_id and bone")


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

def validate_truth(
    doc: Any,
    *,
    group_id: str | None = None,
    member_shots: Iterable[str] | None = None,
    require_offset_consistency: bool = True,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate and normalise a truth document.

    Returns ``(normalised, errors)``; ``errors`` empty means valid. The
    normalised dict has every optional field filled with its default.
    """
    errs = _Errs()
    if not isinstance(doc, dict):
        return {}, [{"path": "", "message": "document must be an object"}]
    errs.unknown("", doc, _TOP_FIELDS)

    if doc.get("version") != VERSION:
        errs.add("version", f"must be {VERSION}")
    gid = doc.get("group_id")
    if not isinstance(gid, str) or not _ID_RE.fullmatch(gid or ""):
        errs.add("group_id", "required id string")
    elif group_id is not None and gid != group_id:
        errs.add("group_id", f"must equal the URL group {group_id!r}")
    ref = doc.get("reference_shot")
    if not isinstance(ref, str) or not ref:
        errs.add("reference_shot", "required string")
    fps = doc.get("fps")
    if not _is_num(fps) or float(fps) <= 0:
        errs.add("fps", "must be a positive number")

    members = set(member_shots) if member_shots is not None else None

    # shots
    shots: list[dict] = []
    offsets: dict[str, int] = {}
    raw_shots = doc.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        errs.add("shots", "must be a non-empty list")
        raw_shots = []
    for i, s in enumerate(raw_shots):
        p = f"shots[{i}]"
        if not isinstance(s, dict):
            errs.add(p, "must be an object")
            continue
        errs.unknown(p, s, {"shot_id", "frame_offset"})
        sid = s.get("shot_id")
        if not isinstance(sid, str) or not sid:
            errs.add(f"{p}.shot_id", "required string")
            continue
        if members is not None and sid not in members:
            errs.add(f"{p}.shot_id", f"{sid!r} is not a member of the group")
        if sid in offsets:
            errs.add(f"{p}.shot_id", f"duplicate shot {sid!r}")
        if not _is_int(s.get("frame_offset")):
            errs.add(f"{p}.frame_offset", "must be an integer")
            continue
        offsets[sid] = int(s["frame_offset"])
        shots.append({"shot_id": sid, "frame_offset": offsets[sid]})
    shot_ids = set(offsets) if offsets else None
    if isinstance(ref, str) and shot_ids is not None and ref not in shot_ids:
        errs.add("reference_shot", "must be one of shots[]")

    outcome = doc.get("outcome", "unknown")
    if outcome not in OUTCOMES:
        errs.add("outcome", f"must be one of {list(OUTCOMES)}")

    # keys
    keys: list[dict] = []
    key_frames: dict[int, str] = {}
    key_ids: dict[str, int] = {}
    raw_keys = doc.get("keys", [])
    if not isinstance(raw_keys, list):
        errs.add("keys", "must be a list")
        raw_keys = []
    for i, k in enumerate(raw_keys):
        p = f"keys[{i}]"
        if not isinstance(k, dict):
            errs.add(p, "must be an object")
            continue
        errs.unknown(p, k, _KEY_FIELDS)
        kid = k.get("id")
        if not isinstance(kid, str) or not _ID_RE.fullmatch(kid):
            errs.add(f"{p}.id", "required id string [A-Za-z0-9_-]+")
            continue
        if kid in key_ids:
            errs.add(f"{p}.id", f"duplicate key id {kid!r}")
        key_ids[kid] = i
        if not _is_int(k.get("frame")):
            errs.add(f"{p}.frame", "must be an integer reference frame")
            continue
        fr = int(k["frame"])
        if fr in key_frames:
            errs.add(f"{p}.frame", f"frame {fr} already used by key {key_frames[fr]!r}")
        key_frames[fr] = kid
        xyz = _vec(k.get("xyz"), 3)
        if xyz is None:
            errs.add(f"{p}.xyz", "must be three finite numbers")
            continue
        src = k.get("source")
        if src not in KEY_SOURCES:
            errs.add(f"{p}.source", f"must be one of {list(KEY_SOURCES)}")
            continue
        constraint = _constraint(k.get("constraint"), f"{p}.constraint", errs)
        obs: list[dict] = []
        raw_obs = k.get("observations", [])
        if not isinstance(raw_obs, list):
            errs.add(f"{p}.observations", "must be a list")
            raw_obs = []
        for j, o in enumerate(raw_obs):
            ob = _obs(o, f"{p}.observations[{j}]", shot_ids, errs)
            if ob is None:
                continue
            obs.append(ob)
            if (
                require_offset_consistency and ob["shot_id"] in offsets
                and ob["shot_frame"] - offsets[ob["shot_id"]] != fr
            ):
                errs.add(
                    f"{p}.observations[{j}]",
                    f"shot_frame {ob['shot_frame']} maps to reference frame "
                    f"{ob['shot_frame'] - offsets[ob['shot_id']]}, not the key frame {fr}",
                )
        resid = k.get("residual_px", {})
        resid_out: dict[str, float] = {}
        if resid is not None:
            if not isinstance(resid, dict) or not all(
                isinstance(a, str) and _is_num(b) for a, b in resid.items()
            ):
                errs.add(f"{p}.residual_px", "must map shot id to number")
            else:
                resid_out = {a: float(b) for a, b in resid.items()}
        note = k.get("note", "")
        if not isinstance(note, str):
            errs.add(f"{p}.note", "must be a string")
            note = ""
        key = {
            "id": kid, "frame": fr, "xyz": xyz, "source": src,
            "constraint": constraint, "observations": obs,
            "residual_px": resid_out, "note": note,
        }
        _check_key_source(key, p, errs)
        keys.append(key)

    # segments
    segments: list[dict] = []
    raw_segs = doc.get("segments", [])
    if not isinstance(raw_segs, list):
        errs.add("segments", "must be a list")
        raw_segs = []
    key_by_id = {k["id"]: k for k in keys}
    seen_from: set[str] = set()
    for i, s in enumerate(raw_segs):
        p = f"segments[{i}]"
        if not isinstance(s, dict):
            errs.add(p, "must be an object")
            continue
        errs.unknown(p, s, _SEGMENT_FIELDS)
        a, b = s.get("from"), s.get("to")
        if a not in key_by_id or b not in key_by_id:
            errs.add(p, "from/to must reference existing key ids")
            continue
        if key_by_id[a]["frame"] >= key_by_id[b]["frame"]:
            errs.add(p, "from.frame must be < to.frame")
            continue
        if s.get("kind") not in SEGMENT_KINDS:
            errs.add(f"{p}.kind", f"must be one of {list(SEGMENT_KINDS)}")
            continue
        if a in seen_from:
            errs.add(p, f"more than one segment starts at key {a!r}")
        seen_from.add(a)
        raw_params = s.get("params") or {}
        params = {
            "drag": True, "cd": None, "magnus": "auto",
            "player_id": None, "bone": None,
        }
        if not isinstance(raw_params, dict):
            errs.add(f"{p}.params", "must be an object")
        else:
            errs.unknown(f"{p}.params", raw_params, _PARAM_FIELDS)
            if "drag" in raw_params:
                if not isinstance(raw_params["drag"], bool):
                    errs.add(f"{p}.params.drag", "must be a boolean")
                else:
                    params["drag"] = raw_params["drag"]
            if raw_params.get("cd") is not None:
                if not _is_num(raw_params["cd"]) or float(raw_params["cd"]) < 0:
                    errs.add(f"{p}.params.cd", "must be a number >= 0 or null")
                else:
                    params["cd"] = float(raw_params["cd"])
            if "magnus" in raw_params:
                if raw_params["magnus"] not in MAGNUS_MODES:
                    errs.add(f"{p}.params.magnus", f"must be one of {list(MAGNUS_MODES)}")
                else:
                    params["magnus"] = raw_params["magnus"]
            for name in ("player_id", "bone"):
                v = raw_params.get(name)
                if v is not None:
                    if not isinstance(v, str) or not v:
                        errs.add(f"{p}.params.{name}", "must be a non-empty string or null")
                    else:
                        params[name] = v
        segments.append({"from": a, "to": b, "kind": s["kind"], "params": params})

    # soft observations
    observations: list[dict] = []
    raw_so = doc.get("observations", [])
    if not isinstance(raw_so, list):
        errs.add("observations", "must be a list")
        raw_so = []
    for i, o in enumerate(raw_so):
        ob = _obs(o, f"observations[{i}]", shot_ids, errs)
        if ob is not None:
            observations.append(ob)

    # events
    events: list[dict] = []
    raw_ev = doc.get("events", [])
    if not isinstance(raw_ev, list):
        errs.add("events", "must be a list")
        raw_ev = []
    for i, e in enumerate(raw_ev):
        p = f"events[{i}]"
        if not isinstance(e, dict):
            errs.add(p, "must be an object")
            continue
        errs.unknown(p, e, _EVENT_FIELDS)
        if not _is_int(e.get("frame")):
            errs.add(f"{p}.frame", "must be an integer reference frame")
            continue
        if e.get("kind") not in EVENT_KINDS:
            errs.add(f"{p}.kind", f"must be one of {list(EVENT_KINDS)}")
            continue
        ev = {
            "frame": int(e["frame"]), "kind": e["kind"],
            "player_id": None, "bone": None, "note": "",
        }
        for name in ("player_id", "bone"):
            v = e.get(name)
            if v is not None:
                if not isinstance(v, str) or not v:
                    errs.add(f"{p}.{name}", "must be a non-empty string or null")
                else:
                    ev[name] = v
        if e.get("note") is not None:
            if not isinstance(e["note"], str):
                errs.add(f"{p}.note", "must be a string")
            else:
                ev["note"] = e["note"]
        events.append(ev)

    # meta
    meta_in = doc.get("meta", {})
    meta = {"authored_by": "operator", "updated_at": None, "notes": "", "status": "draft"}
    if not isinstance(meta_in, dict):
        errs.add("meta", "must be an object")
    else:
        errs.unknown("meta", meta_in, _META_FIELDS)
        if "status" in meta_in:
            if meta_in["status"] not in META_STATUSES:
                errs.add("meta.status", f"must be one of {list(META_STATUSES)}")
            else:
                meta["status"] = meta_in["status"]
        for name in ("authored_by", "notes"):
            if name in meta_in:
                if not isinstance(meta_in[name], str):
                    errs.add(f"meta.{name}", "must be a string")
                else:
                    meta[name] = meta_in[name]
        ua = meta_in.get("updated_at")
        if ua is not None:
            if not isinstance(ua, str):
                errs.add("meta.updated_at", "must be a string or null")
            else:
                meta["updated_at"] = ua

    normalised = {
        "version": VERSION,
        "group_id": gid if isinstance(gid, str) else "",
        "reference_shot": ref if isinstance(ref, str) else "",
        "fps": float(fps) if _is_num(fps) else 0.0,
        "shots": shots,
        "outcome": outcome if outcome in OUTCOMES else "unknown",
        "keys": sorted(keys, key=lambda k: k["frame"]),
        "segments": segments,
        "observations": observations,
        "events": events,
        "meta": meta,
    }
    return normalised, errs.items


def require_valid(doc: Any, **kw: Any) -> dict[str, Any]:
    """``validate_truth`` that raises :class:`BallTruthError` on problems."""
    norm, errors = validate_truth(doc, **kw)
    if errors:
        raise BallTruthError(errors)
    return norm


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def truth_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "ball_truth"


def truth_path(output_dir: Path, group_id: str) -> Path:
    return truth_dir(output_dir) / f"{group_id}_ball_truth.json"


def dense_path(output_dir: Path, group_id: str) -> Path:
    return truth_dir(output_dir) / f"{group_id}_ball_truth_dense.json"


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_truth(output_dir: Path, group_id: str) -> dict[str, Any] | None:
    p = truth_path(output_dir, group_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_truth(output_dir: Path, group_id: str, doc: dict[str, Any]) -> str | None:
    """Atomically write ``doc``; keep the previous file in ``.history/``.

    Returns the history file path relative to ``ball_truth/`` (or None when
    there was no previous version).
    """
    p = truth_path(output_dir, group_id)
    history_rel: str | None = None
    if p.exists():
        hist_dir = truth_dir(output_dir) / ".history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        hist = hist_dir / f"{group_id}_ball_truth.{stamp}.json"
        hist.write_bytes(p.read_bytes())
        history_rel = f".history/{hist.name}"
    _atomic_write_json(p, doc)
    return history_rel


def save_dense(output_dir: Path, group_id: str, dense: dict[str, Any]) -> None:
    _atomic_write_json(dense_path(output_dir, group_id), dense)


def load_dense(output_dir: Path, group_id: str) -> dict[str, Any] | None:
    p = dense_path(output_dir, group_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())

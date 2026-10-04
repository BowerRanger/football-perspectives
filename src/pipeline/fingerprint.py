"""Per-stage freshness fingerprints (``<out>/pipeline_state.json``).

A stage's fingerprint is three independent hashes recorded when it last ran:

* ``config``  - the config slice the stage reads (declared in ``STAGE_TABLE``),
* ``inputs``  - per-file content hashes of the artefacts it consumes
  (upstream outputs *and* operator files such as ball anchors / players.json),
* ``code``    - hash of the stage module source.

Config or input drift makes a completed stage **stale** (re-run). Code-only
drift is reported separately (``code_drift``) because most source edits do
not change results; ``run --stale`` opts in to re-running those. Output dirs
with no recorded state are ``unknown`` - never stale - so legacy runs are not
invalidated wholesale.

The table is central and declarative on purpose: stage modules are untouched.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

STATE_FILENAME = "pipeline_state.json"
BIG_FILE_BYTES = 8 * 1024 * 1024
_EDGE_BYTES = 64 * 1024
_BINARY_SUFFIXES = {".npz", ".npy", ".mp4", ".mov", ".pt", ".pth", ".glb", ".fbx", ".exr", ".wav"}
_REPO_ROOT = Path(__file__).resolve().parents[2]

_MANIFEST = "shots/shots_manifest.json"
_SYNC = "shots/sync_map.json"
_PLAYERS = "players.json"
_CAM_TRACK = "camera/*_camera_track.json"
_REFINED = "refined_poses/*_refined.npz"
_BALL_TRACK = "ball/*_ball_track.json"
_APPEARANCE = "appearance/*.json"

# stage -> config top-level keys (``key`` or ``key.sub``), ignored sub-keys,
# input globs relative to the output dir, and upstream stages (cascade).
STAGE_TABLE: dict[str, dict] = {
    "prepare_shots": {
        "config": ["prepare_shots"], "inputs": [], "upstream": [],
        "module": "src/stages/prepare_shots.py",
    },
    "tracking": {
        "config": ["tracking"],
        "inputs": [_MANIFEST, "shots/*.mp4"],
        "upstream": ["prepare_shots"],
        "module": "src/stages/tracking.py",
    },
    "camera": {
        "config": ["camera", "pitch"],
        "inputs": [_MANIFEST, "camera/*_anchors*.json", "shots/*.mp4"],
        "upstream": ["prepare_shots"],
        "module": "src/stages/camera.py",
    },
    "replay_sync": {
        "config": ["replay_sync"],
        "inputs": [_MANIFEST, _SYNC, "tracks/*_tracks.json", _CAM_TRACK],
        "upstream": ["tracking", "camera"],
        "module": "src/stages/replay_sync.py",
    },
    "hmr_world": {
        "config": ["hmr_world", "pitch"],
        "inputs": [_MANIFEST, "tracks/*_tracks.json", _CAM_TRACK],
        "upstream": ["tracking", "camera", "replay_sync"],
        "module": "src/stages/hmr_world.py",
    },
    "refined_poses": {
        "config": ["refined_poses", "pitch"],
        "inputs": ["hmr_world/*_smpl_world.npz", "hmr_world/*_kp2d.json", _SYNC, _CAM_TRACK],
        "upstream": ["hmr_world"],
        "module": "src/stages/refined_poses.py",
    },
    "ball": {
        "config": ["ball", "pitch"],
        "ignore": ["ball.detection_cache"],
        "inputs": [_MANIFEST, _CAM_TRACK, _REFINED, "ball/*_ball_anchors.json", _SYNC, _PLAYERS],
        "upstream": ["camera", "refined_poses"],
        "module": "src/stages/ball.py",
    },
    "appearance": {
        "config": ["appearance"],
        "inputs": ["tracks/*_tracks.json", "hmr_world/*_kp2d.json", _REFINED, "shots/*.mp4"],
        "upstream": ["tracking", "hmr_world", "refined_poses"],
        "module": "src/stages/appearance.py",
    },
    "export": {
        "config": ["export", "pitch"],
        "inputs": [_REFINED, _BALL_TRACK, _CAM_TRACK, _PLAYERS, _APPEARANCE],
        "upstream": ["refined_poses", "ball", "camera", "appearance"],
        "module": "src/stages/export.py",
    },
    "render": {
        "config": ["render", "export.virtual_cameras", "pitch"],
        "inputs": [_REFINED, _BALL_TRACK, _CAM_TRACK, _PLAYERS, _APPEARANCE, "appearance/kits_operator.json"],
        "upstream": ["refined_poses", "ball", "camera", "appearance"],
        "module": "src/stages/render.py",
    },
    "shorts": {
        "config": ["shorts"],
        "inputs": [_BALL_TRACK, _PLAYERS, "render/*/*.mp4", "shots/*.mp4", "shorts/*_operator.json"],
        "upstream": ["render", "ball"],
        "module": "src/stages/shorts.py",
    },
}


@dataclass(frozen=True)
class Freshness:
    state: str  # "fresh" | "stale" | "unknown"
    reasons: tuple[str, ...] = ()
    code_drift: bool = False


# --------------------------------------------------------------------- hashing

def hash_file(path: Path) -> str:
    """Content hash. JSON/YAML -> canonical JSON sha (formatting-insensitive);
    big or binary files -> size + head/tail sha."""
    try:
        size = path.stat().st_size
        if size > BIG_FILE_BYTES or path.suffix.lower() in _BINARY_SUFFIXES:
            h = hashlib.sha256()
            with path.open("rb") as f:
                h.update(f.read(_EDGE_BYTES))
                if size > 2 * _EDGE_BYTES:
                    f.seek(-_EDGE_BYTES, 2)
                    h.update(f.read(_EDGE_BYTES))
            return f"b{size}:{h.hexdigest()[:16]}"
        raw = path.read_bytes()
        if path.suffix.lower() in (".json", ".yaml", ".yml"):
            try:
                if path.suffix.lower() == ".json":
                    obj = json.loads(raw)
                else:
                    import yaml
                    obj = yaml.safe_load(raw)
                raw = json.dumps(obj, sort_keys=True, default=str).encode()
            except Exception:  # noqa: BLE001 - unparsable -> raw bytes
                pass
        return hashlib.sha256(raw).hexdigest()[:16]
    except OSError:
        return "unreadable"


def _dig(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def config_slice(stage: str, cfg: dict) -> dict:
    spec = STAGE_TABLE[stage]
    sl = {k: _dig(cfg, k) for k in spec["config"]}
    for dotted in spec.get("ignore", []):
        top, _, rest = dotted.partition(".")
        if isinstance(sl.get(top), dict):
            sl[top] = {k: v for k, v in sl[top].items() if k != rest}
    return sl


def config_hash(stage: str, cfg: dict) -> str:
    blob = json.dumps(config_slice(stage, cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def code_hash(stage: str) -> str:
    path = _REPO_ROOT / STAGE_TABLE[stage]["module"]
    if not path.exists():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def input_hashes(stage: str, output_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for pattern in STAGE_TABLE[stage]["inputs"]:
        for p in sorted(output_dir.glob(pattern)):
            if p.is_file():
                found[p.relative_to(output_dir).as_posix()] = hash_file(p)
    return found


def compute(stage: str, output_dir: Path, cfg: dict) -> dict:
    return {
        "config": config_hash(stage, cfg),
        "inputs": input_hashes(stage, output_dir),
        "code": code_hash(stage),
    }


# ----------------------------------------------------------------------- state

def load_state(output_dir: Path) -> dict:
    path = output_dir / STATE_FILENAME
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"stages": {}}
    if not isinstance(data, dict) or not isinstance(data.get("stages"), dict):
        return {"stages": {}}
    return data


def record(output_dir: Path, stage: str, cfg: dict) -> None:
    """Persist the stage's current fingerprint (call right after it ran)."""
    state = load_state(output_dir)
    state["stages"][stage] = {
        **compute(stage, output_dir, cfg),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = output_dir / (STATE_FILENAME + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(output_dir / STATE_FILENAME)


def _diff_inputs(old: dict[str, str], new: dict[str, str]) -> list[str]:
    reasons = [f"input changed: {k}" for k in sorted(new) if k in old and old[k] != new[k]]
    reasons += [f"input added: {k}" for k in sorted(set(new) - set(old))]
    reasons += [f"input removed: {k}" for k in sorted(set(old) - set(new))]
    return reasons


def assess(output_dir: Path, stage: str, cfg: dict) -> Freshness:
    rec = load_state(output_dir)["stages"].get(stage)
    if not isinstance(rec, dict):
        return Freshness("unknown")
    cur = compute(stage, output_dir, cfg)
    reasons: list[str] = []
    if rec.get("config") != cur["config"]:
        reasons.append("config changed")
    reasons += _diff_inputs(rec.get("inputs") or {}, cur["inputs"])
    drift = rec.get("code") != cur["code"]
    return Freshness("stale" if reasons else "fresh", tuple(reasons), drift)


def downstream_of(stage: str) -> list[str]:
    """Transitive dependents of ``stage`` per the table's upstream edges."""
    out: set[str] = set()
    frontier = {stage}
    while frontier:
        nxt = {s for s, spec in STAGE_TABLE.items()
               if s not in out and frontier & set(spec["upstream"])}
        out |= nxt
        frontier = nxt
    return [s for s in STAGE_TABLE if s in out]

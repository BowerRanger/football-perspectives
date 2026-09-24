"""Shared JSON-serializable data types for the ball hybrid-extraction PoC.

Frozen dataclasses for every type named in ``CONTRACT.md``'s "Shared data
types" section (Observation, TruthTrack/TruthFrame/TruthEvent, SynthRun,
Track/TrackFrame), each with explicit ``to_json``/``from_json`` (dict, not
``dataclasses.asdict``, so tuple<->list and Optional fields round-trip
predictably) plus ``save_json``/``load_json`` file helpers.

``Results`` (the viewer's consumed shape) is intentionally a plain dict —
it is method/scenario-keyed and grows across the PoC's phases, so a rigid
dataclass would just be re-litigated every phase. ``validate_results``
checks structure only (missing keys / wrong container types), not values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


def _f2(v: Any) -> Vec2:
    return (float(v[0]), float(v[1]))


def _f3(v: Any) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


@dataclass(frozen=True)
class Observation:
    """One evidence sample: ``{frame, uv, conf, source}`` per CONTRACT.md."""

    frame: int
    uv: Vec2
    conf: float
    source: str

    def to_json(self) -> dict:
        return {
            "frame": self.frame,
            "uv": list(self.uv),
            "conf": self.conf,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Observation":
        return cls(
            frame=int(d["frame"]),
            uv=_f2(d["uv"]),
            conf=float(d["conf"]),
            source=str(d["source"]),
        )


@dataclass(frozen=True)
class TruthFrame:
    """One dense synthetic-truth sample: ``state`` is one of
    ``"ground" | "air" | "contact"``."""

    frame: int
    xyz: Vec3
    state: str

    def to_json(self) -> dict:
        return {"frame": self.frame, "xyz": list(self.xyz), "state": self.state}

    @classmethod
    def from_json(cls, d: dict) -> "TruthFrame":
        return cls(frame=int(d["frame"]), xyz=_f3(d["xyz"]), state=str(d["state"]))


@dataclass(frozen=True)
class TruthEvent:
    """One sparse truth event: ``kind`` is one of
    ``"touch" | "bounce" | "net" | "post" | "rest"``."""

    frame: int
    kind: str
    xyz: Vec3
    player_id: Optional[str] = None
    bone: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "frame": self.frame,
            "kind": self.kind,
            "xyz": list(self.xyz),
            "player_id": self.player_id,
            "bone": self.bone,
        }

    @classmethod
    def from_json(cls, d: dict) -> "TruthEvent":
        return cls(
            frame=int(d["frame"]),
            kind=str(d["kind"]),
            xyz=_f3(d["xyz"]),
            player_id=(str(d["player_id"]) if d.get("player_id") else None),
            bone=(str(d["bone"]) if d.get("bone") else None),
        )


@dataclass(frozen=True)
class TruthTrack:
    """``truth_<scenario>.json`` — synthetic 3-D ground truth seeded ONLY
    from operator data (manual anchors) + reconstructed players; never
    from ball-stage output."""

    clip_id: str
    scenario: str
    fps: float
    frames: tuple[TruthFrame, ...]
    events: tuple[TruthEvent, ...] = ()
    seed_anchor_frames: tuple[int, ...] = ()
    physics: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "scenario": self.scenario,
            "fps": self.fps,
            "frames": [f.to_json() for f in self.frames],
            "events": [e.to_json() for e in self.events],
            "seed_anchor_frames": list(self.seed_anchor_frames),
            "physics": dict(self.physics),
        }

    @classmethod
    def from_json(cls, d: dict) -> "TruthTrack":
        return cls(
            clip_id=str(d["clip_id"]),
            scenario=str(d["scenario"]),
            fps=float(d["fps"]),
            frames=tuple(TruthFrame.from_json(f) for f in d.get("frames", [])),
            events=tuple(TruthEvent.from_json(e) for e in d.get("events", [])),
            seed_anchor_frames=tuple(
                int(x) for x in d.get("seed_anchor_frames", [])),
            physics=dict(d.get("physics", {})),
        )


@dataclass(frozen=True)
class SynthRun:
    """``synth_obs_<scenario>.json`` — a synthetic detector stream derived
    from a ``TruthTrack``. ``anchors`` are loose BallAnchor-like dicts
    (same frames/states as the real manual anchors, ``image_xy`` = truth
    projection + noise) — kept as dicts rather than importing
    ``src.schemas.ball_anchor.BallAnchor`` so this module has no
    dependency on the anchor schema's validation rules."""

    clip_id: str
    scenario: str
    observations: tuple[Observation, ...]
    anchors: tuple[dict, ...] = ()
    noise_model: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "scenario": self.scenario,
            "observations": [o.to_json() for o in self.observations],
            "anchors": [dict(a) for a in self.anchors],
            "noise_model": dict(self.noise_model),
        }

    @classmethod
    def from_json(cls, d: dict) -> "SynthRun":
        return cls(
            clip_id=str(d["clip_id"]),
            scenario=str(d["scenario"]),
            observations=tuple(
                Observation.from_json(o) for o in d.get("observations", [])),
            anchors=tuple(dict(a) for a in d.get("anchors", [])),
            noise_model=dict(d.get("noise_model", {})),
        )


@dataclass(frozen=True)
class TrackFrame:
    """One frame of any method's dense output. ``mode`` is one of
    ``"faithful" | "simulated" | "anchor" | ...`` (method-defined);
    ``xyz`` is ``None`` where the method has no estimate for this frame."""

    frame: int
    xyz: Optional[Vec3]
    mode: str
    conf: Optional[float] = None

    def to_json(self) -> dict:
        return {
            "frame": self.frame,
            "xyz": (list(self.xyz) if self.xyz is not None else None),
            "mode": self.mode,
            "conf": self.conf,
        }

    @classmethod
    def from_json(cls, d: dict) -> "TrackFrame":
        xyz = d.get("xyz")
        conf = d.get("conf")
        return cls(
            frame=int(d["frame"]),
            xyz=(_f3(xyz) if xyz is not None else None),
            mode=str(d["mode"]),
            conf=(float(conf) if conf is not None else None),
        )


@dataclass(frozen=True)
class Track:
    """``track_<method>_<scenario>.json`` — any method's dense per-frame
    output (``method`` e.g. ``"current" | "hybrid"``)."""

    clip_id: str
    method: str
    frames: tuple[TrackFrame, ...]

    def to_json(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "method": self.method,
            "frames": [f.to_json() for f in self.frames],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Track":
        return cls(
            clip_id=str(d["clip_id"]),
            method=str(d["method"]),
            frames=tuple(TrackFrame.from_json(f) for f in d.get("frames", [])),
        )


def save_json(path: Path, obj: Any) -> None:
    """Write ``obj`` (any of the above dataclasses, or a plain dict/list —
    e.g. a ``Results`` payload) as JSON to ``path``, creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = obj.to_json() if hasattr(obj, "to_json") else obj
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def load_json(path: Path) -> Any:
    """Read raw JSON (a dict/list) from ``path``. Callers reconstruct a
    typed dataclass with e.g. ``TruthTrack.from_json(load_json(path))``."""
    with Path(path).open() as fh:
        return json.load(fh)


def validate_results(d: dict) -> list[str]:
    """Loose structural validation of a ``Results``-shaped dict (see
    CONTRACT.md). Returns a list of problems; empty means it looks fine.
    Does not validate metric values, only presence/shape of containers."""
    problems: list[str] = []
    if not isinstance(d, dict):
        return ["results is not a dict"]

    for key in ("clip_id", "fps", "image_size", "scenarios"):
        if key not in d:
            problems.append(f"missing top-level key: {key!r}")

    scenarios = d.get("scenarios")
    if scenarios is not None and not isinstance(scenarios, dict):
        problems.append("'scenarios' must be a dict")
    elif isinstance(scenarios, dict):
        for name, sc in scenarios.items():
            if not isinstance(sc, dict):
                problems.append(f"scenario {name!r} is not a dict")
                continue
            for key in ("truth", "tracks", "metrics"):
                if key not in sc:
                    problems.append(f"scenario {name!r} missing {key!r}")
            tracks = sc.get("tracks")
            if tracks is not None and not isinstance(tracks, dict):
                problems.append(f"scenario {name!r} 'tracks' must be a dict")
            elif isinstance(tracks, dict):
                for method, tr in tracks.items():
                    if not isinstance(tr, dict) or "frames" not in tr:
                        problems.append(
                            f"scenario {name!r} track {method!r} "
                            "missing 'frames'")

    real = d.get("real")
    if real is not None:
        if not isinstance(real, dict):
            problems.append("'real' must be a dict")
        else:
            for key in ("tracks", "metrics"):
                if key not in real:
                    problems.append(f"'real' missing {key!r}")

    camera = d.get("camera")
    if camera is not None and "centre_xyz_per_frame" not in camera:
        problems.append("'camera' missing 'centre_xyz_per_frame'")

    return problems

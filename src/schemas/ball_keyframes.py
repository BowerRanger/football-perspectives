"""Sparse ball keyframe schema — one entry per manual ball anchor,
carrying its resolved 3D position, the clicked camera ray (airborne
anchors only), and semantic metadata. Engine-facing sidecar written by
``BallStage`` alongside the dense ``ball_track.json``; the UE side keys a
transform track only at these frames and tweens between them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

# Mirror of BallAnchorState — the states a keyframe may carry.
KeyframeState = Literal[
    "grounded", "airborne_low", "airborne_mid", "airborne_high",
    "kick", "catch", "bounce", "header", "volley", "chest",
    "player_touch", "goal_impact", "off_screen_flight",
]

_VALID_STATES: frozenset[str] = frozenset(get_args(KeyframeState))

DepthSource = Literal["ground", "ray_physics", "player_bone", "goal_geometry"]

_VALID_DEPTH_SOURCES: frozenset[str] = frozenset(get_args(DepthSource))

# (ray_origin_xyz, ray_dir_xyz); dir is a unit vector in world frame.
Ray = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class BallKeyframe:
    """One sparse keyframe corresponding to a single manual anchor.

    ``world_xyz`` is the artist's default key position (pitch metres);
    ``None`` only for ``off_screen_flight`` (no pixel, no resolved 3D).
    ``image_xy`` is the authoritative clicked pixel (``None`` only for
    ``off_screen_flight``). ``ray`` is the clicked camera ray
    ``((ox,oy,oz),(dx,dy,dz))`` populated for ``airborne_*`` states so the
    engine can re-snap a moved key onto the line of sight. The remaining
    optional fields carry the anchor's semantic tags.
    """

    frame: int
    state: KeyframeState
    depth_source: DepthSource
    world_xyz: tuple[float, float, float] | None = None
    image_xy: tuple[float, float] | None = None
    ray: Ray | None = None
    player_id: str | None = None
    bone: str | None = None
    goal_element: str | None = None
    touch_type: str | None = None
    spin: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class BallKeyframeSet:
    clip_id: str
    fps: float
    image_size: tuple[int, int]
    keyframes: tuple[BallKeyframe, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(
                asdict(self),
                fh,
                indent=2,
                default=lambda v: list(v) if isinstance(v, tuple) else v,
            )

    @classmethod
    def load(cls, path: Path) -> "BallKeyframeSet":
        with path.open() as fh:
            data = json.load(fh)
        keyframes = tuple(_load_keyframe(k) for k in data.get("keyframes", []))
        return cls(
            clip_id=str(data["clip_id"]),
            fps=float(data["fps"]),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            keyframes=keyframes,
        )


def _as_tuple3(v: Sequence[float]) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _load_ray(v) -> Ray | None:
    if v is None:
        return None
    return (_as_tuple3(v[0]), _as_tuple3(v[1]))


def _load_keyframe(k: dict) -> BallKeyframe:
    state = str(k["state"])
    if state not in _VALID_STATES:
        raise ValueError(f"unknown ball keyframe state: {state!r}")
    depth_source = str(k["depth_source"])
    if depth_source not in _VALID_DEPTH_SOURCES:
        raise ValueError(f"unknown depth_source: {depth_source!r}")
    if state == "player_touch":
        if not k.get("player_id"):
            raise ValueError("player_id is required for state 'player_touch'")
        if not k.get("bone"):
            raise ValueError("bone is required for state 'player_touch'")
    goal_element = k.get("goal_element")
    if state == "goal_impact":
        if not goal_element:
            raise ValueError("goal_element is required for state 'goal_impact'")
        # Lazy import mirrors ball_anchor.py BallAnchorSet.load to avoid
        # any potential import-cycle risk.
        from src.utils.ball_anchor_heights import VALID_GOAL_ELEMENTS  # noqa: PLC0415
        if goal_element not in VALID_GOAL_ELEMENTS:
            raise ValueError(
                f"unknown goal_element {goal_element!r}; "
                f"valid: {sorted(VALID_GOAL_ELEMENTS)}"
            )
    world = k.get("world_xyz")
    xy = k.get("image_xy")
    if state == "off_screen_flight":
        if world is not None or xy is not None:
            raise ValueError(
                "off_screen_flight must have null world_xyz and image_xy"
            )
    elif xy is None:
        raise ValueError(f"image_xy is required for state {state!r}")
    return BallKeyframe(
        frame=int(k["frame"]),
        state=state,  # type: ignore[arg-type]
        depth_source=depth_source,  # type: ignore[arg-type]
        world_xyz=_as_tuple3(world) if world is not None else None,
        image_xy=(float(xy[0]), float(xy[1])) if xy is not None else None,
        ray=_load_ray(k.get("ray")),
        player_id=k.get("player_id"),
        bone=k.get("bone"),
        goal_element=goal_element,
        touch_type=k.get("touch_type"),
        spin=k.get("spin"),
        confidence=float(k.get("confidence", 1.0)),
    )

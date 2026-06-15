"""EventResolver — the default ball "back half".

Resolves each merged ball anchor/event to a 3-D ``BallKeyframe`` WITHOUT a
dense trajectory solve, derives the interpolation segments, and renders a
derived dense ``BallTrack`` via the pure reference interpolator. The key
difference from the trajectory solver is touch depth: a ``player_touch`` is
**pinned to the contacting body joint** (depth-stable, occlusion-robust)
rather than having its monocular depth solved.

Kept torch-free and video-free: ``player_ctx`` is duck-typed (anything with
``joint_world(frame, player_id, bone) -> (x, y, z) | None``), so this is
unit-testable with a fake context and a plain pinhole camera.

See ``docs/superpowers/specs/2026-06-15-ball-touch-events-design.md`` §5-§11.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.schemas.ball_keyframes import BallKeyframeSet
from src.schemas.ball_track import FlightSegment
from src.utils.ball_interpolate import interpolate_events
from src.utils.ball_keyframe_builder import build_ball_keyframe_set
from src.utils.ball_possession import detect_carry_spans
from src.utils.ball_segments import derive_segments
from src.utils.camera_projection import project_point_onto_pixel_ray
from src.utils.foot_anchor import ankle_ray_to_pitch


class _PlayerCtx(Protocol):
    def joint_world(
        self, frame: int, player_id: str, bone: str,
    ) -> tuple[float, float, float] | None: ...


@dataclass
class EventResolveResult:
    """Mirrors the trajectory solver's result surface so the ball stage's
    existing emit path can consume it unchanged."""

    world_by_frame: dict[int, tuple[tuple[float, float, float], float]]
    state_by_frame: dict[int, str]
    flight_segments: tuple[FlightSegment, ...]
    diagnostics: dict
    keyframe_set: BallKeyframeSet = field(default=None)  # type: ignore[assignment]


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -np.asarray(R).T @ np.asarray(t, dtype=float)


def _offset_toward_camera(
    point: np.ndarray, R: np.ndarray, t: np.ndarray, ball_radius: float,
) -> np.ndarray:
    """Nudge a body-joint position by the ball radius toward the camera so
    the ball surface (not the joint centre) sits on the line of sight."""
    C = _camera_center(R, t)
    d = C - point
    n = float(np.linalg.norm(d))
    if n <= 1e-9:
        return point
    return point + (d / n) * ball_radius


def _resolve_touch_world(
    anc: BallAnchor,
    fi: int,
    player_ctx: _PlayerCtx,
    K: np.ndarray | None,
    R: np.ndarray | None,
    t: np.ndarray | None,
    distortion: tuple[float, float],
    ball_radius: float,
) -> np.ndarray | None:
    """Body-pinned touch resolution (spec §7)."""
    if not (anc.player_id and anc.bone):
        return None
    bone_world = player_ctx.joint_world(fi, anc.player_id, anc.bone)
    if bone_world is None:
        return None
    base = np.asarray(bone_world, dtype=float)
    have_cam = K is not None and R is not None and t is not None
    # Lateral refine onto the ball pixel ray when a confident pixel exists;
    # otherwise (occlusion) use the joint position directly.
    if anc.image_xy is not None and have_cam:
        uv = (float(anc.image_xy[0]), float(anc.image_xy[1]))
        base = project_point_onto_pixel_ray(base, uv, K, R, t, distortion)
    if have_cam:
        base = _offset_toward_camera(base, R, t, ball_radius)
    return base


def _resolve_waypoint_world(
    anc: BallAnchor,
    fi: int,
    K: np.ndarray | None,
    R: np.ndarray | None,
    t: np.ndarray | None,
    distortion: tuple[float, float],
    ball_radius: float,
    goal_geometry,
) -> np.ndarray | None:
    """Non-touch resolution: ball-pixel ray ∩ ground/goal geometry."""
    if anc.image_xy is None or K is None or R is None or t is None:
        return None
    uv = (float(anc.image_xy[0]), float(anc.image_xy[1]))
    if anc.state == "goal_impact" and anc.goal_element and goal_geometry is not None:
        from src.utils.goal_geometry import resolve_goal_impact_world
        try:
            return np.asarray(resolve_goal_impact_world(
                uv, anc.goal_element, K=K, R=R, t=t,
                distortion=distortion, geometry=goal_geometry,
            ), dtype=float)
        except Exception:  # noqa: BLE001 — fall through to ray-cast
            pass
    from src.utils.ball_anchor_heights import state_to_height
    try:
        plane_z = state_to_height(anc.state)
    except ValueError:
        plane_z = ball_radius
    try:
        return np.asarray(ankle_ray_to_pitch(
            uv, K=K, R=R, t=t, plane_z=plane_z, distortion=distortion,
        ), dtype=float)
    except Exception:  # noqa: BLE001 — unresolvable anchor
        return None


def resolve_events(
    *,
    anchor_by_frame: dict[int, BallAnchor],
    player_ctx: _PlayerCtx,
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ball_radius: float,
    goal_geometry,
    n_frames: int,
    fps: float,
    clip_id: str,
    image_size: tuple[int, int],
) -> EventResolveResult:
    """Resolve events → keyframes + segments → derived dense track."""
    world_for_anchor: dict[int, tuple[float, float, float] | None] = {}
    for fi in sorted(anchor_by_frame):
        anc = anchor_by_frame[fi]
        K = per_frame_K.get(fi)
        R = per_frame_R.get(fi)
        t = per_frame_t.get(fi)
        if anc.state == "off_screen_flight":
            world = None
        elif anc.state == "player_touch":
            world = _resolve_touch_world(
                anc, fi, player_ctx, K, R, t, distortion, ball_radius,
            )
            if world is None:
                world = _resolve_waypoint_world(
                    anc, fi, K, R, t, distortion, ball_radius, goal_geometry,
                )
        else:
            world = _resolve_waypoint_world(
                anc, fi, K, R, t, distortion, ball_radius, goal_geometry,
            )
        world_for_anchor[fi] = (
            None if world is None
            else (float(world[0]), float(world[1]), float(world[2]))
        )

    # Force player_bone depth_source for every touch (body-pinned, even on
    # ground touches) by passing an empty ground_touch_frames set.
    keyframe_set = build_ball_keyframe_set(
        clip_id=clip_id,
        fps=fps,
        image_size=image_size,
        anchor_by_frame=anchor_by_frame,
        world_by_frame=world_for_anchor,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=distortion,
        ground_touch_frames=set(),
        flight_segments=(),
    )
    carry_spans = detect_carry_spans(keyframe_set.keyframes)
    segments = derive_segments(
        keyframe_set.keyframes, n_frames=n_frames, fps=fps,
        carry_spans=carry_spans,
    )
    keyframe_set = dataclasses.replace(keyframe_set, segments=segments)

    track = interpolate_events(
        keyframe_set, n_frames=n_frames, ball_radius_m=ball_radius,
    )
    world_by_frame = {
        f.frame: (f.world_xyz, f.confidence)
        for f in track.frames if f.world_xyz is not None
    }
    state_by_frame = {f.frame: f.state for f in track.frames}

    diagnostics = {
        "underconstrained_spans": [],
        "segments": [
            {"start": s.start_frame, "end": s.end_frame, "kind": s.kind}
            for s in segments
        ],
        "bounces": [],
        "splits": 0,
    }
    return EventResolveResult(
        world_by_frame=world_by_frame,
        state_by_frame=state_by_frame,
        flight_segments=track.flight_segments,
        diagnostics=diagnostics,
        keyframe_set=keyframe_set,
    )


__all__ = ["resolve_events", "EventResolveResult"]

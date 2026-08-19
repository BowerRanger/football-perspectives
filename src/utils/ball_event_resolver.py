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


# A ground-clamped touch may snap to the on-ray ground point only when that
# point is genuinely within reach of the contacting joint.
_CLAMP_CONTACT_MAX_M = 0.6


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
) -> tuple[np.ndarray | None, bool]:
    """Body-pinned touch resolution (spec §7).

    Returns ``(world | None, ground_clamped)``. The ball centre can never
    sit below z = ball_radius: FK foot joints dip under the pitch and would
    otherwise carry that error straight into the touch keyframe (sub-20cm
    campaign W2a). With a pixel the clamp stays on the clicked ray
    (ray ∩ z = ball_radius); without one it lifts vertically.
    """
    if not (anc.player_id and anc.bone):
        return None, False
    bone_world = player_ctx.joint_world(fi, anc.player_id, anc.bone)
    if bone_world is None:
        return None, False
    base = np.asarray(bone_world, dtype=float)
    have_cam = K is not None and R is not None and t is not None
    # With a confident ball pixel: refine laterally onto its ray, then push
    # the ball centre a radius toward the camera so the surface (not the
    # joint centre) sits on the sight-line. Occluded (no pixel): resolve
    # straight to the joint position — there is no ray to define the offset
    # direction meaningfully (spec §7, occlusion robustness).
    if anc.image_xy is not None and have_cam:
        uv = (float(anc.image_xy[0]), float(anc.image_xy[1]))
        base = project_point_onto_pixel_ray(base, uv, K, R, t, distortion)
        base = _offset_toward_camera(base, R, t, ball_radius)
    if base[2] >= ball_radius:
        return base, False
    # Clamp with the MINIMUM move. The on-ray ground point (ray ∩ z=r) is
    # ideal — pixel-faithful AND physical — but on a shallow broadcast ray
    # it can sit metres away in depth from the actual foot (small height
    # errors amplify by 1/sin(elevation)). Only take it when it stays in
    # contact range of the joint; otherwise lift vertically at the joint.
    clamped: np.ndarray | None = None
    if anc.image_xy is not None and have_cam:
        try:
            ground_pt = np.asarray(ankle_ray_to_pitch(
                (float(anc.image_xy[0]), float(anc.image_xy[1])),
                K=K, R=R, t=t, plane_z=ball_radius, distortion=distortion,
            ), dtype=float)
            joint = np.asarray(bone_world, dtype=float)
            if float(np.linalg.norm(ground_pt - joint)) <= _CLAMP_CONTACT_MAX_M:
                clamped = ground_pt
        except Exception:  # noqa: BLE001 — grazing ray: fall through
            clamped = None
    if clamped is None:
        clamped = np.array([base[0], base[1], ball_radius])
    return clamped, True


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
    if anc.state == "grounded" and anc.landmark:
        from src.utils.ball_landmark_fix import resolve_landmark_world
        world = resolve_landmark_world(
            anc.image_xy, anc.landmark, K=K, R=R, t=t,
            distortion=distortion, ball_radius=ball_radius,
        )
        if world is not None:
            return world
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


_HARD_EVIDENCE_SOURCES = frozenset({"detector", "second_pass", "foot_guided"})
_EVIDENCE_MIN_CONF = 0.3


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
    steps=None,
    confidences: dict[int, float] | None = None,
    sources: dict[int, str] | None = None,
    manual_frames: frozenset[int] | None = None,
) -> EventResolveResult:
    """Resolve events → keyframes + segments → derived dense track.

    ``steps``/``confidences``/``sources`` (W4) carry the shared core's
    observation stream; frames whose observation came from a real detector
    pass become rendering evidence: roll spans follow their ground points
    and airborne chain fits consume their pixels as extra ray constraints.
    """
    hard_obs: dict[int, tuple[float, float]] = {}
    if steps is not None and sources:
        conf = confidences or {}
        for s in steps:
            if s.uv is None:
                continue
            f = s.frame
            if (sources.get(f) in _HARD_EVIDENCE_SOURCES
                    and conf.get(f, 0.0) >= _EVIDENCE_MIN_CONF):
                hard_obs[f] = (float(s.uv[0]), float(s.uv[1]))
    world_for_anchor: dict[int, tuple[float, float, float] | None] = {}
    n_ground_clamped = 0
    for fi in sorted(anchor_by_frame):
        anc = anchor_by_frame[fi]
        K = per_frame_K.get(fi)
        R = per_frame_R.get(fi)
        t = per_frame_t.get(fi)
        if anc.state == "off_screen_flight":
            world = None
        elif anc.state == "player_touch":
            world, was_clamped = _resolve_touch_world(
                anc, fi, player_ctx, K, R, t, distortion, ball_radius,
            )
            n_ground_clamped += int(was_clamped)
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

    # W3 — airborne chains: replace bucket-height placeholders with a
    # gravity-arc fit through the bracketing hard knots (pixels as rays;
    # W4: real in-span detections densify the fit).
    from src.utils.ball_flight_chains import refit_airborne_chains
    chain_updates, chain_diags = refit_airborne_chains(
        anchor_by_frame=anchor_by_frame,
        world_for_anchor=world_for_anchor,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R,
        per_frame_t=per_frame_t, distortion=distortion, fps=fps,
        extra_observations=hard_obs or None,
        manual_frames=manual_frames,
    )
    # Refit keyframes stay ray-faithful: snap each onto its clicked ray at
    # the fitted depth BEFORE rendering, so interpolation and the C4 stage
    # pass agree (no kinks between an anchor frame and its neighbours).
    for fi, w in list(chain_updates.items()):
        anc = anchor_by_frame.get(fi)
        K = per_frame_K.get(fi)
        R = per_frame_R.get(fi)
        t = per_frame_t.get(fi)
        if (anc is not None and anc.image_xy is not None
                and K is not None and R is not None and t is not None):
            snapped = project_point_onto_pixel_ray(
                np.asarray(w, dtype=float),
                (float(anc.image_xy[0]), float(anc.image_xy[1])),
                K, R, t, distortion,
            )
            chain_updates[fi] = (float(snapped[0]), float(snapped[1]),
                                 float(snapped[2]))
    world_for_anchor.update(chain_updates)

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
    # Segment kinds are state-based here; a roll span the IMM confidently
    # calls flight is only FLIPPED to ballistic below when a detection-
    # fitted arc actually accepts (classification follows fit success —
    # flipping without a firing fit measurably hurt: kroupi01 dense
    # 0.57→0.91 m, both alone and with fits that never fired).
    p_flight_by_frame = (
        {s.frame: float(getattr(s, "p_flight", 0.0)) for s in steps}
        if steps is not None else {}
    )
    segments = derive_segments(
        keyframe_set.keyframes, n_frames=n_frames, fps=fps,
        carry_spans=carry_spans,
    )
    # W5l — segment-level flight refits: a ballistic span with no interior
    # anchors (deep crosses) fits its arc from in-span detections; manual
    # endpoints stay near-hard, auto body-pins soft. The fitted arc rides
    # in the segment hints so the renderer (and UE) draw the same shape.
    if hard_obs:
        from src.utils.ball_flight_chains import refit_ballistic_segment
        kf_world2 = {kf.frame: kf.world_xyz for kf in keyframe_set.keyframes}

        def _flight_candidate(seg) -> bool:
            if seg.kind == "ballistic":
                return True
            if seg.kind != "roll":
                return False
            interior = [p_flight_by_frame[f]
                        for f in range(seg.start_frame + 1, seg.end_frame)
                        if f in p_flight_by_frame]
            return (bool(interior)
                    and sum(interior) / len(interior) > 0.8)

        new_segments = []
        segment_fit_diags: list[dict] = []
        for seg in segments:
            wa = kf_world2.get(seg.start_frame)
            wb = kf_world2.get(seg.end_frame)
            n_obs_in = sum(1 for f in hard_obs
                           if seg.start_frame < f < seg.end_frame)
            if (not _flight_candidate(seg) or wa is None or wb is None
                    or seg.end_frame - seg.start_frame < 6):
                if seg.kind == "ballistic":
                    segment_fit_diags.append({
                        "span": [seg.start_frame, seg.end_frame],
                        "kind": seg.kind, "n_obs": n_obs_in,
                        "attempted": False})
                new_segments.append(seg)
                continue
            conf_of = {
                f: float(getattr(anchor_by_frame.get(f), "confidence", 1.0)
                         or 1.0)
                for f in (seg.start_frame, seg.end_frame)
            }
            is_manual = {
                f: (manual_frames is None or f in manual_frames)
                for f in (seg.start_frame, seg.end_frame)
            }
            fit = refit_ballistic_segment(
                start_frame=seg.start_frame, end_frame=seg.end_frame,
                start_world=wa, end_world=wb,
                start_is_manual=is_manual[seg.start_frame],
                end_is_manual=is_manual[seg.end_frame],
                start_confidence=conf_of[seg.start_frame],
                end_confidence=conf_of[seg.end_frame],
                extra_observations=hard_obs,
                per_frame_K=per_frame_K, per_frame_R=per_frame_R,
                per_frame_t=per_frame_t, distortion=distortion, fps=fps,
            )
            split = None
            if fit is None and n_obs_in >= 6:
                from src.utils.ball_flight_chains import refit_split_segment
                split = refit_split_segment(
                    start_frame=seg.start_frame, end_frame=seg.end_frame,
                    start_world=wa, end_world=wb,
                    start_is_manual=is_manual[seg.start_frame],
                    end_is_manual=is_manual[seg.end_frame],
                    start_confidence=conf_of[seg.start_frame],
                    end_confidence=conf_of[seg.end_frame],
                    extra_observations=hard_obs,
                    per_frame_K=per_frame_K, per_frame_R=per_frame_R,
                    per_frame_t=per_frame_t, distortion=distortion,
                    fps=fps,
                )
            segment_fit_diags.append({
                "span": [seg.start_frame, seg.end_frame],
                "kind": seg.kind, "n_obs": n_obs_in,
                "attempted": True, "accepted": fit is not None,
                "split": None if split is None else split[0]})
            if split is not None:
                s_frame, (pa, va), (pb, vb) = split
                base_h = {k: v for k, v in (seg.hints or {}).items()}
                new_segments.append(dataclasses.replace(
                    seg, kind="ballistic", end_frame=s_frame,
                    hints={**base_h, "gravity": -9.81, "split_fit": True,
                           "fit_p0": list(pa), "fit_v0": list(va)}))
                new_segments.append(dataclasses.replace(
                    seg, kind="ballistic", start_frame=s_frame,
                    hints={**base_h, "gravity": -9.81, "split_fit": True,
                           "fit_p0": list(pb), "fit_v0": list(vb)}))
                continue
            if fit is not None:
                new_segments.append(dataclasses.replace(
                    seg, kind="ballistic",
                    hints={**(seg.hints or {}),
                           "gravity": -9.81,
                           "fit_p0": list(fit[0]),
                           "fit_v0": list(fit[1])},
                ))
            else:
                # No accepted fit: an unflipped roll stays a roll.
                new_segments.append(seg)
        segments = tuple(new_segments)

    keyframe_set = dataclasses.replace(keyframe_set, segments=segments)

    # W4b — carry spans follow the owning player's foot path (the §10
    # design target): ball = foot midpoint + endpoint-blended offset, so a
    # dribbler's curved run no longer renders as a straight chord.
    carry_worlds: dict[int, tuple[float, float, float]] = {}
    kf_world = {kf.frame: kf.world_xyz for kf in keyframe_set.keyframes}

    def _foot_mid(f: int, pid: str) -> np.ndarray | None:
        feet = [player_ctx.joint_world(f, pid, b) for b in ("l_foot", "r_foot")]
        feet = [np.asarray(x, dtype=float) for x in feet if x is not None]
        if not feet:
            return None
        return np.mean(np.stack(feet), axis=0)

    for seg in segments:
        if seg.kind != "carry":
            continue
        pid = (seg.hints or {}).get("player_id")
        pa = kf_world.get(seg.start_frame)
        pb = kf_world.get(seg.end_frame)
        if not pid or pa is None or pb is None:
            continue
        fa = _foot_mid(seg.start_frame, pid)
        fb = _foot_mid(seg.end_frame, pid)
        if fa is None or fb is None:
            continue
        off_a = np.asarray(pa, dtype=float) - fa
        off_b = np.asarray(pb, dtype=float) - fb
        span = seg.end_frame - seg.start_frame
        for f in range(seg.start_frame + 1, seg.end_frame):
            ff = _foot_mid(f, pid)
            if ff is None:
                continue
            s = (f - seg.start_frame) / span
            w = ff + (1.0 - s) * off_a + s * off_b
            carry_worlds[f] = (float(w[0]), float(w[1]),
                               float(max(w[2], ball_radius)))

    # W4 — rendering evidence: real detections' ground points steer roll
    # spans (the interpolator ignores them on flight spans, whose shape the
    # chain fits already own).
    evidence_worlds: dict[int, tuple[float, float, float]] = {}
    for fi, uv in hard_obs.items():
        if fi in anchor_by_frame:
            continue
        K = per_frame_K.get(fi)
        R = per_frame_R.get(fi)
        t = per_frame_t.get(fi)
        if K is None or R is None or t is None:
            continue
        try:
            w = ankle_ray_to_pitch(uv, K=K, R=R, t=t, plane_z=ball_radius,
                                   distortion=distortion)
            evidence_worlds[fi] = (float(w[0]), float(w[1]), float(w[2]))
        except Exception:  # noqa: BLE001 — grazing ray: no evidence point
            continue

    track = interpolate_events(
        keyframe_set, n_frames=n_frames, ball_radius_m=ball_radius,
        evidence_worlds=evidence_worlds or None,
        carry_worlds=carry_worlds or None,
    )
    world_by_frame = {
        f.frame: (f.world_xyz, f.confidence)
        for f in track.frames if f.world_xyz is not None
    }
    state_by_frame = {f.frame: f.state for f in track.frames}

    diagnostics = {
        "underconstrained_spans": [
            {
                "start": min(d["air_frames"]),
                "end": max(d["air_frames"]),
                "residual_px": None,
                "note": d.get("note", ""),
            }
            for d in chain_diags
            if d.get("kind") == "underconstrained_chain" and d["air_frames"]
        ],
        "segments": [
            {"start": s.start_frame, "end": s.end_frame, "kind": s.kind}
            for s in segments
        ],
        "bounces": [],
        "splits": 0,
        "touch_ground_clamped": n_ground_clamped,
        "segment_fits": (segment_fit_diags
                         if hard_obs else []),
        "roll_evidence": [
            {"span": [sg.start_frame, sg.end_frame],
             "n_evidence": sum(1 for f in evidence_worlds
                               if sg.start_frame < f < sg.end_frame)}
            for sg in segments if sg.kind == "roll"
        ],
        "flight_chains": chain_diags,
        "airborne_refit_frames": sorted(chain_updates),
    }
    return EventResolveResult(
        world_by_frame=world_by_frame,
        state_by_frame=state_by_frame,
        flight_segments=track.flight_segments,
        diagnostics=diagnostics,
        keyframe_set=keyframe_set,
    )


__all__ = ["resolve_events", "EventResolveResult"]

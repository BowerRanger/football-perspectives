"""Automatic ball event detection from pixel kinematics + player poses.

Events are the frames where the ball's velocity changes hands: player
touches (including keeper saves), ground bounces, goal-frame impacts,
and stationary spans (free kicks, kick-offs). They are detected purely
from already-computed signals — the IMM-smoothed pixel track, the
players' FK contact joints (``PlayerContext``), and the goal geometry —
so detection is cheap and runs after the per-frame detection loop.

Detected events feed two consumers:
  * auto-anchor generation (``ball_auto_anchor``) — events become
    BallAnchor records that bracket trajectory segments, giving the
    monocular solver the depth information it otherwise lacks;
  * the piecewise solver's split-and-retry — a failed flight fit is
    split at the strongest interior event and refit.

Classification precedence at a velocity break, most-specific first:
  1. post/crossbar hit (tight line residual — strong geometric evidence)
  2. player touch (joint within pixel radius)
  3. net hit (ray crosses a net plane — weak geometric evidence on its
     own, so it additionally requires a speed collapse)
  4. bounce (vertical pixel-velocity flip, nobody nearby)
  5. generic velocity_break (solver split hint; never becomes an anchor)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from src.utils.goal_geometry import GoalGeometry, goal_element_candidates

logger = logging.getLogger(__name__)

_LINE_ELEMENTS = ("post", "crossbar")


class _SupportsJointLookup(Protocol):
    def joints_near_pixel(
        self, frame: int, uv: tuple[float, float], radius_px: float
    ) -> list: ...


class _SupportsUv(Protocol):
    frame: int
    uv: tuple[float, float] | None


@dataclass(frozen=True)
class AutoEventCfg:
    """Thresholds for event detection. Pixel units are px/frame."""

    touch_max_px: float = 25.0
    min_direction_change_deg: float = 25.0
    min_speed_change_px: float = 4.0
    # Direction change is only meaningful when the ball is actually
    # moving on both sides of the candidate frame.
    min_break_speed_px: float = 1.0
    event_window_frames: int = 3
    merge_window_frames: int = 4
    bounce_min_vy_px: float = 0.5
    stationary_max_speed_px: float = 0.7
    stationary_min_frames: int = 8
    stationary_min_conf: float = 0.4
    # Goal-impact gates: line hits need the ray within tolerance of the
    # post/crossbar; net hits need the outbound speed to collapse to
    # this fraction of the inbound speed.
    goal_line_tolerance_m: float = 0.35
    goal_net_speed_drop_ratio: float = 0.55
    goal_min_direction_change_deg: float = 45.0


@dataclass(frozen=True)
class BallEvent:
    """One detected event. ``end_frame`` is set for stationary spans."""

    frame: int
    kind: str  # touch | bounce | goal_impact | stationary | velocity_break
    score: float
    player_id: str | None = None
    bone: str | None = None
    goal_element: str | None = None
    end_frame: int | None = None


@dataclass(frozen=True)
class _Break:
    frame: int
    strength: float
    dir_change_deg: float
    dspeed_px: float
    speed_before: float
    speed_after: float
    vy_before: float
    vy_after: float


def _window_velocity(
    uvs: dict[int, np.ndarray], f: int, w: int, sign: int
) -> np.ndarray | None:
    """Mean pixel velocity (px/frame, forward-in-time) over the window
    before (sign=-1) or after (sign=+1) frame ``f``. Prefers the longest
    available baseline; needs at least 2 frames of separation."""
    base = uvs.get(f)
    if base is None:
        return None
    for off in range(w, 1, -1):
        other = uvs.get(f + sign * off)
        if other is not None:
            return (other - base) * (sign / off)
    return None


def _find_breaks(
    uvs: dict[int, np.ndarray], cfg: AutoEventCfg
) -> list[_Break]:
    """Velocity-break candidates with non-max suppression by strength."""
    candidates: list[_Break] = []
    for f in sorted(uvs):
        v_b = _window_velocity(uvs, f, cfg.event_window_frames, -1)
        v_a = _window_velocity(uvs, f, cfg.event_window_frames, +1)
        if v_b is None or v_a is None:
            continue
        sb = float(np.linalg.norm(v_b))
        sa = float(np.linalg.norm(v_a))
        dspeed = abs(sa - sb)
        if min(sb, sa) >= cfg.min_break_speed_px:
            cosang = float(np.dot(v_b, v_a)) / max(sb * sa, 1e-9)
            dir_change = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        else:
            dir_change = 0.0
        if (
            dir_change < cfg.min_direction_change_deg
            and dspeed < cfg.min_speed_change_px
        ):
            continue
        strength = min(1.0, 0.5 * (dir_change / 90.0)
                       + 0.5 * (dspeed / (3.0 * cfg.min_speed_change_px)))
        candidates.append(_Break(
            frame=f, strength=strength, dir_change_deg=dir_change,
            dspeed_px=dspeed, speed_before=sb, speed_after=sa,
            vy_before=float(v_b[1]), vy_after=float(v_a[1]),
        ))
    # Non-max suppression: strongest first, suppress within merge window.
    kept: list[_Break] = []
    for cand in sorted(candidates, key=lambda b: -b.strength):
        if all(abs(cand.frame - k.frame) > cfg.merge_window_frames for k in kept):
            kept.append(cand)
    kept.sort(key=lambda b: b.frame)
    return kept


def _classify_goal_line(
    brk: _Break,
    uv: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    geometry: GoalGeometry,
    cfg: AutoEventCfg,
) -> BallEvent | None:
    """Post/crossbar hit: tight line residual + a hard velocity break."""
    if (
        brk.dir_change_deg < cfg.goal_min_direction_change_deg
        and brk.dspeed_px < cfg.min_speed_change_px
    ):
        return None
    hits = goal_element_candidates(
        (float(uv[0]), float(uv[1])),
        K=K, R=R, t=t, distortion=distortion, geometry=geometry,
    )
    line_hits = [
        h for h in hits
        if h[0] in _LINE_ELEMENTS and h[1] <= cfg.goal_line_tolerance_m
    ]
    if not line_hits:
        return None
    element, residual, _, _ = line_hits[0]
    score = brk.strength * (1.0 - 0.5 * residual / cfg.goal_line_tolerance_m)
    return BallEvent(
        frame=brk.frame, kind="goal_impact", score=float(score),
        goal_element=element,
    )


def _classify_touch(
    brk: _Break,
    uv: np.ndarray,
    player_ctx: _SupportsJointLookup,
    cfg: AutoEventCfg,
) -> BallEvent | None:
    """Nearest contact joint within the touch radius around the break.

    Probes the break frame and its immediate neighbours — pose sampling
    and break localisation can disagree by a frame.
    """
    best: tuple[float, object] | None = None
    for probe in (0, -1, 1):
        for s in player_ctx.joints_near_pixel(
            brk.frame + probe, (float(uv[0]), float(uv[1])), cfg.touch_max_px
        ):
            d = float(np.hypot(s.uv[0] - uv[0], s.uv[1] - uv[1]))
            if best is None or d < best[0]:
                best = (d, s)
    if best is None:
        return None
    d, sample = best
    score = 0.5 * (1.0 - d / cfg.touch_max_px) + 0.5 * brk.strength
    return BallEvent(
        frame=brk.frame, kind="touch", score=float(score),
        player_id=sample.player_id, bone=sample.bone,
    )


def _classify_net(
    brk: _Break,
    uv: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    geometry: GoalGeometry,
    cfg: AutoEventCfg,
) -> BallEvent | None:
    """Net hit: ray crosses a net plane AND the ball's speed collapses."""
    if brk.speed_before < 2.0 * cfg.min_break_speed_px:
        return None
    if brk.speed_after > cfg.goal_net_speed_drop_ratio * brk.speed_before:
        return None
    hits = goal_element_candidates(
        (float(uv[0]), float(uv[1])),
        K=K, R=R, t=t, distortion=distortion, geometry=geometry,
    )
    net_hits = [h for h in hits if h[0] in ("back_net", "side_net")]
    if not net_hits:
        return None
    element = net_hits[0][0]
    return BallEvent(
        frame=brk.frame, kind="goal_impact", score=float(brk.strength),
        goal_element=element,
    )


def _classify_bounce(brk: _Break, cfg: AutoEventCfg) -> BallEvent | None:
    """Vertical pixel-velocity flip (down -> up in image coordinates)."""
    if brk.vy_before > cfg.bounce_min_vy_px and brk.vy_after < -cfg.bounce_min_vy_px:
        return BallEvent(
            frame=brk.frame, kind="bounce", score=float(brk.strength),
        )
    return None


def _stationary_spans(
    uvs: dict[int, np.ndarray],
    confidences: dict[int, float],
    cfg: AutoEventCfg,
) -> list[BallEvent]:
    events: list[BallEvent] = []
    run_start: int | None = None
    prev_frame: int | None = None
    confs: list[float] = []

    def _flush(end_frame: int) -> None:
        nonlocal run_start, confs
        if run_start is not None:
            length = end_frame - run_start + 1
            mean_conf = float(np.mean(confs)) if confs else 0.0
            if (
                length >= cfg.stationary_min_frames
                and mean_conf >= cfg.stationary_min_conf
            ):
                events.append(BallEvent(
                    frame=run_start, kind="stationary",
                    score=min(1.0, mean_conf), end_frame=end_frame,
                ))
        run_start = None
        confs = []

    for f in sorted(uvs):
        nxt = uvs.get(f + 1)
        contiguous = prev_frame is not None and f == prev_frame + 1
        if nxt is None:
            if run_start is not None:
                confs.append(float(confidences.get(f, 0.5)))
                _flush(f)
            prev_frame = f
            continue
        speed = float(np.linalg.norm(nxt - uvs[f]))
        if speed <= cfg.stationary_max_speed_px:
            if run_start is None or not contiguous:
                _flush(f - 1)
                run_start = f
                confs = []
            confs.append(float(confidences.get(f, 0.5)))
        else:
            _flush(f)
        prev_frame = f
    if run_start is not None:
        _flush(prev_frame if prev_frame is not None else run_start)
    return events


def detect_events(
    *,
    steps: Sequence[_SupportsUv],
    confidences: dict[int, float],
    player_ctx: _SupportsJointLookup,
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float] = (0.0, 0.0),
    goal_geometry: GoalGeometry | None = None,
    cfg: AutoEventCfg | None = None,
) -> tuple[BallEvent, ...]:
    """Detect ball events for one shot. See module docstring."""
    cfg = cfg or AutoEventCfg()
    uvs: dict[int, np.ndarray] = {
        s.frame: np.asarray(s.uv, dtype=float)
        for s in steps if s.uv is not None
    }
    events: list[BallEvent] = []
    for brk in _find_breaks(uvs, cfg):
        uv = uvs[brk.frame]
        K = per_frame_K.get(brk.frame)
        R = per_frame_R.get(brk.frame)
        t = per_frame_t.get(brk.frame)
        has_cam = K is not None and R is not None and t is not None
        event: BallEvent | None = None
        if goal_geometry is not None and has_cam:
            event = _classify_goal_line(
                brk, uv, K, R, t, distortion, goal_geometry, cfg
            )
        if event is None:
            event = _classify_touch(brk, uv, player_ctx, cfg)
        if event is None and goal_geometry is not None and has_cam:
            event = _classify_net(
                brk, uv, K, R, t, distortion, goal_geometry, cfg
            )
        if event is None:
            event = _classify_bounce(brk, cfg)
        if event is None:
            event = BallEvent(
                frame=brk.frame, kind="velocity_break",
                score=float(0.5 * brk.strength),
            )
        events.append(event)
    events.extend(_stationary_spans(uvs, confidences, cfg))
    events.sort(key=lambda e: (e.frame, e.kind))
    if events:
        logger.info(
            "ball auto-events: %d detected (%s)",
            len(events),
            ", ".join(f"{e.kind}@{e.frame}" for e in events),
        )
    return tuple(events)

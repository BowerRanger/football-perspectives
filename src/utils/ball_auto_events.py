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

Public API
----------
``detect_events`` — production greedy-NMS path (unchanged).
``detect_event_candidates`` — soft-NMS / top-K-per-window path that
    returns every plausible candidate (used by the global mode-search
    beam to populate breakpoint candidates).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

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
    # Phase B: derive direction-change breaks from a global robust piecewise
    # fit (ball_traj_segment) instead of fragile local velocity windows.
    use_segmentation: bool = True
    segment_max_residual_px: float = 6.0
    # Phase C: pose-anchored touch attribution. Relax the joint-proximity gate
    # and rank candidate joints by a kinematic bonus (a fast-moving foot near a
    # direction change is the likely toucher). High recall — prune in editor.
    use_pose_touch: bool = True
    touch_relaxed_px: float = 60.0
    kinematic_bonus_weight: float = 0.3


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


def _raw_break_candidates(
    uvs: dict[int, np.ndarray], cfg: AutoEventCfg
) -> list[_Break]:
    """Build all velocity-break candidates WITHOUT any NMS/merge step.

    Applies the direction-change and speed-change gates from *cfg* but
    performs no suppression — every frame that clears the thresholds is
    returned, ordered by frame number.  Both ``detect_events`` and
    ``detect_event_candidates`` call this; they differ only in how they
    thin the resulting list.
    """
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
    candidates.sort(key=lambda b: b.frame)
    return candidates


def _find_breaks(
    uvs: dict[int, np.ndarray], cfg: AutoEventCfg
) -> list[_Break]:
    """Velocity-break candidates with non-max suppression by strength.

    Calls ``_raw_break_candidates`` then applies the original greedy NMS
    (strongest-first, suppress within merge_window_frames).  Output is
    identical to the pre-refactor implementation.
    """
    candidates = _raw_break_candidates(uvs, cfg)
    return _greedy_nms(candidates, cfg)


def _greedy_nms(candidates: list[_Break], cfg: AutoEventCfg) -> list[_Break]:
    """Strongest-first suppression within ``merge_window_frames``."""
    kept: list[_Break] = []
    for cand in sorted(candidates, key=lambda b: -b.strength):
        if all(abs(cand.frame - k.frame) > cfg.merge_window_frames for k in kept):
            kept.append(cand)
    kept.sort(key=lambda b: b.frame)
    return kept


def _select_breaks(uvs: dict[int, np.ndarray], cfg: AutoEventCfg) -> list[_Break]:
    """Direction-change breaks: robust global segmentation when
    ``cfg.use_segmentation`` (Phase B), else the local velocity-break path.
    Both are greedy-NMS'd identically."""
    if getattr(cfg, "use_segmentation", False):
        from src.utils.ball_traj_segment import segment_track  # lazy: avoid cycle
        raw = segment_track(
            uvs, cfg=cfg, max_residual_px=cfg.segment_max_residual_px
        )
        return _greedy_nms(raw, cfg)
    return _find_breaks(uvs, cfg)


def _dispatch_touch(brk, uv, player_ctx, cfg):
    """Pose-anchored attribution (Phase C) when enabled, else the legacy
    nearest-joint-within-25px classifier."""
    if getattr(cfg, "use_pose_touch", False):
        from src.utils.ball_pose_touch import (  # lazy import: avoid cycle
            classify_touch as _pose_classify_touch,
        )
        return _pose_classify_touch(brk, uv, player_ctx, cfg)
    return _classify_touch(brk, uv, player_ctx, cfg)


def _top_k_per_window(
    candidates: list[_Break], merge_window_frames: int, k: int = 2
) -> list[_Break]:
    """Soft-NMS: within each merge window, keep up to *k* candidates
    (ranked by strength) rather than suppressing all but the strongest.

    Windows are formed by a greedy pass identical to the hard-NMS, but
    instead of keeping exactly one winner per window we keep the top *k*
    by strength.  Frames that fall outside any existing window are always
    kept.  The returned list is sorted by frame number.
    """
    if not candidates:
        return []
    # Sort descending by strength so the first candidate in each window
    # is the strongest (same order as the greedy NMS pass).
    by_strength = sorted(candidates, key=lambda b: -b.strength)
    # windows: list of (centre_frame, [kept breaks in this window])
    windows: list[tuple[int, list[_Break]]] = []

    for cand in by_strength:
        # Find the first existing window this candidate falls into.
        assigned = False
        for centre, members in windows:
            if abs(cand.frame - centre) <= merge_window_frames:
                if len(members) < k:
                    members.append(cand)
                assigned = True
                break
        if not assigned:
            # Open a new window centred on this candidate's frame.
            windows.append((cand.frame, [cand]))

    kept = [b for _, members in windows for b in members]
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


def _permissive_cfg(cfg: AutoEventCfg) -> AutoEventCfg:
    """Return a copy of *cfg* with detection thresholds lowered for the
    permissive profile.  Only the fields that gate whether a candidate is
    *generated at all* are relaxed; scoring weights are unchanged so the
    returned scores remain comparable with the default profile.
    """
    return AutoEventCfg(
        touch_max_px=cfg.touch_max_px,
        # Lower speed and direction gates so subtler breaks surface.
        min_direction_change_deg=cfg.min_direction_change_deg * 0.5,
        min_speed_change_px=cfg.min_speed_change_px * 0.5,
        min_break_speed_px=cfg.min_break_speed_px * 0.5,
        event_window_frames=cfg.event_window_frames,
        merge_window_frames=cfg.merge_window_frames,
        bounce_min_vy_px=cfg.bounce_min_vy_px * 0.5,
        stationary_max_speed_px=cfg.stationary_max_speed_px,
        stationary_min_frames=cfg.stationary_min_frames,
        stationary_min_conf=cfg.stationary_min_conf,
        goal_line_tolerance_m=cfg.goal_line_tolerance_m,
        goal_net_speed_drop_ratio=cfg.goal_net_speed_drop_ratio,
        goal_min_direction_change_deg=cfg.goal_min_direction_change_deg * 0.5,
    )


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
    for brk in _select_breaks(uvs, cfg):
        uv = uvs.get(brk.frame)
        if uv is None:
            # Segmentation can place a corner on a frame with no raw obs
            # (between two sparse points); skip cam/touch lookups that need
            # the pixel and emit a bare velocity_break.
            events.append(BallEvent(
                frame=brk.frame, kind="velocity_break",
                score=float(0.5 * brk.strength),
            ))
            continue
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
            event = _dispatch_touch(brk, uv, player_ctx, cfg)
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


def _classify_breaks_to_events(
    breaks: list[_Break],
    uvs: dict[int, np.ndarray],
    player_ctx: _SupportsJointLookup,
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    goal_geometry: GoalGeometry | None,
    cfg: AutoEventCfg,
) -> list[BallEvent]:
    """Classify a list of ``_Break`` objects into ``BallEvent`` records.

    Used by ``detect_event_candidates`` after soft-NMS thinning.
    ``detect_events`` uses an equivalent inline loop (kept separate to
    preserve byte-identical output).
    """
    events: list[BallEvent] = []
    for brk in breaks:
        uv = uvs.get(brk.frame)
        if uv is None:
            continue
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
            event = _dispatch_touch(brk, uv, player_ctx, cfg)
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
    return events


def detect_event_candidates(
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
    profile: Literal["default", "permissive"] = "default",
) -> tuple[BallEvent, ...]:
    """Return ALL plausible ball-event candidates for the global mode-search.

    Unlike ``detect_events`` (which applies greedy NMS so that at most one
    event survives each ``merge_window_frames``-wide window), this function
    uses a **soft-NMS / top-K-per-window** policy: up to
    ``_TOP_K_PER_WINDOW`` candidates per window are kept so that two events
    that are only a few frames apart can both survive.

    Parameters
    ----------
    profile:
        ``"default"`` — use *cfg* thresholds unchanged (same gates as
        ``detect_events``, but without the greedy merge step).
        ``"permissive"`` — lower ``min_speed_change_px``,
        ``min_direction_change_deg``, and related gates by 50 % so
        subtler velocity breaks surface as low-score candidates.

    Notes
    -----
    * Velocity-break events at synthetic clip boundaries (frame 0 and the
      last frame in ``steps``) are never emitted — ``_window_velocity``
      requires a valid window on both sides (F17). A stationary span that
      begins at frame 0 is still reported (its ``frame`` is the span start,
      a legitimate breakpoint, not a synthetic event).
    * All returned ``score`` values are in [0, 1].
    * The returned tuple is sorted by ``(frame, kind)``.
    * Stationary spans are always returned (not subject to merge
      suppression) because they cover ranges, not single frames.
    """
    _TOP_K_PER_WINDOW = 2

    base_cfg = cfg or AutoEventCfg()
    effective_cfg = _permissive_cfg(base_cfg) if profile == "permissive" else base_cfg

    uvs: dict[int, np.ndarray] = {
        s.frame: np.asarray(s.uv, dtype=float)
        for s in steps if s.uv is not None
    }

    # Soft-NMS: keep top-K breaks per merge window instead of top-1.
    # Note: boundary frames (0 / last) are already excluded naturally because
    # _raw_break_candidates requires valid velocity windows on both sides of
    # the candidate frame (v_b and v_a both None-checked).
    raw_breaks = _raw_break_candidates(uvs, effective_cfg)
    thinned_breaks = _top_k_per_window(
        raw_breaks, effective_cfg.merge_window_frames, k=_TOP_K_PER_WINDOW
    )

    events: list[BallEvent] = _classify_breaks_to_events(
        thinned_breaks,
        uvs=uvs,
        player_ctx=player_ctx,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=distortion,
        goal_geometry=goal_geometry,
        cfg=effective_cfg,
    )

    # Stationary spans: not subject to merge suppression.
    events.extend(_stationary_spans(uvs, confidences, effective_cfg))

    events.sort(key=lambda e: (e.frame, e.kind))
    if events:
        logger.debug(
            "ball event candidates (%s): %d (%s)",
            profile,
            len(events),
            ", ".join(f"{e.kind}@{e.frame}" for e in events),
        )
    return tuple(events)

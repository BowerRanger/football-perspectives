"""Automatic ball-anchor generation (camera auto-anchor analogy).

Turns detected events (``ball_auto_events``) plus confidently-grounded
detection spans into ``BallAnchor`` records — the same schema the manual
anchor editor writes — validated against simple physical gates before
they are allowed to constrain the trajectory solver:

  * contact gap: a ``player_touch`` is only trusted when the named joint
    is within ``contact_max_gap_m`` of the ball's camera ray;
  * on-pitch: a candidate whose ground projection lands outside the
    pitch (+margin) is detector noise;
  * reachability: consecutive candidates must not imply impossible
    speeds; the lower-scored offender is dropped.

Auto anchors are persisted to ``{shot}_ball_anchors_auto.json`` next to
the manual ``{shot}_ball_anchors.json``. At solve time
:func:`merge_anchors` combines the two with manual anchors always
winning, and any auto anchor within ``suppress_radius_frames`` of a
manual one dropped — the operator has looked at that moment.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.schemas.ball_anchor import BallAnchor, DismissedAuto
from src.utils.ball_auto_events import BallEvent
from src.utils.camera_projection import point_to_pixel_ray_distance
from src.utils.foot_anchor import ankle_ray_to_pitch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoAnchorCfg:
    enabled: bool = True
    min_event_score: float = 0.25
    # Grounded keyframe sampling (camera keyframe-interval analogy).
    grounded_interval: int = 25
    grounded_min_conf: float = 0.55
    grounded_max_p_flight: float = 0.3
    # A touch is only trusted when the joint sits this close to the
    # ball's camera ray (HMR depth drift otherwise poisons the knot).
    contact_max_gap_m: float = 0.6
    # Outbound pixel speed (px/frame) above which a touch is a shot.
    shot_speed_px: float = 12.0
    # Validation gates. Ground-level candidate pairs use the tighter
    # rolling cap: a lob the IMM missed projects to ground positions
    # whose implied roll speed is impossible, and that is often the only
    # signal that the sample is airborne.
    max_speed_m_s: float = 45.0
    max_ground_speed_m_s: float = 35.0
    pitch_margin_m: float = 3.0
    # Auto anchors this close to a manual anchor defer to the operator.
    suppress_radius_frames: int = 3
    # No grounded sampling for this long after a touch/bounce/impact:
    # the ball is likely airborne and the IMM posterior lags the launch,
    # so early post-event samples are the classic bogus ground anchor.
    post_event_suppress_frames: int = 8
    ball_radius_m: float = 0.11
    # Evidence gate (sub-20cm campaign W2b). Touch/bounce/goal events found
    # on a synthetic pixel track (anchor-interp / bridge / gap-fill only)
    # must not become body-pinned keyframes: they add no information over
    # interpolation and a mis-attributed joint drags the track metres off.
    # Requires >= 1 frame whose observation came from a real detector pass
    # within the window. Grounded sampling additionally accepts `bridge`
    # (on-image template evidence) but never source-less synthetic frames.
    # Both apply only when a `sources` map is provided.
    require_event_evidence: bool = True
    event_evidence_window: int = 3
    event_evidence_sources: tuple[str, ...] = (
        "detector", "second_pass", "foot_guided",
    )
    grounded_evidence_sources: tuple[str, ...] = (
        "detector", "second_pass", "foot_guided", "bridge", "anchor",
    )


def auto_anchor_path(ball_dir: Path, shot_id: str) -> Path:
    """Sidecar path for a shot's auto anchors (``ball_dir`` is the
    directory the manual ``{shot}_ball_anchors.json`` lives in)."""
    name = (
        f"{shot_id}_ball_anchors_auto.json"
        if shot_id else "ball_anchors_auto.json"
    )
    return Path(ball_dir) / name


@dataclass(frozen=True)
class _Candidate:
    anchor: BallAnchor
    score: float


def _uv_at(steps_by_frame: Mapping[int, tuple[float, float]], frame: int):
    uv = steps_by_frame.get(frame)
    return (float(uv[0]), float(uv[1])) if uv is not None else None


def _outbound_speed_px(
    steps_by_frame: Mapping[int, tuple[float, float]],
    frame: int,
    window: int = 3,
) -> float:
    base = steps_by_frame.get(frame)
    if base is None:
        return 0.0
    for off in range(window, 0, -1):
        other = steps_by_frame.get(frame + off)
        if other is not None:
            return float(np.hypot(other[0] - base[0], other[1] - base[1])) / off
    return 0.0


def _event_candidates(
    events: Sequence[BallEvent],
    steps_by_frame: Mapping[int, tuple[float, float]],
    player_ctx,
    per_frame_K: Mapping[int, np.ndarray],
    per_frame_R: Mapping[int, np.ndarray],
    per_frame_t: Mapping[int, np.ndarray],
    distortion: tuple[float, float],
    cfg: AutoAnchorCfg,
) -> list[_Candidate]:
    out: list[_Candidate] = []
    for ev in events:
        if ev.score < cfg.min_event_score:
            continue
        if ev.kind == "stationary":
            for f in {ev.frame, ev.end_frame if ev.end_frame is not None else ev.frame}:
                uv = _uv_at(steps_by_frame, f)
                if uv is not None:
                    out.append(_Candidate(
                        BallAnchor(frame=f, image_xy=uv, state="grounded"),
                        ev.score,
                    ))
            continue
        uv = _uv_at(steps_by_frame, ev.frame)
        if uv is None:
            continue
        if ev.kind == "touch":
            if not ev.player_id or not ev.bone:
                continue
            joint = player_ctx.joint_world(ev.frame, ev.player_id, ev.bone)
            K = per_frame_K.get(ev.frame)
            R = per_frame_R.get(ev.frame)
            t = per_frame_t.get(ev.frame)
            if joint is None or K is None or R is None or t is None:
                continue
            gap = point_to_pixel_ray_distance(joint, uv, K, R, t, distortion)
            if gap > cfg.contact_max_gap_m:
                logger.info(
                    "ball auto-anchor: touch at frame %d rejected — joint "
                    "%s/%s is %.2f m off the ball ray (> %.2f m)",
                    ev.frame, ev.player_id, ev.bone, gap, cfg.contact_max_gap_m,
                )
                continue
            touch_type = (
                "shot"
                if _outbound_speed_px(steps_by_frame, ev.frame) >= cfg.shot_speed_px
                else None
            )
            out.append(_Candidate(
                BallAnchor(
                    frame=ev.frame, image_xy=uv, state="player_touch",
                    player_id=ev.player_id, bone=ev.bone,
                    touch_type=touch_type,
                ),
                ev.score,
            ))
        elif ev.kind == "bounce":
            out.append(_Candidate(
                BallAnchor(frame=ev.frame, image_xy=uv, state="bounce"),
                ev.score,
            ))
        elif ev.kind == "goal_impact":
            if not ev.goal_element:
                continue
            out.append(_Candidate(
                BallAnchor(
                    frame=ev.frame, image_xy=uv, state="goal_impact",
                    goal_element=ev.goal_element,
                ),
                ev.score,
            ))
        # velocity_break: solver split hint only — never an anchor.
    return out


def _grounded_candidates(
    steps,
    confidences: Mapping[int, float],
    taken_frames: set[int],
    cfg: AutoAnchorCfg,
    sources: Mapping[int, str] | None = None,
) -> list[_Candidate]:
    out: list[_Candidate] = []
    last_emitted: int | None = None
    for step in steps:
        if step.uv is None or getattr(step, "is_gap_fill", False):
            continue
        f = step.frame
        if (sources is not None
                and sources.get(f) not in cfg.grounded_evidence_sources):
            continue
        if getattr(step, "p_flight", 0.0) > cfg.grounded_max_p_flight:
            continue
        conf = float(confidences.get(f, 0.0))
        if conf < cfg.grounded_min_conf:
            continue
        if any(abs(f - tf) <= cfg.suppress_radius_frames for tf in taken_frames):
            continue
        if any(
            0 < f - tf <= cfg.post_event_suppress_frames
            for tf in taken_frames
        ):
            continue
        if last_emitted is not None and f - last_emitted < cfg.grounded_interval:
            continue
        out.append(_Candidate(
            BallAnchor(
                frame=f,
                image_xy=(float(step.uv[0]), float(step.uv[1])),
                state="grounded",
            ),
            conf,
        ))
        last_emitted = f
    return out


def _resolve_for_gate(
    cand: _Candidate,
    per_frame_K: Mapping[int, np.ndarray],
    per_frame_R: Mapping[int, np.ndarray],
    per_frame_t: Mapping[int, np.ndarray],
    distortion: tuple[float, float],
    player_ctx,
    cfg: AutoAnchorCfg,
) -> np.ndarray | None:
    """Approximate world position used only for validation gates.

    The solver later resolves anchors exactly (goal geometry, bone
    projection); here a ground-plane ray-cast is enough to catch
    off-pitch noise and impossible speeds.
    """
    a = cand.anchor
    f = a.frame
    K, R, t = per_frame_K.get(f), per_frame_R.get(f), per_frame_t.get(f)
    if K is None or R is None or t is None or a.image_xy is None:
        return None
    if a.state == "player_touch" and a.player_id and a.bone:
        joint = player_ctx.joint_world(f, a.player_id, a.bone)
        if joint is not None:
            return np.asarray(joint, dtype=float)
    plane_z = cfg.ball_radius_m if a.state != "goal_impact" else 1.2
    try:
        return np.asarray(ankle_ray_to_pitch(
            a.image_xy, K=K, R=R, t=t, plane_z=plane_z, distortion=distortion,
        ), dtype=float)
    except Exception:
        return None


def _apply_gates(
    candidates: list[_Candidate],
    per_frame_K: Mapping[int, np.ndarray],
    per_frame_R: Mapping[int, np.ndarray],
    per_frame_t: Mapping[int, np.ndarray],
    distortion: tuple[float, float],
    player_ctx,
    fps: float,
    pitch_cfg: Mapping[str, float],
    cfg: AutoAnchorCfg,
) -> list[_Candidate]:
    length = float(pitch_cfg.get("length_m", 105.0))
    width = float(pitch_cfg.get("width_m", 68.0))
    margin = cfg.pitch_margin_m

    resolved: list[tuple[_Candidate, np.ndarray]] = []
    for cand in sorted(candidates, key=lambda c: c.anchor.frame):
        world = _resolve_for_gate(
            cand, per_frame_K, per_frame_R, per_frame_t,
            distortion, player_ctx, cfg,
        )
        if world is None:
            continue
        if not (
            -margin <= world[0] <= length + margin
            and -margin <= world[1] <= width + margin
        ):
            logger.info(
                "ball auto-anchor: %s at frame %d rejected — off-pitch "
                "(%.1f, %.1f)",
                cand.anchor.state, cand.anchor.frame, world[0], world[1],
            )
            continue
        resolved.append((cand, world))

    # Reachability: walk in frame order; drop the lower-scored member of
    # any pair implying an impossible speed.
    _GROUND_STATES = ("grounded", "bounce")
    kept: list[tuple[_Candidate, np.ndarray]] = []
    for cand, world in resolved:
        if kept:
            prev_cand, prev_world = kept[-1]
            df = cand.anchor.frame - prev_cand.anchor.frame
            if df > 0:
                both_ground = (
                    cand.anchor.state in _GROUND_STATES
                    and prev_cand.anchor.state in _GROUND_STATES
                )
                cap = (
                    cfg.max_ground_speed_m_s if both_ground
                    else cfg.max_speed_m_s
                )
                speed = float(np.linalg.norm(world - prev_world)) * fps / df
                if speed > cap:
                    if cand.score <= prev_cand.score:
                        logger.info(
                            "ball auto-anchor: %s at frame %d rejected — "
                            "%.0f m/s to previous anchor",
                            cand.anchor.state, cand.anchor.frame, speed,
                        )
                        continue
                    kept.pop()
        kept.append((cand, world))
    return [cand for cand, _ in kept]


def generate_auto_anchors(
    *,
    events: Sequence[BallEvent],
    steps,
    confidences: Mapping[int, float],
    player_ctx,
    per_frame_K: Mapping[int, np.ndarray],
    per_frame_R: Mapping[int, np.ndarray],
    per_frame_t: Mapping[int, np.ndarray],
    distortion: tuple[float, float],
    fps: float,
    pitch_cfg: Mapping[str, float],
    cfg: AutoAnchorCfg | None = None,
    sources: Mapping[int, str] | None = None,
) -> tuple[BallAnchor, ...]:
    """Events + grounded sampling -> validated auto anchors, frame order."""
    cfg = cfg or AutoAnchorCfg()
    if not cfg.enabled:
        return ()

    def _not_second_pass(c: _Candidate) -> bool:
        return sources is None or sources.get(c.anchor.frame) != "second_pass"

    steps_by_frame: dict[int, tuple[float, float]] = {
        s.frame: s.uv for s in steps if s.uv is not None
    }
    candidates = _event_candidates(
        events, steps_by_frame, player_ctx,
        per_frame_K, per_frame_R, per_frame_t, distortion, cfg,
    )
    # Second-pass detections densify solver evidence but never mint
    # constraints (ball v2 design, Phase 1). Filter event candidates BEFORE
    # computing `taken` so a filtered-out second-pass event never suppresses
    # nearby grounded candidates.
    if sources is not None:
        candidates = [c for c in candidates if _not_second_pass(c)]
    # W2b evidence gate: an event candidate with no real detector evidence
    # anywhere near its frame was found on a synthetic pixel track — keep it
    # out of the anchor set (it would body-pin the track to a guess).
    if sources is not None and cfg.require_event_evidence:
        hard = set(cfg.event_evidence_sources)
        w = int(cfg.event_evidence_window)

        def _has_hard_evidence(frame: int) -> bool:
            return any(sources.get(f) in hard
                       for f in range(frame - w, frame + w + 1))

        before = len(candidates)
        candidates = [
            c for c in candidates if _has_hard_evidence(c.anchor.frame)
        ]
        if before != len(candidates):
            logger.info(
                "ball auto-anchor: evidence gate dropped %d/%d event "
                "candidates (no %s frame within ±%d)",
                before - len(candidates), before,
                "/".join(sorted(hard)), w,
            )
    taken = {c.anchor.frame for c in candidates}
    grounded = _grounded_candidates(steps, confidences, taken, cfg,
                                    sources=sources)
    if sources is not None:
        grounded = [c for c in grounded if _not_second_pass(c)]
    candidates.extend(grounded)
    gated = _apply_gates(
        candidates, per_frame_K, per_frame_R, per_frame_t,
        distortion, player_ctx, fps, pitch_cfg, cfg,
    )
    # One anchor per frame. Specific beats generic regardless of score —
    # a touch/impact carries strictly more information than a grounded
    # sample of the same instant; scores only break ties within a rank.
    _STATE_RANK = {
        "goal_impact": 3, "player_touch": 2, "bounce": 2, "grounded": 1,
    }
    by_frame: dict[int, _Candidate] = {}
    for cand in gated:
        existing = by_frame.get(cand.anchor.frame)
        if existing is None:
            by_frame[cand.anchor.frame] = cand
            continue
        new_key = (_STATE_RANK.get(cand.anchor.state, 0), cand.score)
        old_key = (_STATE_RANK.get(existing.anchor.state, 0), existing.score)
        if new_key > old_key:
            by_frame[cand.anchor.frame] = cand
    # Carry each candidate's detector score onto the anchor as its
    # confidence (clamped to [0, 1]) so the web editor can render auto
    # suggestions distinctly from confirmed/manual anchors.
    anchors = tuple(
        replace(
            by_frame[f].anchor,
            confidence=min(1.0, max(0.0, float(by_frame[f].score))),
        )
        for f in sorted(by_frame)
    )
    logger.info("ball auto-anchor: %d anchors generated", len(anchors))
    return anchors


def merge_anchors(
    manual: Mapping[int, BallAnchor],
    auto: Mapping[int, BallAnchor],
    suppress_radius_frames: int,
    dismissed: Collection[DismissedAuto] = (),
) -> dict[int, BallAnchor]:
    """Manual anchors win; auto anchors near a manual frame are dropped;
    auto anchors exactly matching an operator dismissal are dropped."""
    dismissed_keys = {
        (d.frame, d.state, d.player_id, d.bone) for d in dismissed
    }
    merged: dict[int, BallAnchor] = dict(manual)
    for f, anchor in auto.items():
        if any(abs(f - mf) <= suppress_radius_frames for mf in manual):
            continue
        if (anchor.frame, anchor.state, anchor.player_id,
                anchor.bone) in dismissed_keys:
            continue
        merged[f] = anchor
    return merged

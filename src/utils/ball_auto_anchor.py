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
        "detector", "second_pass", "foot_guided", "strike_window",
    )
    grounded_evidence_sources: tuple[str, ...] = (
        "detector", "second_pass", "foot_guided", "strike_window",
        "bridge", "anchor",
    )
    # Synthetic-born events (no hard evidence in window) may REFINE the
    # operator's path but never REWRITE it: kept only when the resolved pin
    # lies within this distance of the path interpolated between the
    # bracketing manual anchors (goal_impact exempt — resolved against
    # known goal geometry, its position is bounded by the goal frame).
    synthetic_event_max_path_dev_m: float = 1.5
    synthetic_event_bracket_max_frames: int = 120
    # A flying ball passes OVER players: their feet/knees sit near its
    # camera RAY (the contact gap is depth-blind) and would mint phantom
    # touches. While the IMM says flight, only aerial contacts (header/
    # chest/shoulder) may mint (sub-20cm campaign W5f, kroupi01).
    flight_touch_max_p_flight: float = 0.8
    # Adjacent-frame same-player touch duplicates (FK jitter under a
    # moving ball) collapse to the strongest. Kept to 1 frame: genuine
    # consecutive micro-touches during ball control are 2+ frames apart
    # (measured on gberch's manual touches at f43/45/49).
    touch_burst_nms_frames: int = 1
    # Physics-consistency tie-break (sub-20cm campaign W2, foot-contact
    # locomotion regression recovery). The reachability gate below drops
    # the lower-SCORED of two candidates whose implied speed is
    # impossible; when the pair is two DIFFERENT-player `player_touch`
    # candidates disputing the same moment (a "scramble" — two players'
    # limbs both pass the contact-gap gate a frame or two apart, e.g.
    # gberch f55 P019 vs f56 P020), raw kin-strength score is a weak
    # arbiter — it favours whichever limb was moving fastest, not
    # whichever one actually redirected the ball. Physical consistency
    # with the BALL'S OWN velocity change is a stronger, principled
    # signal, blending two terms (validated on gberch f55/f56: the real
    # toucher scores 0.98 direction / 0.96 kink vs the bystander's
    # 0.92 / 0.03):
    #
    #   * direction — does the ball's trajectory after this candidate's
    #     frame move AWAY from the candidate's own joint (an impulse
    #     pushes the ball away from the striking limb)?
    #   * kink — does the ball's pixel trajectory actually change
    #     direction around this candidate's frame at all (a bystander
    #     one frame before the true contact sits on the smooth incoming
    #     glide, not the reversal)?
    #
    # When enabled, the reachability comparison uses
    # `score + physics_tiebreak_bonus_weight * term` instead of raw score
    # (term = dir_weight*direction + kink_weight*kink, weights documented
    # to sum to 1.0). Disabled -> exact legacy behaviour (raw score only).
    physics_tiebreak_enabled: bool = True
    # Two candidates more than this many frames apart never engage the
    # tie-break — a "scramble" is a same-instant dispute, not two
    # genuinely separate touches accidentally tripping the speed cap.
    physics_tiebreak_max_candidate_gap_frames: int = 4
    # +/- frames around each candidate's OWN frame used to sample the
    # ball's incoming/outgoing pixel-velocity direction.
    physics_tiebreak_velocity_window: int = 3
    # direction_term + kink_term weights (sum to 1.0 by convention).
    physics_tiebreak_dir_weight: float = 0.6
    physics_tiebreak_kink_weight: float = 0.4
    # Scales the [0,1] physics term into score units before comparison.
    physics_tiebreak_bonus_weight: float = 0.5


_AERIAL_TOUCH_BONES = frozenset({"head", "chest", "l_shoulder", "r_shoulder"})


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
    # Contact gap (3-D bone<->ball-ray distance, metres) at proposal time;
    # set only for `player_touch` candidates, None otherwise. Reported in
    # the ranked-candidates sidecar (W3) and available to the physics
    # tie-break (W2) — recomputing it there would need the same inputs
    # twice for no benefit.
    gap_m: float | None = None


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


def _burst_nms(candidates: list[_Candidate], window: int) -> list[_Candidate]:
    """Collapse same-player-AND-bone touch runs within ``window`` frames to
    the strongest; non-touch candidates pass through untouched.

    Keyed by ``(player_id, bone)``, not player alone: a player-only key
    also collapses genuinely AMBIGUOUS same-instant candidates on
    DIFFERENT bones (both feet near the ball during close control — the
    kinematic proposer legitimately proposes both when gaps are
    comparable), silently discarding whichever bone scored lower even
    when it is the operator's correct label. Keying on bone too still
    collapses the FK-jitter case this was built for (the same bone
    firing on adjacent frames for one physical touch — gberch f43/45/49
    stay 2+ frames apart and are unaffected) without discarding a
    legitimate alternate-bone candidate at the same instant.
    """
    touches = [c for c in candidates if c.anchor.state == "player_touch"]
    others = [c for c in candidates if c.anchor.state != "player_touch"]
    kept: list[_Candidate] = []
    for c in sorted(touches, key=lambda c: -c.score):
        if any(k.anchor.player_id == c.anchor.player_id
               and k.anchor.bone == c.anchor.bone
               and abs(k.anchor.frame - c.anchor.frame) <= window
               for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda c: c.anchor.frame)
    return [*others, *kept]


def _event_candidates(
    events: Sequence[BallEvent],
    steps_by_frame: Mapping[int, tuple[float, float]],
    player_ctx,
    per_frame_K: Mapping[int, np.ndarray],
    per_frame_R: Mapping[int, np.ndarray],
    per_frame_t: Mapping[int, np.ndarray],
    distortion: tuple[float, float],
    cfg: AutoAnchorCfg,
    p_flight_by_frame: Mapping[int, float] | None = None,
) -> list[_Candidate]:
    out: list[_Candidate] = []
    pf_by_frame = p_flight_by_frame or {}
    for ev in events:
        if ev.score < cfg.min_event_score:
            continue
        if (ev.kind == "touch"
                and pf_by_frame.get(ev.frame, 0.0)
                > cfg.flight_touch_max_p_flight
                and ev.bone not in _AERIAL_TOUCH_BONES):
            logger.info(
                "ball auto-anchor: touch at frame %d (%s/%s) rejected — "
                "ball in flight (p_flight %.2f)",
                ev.frame, ev.player_id, ev.bone,
                pf_by_frame.get(ev.frame, 0.0),
            )
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
                gap_m=gap,
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
    return _burst_nms(out, cfg.touch_burst_nms_frames)


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


def _manual_reference_points(
    manual_anchors: Mapping[int, BallAnchor],
    per_frame_K: Mapping[int, np.ndarray],
    per_frame_R: Mapping[int, np.ndarray],
    per_frame_t: Mapping[int, np.ndarray],
    distortion: tuple[float, float],
) -> list[tuple[int, np.ndarray]]:
    """Coarse world positions of the operator's anchors (state-height
    ray-casts) — the reference path synthetic events are checked against."""
    from src.utils.ball_anchor_heights import state_to_height

    pts: list[tuple[int, np.ndarray]] = []
    for f in sorted(manual_anchors):
        a = manual_anchors[f]
        if a.image_xy is None:
            continue
        K, R, t = per_frame_K.get(f), per_frame_R.get(f), per_frame_t.get(f)
        if K is None or R is None or t is None:
            continue
        try:
            z = state_to_height(a.state)
        except ValueError:
            continue
        try:
            pts.append((f, np.asarray(ankle_ray_to_pitch(
                a.image_xy, K=K, R=R, t=t, plane_z=z, distortion=distortion,
            ), dtype=float)))
        except Exception:  # noqa: BLE001 — grazing ray
            continue
    return pts


def _path_deviation_m(
    frame: int,
    world: np.ndarray,
    ref_points: list[tuple[int, np.ndarray]],
    max_bracket_frames: int,
) -> float | None:
    """Distance from ``world`` to the linear reference path at ``frame``;
    None when no adequate bracketing reference exists."""
    prev = next_ = None
    for f, p in ref_points:
        if f <= frame:
            prev = (f, p)
        elif next_ is None:
            next_ = (f, p)
            break
    if prev is None or next_ is None:
        return None
    (f0, p0), (f1, p1) = prev, next_
    if f1 - f0 > max_bracket_frames or f1 == f0:
        return None
    s = (frame - f0) / (f1 - f0)
    ref = p0 + (p1 - p0) * s
    return float(np.linalg.norm(np.asarray(world, dtype=float) - ref))


def _physics_consistency_term(
    cand: _Candidate,
    steps_by_frame: Mapping[int, tuple[float, float]],
    player_ctx,
    cfg: AutoAnchorCfg,
) -> float | None:
    """[0, 1] consistency of a ``player_touch`` candidate with the ball's
    own pixel-velocity change around the candidate's OWN frame.

    Two terms (weighted, see :class:`AutoAnchorCfg`), both sampled from
    the ball pixel track over +/- ``physics_tiebreak_velocity_window``
    frames of ``cand.anchor.frame``:

    * ``direction_term`` — cosine similarity (rescaled to [0,1]) between
      the ball's outgoing pixel-velocity direction and the direction from
      the candidate's joint pixel to the ball's post-window pixel. High
      when the ball moves away from the joint — the "an impulse pushes
      the ball away from the striking limb" signature.
    * ``kink_term`` — how much the ball's pixel-velocity direction
      actually CHANGES across the candidate's frame (1 - cosine of
      incoming vs. outgoing direction, rescaled to [0,1]). A bystander
      whose local-minimum frame sits just before the true contact is
      still on the smooth incoming glide (little to no direction
      change); the real toucher's frame straddles the reversal.

    Empirically separates gberch's f55 (P019, bystander: direction 0.92,
    kink 0.03) from f56 (P020, real toucher: direction 0.99, kink 0.96).

    Returns None when the ball pixel track or the candidate's joint pixel
    aren't available in the window — callers must fall back to raw score
    in that case (no discriminating signal).
    """
    a = cand.anchor
    if a.state != "player_touch" or not a.player_id or not a.bone:
        return None
    w = cfg.physics_tiebreak_velocity_window
    v0 = _uv_at(steps_by_frame, a.frame - w)
    b0 = _uv_at(steps_by_frame, a.frame)
    v1 = _uv_at(steps_by_frame, a.frame + w)
    if v0 is None or b0 is None or v1 is None:
        return None
    joint_uv = None
    for s in player_ctx.joints_at(a.frame):
        if s.player_id == a.player_id and s.bone == a.bone and s.uv is not None:
            joint_uv = s.uv
            break
    if joint_uv is None:
        return None

    in_dir = np.array([b0[0] - v0[0], b0[1] - v0[1]])
    out_dir = np.array([v1[0] - b0[0], v1[1] - b0[1]])
    away = np.array([v1[0] - joint_uv[0], v1[1] - joint_uv[1]])
    in_norm = float(np.linalg.norm(in_dir))
    out_norm = float(np.linalg.norm(out_dir))
    away_norm = float(np.linalg.norm(away))

    if out_norm < 1e-6 or away_norm < 1e-6:
        direction_term = 0.5  # no discriminating signal -> neutral
    else:
        cos = float(np.dot(out_dir, away) / (out_norm * away_norm))
        direction_term = max(0.0, min(1.0, (cos + 1.0) / 2.0))

    if in_norm < 1e-6 or out_norm < 1e-6:
        kink_term = 0.5  # ball near-stationary on one side -> ambiguous
    else:
        cos = float(np.dot(in_dir, out_dir) / (in_norm * out_norm))
        kink_term = max(0.0, min(1.0, (1.0 - cos) / 2.0))

    return (
        cfg.physics_tiebreak_dir_weight * direction_term
        + cfg.physics_tiebreak_kink_weight * kink_term
    )


def _reachability_winner(
    cand: _Candidate,
    prev_cand: _Candidate,
    steps_by_frame: Mapping[int, tuple[float, float]],
    player_ctx,
    cfg: AutoAnchorCfg,
) -> _Candidate:
    """Which of two reachability-conflicting candidates survives.

    Raw score wins by default (a tie favours ``prev_cand``, matching the
    walk-in-frame-order behaviour this replaces). When both are
    `player_touch` candidates for DIFFERENT players within
    ``physics_tiebreak_max_candidate_gap_frames`` of each other (a
    same-instant "scramble" dispute, not two unrelated touches), the
    comparison adds a physics-consistency bonus to each raw score first
    (sub-20cm campaign W2 — recovers scramble-ambiguity misses like
    gberch f56, where a same-neighbourhood bystander candidate used to
    beat the real toucher purely on kin-strength score).
    """
    cand_score, prev_score = cand.score, prev_cand.score
    if (
        cfg.physics_tiebreak_enabled
        and cand.anchor.state == "player_touch"
        and prev_cand.anchor.state == "player_touch"
        and cand.anchor.player_id != prev_cand.anchor.player_id
        and abs(cand.anchor.frame - prev_cand.anchor.frame)
        <= cfg.physics_tiebreak_max_candidate_gap_frames
    ):
        cand_term = _physics_consistency_term(
            cand, steps_by_frame, player_ctx, cfg)
        prev_term = _physics_consistency_term(
            prev_cand, steps_by_frame, player_ctx, cfg)
        if cand_term is not None and prev_term is not None:
            cand_score = cand.score + cfg.physics_tiebreak_bonus_weight * cand_term
            prev_score = prev_cand.score + cfg.physics_tiebreak_bonus_weight * prev_term
    return prev_cand if cand_score <= prev_score else cand


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
    steps_by_frame: Mapping[int, tuple[float, float]] | None = None,
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

    # Reachability: walk in frame order; drop the loser of any pair
    # implying an impossible speed (raw score by default; a physics-
    # consistency bonus breaks DIFFERENT-player player_touch ties in a
    # same-instant scramble — see _reachability_winner).
    _GROUND_STATES = ("grounded", "bounce")
    steps_by_frame = steps_by_frame or {}
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
                    winner = _reachability_winner(
                        cand, prev_cand, steps_by_frame, player_ctx, cfg,
                    )
                    if winner is prev_cand:
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
    manual_anchors: Mapping[int, BallAnchor] | None = None,
    candidates_out: dict[int, list[dict]] | None = None,
) -> tuple[BallAnchor, ...]:
    """Events + grounded sampling -> validated auto anchors, frame order.

    ``candidates_out``, when given, is populated (mutated in place) with
    the full ranked ``player_touch`` candidate pool per frame that
    survived gating — not just the single winner minted into the
    returned anchors — as
    ``{frame: [{"player_id", "bone", "gap_m", "score"}, ...]}``, best
    first. Sub-20cm campaign W3: a one-anchor-per-frame collapse can
    correctly discard a same-frame alternate for the SOLVE (only one
    knot per frame makes sense) while still losing information a
    downstream attribution pass or the anchor-editor UI could use to
    re-pick — e.g. gberch f192, where the higher-kin-score r_foot wins
    the frame over the smaller-gap l_foot the operator actually clicked.
    """
    cfg = cfg or AutoAnchorCfg()
    if candidates_out is not None:
        candidates_out.clear()
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
        p_flight_by_frame={
            s.frame: float(getattr(s, "p_flight", 0.0)) for s in steps
        },
    )
    # Second-pass detections densify solver evidence but never mint
    # constraints (ball v2 design, Phase 1). Filter event candidates BEFORE
    # computing `taken` so a filtered-out second-pass event never suppresses
    # nearby grounded candidates.
    if sources is not None:
        candidates = [c for c in candidates if _not_second_pass(c)]
    # W2b/W2c evidence gate: an event candidate with no real detector
    # evidence near its frame was found on a synthetic pixel track. Such a
    # candidate may REFINE the operator's path (kept when its resolved pin
    # stays close to the manual-anchor-interpolated reference) but never
    # REWRITE it (dropped when it would drag the track away, or when no
    # reference brackets it). goal_impact is exempt: it resolves against
    # known goal geometry, so its position is bounded by the goal frame.
    if sources is not None and cfg.require_event_evidence:
        hard = set(cfg.event_evidence_sources)
        w = int(cfg.event_evidence_window)
        ref_points = _manual_reference_points(
            manual_anchors or {}, per_frame_K, per_frame_R, per_frame_t,
            distortion,
        )

        def _has_hard_evidence(frame: int) -> bool:
            return any(sources.get(f) in hard
                       for f in range(frame - w, frame + w + 1))

        def _synthetic_ok(c: _Candidate) -> bool:
            if c.anchor.state == "goal_impact":
                return True
            world = _resolve_for_gate(
                c, per_frame_K, per_frame_R, per_frame_t, distortion,
                player_ctx, cfg,
            )
            if world is None:
                return False
            dev = _path_deviation_m(
                c.anchor.frame, world, ref_points,
                cfg.synthetic_event_bracket_max_frames,
            )
            return dev is not None and dev <= cfg.synthetic_event_max_path_dev_m

        before = len(candidates)
        candidates = [
            c for c in candidates
            if _has_hard_evidence(c.anchor.frame) or _synthetic_ok(c)
        ]
        if before != len(candidates):
            logger.info(
                "ball auto-anchor: evidence gate dropped %d/%d event "
                "candidates (no %s frame within ±%d and off the manual "
                "reference path)",
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
        steps_by_frame=steps_by_frame,
    )
    if candidates_out is not None:
        touch_by_frame: dict[int, list[_Candidate]] = {}
        for cand in gated:
            if cand.anchor.state == "player_touch":
                touch_by_frame.setdefault(cand.anchor.frame, []).append(cand)
        for f, cands in touch_by_frame.items():
            ranked = sorted(cands, key=lambda c: -c.score)
            candidates_out[f] = [
                {
                    "player_id": c.anchor.player_id,
                    "bone": c.anchor.bone,
                    "gap_m": (round(float(c.gap_m), 4)
                              if c.gap_m is not None else None),
                    "score": round(float(c.score), 4),
                }
                for c in ranked
            ]
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

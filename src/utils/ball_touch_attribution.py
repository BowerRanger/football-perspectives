"""Touch bone-attribution refinement (spec §4.4).

Validated on gberch: with the same ±2-frame tolerance, ignoring the bone
claim lifts touch recall from 0.25 to 0.50 — half the touch moments are
found but pinned to the wrong body part. The original attribution happens
at the (noisy) break/proposal moment; this post-pass re-picks each touch
event's (player, bone) as the joint with the smallest 3-D bone↔ball-ray
gap over a small window around the event frame, keeping the original when
the improvement is within an ambiguity margin. It relabels ONLY — never
adds, removes, re-frames, or re-scores events. Pure and torch-free.

Default-ON since detector fine-tune v1 (2026-07-04): relabelling now helps
(gberch union recall 0.500 -> 0.625). History: this was default-off from
2026-07-02 through the fine-tune, because relabelling trusts the ball
pixel, which is exactly what was unreliable on the stock (pre-fine-tune)
detector at touch moments.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Collection, Mapping, Sequence

import numpy as np

from src.utils.ball_auto_anchor import pixel_velocity_consistency_term
from src.utils.ball_kinematic_touch import point_to_pixel_ray_distance

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.utils.ball_auto_events import BallEvent


@dataclass(frozen=True)
class TouchAttributionCfg:
    enabled: bool = True
    window: int = 2          # +/- frames considered around the event frame
    max_gap_m: float = 0.45  # candidate joints beyond this never relabel
    margin_m: float = 0.05   # new joint must beat the current one by this
    min_fk_conf: float = 0.3
    # The ray gap is DEPTH-BLIND: a joint near the camera↔ball line passes
    # even when it sits metres from the ball along the ray (the kicker's
    # foot stealing the true toucher's label — sub-20cm campaign W5d).
    # When an expected ball world is available, each candidate's score adds
    # depth_weight × |along-ray depth mismatch|.
    depth_weight: float = 0.5
    # W3 (ball-auto-anchor ranked candidates): when the standard relabel
    # check above doesn't trigger (margin not met, or the depth-weighted
    # score disagrees), fall back to the PURE 3-D ray-gap ranking minting
    # already computed and discarded by generate_auto_anchors's one-
    # anchor-per-frame collapse — an independent second opinion untainted
    # by the depth term's expected-world assumption. Only trusted when it
    # beats the current label's own minted gap.
    consider_ranked_candidates: bool = True
    # Cross-player physics guard (touch-gate calibration follow-up,
    # 2026-09-03): both relabel paths above compare raw bone<->ball-ray
    # gaps only. That is depth-blind AND kink-blind — a bystander whose
    # torso/hand happens to sit nearer the ball's pixel ray than the true
    # toucher's (motion-blurred, FK-noisy) foot can win purely on
    # geometry, and when the winner is a DIFFERENT PLAYER this silently
    # reassigns the touch to the wrong person (gberch f343: production
    # relabels a correctly-player-attributed-but-wrong-bone event from
    # P006 to a bystander P009 sitting right on the ball's ray). The
    # reachability tie-break at minting (ball_auto_anchor's
    # `_reachability_winner`, sub-20cm campaign W2) already guards the
    # analogous same-instant cross-player dispute with a pixel-velocity
    # consistency term (does the ball's own trajectory actually kink
    # around the candidate's frame, and move away from its joint?);
    # this reuses the identical term (`pixel_velocity_consistency_term`)
    # to guard attribution's cross-player flips the same way. Only
    # engages when the winning alternate's player_id differs from the
    # event's current player_id; same-player bone corrections (the
    # common case) are never gated by it. When either side's term is
    # unavailable (ball track/joint pixel missing in the velocity
    # window) there is no discriminating signal, so the flip proceeds
    # exactly as before (matches `_reachability_winner`'s fallback).
    cross_player_physics_guard: bool = True
    cross_player_physics_window: int = 3
    cross_player_physics_dir_weight: float = 0.6
    cross_player_physics_kink_weight: float = 0.4
    # The alternate's term must clear the current label's term minus this
    # slack — 0.0 means "at least as physically consistent", not
    # "strictly better", so a genuine tie doesn't spuriously block a
    # relabel the raw-gap gates already found convincing.
    cross_player_physics_slack: float = 0.0


def _best_gaps_in_window(
    frame: int,
    *,
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    cfg: TouchAttributionCfg,
    expected_world_by_frame: dict | None = None,
) -> dict[tuple[str, str], tuple[float, int]]:
    """Per-(player, bone) minimal (score, frame) over the window around
    ``frame`` — the frame is the specific sighting that achieved the
    minimum, needed by the cross-player physics guard to sample the ball
    track around that candidate's own best occurrence.

    Score = ray gap + ``depth_weight`` × along-ray depth mismatch against
    the expected ball world (when one exists at that frame).
    """
    from src.utils.camera_projection import pixel_ray

    best: dict[tuple[str, str], tuple[float, int]] = {}
    for f in range(frame - cfg.window, frame + cfg.window + 1):
        ball_uv = ball_uvs.get(f)
        K, R, t = per_frame_K.get(f), per_frame_R.get(f), per_frame_t.get(f)
        if ball_uv is None or K is None or R is None or t is None:
            continue
        expected = (expected_world_by_frame or {}).get(f)
        C = d_hat = exp_depth = None
        if expected is not None:
            C, d_hat = pixel_ray(
                (float(ball_uv[0]), float(ball_uv[1])), K, R, t, distortion)
            exp_depth = float(np.dot(
                np.asarray(expected, dtype=float) - C, d_hat))
        for s in player_ctx.joints_at(f):
            if s.confidence < cfg.min_fk_conf or s.world_xyz is None:
                continue
            joint = np.asarray(s.world_xyz, dtype=float)
            score = float(point_to_pixel_ray_distance(
                joint, ball_uv, K, R, t, distortion,
            ))
            if exp_depth is not None:
                joint_depth = float(np.dot(joint - C, d_hat))
                score += cfg.depth_weight * abs(joint_depth - exp_depth)
            key = (s.player_id, s.bone)
            if score < best.get(key, (float("inf"), f))[0]:
                best[key] = (score, f)
    return best


def _joint_uv_at(player_ctx, frame: int, player_id: str, bone: str):
    """The pixel coordinate of ``(player_id, bone)`` at ``frame``, or None
    when that joint isn't present there."""
    for s in player_ctx.joints_at(frame):
        if s.player_id == player_id and s.bone == bone and s.uv is not None:
            return s.uv
    return None


def _cross_player_physics_ok(
    *,
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    cfg: TouchAttributionCfg,
    cur_pid: str, cur_bone: str, cur_frame: int,
    cand_pid: str, cand_bone: str, cand_frame: int,
) -> bool:
    """Whether a cross-player relabel from ``(cur_pid, cur_bone)`` to
    ``(cand_pid, cand_bone)`` is corroborated by the ball's own
    pixel-velocity signature — the same `direction`/`kink` terms
    ``ball_auto_anchor``'s minting-time reachability tie-break uses (see
    :data:`TouchAttributionCfg.cross_player_physics_guard`).

    True (flip allowed) whenever either side's term can't be computed —
    no discriminating signal, so the raw-gap gates decide alone, matching
    ``_reachability_winner``'s fallback."""
    steps_by_frame = {f: tuple(uv) for f, uv in ball_uvs.items()}
    kw = dict(
        window=cfg.cross_player_physics_window,
        dir_weight=cfg.cross_player_physics_dir_weight,
        kink_weight=cfg.cross_player_physics_kink_weight,
    )
    cur_term = pixel_velocity_consistency_term(
        steps_by_frame, cur_frame,
        _joint_uv_at(player_ctx, cur_frame, cur_pid, cur_bone), **kw)
    cand_term = pixel_velocity_consistency_term(
        steps_by_frame, cand_frame,
        _joint_uv_at(player_ctx, cand_frame, cand_pid, cand_bone), **kw)
    if cur_term is None or cand_term is None:
        return True
    return cand_term >= cur_term - cfg.cross_player_physics_slack


def _corroborated_alternate(
    e: "BallEvent",
    ranked_candidates: Mapping[int, Sequence[Mapping]],
    cfg: TouchAttributionCfg,
    *,
    player_ctx=None,
    ball_uvs: "dict[int, np.ndarray] | None" = None,
) -> tuple[str, str] | None:
    """A ranked touch-candidate alternate for ``e`` — an independent
    second opinion from generate_auto_anchors's minting-time candidate
    pool (pure 3-D ray gap, no depth weighting) — when one beats the
    current label's own minted gap.

    Looks across ``+/- cfg.window`` frames (the same neighbourhood the
    ray-gap check above considers) for the smallest-gap candidate whose
    (player, bone) differs from ``e``'s current label and whose gap
    clears ``cfg.max_gap_m``; returns it only when it is strictly better
    than the current label's own minted gap (when known — an unminted
    current label has nothing to lose to). The current label's gap is
    the TRUE minimum across every sighting in the window, preferring an
    exact sighting at ``e.frame`` itself when one exists (the frame the
    event actually claims) over any off-frame sighting. None when no
    alternate qualifies.

    When the winning alternate belongs to a DIFFERENT player and
    ``player_ctx``/``ball_uvs`` are given, the cross-player physics guard
    (:func:`_cross_player_physics_ok`) must also clear before it is
    returned.
    """
    current_gap: float | None = None
    current_gap_own_frame: float | None = None
    best: tuple[str, str] | None = None
    best_gap = float("inf")
    best_frame: int | None = None
    for f in range(e.frame - cfg.window, e.frame + cfg.window + 1):
        for c in ranked_candidates.get(f, ()):
            pid, bone, gap = c.get("player_id"), c.get("bone"), c.get("gap_m")
            if pid is None or bone is None or gap is None:
                continue
            gap = float(gap)
            if (pid, bone) == (e.player_id, e.bone):
                if f == e.frame:
                    current_gap_own_frame = (
                        gap if current_gap_own_frame is None
                        else min(current_gap_own_frame, gap)
                    )
                else:
                    current_gap = (
                        gap if current_gap is None else min(current_gap, gap)
                    )
                continue
            if gap > cfg.max_gap_m:
                continue
            if gap < best_gap:
                best_gap, best, best_frame = gap, (pid, bone), f
    if current_gap_own_frame is not None:
        current_gap = current_gap_own_frame
    if best is None:
        return None
    if current_gap is not None and best_gap >= current_gap:
        return None
    if (
        cfg.cross_player_physics_guard
        and best[0] != e.player_id
        and player_ctx is not None
        and ball_uvs is not None
        and not _cross_player_physics_ok(
            player_ctx=player_ctx, ball_uvs=ball_uvs, cfg=cfg,
            cur_pid=e.player_id, cur_bone=e.bone, cur_frame=e.frame,
            cand_pid=best[0], cand_bone=best[1], cand_frame=best_frame,
        )
    ):
        return None
    return best


def refine_touch_attribution(
    events: "Sequence[BallEvent]",
    *,
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    cfg: TouchAttributionCfg,
    expected_world_by_frame: dict | None = None,
    ranked_candidates: "Mapping[int, Sequence[Mapping]] | None" = None,
) -> "tuple[BallEvent, ...]":
    """Relabel touch events to the best-scoring joint; everything else
    passes through untouched (same order, same length). With
    ``expected_world_by_frame`` the score is depth-aware (W5d), so a
    joint lying on the ray but metres from the ball never wins.

    ``ranked_candidates`` (W3), when given, is consulted ONLY when the
    depth-aware check above does not already relabel: a genuinely
    corroborated alternate from the minting-time candidate pool (see
    :func:`_corroborated_alternate`) can still flip the label. Always
    relabels — never adds, removes, or reorders events."""
    if not cfg.enabled:
        return tuple(events)
    out: "list[BallEvent]" = []
    for e in events:
        if e.kind != "touch" or not e.player_id or not e.bone:
            out.append(e)
            continue
        gaps = _best_gaps_in_window(
            e.frame, player_ctx=player_ctx, ball_uvs=ball_uvs,
            per_frame_K=per_frame_K, per_frame_R=per_frame_R,
            per_frame_t=per_frame_t, distortion=distortion, cfg=cfg,
            expected_world_by_frame=expected_world_by_frame,
        )
        best_pid = best_bone = None
        relabel = False
        if gaps:
            (best_pid, best_bone), (best_gap, best_frame) = min(
                gaps.items(), key=lambda kv: (kv[1][0], kv[0]))
            current = gaps.get((e.player_id, e.bone))
            current_gap, current_frame = (
                current if current is not None else (None, e.frame))
            relabel = (
                best_gap <= cfg.max_gap_m
                and (best_pid, best_bone) != (e.player_id, e.bone)
                and (current_gap is None or best_gap + cfg.margin_m < current_gap)
            )
            if (relabel and cfg.cross_player_physics_guard
                    and best_pid != e.player_id
                    and not _cross_player_physics_ok(
                        player_ctx=player_ctx, ball_uvs=ball_uvs, cfg=cfg,
                        cur_pid=e.player_id, cur_bone=e.bone,
                        cur_frame=current_frame,
                        cand_pid=best_pid, cand_bone=best_bone,
                        cand_frame=best_frame,
                    )):
                relabel = False
        if not relabel and cfg.consider_ranked_candidates and ranked_candidates:
            alt = _corroborated_alternate(
                e, ranked_candidates, cfg,
                player_ctx=player_ctx, ball_uvs=ball_uvs)
            if alt is not None:
                best_pid, best_bone = alt
                relabel = True
        if relabel:
            out.append(dataclasses.replace(
                e, player_id=best_pid, bone=best_bone))
        else:
            out.append(e)
    return tuple(out)


def context_expected_worlds(
    world_by_frame: dict,
    *,
    touch_frames: Collection[int],
    window: int = 2,
    max_bridge_frames: int = 30,
) -> dict[int, tuple[float, float, float]]:
    """Expected ball worlds from a resolved track, with every touch's
    ±``window`` neighbourhood re-interpolated from the clean context
    outside it (two-pass attribution, sub-20cm campaign).

    A wrong first-pass body-pin drags the track toward the wrong joint at
    the touch itself, so the expectation there must come from where the
    ball was coming from and going to — never from the disputed pin.
    """
    out = {f: tuple(float(x) for x in w)
           for f, w in world_by_frame.items() if w is not None}
    frames = sorted(out)
    if not frames:
        return out
    for tf in touch_frames:
        lo = next((f for f in reversed(frames)
                   if f < tf - window and tf - f <= max_bridge_frames), None)
        hi = next((f for f in frames
                   if f > tf + window and f - tf <= max_bridge_frames), None)
        if lo is None or hi is None or hi <= lo:
            continue
        pa = np.asarray(out[lo], dtype=float)
        pb = np.asarray(out[hi], dtype=float)
        for f in range(max(lo + 1, tf - window), min(hi, tf + window + 1)):
            s = (f - lo) / (hi - lo)
            out[f] = tuple(float(x) for x in (pa + (pb - pa) * s))
    return out


def expected_ball_worlds(
    anchor_by_frame: dict,
    *,
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ball_radius: float,
    max_gap_frames: int = 60,
) -> dict[int, tuple[float, float, float]]:
    """Sparse expected ball worlds from ground-level anchors, linearly
    interpolated between consecutive anchors (no extrapolation).

    Coarse by design — it exists to give the attribution score a depth
    reference so a joint metres off along the ray cannot win.
    """
    from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES
    from src.utils.camera_projection import pixel_ray

    ground_states = frozenset(GROUND_LEVEL_STATES) | {"bounce"}
    pts: list[tuple[int, np.ndarray]] = []
    for f in sorted(anchor_by_frame):
        a = anchor_by_frame[f]
        if a.state not in ground_states or a.image_xy is None:
            continue
        K, R, t = per_frame_K.get(f), per_frame_R.get(f), per_frame_t.get(f)
        if K is None or R is None or t is None:
            continue
        C, d = pixel_ray(a.image_xy, K, R, t, distortion)
        dz = float(d[2])
        if abs(dz) < 1e-9:
            continue
        s = (ball_radius - float(C[2])) / dz
        if s <= 0:
            continue
        pts.append((f, C + s * d))
    worlds: dict[int, tuple[float, float, float]] = {}
    for (fa, pa), (fb, pb) in zip(pts, pts[1:]):
        if fb - fa > max_gap_frames:
            worlds[fa] = tuple(float(x) for x in pa)
            continue
        for f in range(fa, fb + 1):
            s = (f - fa) / (fb - fa) if fb > fa else 0.0
            worlds[f] = tuple(float(x) for x in (pa + (pb - pa) * s))
    if pts:
        worlds[pts[-1][0]] = tuple(float(x) for x in pts[-1][1])
    return worlds

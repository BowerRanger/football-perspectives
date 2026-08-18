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
from typing import TYPE_CHECKING, Collection, Sequence

import numpy as np

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
) -> dict[tuple[str, str], float]:
    """Per-(player, bone) minimal score over the window around ``frame``.

    Score = ray gap + ``depth_weight`` × along-ray depth mismatch against
    the expected ball world (when one exists at that frame).
    """
    from src.utils.camera_projection import pixel_ray

    best: dict[tuple[str, str], float] = {}
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
            if score < best.get(key, float("inf")):
                best[key] = score
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
) -> "tuple[BallEvent, ...]":
    """Relabel touch events to the best-scoring joint; everything else
    passes through untouched (same order, same length). With
    ``expected_world_by_frame`` the score is depth-aware (W5d), so a
    joint lying on the ray but metres from the ball never wins."""
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
        if not gaps:
            out.append(e)
            continue
        (best_pid, best_bone), best_gap = min(
            gaps.items(), key=lambda kv: (kv[1], kv[0]))
        current_gap = gaps.get((e.player_id, e.bone))
        relabel = (
            best_gap <= cfg.max_gap_m
            and (best_pid, best_bone) != (e.player_id, e.bone)
            and (current_gap is None or best_gap + cfg.margin_m < current_gap)
        )
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

"""Gravity-arc refit of airborne anchor chains (sub-20cm campaign W3).

In events mode, ``airborne_low/mid/high`` anchors historically resolved via
ray ∩ a fixed height plane (z = 1.0/6.0/15.0) — bucket placeholders with
metre-scale depth error. This module re-resolves every maximal run of
airborne anchors bracketed by two hard knots (touch/kick/bounce/goal/…,
whose worlds are already resolved) with a single gravity-arc fit:

- the two hard-knot worlds are (near-)hard constraints,
- every chain anchor's clicked pixel is a reprojection observation (a ray,
  not a plane),
- the airborne buckets demote to one-sided z-range hinges.

Two hard 3-D knots + gravity fully determine the 6-DOF arc, depth included
(the 2026-05-31 C2 result), so interior airborne positions — and every
in-between frame the interpolator later renders — inherit physical depth.

Chains missing a bracket, or whose fit residual exceeds the gate, keep
their bucket resolution and are flagged in the diagnostics instead.
Pure numpy/scipy via ``bundle_adjust`` — no torch, no video.
"""

from __future__ import annotations

import logging

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_anchor_heights import (
    AIRBORNE_BUCKETS,
    HARD_KNOT_STATES,
    airborne_bucket_range,
)
from src.utils.bundle_adjust import fit_parabola_to_image_observations
from src.utils.camera_projection import project_world_to_image

logger = logging.getLogger(__name__)

# Sanity caps on an accepted arc (anything beyond is a mis-fit, not football).
_MAX_LAUNCH_SPEED_M_S = 60.0
_MAX_ARC_Z_M = 40.0
# An extra (detection) observation joins the stage-2 refit only when the
# anchors-only arc already reprojects within this of it — junk detections
# (players, static lock-ons) never poison the fit.
_EXTRA_OBS_GATE_PX = 30.0
# Soft-constraint weights for AUTO knot worlds (px per metre of deviation,
# scaled by anchor confidence). Measured on the corrupt-pin fixture: with
# ≥1 manual knot, gravity + the rays fully determine the arc and any pin
# strength only imports attribution error (0.5 → 0.11 m, 2.0 → 0.63 m),
# so auto pins are a near-free tiebreak there. With NO manual knot the
# pins are the only along-ray depth anchor and must carry real weight.
_AUTO_KNOT_WEIGHT_TIEBREAK = 0.5
_AUTO_KNOT_WEIGHT_SOLE_DEPTH = 5.0


def _arc_residual_px(p0, v0, obs, Ks, Rs, ts, distortion, fps, f0) -> float:
    """Median reprojection error of the arc against ``obs`` pixels."""
    errs = []
    g = np.array([0.0, 0.0, -9.81])
    for (f, uv), K, R, t in zip(obs, Ks, Rs, ts):
        tt = (f - f0) / fps
        w = p0 + v0 * tt + 0.5 * g * tt * tt
        proj = project_world_to_image(K, R, t, distortion,
                                      np.asarray(w).reshape(1, 3))[0]
        errs.append(float(np.linalg.norm(proj - np.asarray(uv, dtype=float))))
    return float(np.median(errs)) if errs else float("inf")


def _chains(anchor_by_frame, world_for_anchor):
    """Yield (start_hard_frame, [airborne frames...], end_hard_frame)."""
    frames = sorted(anchor_by_frame)
    runs: list[tuple[int | None, list[int], int | None]] = []
    current_air: list[int] = []
    last_hard: int | None = None
    for f in frames:
        state = anchor_by_frame[f].state
        if state in AIRBORNE_BUCKETS:
            current_air.append(f)
            continue
        is_hard = (state in HARD_KNOT_STATES
                   and world_for_anchor.get(f) is not None)
        if current_air:
            runs.append((last_hard, current_air, f if is_hard else None))
            current_air = []
        if is_hard:
            last_hard = f
        elif state not in AIRBORNE_BUCKETS:
            # off_screen_flight or an unresolved hard state breaks the chain.
            last_hard = None
    if current_air:
        runs.append((last_hard, current_air, None))
    return runs


def refit_airborne_chains(
    *,
    anchor_by_frame: dict[int, BallAnchor],
    world_for_anchor: dict[int, tuple[float, float, float] | None],
    per_frame_K,
    per_frame_R,
    per_frame_t,
    distortion: tuple[float, float],
    fps: float,
    max_residual_px: float = 5.0,
    extra_observations: dict[int, tuple[float, float]] | None = None,
    manual_frames: frozenset[int] | None = None,
    world_fixes: dict[int, tuple] | None = None,
) -> tuple[dict[int, tuple[float, float, float]], list[dict]]:
    """Return ``({airborne_frame: refit_world}, diagnostics)``.

    ``world_fixes`` maps frames to ``(xyz, weight)`` absolute constraints
    (cross-replay triangulation) — the only monocular-external depth truth
    available; in-span entries join every fit (sub-20cm campaign W5u).

    Only interior airborne-bucket anchors of successfully fitted chains
    appear in the updates map; everything else is untouched.
    ``extra_observations`` (W4) are real in-span detection pixels — they
    densify the fit and keep chains determined when interior anchor clicks
    are absent (e.g. a hold-out run or a lightly-anchored clip).
    ``manual_frames`` (W5c): operator-anchored frames. Manual knot worlds
    are near-hard; AUTO knot worlds (body-pins that may carry attribution
    error) become confidence-weighted soft constraints, so the ray
    evidence can pull the arc away from a mis-attributed joint. ``None``
    treats every knot as manual (legacy behaviour).
    """
    updates: dict[int, tuple[float, float, float]] = {}
    diags: list[dict] = []
    extra_observations = extra_observations or {}
    strong_fixes = {
        f: entry for f, entry in (world_fixes or {}).items()
    }
    for start, air_frames, end in _chains(anchor_by_frame, world_for_anchor):
        if start is None or end is None:
            n_strong = sum(1 for f in strong_fixes
                           if air_frames and
                           air_frames[0] - 3 <= f <= air_frames[-1] + 3)
            if n_strong < 2 or not air_frames:
                diags.append({
                    "kind": "underconstrained_chain",
                    "air_frames": list(air_frames),
                    "note": "airborne run lacks a bracketing hard knot; "
                            "bucket heights kept — add a bracketing anchor",
                })
                continue
            # W5w: >=2 absolute cross-replay fixes ARE the missing depth
            # anchors — synthesize the bracket from the run itself.
            start = start if start is not None else air_frames[0]
            end = end if end is not None else air_frames[-1]
        chain_frames = [start, *air_frames, end]
        anchor_obs = []
        for f in chain_frames:
            anc = anchor_by_frame[f]
            if (anc.image_xy is not None and f in per_frame_K
                    and f in per_frame_R and f in per_frame_t):
                anchor_obs.append((f, (float(anc.image_xy[0]),
                                       float(anc.image_xy[1]))))
        extras = []
        anchored = {f for f, _ in anchor_obs}
        for f in sorted(extra_observations):
            if (start < f < end and f not in anchored
                    and f in per_frame_K and f in per_frame_R
                    and f in per_frame_t):
                uv = extra_observations[f]
                extras.append((f, (float(uv[0]), float(uv[1]))))
        if (len(anchor_obs) < 2 or anchor_obs[0][0] != start
                or anchor_obs[-1][0] != end
                or len(anchor_obs) + len(extras) < 3):
            diags.append({
                "kind": "underconstrained_chain",
                "air_frames": list(air_frames),
                "note": "insufficient pixel observations for a chain fit",
            })
            continue

        def _cams_for(o):
            return ([np.asarray(per_frame_K[f]) for f, _ in o],
                    [np.asarray(per_frame_R[f]) for f, _ in o],
                    [np.asarray(per_frame_t[f]) for f, _ in o])

        f0 = start
        knots: dict[int, np.ndarray] = {}
        soft_fixes: list[tuple[int, np.ndarray, float]] = []
        # >=2 absolute in-span fixes fully determine depth with the rays
        # and gravity: EVERY monocular knot (bucket OR body-pin — player
        # reconstruction depth is unreliable at distance) demotes to a
        # soft tiebreak; the clicks stay ray-hard via the observations
        # (sub-20cm campaign W5x).
        n_span_fixes = sum(1 for f in (world_fixes or {})
                           if start <= f <= end)
        fixes_rule = n_span_fixes >= 2
        n_manual_knots = sum(
            1 for kf in (start, end)
            if manual_frames is None or kf in manual_frames)
        auto_weight = (_AUTO_KNOT_WEIGHT_TIEBREAK if n_manual_knots
                       else _AUTO_KNOT_WEIGHT_SOLE_DEPTH)
        for kf in (start, end):
            w = np.asarray(world_for_anchor[kf], dtype=float)
            state = getattr(anchor_by_frame.get(kf), "state", "")
            depth_soft = state in AIRBORNE_BUCKETS   # bucket z is a guess
            if (manual_frames is None or kf in manual_frames) \
                    and not depth_soft and not fixes_rule:
                knots[kf - f0] = w
            else:
                conf = float(getattr(anchor_by_frame[kf], "confidence", 1.0))
                wt = (_AUTO_KNOT_WEIGHT_TIEBREAK if depth_soft
                      else auto_weight)
                soft_fixes.append((kf - f0, w, wt * conf))
        z_ranges = {
            f - f0: bucket
            for f in air_frames
            if (bucket := airborne_bucket_range(
                anchor_by_frame[f].state)) is not None
        }

        def _fit(o):
            Ks, Rs, ts = _cams_for(o)
            shift = o[0][0] - f0
            wf = [(k - shift, w, wt) for k, w, wt in soft_fixes]
            for f, entry in (world_fixes or {}).items():
                if start < f < end:
                    xyz, wt = entry
                    wf.append((f - f0 - shift,
                               np.asarray(xyz, dtype=float), float(wt)))
            return fit_parabola_to_image_observations(
                o, Ks=Ks, Rs=Rs, t_world=ts, fps=fps,
                distortion=distortion, p0_fixed=None,
                knot_frames={k - shift: w for k, w in knots.items()} or None,
                z_range_frames={k - shift: b for k, b in z_ranges.items()},
                world_fixes=wf or None,
            )

        # W5y — preferred path: ≥3 absolute in-chain fixes determine the
        # arc in closed form; accept when it reprojects onto the anchor
        # clicks (which are ray-hard truth).
        span_fixes = {f: e for f, e in (world_fixes or {}).items()
                      if start <= f <= end}
        if len(span_fixes) >= 3:
            cf = fit_arc_to_fixes(span_fixes, fps=fps)
            if cf is not None:
                ff0, (cp0, cv0) = cf
                lo_fix = min(span_fixes) - 2
                hi_fix = max(span_fixes) + 2
                # The arc claims only the fix-covered interval: gating on
                # (or updating) frames it must EXTRAPOLATE to imports
                # pre-/post-event structure it cannot know about.
                in_range_obs = [(f, uv) for f, uv in anchor_obs
                                if lo_fix <= f <= hi_fix]
                gate_obs = in_range_obs if len(in_range_obs) >= 2 \
                    else anchor_obs
                aK, aR, aT = _cams_for(gate_obs)
                med_c = _arc_residual_px(
                    np.asarray(cp0), np.asarray(cv0), gate_obs,
                    aK, aR, aT, distortion, fps, ff0)
                if np.isfinite(med_c) and med_c <= 3 * max_residual_px:
                    g2 = np.array([0.0, 0.0, -9.81])
                    for f in air_frames:
                        if not (lo_fix <= f <= hi_fix):
                            continue
                        t2 = (f - ff0) / fps
                        w2 = (np.asarray(cp0) + np.asarray(cv0) * t2
                              + 0.5 * g2 * t2 * t2)
                        updates[f] = (float(w2[0]), float(w2[1]),
                                      float(w2[2]))
                    diags.append({
                        "kind": "chain_fit", "accepted": True,
                        "span": [start, end], "n_air": len(air_frames),
                        "mode": "fix_arc", "n_fixes": len(span_fixes),
                        "residual_px": float(med_c),
                    })
                    continue
                split2 = (fit_split_arcs_to_fixes(span_fixes, fps=fps)
                          if len(span_fixes) >= 6 else None)
                if split2 is not None:
                    fa0, s_frame, (pa2, va2), (pb2, vb2) = split2
                    g3 = np.array([0.0, 0.0, -9.81])
                    lo2 = min(span_fixes) - 2
                    hi2 = max(span_fixes) + 2
                    for f in air_frames:
                        if not (lo2 <= f <= hi2):
                            continue
                        if f <= s_frame:
                            t3 = (f - fa0) / fps
                            w3 = (np.asarray(pa2) + np.asarray(va2) * t3
                                  + 0.5 * g3 * t3 * t3)
                        else:
                            t3 = (f - s_frame) / fps
                            w3 = (np.asarray(pb2) + np.asarray(vb2) * t3
                                  + 0.5 * g3 * t3 * t3)
                        updates[f] = (float(w3[0]), float(w3[1]),
                                      float(w3[2]))
                    diags.append({
                        "kind": "chain_fit", "accepted": True,
                        "span": [start, end], "n_air": len(air_frames),
                        "mode": "fix_split_arc", "split": int(s_frame),
                        "n_fixes": len(span_fixes),
                        "residual_px": float(med_c),
                    })
                    continue
                diags.append({
                    "kind": "chain_fit_fix_arc_rejected",
                    "span": [start, end],
                    "residual_px": float(med_c),
                })
        # Two-stage robust fit: anchors first (operator clicks are trusted),
        # then only extras the anchors-only arc already agrees with — and
        # the refit must not degrade the anchor residual (W5b).
        n_extra_used = 0
        try:
            p0, v0, residual = _fit(anchor_obs)
            aK, aR, aT = _cams_for(anchor_obs)
            anchor_res = _arc_residual_px(
                p0, v0, anchor_obs, aK, aR, aT, distortion, fps, f0)
            inliers = []
            for f, uv in extras:
                K = np.asarray(per_frame_K[f])
                R = np.asarray(per_frame_R[f])
                t = np.asarray(per_frame_t[f])
                if _arc_residual_px(p0, v0, [(f, uv)], [K], [R], [t],
                                    distortion, fps, f0) <= _EXTRA_OBS_GATE_PX:
                    inliers.append((f, uv))
            if inliers:
                merged = sorted([*anchor_obs, *inliers], key=lambda o: o[0])
                p0_2, v0_2, residual_2 = _fit(merged)
                anchor_res_2 = _arc_residual_px(
                    p0_2, v0_2, anchor_obs, aK, aR, aT, distortion, fps, f0)
                if anchor_res_2 <= anchor_res + 1.0:
                    p0, v0 = p0_2, v0_2
                    anchor_res = anchor_res_2
                    n_extra_used = len(inliers)
            # Acceptance grades the arc against the trusted anchor clicks.
            residual = float(anchor_res)
        except Exception as exc:  # noqa: BLE001 — fit failure keeps buckets
            diags.append({
                "kind": "chain_fit", "accepted": False,
                "span": [start, end], "error": str(exc),
            })
            continue
        arc_z = [
            float(p0[2] + v0[2] * ((f - f0) / fps)
                  - 0.5 * 9.81 * ((f - f0) / fps) ** 2)
            for f in chain_frames
        ]
        ok = (
            np.isfinite(residual)
            and residual <= max_residual_px
            and float(np.linalg.norm(v0)) <= _MAX_LAUNCH_SPEED_M_S
            and all(-1.0 <= z <= _MAX_ARC_Z_M for z in arc_z)
        )
        diags.append({
            "kind": "chain_fit", "accepted": bool(ok),
            "span": [start, end], "n_air": len(air_frames),
            "n_extra_obs": len(extras), "n_extra_used": n_extra_used,
            "residual_px": float(residual),
        })
        if not ok:
            logger.info(
                "ball flight-chain [%d,%d]: fit rejected "
                "(residual %.1fpx > %.1f or implausible arc) — "
                "bucket heights kept", start, end, residual, max_residual_px,
            )
            continue
        g = np.array([0.0, 0.0, -9.81])
        for f in air_frames:
            t = (f - f0) / fps
            w = p0 + v0 * t + 0.5 * g * t * t
            updates[f] = (float(w[0]), float(w[1]), float(w[2]))
    return updates, diags


# Minimum in-span detections for a segment-level fit: with only the two
# endpoints the naive parabola is already the unique arc, so a refit only
# adds information when several rays constrain the span interior.
_SEGMENT_FIT_MIN_OBS = 3
_SEGMENT_FIT_MAX_RESIDUAL_PX = 8.0


def refit_ballistic_segment(
    *,
    start_frame: int,
    end_frame: int,
    start_world,
    end_world,
    start_is_manual: bool,
    end_is_manual: bool,
    end_confidence: float = 1.0,
    start_confidence: float = 1.0,
    extra_observations: dict[int, tuple[float, float]],
    per_frame_K,
    per_frame_R,
    per_frame_t,
    distortion: tuple[float, float],
    fps: float,
    world_fixes: dict[int, tuple] | None = None,
    start_depth_soft: bool = False,
    end_depth_soft: bool = False,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Gravity-arc fit for a ballistic SEGMENT with no interior anchors.

    kroupi-class deep flights sit between two keyframes with nothing in
    between; in-span detections are the only interior evidence. Manual
    endpoint worlds are near-hard; AUTO endpoints (body-pins that may
    carry attribution error) are soft, so the rays can override them.
    Returns ``(p0, v0)`` relative to ``start_frame`` on success, else
    ``None`` (render falls back to the naive endpoint parabola).
    """
    obs = [(f, (float(uv[0]), float(uv[1])))
           for f, uv in sorted(extra_observations.items())
           if start_frame < f < end_frame
           and f in per_frame_K and f in per_frame_R and f in per_frame_t]
    if len(obs) < _SEGMENT_FIT_MIN_OBS:
        return None
    n_manual = int(start_is_manual) + int(end_is_manual)
    auto_weight = (_AUTO_KNOT_WEIGHT_TIEBREAK if n_manual
                   else _AUTO_KNOT_WEIGHT_SOLE_DEPTH)
    n_span_fixes = sum(1 for f in (world_fixes or {})
                       if start_frame <= f <= end_frame)
    if n_span_fixes >= 2:
        # Absolute fixes own the depth; all monocular knots soften (W5x).
        start_depth_soft = True
        end_depth_soft = True
    knots: dict[int, np.ndarray] = {}
    fixes: list[tuple[int, np.ndarray, float]] = []
    for f, w, is_manual, conf, depth_soft in (
        (start_frame, start_world, start_is_manual, start_confidence,
         start_depth_soft),
        (end_frame, end_world, end_is_manual, end_confidence,
         end_depth_soft),
    ):
        arr = np.asarray(w, dtype=float)
        # An AIRBORNE anchor's keyframe world is a bucket-height guess:
        # the operator's click constrains the RAY (already an observation),
        # never the depth — bucket worlds must not out-weigh real depth
        # constraints (sub-20cm campaign W5w).
        if is_manual and not depth_soft:
            knots[f - start_frame] = arr
        else:
            wt = (auto_weight if not depth_soft
                  else _AUTO_KNOT_WEIGHT_TIEBREAK)
            fixes.append((f - start_frame, arr, wt * float(conf)))
    f0 = obs[0][0]
    shift = f0 - start_frame
    Ks = [np.asarray(per_frame_K[f]) for f, _ in obs]
    Rs = [np.asarray(per_frame_R[f]) for f, _ in obs]
    ts = [np.asarray(per_frame_t[f]) for f, _ in obs]
    wf = [(k - shift, w, wt) for k, w, wt in fixes]
    for f, entry in (world_fixes or {}).items():
        if start_frame < f < end_frame:
            xyz, wt = entry
            wf.append((f - f0, np.asarray(xyz, dtype=float), float(wt)))
    try:
        p0, v0, residual = fit_parabola_to_image_observations(
            obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps, distortion=distortion,
            p0_fixed=None,
            knot_frames={k - shift: w for k, w in knots.items()} or None,
            world_fixes=wf or None,
        )
    except Exception:  # noqa: BLE001 — fit failure keeps the naive parabola
        return None
    med = _arc_residual_px(p0, v0, obs, Ks, Rs, ts, distortion, fps, f0)
    if not (np.isfinite(med) and med <= _SEGMENT_FIT_MAX_RESIDUAL_PX
            and float(np.linalg.norm(v0)) <= _MAX_LAUNCH_SPEED_M_S):
        logger.info(
            "ball segment-fit [%d,%d]: rejected (median %.1fpx, |v0| %.1f)",
            start_frame, end_frame, med, float(np.linalg.norm(v0)))
        return None
    # Report p0/v0 at start_frame (rebased from the first observation).
    g = np.array([0.0, 0.0, -9.81])
    dt = (start_frame - f0) / fps
    p0s = p0 + v0 * dt + 0.5 * g * dt * dt
    v0s = v0 + g * dt
    return (tuple(float(x) for x in p0s), tuple(float(x) for x in v0s))


# A split's two arcs must meet within this at the shared frame (they were
# fitted independently; a genuine bounce/touch is position-continuous).
_SPLIT_JOIN_TOL_M = 0.6
_SPLIT_MIN_HALF_OBS = 3


def refit_split_segment(
    *,
    start_frame: int,
    end_frame: int,
    start_world,
    end_world,
    start_is_manual: bool,
    end_is_manual: bool,
    extra_observations: dict[int, tuple[float, float]],
    per_frame_K,
    per_frame_R,
    per_frame_t,
    distortion: tuple[float, float],
    fps: float,
    start_confidence: float = 1.0,
    end_confidence: float = 1.0,
):
    """Two-arc fit for a span hiding one interior event (kroupi class).

    Tries every viable split frame (observation frames with enough
    observations on both sides); each half fits independently with its
    own endpoint knot; a candidate is accepted when both halves pass the
    residual gate AND the arcs meet within ``_SPLIT_JOIN_TOL_M`` at the
    split (a real bounce/touch is position-continuous). Returns
    ``(split_frame, (p0_a, v0_a), (p0_b, v0_b))`` — each arc's state at
    its own start frame (``start_frame`` and ``split_frame``) — or None.
    """
    obs_frames = sorted(f for f in extra_observations
                        if start_frame < f < end_frame)
    if len(obs_frames) < 2 * _SPLIT_MIN_HALF_OBS:
        return None
    g = np.array([0.0, 0.0, -9.81])
    best = None
    for s in obs_frames[_SPLIT_MIN_HALF_OBS - 1:
                        len(obs_frames) - (_SPLIT_MIN_HALF_OBS - 1)]:
        left = {f: extra_observations[f] for f in obs_frames if f <= s}
        right = {f: extra_observations[f] for f in obs_frames if f >= s}
        if len(left) < _SPLIT_MIN_HALF_OBS or len(right) < _SPLIT_MIN_HALF_OBS:
            continue
        fa = _fit_half(
            start_frame, s, start_world, start_is_manual,
            start_confidence, left, per_frame_K, per_frame_R,
            per_frame_t, distortion, fps, knot_at_start=True)
        fb = _fit_half(
            s, end_frame, end_world, end_is_manual,
            end_confidence, right, per_frame_K, per_frame_R,
            per_frame_t, distortion, fps, knot_at_start=False)
        if fa is None or fb is None:
            continue
        (pa, va, res_a) = fa
        (pb, vb, res_b) = fb
        tt = (s - start_frame) / fps
        a_at_s = np.asarray(pa) + np.asarray(va) * tt + 0.5 * g * tt * tt
        join = float(np.linalg.norm(a_at_s - np.asarray(pb)))
        if join > _SPLIT_JOIN_TOL_M:
            continue
        score = res_a + res_b + join
        if best is None or score < best[0]:
            best = (score, s,
                    (tuple(float(x) for x in pa),
                     tuple(float(x) for x in va)),
                    (tuple(float(x) for x in pb),
                     tuple(float(x) for x in vb)))
    if best is None:
        return None
    return best[1], best[2], best[3]


def _fit_half(f_start, f_end, knot_world, knot_is_manual, knot_conf,
              obs_dict, per_frame_K, per_frame_R, per_frame_t,
              distortion, fps, *, knot_at_start):
    """One half of a split: single knot at the anchored end, rays free."""
    obs = [(f, (float(uv[0]), float(uv[1])))
           for f, uv in sorted(obs_dict.items())
           if f in per_frame_K and f in per_frame_R and f in per_frame_t]
    if len(obs) < _SPLIT_MIN_HALF_OBS:
        return None
    f0 = obs[0][0]
    Ks = [np.asarray(per_frame_K[f]) for f, _ in obs]
    Rs = [np.asarray(per_frame_R[f]) for f, _ in obs]
    ts = [np.asarray(per_frame_t[f]) for f, _ in obs]
    knot_frame = f_start if knot_at_start else f_end
    knots = {}
    fixes = []
    w = np.asarray(knot_world, dtype=float)
    if knot_is_manual:
        knots[knot_frame - f0] = w
    else:
        fixes.append((knot_frame - f0, w,
                      _AUTO_KNOT_WEIGHT_SOLE_DEPTH * float(knot_conf)))
    try:
        p0, v0, _ = fit_parabola_to_image_observations(
            obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps, distortion=distortion,
            p0_fixed=None, knot_frames=knots or None,
            world_fixes=fixes or None,
        )
    except Exception:  # noqa: BLE001
        return None
    med = _arc_residual_px(p0, v0, obs, Ks, Rs, ts, distortion, fps, f0)
    if not (np.isfinite(med) and med <= _SEGMENT_FIT_MAX_RESIDUAL_PX
            and float(np.linalg.norm(v0)) <= _MAX_LAUNCH_SPEED_M_S):
        return None
    # Rebase the arc state to the half's own start frame.
    g = np.array([0.0, 0.0, -9.81])
    dt = (f_start - f0) / fps
    p0s = p0 + v0 * dt + 0.5 * g * dt * dt
    v0s = v0 + g * dt
    return p0s, v0s, float(med)


def fit_arc_to_fixes(
    fixes: dict[int, tuple],
    *,
    fps: float,
    min_fixes: int = 3,
):
    """Closed-form gravity arc through absolute fixes (W5y).

    ``fixes`` maps frame → ``(xyz, weight)``. With ≥3 fixes the 6-DOF arc
    is a plain weighted linear least-squares problem in (p0, v0) — no LM,
    no seeding, immune to the near-parallel-ray cost landscape that
    stalls pixel-based fits at broadcast depths. Returns
    ``(f0, (p0, v0))`` at the earliest fix frame, or None.
    """
    items = sorted(fixes.items())
    if len(items) < min_fixes:
        return None
    f0 = items[0][0]
    g = np.array([0.0, 0.0, -9.81])
    A_rows, b_rows, w_rows = [], [], []
    for f, entry in items:
        xyz, wt = entry
        t = (f - f0) / fps
        # xyz - 0.5 g t^2 = p0 + v0 t   (per axis)
        target = np.asarray(xyz, dtype=float) - 0.5 * g * t * t
        A_rows.append((1.0, t))
        b_rows.append(target)
        w_rows.append(float(wt))
    A = np.asarray(A_rows)              # (n, 2)
    B = np.stack(b_rows)                # (n, 3)
    W = np.sqrt(np.asarray(w_rows))[:, None]
    sol, *_ = np.linalg.lstsq(A * W, B * W, rcond=None)
    p0, v0 = sol[0], sol[1]
    return f0, (tuple(float(x) for x in p0), tuple(float(x) for x in v0))


def fit_split_arcs_to_fixes(
    fixes: dict[int, tuple],
    *,
    fps: float,
    min_half: int = 3,
    join_tol_m: float = 0.8,
    max_fit_res_m: float = 0.35,
):
    """Two closed-form arcs through a fix run hiding one deflection.

    Tries every split with ``min_half`` fixes per side; each half fits by
    :func:`fit_arc_to_fixes`; a candidate needs per-half RMS fix residual
    ≤ ``max_fit_res_m`` and position continuity at the split within
    ``join_tol_m``. Returns ``(split_frame, (p0_a, v0_a), (p0_b, v0_b))``
    with each arc's state at its own first-fix frame, or None.
    """
    frames = sorted(fixes)
    if len(frames) < 2 * min_half:
        return None
    g = np.array([0.0, 0.0, -9.81])

    def _rms(fit, sub):
        f0, (p0, v0) = fit
        errs = []
        for f, (xyz, _w) in sub.items():
            t = (f - f0) / fps
            w = np.asarray(p0) + np.asarray(v0) * t + 0.5 * g * t * t
            errs.append(float(np.linalg.norm(w - np.asarray(xyz))))
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else np.inf

    best = None
    for i in range(min_half - 1, len(frames) - min_half):
        s_frame = frames[i]
        left = {f: fixes[f] for f in frames if f <= s_frame}
        right = {f: fixes[f] for f in frames if f > s_frame}
        fa = fit_arc_to_fixes(left, fps=fps, min_fixes=min_half)
        fb = fit_arc_to_fixes(right, fps=fps, min_fixes=min_half)
        if fa is None or fb is None:
            continue
        ra, rb = _rms(fa, left), _rms(fb, right)
        if ra > max_fit_res_m or rb > max_fit_res_m:
            continue
        fa0, (pa, va) = fa
        fb0, (pb, vb) = fb
        ta = (s_frame - fa0) / fps
        a_end = np.asarray(pa) + np.asarray(va) * ta + 0.5 * g * ta * ta
        tb = (s_frame - fb0) / fps
        b_at = np.asarray(pb) + np.asarray(vb) * tb + 0.5 * g * tb * tb
        join = float(np.linalg.norm(a_end - b_at))
        if join > join_tol_m:
            continue
        score = ra + rb + join
        if best is None or score < best[0]:
            best = (score, s_frame, (pa, va), (pb, vb), fb0)
    if best is None:
        return None
    _, s_frame, (pa, va), (pb, vb), fb0 = best
    # Rebase arc B to the split frame for a uniform contract.
    tb = (s_frame - fb0) / fps
    pb2 = np.asarray(pb) + np.asarray(vb) * tb + 0.5 * g * tb * tb
    vb2 = np.asarray(vb) + g * tb
    return (frames[0], s_frame,
            (tuple(float(x) for x in pa), tuple(float(x) for x in va)),
            (tuple(float(x) for x in pb2), tuple(float(x) for x in vb2)))


__all__ = ["refit_airborne_chains", "refit_ballistic_segment",
           "refit_split_segment", "fit_arc_to_fixes",
           "fit_split_arcs_to_fixes"]

"""Synthetic detector stream derived from a ``TruthTrack``.

Produces a ``SynthRun`` — anchors (same frames/states as the real manual
anchors, image positions re-derived from truth + small noise) and a dense
per-frame observation stream (truth projection + calibrated noise, with a
miss model and replayed real-detector false positives) that a candidate
extraction method consumes exactly like the real ``<shot>_ball_observations
.json`` sidecar.

Deliberately reads the real ``<shot>_ball_observations.json`` and
``tracks/<shot>_tracks.json`` sidecars (real detector evidence / player
boxes, both legitimate per CONTRACT.md) but NEVER the ball stage's own
dense-track, auto-anchor or sparse-keyframe outputs — those are the thing
under test in this PoC and a test greps this file to enforce that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .ctx import ClipContext
from .types import Observation, SynthRun, TruthTrack

# Mirrors scripts/eval_ball_accuracy.py's _DENSE_EVAL_SOURCES / veto radius —
# duplicated as small constants (not an import of solver/physics code) so
# this module can independently classify real detections as "on the click
# path" vs. "junk" for false-positive replay.
_DENSE_EVAL_SOURCES = frozenset(
    {"detector", "second_pass", "foot_guided", "strike_window"})
_ANCHOR_VETO_PX = 60.0
_MAX_INTERP_GAP_FRAMES = 6
_SIGMA_FLOOR_PX = 1.5
_WEAK_CANDIDATE_PROB = 0.35
_FP_BASE_PROB = 0.5


def _expected_click_path(anchors, max_gap: int = _MAX_INTERP_GAP_FRAMES):
    """Frame -> expected click pixel, exact at anchor frames and linearly
    interpolated between anchors <= ``max_gap`` frames apart."""
    clicks = sorted((a.frame, a.image_xy) for a in anchors
                    if a.image_xy is not None)
    expected: dict[int, tuple[float, float]] = dict(clicks)
    for (fa, ua), (fb, ub) in zip(clicks, clicks[1:]):
        if 0 < fb - fa <= max_gap:
            for f in range(fa + 1, fb):
                s = (f - fa) / (fb - fa)
                expected[f] = (ua[0] + (ub[0] - ua[0]) * s,
                              ua[1] + (ub[1] - ua[1]) * s)
    return expected


def _calibrate_sigma(ctx: ClipContext) -> float:
    """Detector-vs-click px residual std at frames where a real detector
    observation coincides exactly with a manual anchor frame."""
    anchor_uv = {a.frame: a.image_xy for a in ctx.anchors.anchors
                if a.image_xy is not None}
    residuals = []
    for obs in ctx.observations:
        uv_a = anchor_uv.get(obs.frame)
        if uv_a is None:
            continue
        residuals.append(float(np.hypot(obs.uv[0] - uv_a[0],
                                        obs.uv[1] - uv_a[1])))
    if not residuals:
        return _SIGMA_FLOOR_PX
    return max(float(np.std(residuals)), _SIGMA_FLOOR_PX)


def _load_raw_observation_frames(ctx: ClipContext):
    path = ctx.output_dir / "ball" / f"{ctx.shot_id}_ball_observations.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = []
    for e in data.get("frames", []):
        uv = e.get("uv")
        if (uv is None or e.get("frame") is None or e.get("gap_fill")
                or str(e.get("source")) not in _DENSE_EVAL_SOURCES):
            continue
        out.append((int(e["frame"]), (float(uv[0]), float(uv[1])),
                   float(e.get("confidence", 0.0)), str(e.get("source"))))
    return out


def _junk_detections(ctx: ClipContext, veto_px: float = _ANCHOR_VETO_PX):
    """Real detections far from the manual-anchor interpolated pixel path —
    genuine detector junk (static lock-ons, player socks, etc.)."""
    raw = _load_raw_observation_frames(ctx)
    expected = _expected_click_path(ctx.anchors.anchors)
    junk = []
    for frame, uv, conf, source in raw:
        exp = expected.get(frame)
        if exp is None:
            continue
        dist2 = (uv[0] - exp[0]) ** 2 + (uv[1] - exp[1]) ** 2
        if dist2 > veto_px ** 2:
            junk.append((frame, uv, conf, source))
    return junk


def _load_player_boxes(ctx: ClipContext):
    """Best-effort per-frame player bboxes from ``tracks/<shot>_tracks
    .json``. Returns ``None`` (occlusion signal skipped) if the sidecar is
    missing or not in the expected shape, per CONTRACT.md."""
    path = ctx.output_dir / "tracks" / f"{ctx.shot_id}_tracks.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        out: dict[int, list[tuple[float, float, float, float]]] = {}
        for tr in data["tracks"]:
            for fr in tr.get("frames", []):
                bbox = fr.get("bbox")
                frame = fr.get("frame")
                if bbox is None or frame is None:
                    continue
                out.setdefault(int(frame), []).append(
                    tuple(float(v) for v in bbox))
        return out
    except Exception:  # noqa: BLE001 — occlusion is a nice-to-have signal
        return None


def _in_any_box(uv, boxes) -> bool:
    u, v = float(uv[0]), float(uv[1])
    return any(x1 <= u <= x2 and y1 <= v <= y2 for (x1, y1, x2, y2) in boxes)


@dataclass(frozen=True)
class _MissModel:
    base_p: float
    speed_scale: float
    occlusion_bonus: float
    speed_ref_px_frame: float

    def p_miss(self, speed_px_frame: float, occluded: bool) -> float:
        p = self.base_p + self.speed_scale * min(
            1.0, speed_px_frame / self.speed_ref_px_frame)
        if occluded:
            p += self.occlusion_bonus
        return float(np.clip(p, 0.0, 0.95))


def _calibrate_base_p(target_coverage: float, speeds, occluded,
                      speed_scale: float, occlusion_bonus: float,
                      speed_ref: float) -> float:
    """Bisect ``base_p`` so mean detection probability over the clip's
    frames matches ``target_coverage`` (mean p_miss is monotonic
    increasing in ``base_p``, so mean coverage is monotonic decreasing)."""
    def mean_coverage(base_p: float) -> float:
        model = _MissModel(base_p, speed_scale, occlusion_bonus, speed_ref)
        misses = [model.p_miss(s, o) for s, o in zip(speeds, occluded)]
        return 1.0 - float(np.mean(misses)) if misses else 1.0

    lo, hi = 0.0, 0.95
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if mean_coverage(mid) > target_coverage:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_synth_run(ctx: ClipContext, truth: TruthTrack, scenario: str,
                   seed: int = 0) -> SynthRun:
    rng = np.random.default_rng(seed)
    sigma_px = _calibrate_sigma(ctx)

    frames = sorted(truth.frames, key=lambda f: f.frame)
    frame_ids = [tf.frame for tf in frames]
    uv_by_frame = {tf.frame: ctx.project(tf.frame, np.asarray(tf.xyz))
                  for tf in frames}

    speed_px: dict[int, float] = {}
    for i, f in enumerate(frame_ids):
        if i == 0:
            speed_px[f] = 0.0
            continue
        prev = frame_ids[i - 1]
        gap = f - prev
        diff = uv_by_frame[f] - uv_by_frame[prev]
        speed_px[f] = float(np.hypot(diff[0], diff[1])) / gap if gap > 0 else 0.0

    boxes_by_frame = _load_player_boxes(ctx)
    occlusion_available = boxes_by_frame is not None
    occluded = {
        f: (occlusion_available
           and _in_any_box(uv_by_frame[f], boxes_by_frame.get(f, [])))
        for f in frame_ids
    }

    real_coverage = min(1.0, len(ctx.observations) / max(1, len(frame_ids)))
    target_coverage = real_coverage * (0.5 if scenario == "sparse" else 1.0)

    speed_ref = 25.0
    speed_scale = 0.35
    occlusion_bonus = 0.25 if occlusion_available else 0.0
    base_p = _calibrate_base_p(
        target_coverage, [speed_px[f] for f in frame_ids],
        [occluded[f] for f in frame_ids], speed_scale, occlusion_bonus,
        speed_ref)
    model = _MissModel(base_p, speed_scale, occlusion_bonus, speed_ref)

    observations: list[Observation] = []
    weak_candidates: list[dict] = []
    n_missed = 0
    for f in frame_ids:
        p_miss = model.p_miss(speed_px[f], occluded[f])
        if rng.uniform() < p_miss:
            n_missed += 1
            if rng.uniform() < _WEAK_CANDIDATE_PROB:
                noise = rng.normal(0.0, sigma_px * 3.0, size=2)
                uv = uv_by_frame[f] + noise
                weak_candidates.append({
                    "frame": f, "uv": [float(uv[0]), float(uv[1])],
                    "score": float(rng.uniform(0.1, 0.3)),
                })
            continue
        noise = rng.normal(0.0, sigma_px, size=2)
        uv = uv_by_frame[f] + noise
        conf = float(np.clip(rng.normal(0.75, 0.12), 0.3, 0.99))
        observations.append(Observation(
            frame=f, uv=(float(uv[0]), float(uv[1])), conf=conf,
            source="detector"))

    junk = _junk_detections(ctx)
    fp_prob = min(1.0, _FP_BASE_PROB * (2.0 if scenario == "sparse" else 1.0))
    has_true_obs = {o.frame for o in observations}
    n_fp = n_fp_sole = n_fp_extra = 0
    for frame, uv, conf, _source in junk:
        if not (frame_ids[0] <= frame <= frame_ids[-1]):
            continue
        if rng.uniform() >= fp_prob:
            continue
        observations.append(Observation(frame=frame, uv=uv, conf=conf,
                                        source="detector"))
        n_fp += 1
        if frame in has_true_obs:
            n_fp_extra += 1
        else:
            n_fp_sole += 1

    observations.sort(key=lambda o: (o.frame, -o.conf))

    truth_by_frame = {tf.frame: tf.xyz for tf in frames}
    anchors_out = []
    for a in ctx.anchors.anchors:
        image_xy = None
        if a.image_xy is not None:
            txyz = truth_by_frame.get(a.frame)
            if txyz is not None:
                uv = ctx.project(a.frame, np.asarray(txyz)) + rng.normal(
                    0.0, 1.5, size=2)
            else:
                # Outside the truth span (shouldn't happen — truth spans
                # first..last anchor frame — kept defensively).
                uv = np.asarray(a.image_xy, dtype=float)
            image_xy = [float(uv[0]), float(uv[1])]
        anchors_out.append({
            "frame": a.frame, "image_xy": image_xy, "state": a.state,
            "player_id": a.player_id, "bone": a.bone,
            "goal_element": a.goal_element, "touch_type": a.touch_type,
            "spin": a.spin, "confidence": 1.0, "end_frame": a.end_frame,
        })

    noise_model = {
        "sigma_px": sigma_px,
        "sigma_px_floor": _SIGMA_FLOOR_PX,
        "real_coverage_reference": real_coverage,
        "target_coverage": target_coverage,
        "base_miss_p": base_p,
        "speed_scale": speed_scale,
        "speed_ref_px_frame": speed_ref,
        "occlusion_bonus": occlusion_bonus,
        "occlusion_available": occlusion_available,
        "n_frames": len(frame_ids),
        "n_missed": n_missed,
        "n_detected": len(frame_ids) - n_missed,
        "achieved_coverage": (len(frame_ids) - n_missed) / max(1, len(frame_ids)),
        "n_junk_candidates": len(junk),
        "anchor_veto_px": _ANCHOR_VETO_PX,
        "fp_inclusion_prob": fp_prob,
        "n_fp_inserted": n_fp,
        "n_fp_as_sole_observation": n_fp_sole,
        "n_fp_as_extra_alternative": n_fp_extra,
        "weak_candidate_prob": _WEAK_CANDIDATE_PROB,
        "weak_candidates": weak_candidates,
        "scenario": scenario,
        "seed": seed,
    }

    return SynthRun(clip_id=ctx.clip_id, scenario=scenario,
                    observations=tuple(observations),
                    anchors=tuple(anchors_out), noise_model=noise_model)


__all__ = ["make_synth_run"]

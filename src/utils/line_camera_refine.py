"""Per-frame camera refinement from detected painted lines.

Wraps the line detector (``line_detector.py``) and the line-residual LM
(``anchor_solver._line_residuals``) into a single detect-and-solve loop
that refines one frame's camera against the painted pitch lines visible
in it.

This is the production entry point for the experimental
``camera.line_extraction`` path — see
``docs/superpowers/notes/2026-05-14-camera-1px-experiment.md`` for the
research write-up. It is deliberately a separate module from
``line_detector.py`` (pure detection) and ``anchor_solver.py`` (anchor
solve) so the camera stage can opt into it without pulling solver code
into the detector or vice versa.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.schemas.anchor import LandmarkObservation, LineObservation
from src.utils.anchor_solver import (
    _line_residuals,
    _make_K,
    _point_residuals_distorted,
)
from src.utils.line_detector import (
    DetectorConfig,
    detect_painted_lines_in_frame,
)
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE


logger = logging.getLogger(__name__)


def is_pitch_line(
    segment: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    """True if both endpoints are painted-pitch-surface points (z=0,
    inside the 0–105 × 0–68 pitch rectangle). Excludes ad-board lines,
    goal-frame lines, and anything off the playing surface — those need
    different detection logic (different background, not white-on-grass).
    """
    a, b = segment
    return all(
        0.0 <= p[0] <= 105.0 and 0.0 <= p[1] <= 68.0 and p[2] == 0.0
        for p in (a, b)
    )


# Catalogue of just the painted pitch-surface lines — computed once.
PITCH_LINE_CATALOGUE: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {n: s for n, s in LINE_CATALOGUE.items() if is_pitch_line(s)}


class FrameRefinement(NamedTuple):
    line_rms_px: float
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    detected_lines: list[LineObservation]
    n_detections: int
    corridor_deviation_m: float = 0.0
    """|dC| at the accepted pose when ``corridor_centre`` was supplied —
    how far this frame's camera centre sits from the anchor-interpolated
    corridor. ``0.0`` when no corridor was requested (legacy free solve)."""
    speed_clamped: bool = False
    """True when ``prev_centre``/``max_step_m`` were supplied AND the
    fit's own centre exceeded the step budget, so the returned pose was
    pulled back to the budget boundary (with ``line_rms_px`` honestly
    recomputed there)."""


def refine_camera_from_lines(
    frame_bgr: np.ndarray,
    K_init: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    distortion: tuple[float, float],
    *,
    point_hint_landmarks: list[LandmarkObservation] | None = None,
    detector_cfg: DetectorConfig | None = None,
    max_iters: int = 4,
    min_confidence: float = 0.5,
    min_n_samples: int = 40,
    point_hint_weight: float = 0.3,
    corridor_centre: np.ndarray | None = None,
    max_corridor_deviation_m: float = 5.0,
    prev_centre: np.ndarray | None = None,
    max_step_m: float | None = None,
) -> FrameRefinement:
    """Detect painted lines in ``frame_bgr`` and refine ``(K, R, t)`` to
    fit them.

    Iterates: detect lines using the current camera as bootstrap → LM-
    solve against the line residuals (plus an optional low-weight
    point-landmark hint) → repeat with the improved camera so detection
    windows tighten onto the true painted line.

    Without ``corridor_centre``, the solve is free over ``(rvec, tvec,
    fx)`` with a loose ±300m ``tvec`` bound — fine for a well-constrained
    (many-line) frame, but a frame with only 1-2 detected lines is
    under-determined (4 residuals for a 7-DOF problem) and that bound
    does nothing to stop the camera CENTRE (``-R^T @ t``, not the raw
    ``tvec`` OpenCV vets) from wandering tens to hundreds of metres to a
    spurious low-RMS local minimum — a real-clip regression (gberch-2):
    10/181 frames wandered up to 212m off the anchor-interpolated
    corridor at reported confidence 0.95-1.00. See
    ``docs/superpowers/specs/2026-09-09-moving-camera-support.md``'s
    corridor-drift addendum.

    With ``corridor_centre`` supplied (the camera stage passes the
    anchor-interpolated centre for every non-anchor frame in moving
    mode), the solve instead parameterises ``(rvec, dC, fx)`` with
    ``t = -R @ (corridor_centre + dC)`` and ``|dC|_∞ ≤
    max_corridor_deviation_m`` — the camera CENTRE is bounded near the
    interpolated rig path BY CONSTRUCTION of the optimiser's own box
    bounds (a scipy guarantee), not a post-hoc best-effort check.

    ``prev_centre`` + ``max_step_m``, if both given, additionally clamp
    the ACCEPTED pose's centre to within ``max_step_m`` of
    ``prev_centre`` (frame-to-frame speed enforcement, for runs of
    skipped/failed frames where the corridor bound alone might still
    permit too large a jump between two accepted frames). Line RMS is
    honestly recalculated at the clamped pose, so confidence never
    reflects the pre-clamp (spuriously better) fit.

    ``point_hint_landmarks`` — when supplied (e.g. the anchor's clicked
    landmarks on an anchor frame), they're added to the cost at
    ``point_hint_weight`` so the line solve doesn't drift into a
    geometrically wrong basin that still fits the (few) detected lines.
    On non-anchor frames pass ``None``.

    Returns the iteration with the lowest line RMS. If detection never
    finds ≥2 usable lines, returns the input camera unchanged with
    ``n_detections=0``.
    """
    if detector_cfg is None:
        detector_cfg = DetectorConfig()
    cx, cy = float(K_init[0, 2]), float(K_init[1, 2])
    K = K_init.copy()
    R = R_init.copy()
    t = t_init.astype(np.float64).copy()
    best = FrameRefinement(float("inf"), K.copy(), R.copy(), t.copy(), [], 0)
    corridor = (
        np.asarray(corridor_centre, dtype=np.float64)
        if corridor_centre is not None else None
    )

    for _it in range(max_iters):
        all_dets = detect_painted_lines_in_frame(
            frame_bgr, K, R, t, distortion, PITCH_LINE_CATALOGUE, detector_cfg,
        )
        dets = [
            d for d in all_dets
            if d.confidence >= min_confidence and d.n_samples >= min_n_samples
        ]
        if len(dets) < 2:
            break
        line_obs = [
            LineObservation(
                name=d.name, image_segment=d.image_segment,
                world_segment=d.world_segment,
            )
            for d in dets
        ]

        rvec_init, _ = cv2.Rodrigues(R)
        fx0 = float(K[0, 0])

        if corridor is not None:
            # Box bound per dC component such that the worst-case L2 stays
            # at the budget: the LM optimises inside [-c, c]^3 where
            # c = max_corridor_deviation_m / sqrt(3), so |dC|_2 <=
            # max_corridor_deviation_m (matches refine_with_bounded_
            # motion's identical L2-vs-L_inf box-bound conversion).
            c_bound = max_corridor_deviation_m / float(np.sqrt(3.0))
            C_init = -R.astype(np.float64).T @ t
            dC_init = np.clip(C_init - corridor, -c_bound, c_bound)

            def _residuals(p: np.ndarray) -> np.ndarray:
                rvec = p[0:3]
                dC = p[3:6]
                fx = float(p[6])
                R_m, _ = cv2.Rodrigues(rvec)
                t_m = -R_m @ (corridor + dC)
                K_m = _make_K(fx, cx, cy)
                parts = [_line_residuals(line_obs, K_m, R_m, t_m)]
                if point_hint_landmarks:
                    parts.append(point_hint_weight * _point_residuals_distorted(
                        point_hint_landmarks, K_m, rvec, t_m, distortion,
                    ))
                return np.concatenate(parts)

            p0 = np.concatenate([rvec_init.reshape(3), dC_init, [fx0]])
            lower = np.array([-np.pi] * 3 + [-c_bound] * 3 + [fx0 * 0.5])
            upper = np.array([np.pi] * 3 + [c_bound] * 3 + [fx0 * 2.0])
        else:
            def _residuals(p: np.ndarray) -> np.ndarray:
                rvec = p[0:3]
                tvec = p[3:6]
                fx = float(p[6])
                R_m, _ = cv2.Rodrigues(rvec)
                K_m = _make_K(fx, cx, cy)
                parts = [_line_residuals(line_obs, K_m, R_m, tvec)]
                if point_hint_landmarks:
                    parts.append(point_hint_weight * _point_residuals_distorted(
                        point_hint_landmarks, K_m, rvec, tvec, distortion,
                    ))
                return np.concatenate(parts)

            p0 = np.concatenate([rvec_init.reshape(3), t, [fx0]])
            lower = np.array([-np.pi]*3 + [-300.0]*3 + [fx0 * 0.5])
            upper = np.array([np.pi]*3 + [300.0]*3 + [fx0 * 2.0])

        try:
            result = least_squares(
                _residuals, p0, bounds=(lower, upper),
                method="trf", loss="huber", f_scale=2.0, max_nfev=2000,
            )
        except Exception as exc:
            logger.warning("line-extraction frame solve failed: %s", exc)
            break

        R, _ = cv2.Rodrigues(result.x[0:3])
        if corridor is not None:
            dC = result.x[3:6]
            t = -R @ (corridor + dC)
            deviation = float(np.linalg.norm(dC))
        else:
            t = result.x[3:6].copy()
            deviation = 0.0
        K = _make_K(float(result.x[6]), cx, cy)
        line_rms = float(np.sqrt(
            (_line_residuals(line_obs, K, R, t) ** 2).mean()
        ))
        if line_rms < best.line_rms_px:
            best = FrameRefinement(
                line_rms_px=line_rms,
                K=K.copy(), R=R.copy(), t=t.copy(),
                detected_lines=list(line_obs),
                n_detections=len(line_obs),
                corridor_deviation_m=deviation,
            )

    if best.n_detections == 0:
        # No usable detections — hand back the input camera unchanged.
        return FrameRefinement(
            line_rms_px=float("nan"),
            K=K_init.copy(), R=R_init.copy(), t=t_init.astype(np.float64).copy(),
            detected_lines=[], n_detections=0,
        )

    # Frame-to-frame speed clamp: even a corridor-bounded fit could imply
    # an unreasonable step if the previous accepted frame is several
    # indices away (e.g. a run of failed detections in between). Clamp
    # the CENTRE (not the raw pose) and honestly re-derive line RMS from
    # the clamped pose so confidence is never scored against the
    # pre-clamp (spuriously better) fit.
    if prev_centre is not None and max_step_m is not None and max_step_m > 0:
        prev_c = np.asarray(prev_centre, dtype=np.float64)
        C_best = -best.R.astype(np.float64).T @ best.t.astype(np.float64)
        step = C_best - prev_c
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step_m and step_norm > 1e-9:
            C_clamped = prev_c + step * (max_step_m / step_norm)
            t_clamped = -best.R @ C_clamped
            line_rms_clamped = (
                float(np.sqrt(
                    (_line_residuals(best.detected_lines, best.K, best.R, t_clamped) ** 2).mean()
                ))
                if best.detected_lines else best.line_rms_px
            )
            best = best._replace(
                t=t_clamped, line_rms_px=line_rms_clamped, speed_clamped=True,
            )

    return best


def detect_lines_for_frames(
    frames_bgr: dict[int, np.ndarray],
    cameras: dict[int, dict[str, np.ndarray]],
    distortion: tuple[float, float],
    detector_cfg: DetectorConfig | None = None,
    *,
    min_confidence: float = 0.5,
    min_n_samples: int = 40,
    min_lines: int = 2,
) -> dict[int, list[LineObservation]]:
    """Detect painted pitch lines across many frames using per-frame
    bootstrap cameras.

    ``frames_bgr`` and ``cameras`` are both keyed by frame id. A frame
    is included in the output only if it has a bootstrap camera, a
    decoded image, and at least ``min_lines`` detections passing the
    confidence / sample-count gates. Frames that fail any check are
    silently dropped — callers keep their propagated camera for those.
    """
    if detector_cfg is None:
        detector_cfg = DetectorConfig()
    out: dict[int, list[LineObservation]] = {}
    for fid, frame in frames_bgr.items():
        cam = cameras.get(fid)
        if cam is None:
            continue
        dets = detect_painted_lines_in_frame(
            frame, cam["K"], cam["R"], cam["t"], distortion,
            PITCH_LINE_CATALOGUE, detector_cfg,
        )
        usable = [
            d for d in dets
            if d.confidence >= min_confidence and d.n_samples >= min_n_samples
        ]
        if len(usable) >= min_lines:
            out[fid] = [
                LineObservation(
                    name=d.name,
                    image_segment=d.image_segment,
                    world_segment=d.world_segment,
                )
                for d in usable
            ]
    return out


def drop_underdetermined_frames(
    per_frame_lines: dict[int, list], min_lines: int
) -> dict[int, list]:
    """Drop frames whose detected-line count is below ``min_lines``.

    A per-frame static-camera solve recovers 4 DOF (rotation + focal length).
    With fewer than ~4 line correspondences — often near-parallel touchlines —
    the solve is under-determined and can converge to a non-physical camera
    (extreme focal, large rotation flip) that still fits the few lines with a
    low residual. Excluding these frames from the solve makes them fall back
    to the smooth interpolated camera instead of a wild per-frame fit, which
    removes single-frame glitches. ``min_lines <= 1`` is a no-op on non-empty
    frames.
    """
    return {
        f: lines for f, lines in per_frame_lines.items() if len(lines) >= min_lines
    }

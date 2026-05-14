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
) -> FrameRefinement:
    """Detect painted lines in ``frame_bgr`` and refine ``(K, R, t)`` to
    fit them.

    Iterates: detect lines using the current camera as bootstrap → LM-
    solve ``(rvec, tvec, fx)`` against the line residuals (plus an
    optional low-weight point-landmark hint) → repeat with the improved
    camera so detection windows tighten onto the true painted line.

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

        rvec_init, _ = cv2.Rodrigues(R)
        fx0 = float(K[0, 0])
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
        t = result.x[3:6].copy()
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
            )

    if best.n_detections == 0:
        # No usable detections — hand back the input camera untouched.
        return FrameRefinement(
            line_rms_px=float("nan"),
            K=K_init.copy(), R=R_init.copy(), t=t_init.astype(np.float64).copy(),
            detected_lines=[], n_detections=0,
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

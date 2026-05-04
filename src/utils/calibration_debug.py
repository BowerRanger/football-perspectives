"""Calibration verification overlay.

Projects canonical FIFA pitch markings through a recovered camera
calibration and draws them on the source frame.  When the projected
lines align with the painted pitch markings, the calibration is good.
When they don't, the misalignment direction tells us which DOF is
biased (focal length, rotation, position).

Used as both a developer tool (inspect a shot's calibration manually)
and as a downstream stage hook (generate annotated frames after the
calibration stage runs).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult
from src.utils.pitch_lines import pitch_polylines

logger = logging.getLogger(__name__)


# BGR colour palette per polyline class — keeps the overlay readable.
_LINE_COLOURS_BGR: list[tuple[int, int, int]] = [
    (0, 255, 255),    # 0  near touchline    — yellow
    (0, 255, 255),    # 1  far touchline     — yellow
    (0, 255, 255),    # 2  left goal line    — yellow
    (0, 255, 255),    # 3  right goal line   — yellow
    (0, 200, 255),    # 4  halfway           — orange
    (0, 200, 0),      # 5  centre circle     — green
    (0, 200, 0),      # 6  L 18-yard         — green
    (0, 200, 0),      # 7  R 18-yard         — green
    (0, 200, 0),      # 8  L 6-yard          — green
    (0, 200, 0),      # 9  R 6-yard          — green
    (255, 200, 0),    # 10 L penalty arc     — cyan
    (255, 200, 0),    # 11 R penalty arc     — cyan
    (255, 0, 255),    # 12 L goal frame      — magenta
    (255, 0, 255),    # 13 R goal frame      — magenta
]


def project_pitch_lines(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """Project all pitch polylines into pixel coordinates.

    Args:
        K, rvec, tvec: camera intrinsics + extrinsics.
        image_size: ``(width, height)`` of the destination frame.
            Returned polylines are not clipped to this rectangle —
            callers can decide whether to clip or let the line fall
            off-canvas as a diagnostic cue.

    Returns:
        List of ``(N, 2)`` int32 arrays, one per polyline.  Polylines
        whose points all sit behind the camera (``z_cam < 0``) are
        skipped entirely; polylines that straddle the camera plane
        are returned with the behind-camera segments removed (so the
        projected line doesn't wrap around).
    """
    K = np.asarray(K, dtype=np.float64)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(rvec)

    out: list[np.ndarray] = []
    for poly in pitch_polylines():
        cam = (R @ poly.T).T + tvec  # (N, 3)
        in_front = cam[:, 2] > 0.05
        if not np.any(in_front):
            out.append(np.empty((0, 2), dtype=np.int32))
            continue
        # Project only the in-front points; OpenCV would still try to
        # divide by negative z and produce nonsense for behind-camera
        # vertices.
        front_world = poly[in_front]
        projected, _ = cv2.projectPoints(front_world, rvec, tvec, K, None)
        projected = projected.reshape(-1, 2).astype(np.int32)
        out.append(projected)
    return out


def draw_overlay(
    frame: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    *,
    line_thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with projected pitch lines drawn on top.

    Args:
        frame: BGR source image (H, W, 3).
        K, rvec, tvec: per-frame calibration.
        line_thickness: stroke width in pixels.
        label: optional short text drawn in the top-left corner —
            useful for tagging frames with `"shot_id frame N"`.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    polylines = project_pitch_lines(K, rvec, tvec, (w, h))

    for colour, poly in zip(_LINE_COLOURS_BGR, polylines):
        if poly.shape[0] < 2:
            continue
        # cv2.polylines wants (N, 1, 2) int32
        cv2.polylines(
            out,
            [poly.reshape(-1, 1, 2)],
            isClosed=False,
            color=colour,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    if label:
        cv2.rectangle(out, (8, 8), (8 + 11 * len(label), 32), (0, 0, 0), -1)
        cv2.putText(
            out, label, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def render_shot_overlays(
    output_dir: Path,
    shot_id: str,
    *,
    n_frames: int = 6,
    line_thickness: int = 2,
    cal_dir: Path | None = None,
    debug_root: Path | None = None,
) -> list[Path]:
    """Render annotated frames for a shot's calibration.

    Picks ``n_frames`` evenly-spaced keyframes from the shot's
    calibration and writes one JPEG per keyframe to
    ``<debug_root>/<shot_id>/``.  Returns the list of written paths so
    callers can surface them in the web UI.

    By default reads ``output_dir/calibration/<shot_id>_calibration.json``
    and writes to ``output_dir/calibration/debug/<shot_id>/``.  Pass
    ``cal_dir`` to read a calibration from a backend-specific subdir
    (e.g. ``output_dir/calibration/tvcalib``), and ``debug_root`` to
    write overlays beside that subdir.
    """
    if cal_dir is None:
        cal_dir = output_dir / "calibration"
    if debug_root is None:
        debug_root = cal_dir / "debug"

    cal_path = cal_dir / f"{shot_id}_calibration.json"
    if not cal_path.exists():
        logger.warning("calibration_debug: no calibration for %s", shot_id)
        return []
    cal = CalibrationResult.load(cal_path)
    if not cal.frames:
        logger.warning("calibration_debug: %s has empty calibration", shot_id)
        return []

    clip_path = output_dir / "shots" / f"{shot_id}.mp4"
    if not clip_path.exists():
        logger.warning("calibration_debug: clip not found %s", clip_path)
        return []

    debug_dir = debug_root / shot_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale frames so a re-render doesn't leave old ones behind.
    for stale in debug_dir.glob("*.jpg"):
        stale.unlink()

    # Pick n_frames evenly-spaced keyframes from the calibration.  The
    # calibration only stores frames where PnLCalib succeeded; we want
    # to inspect those specifically so the overlay shows what the
    # calibration *actually* believes.
    n_kf = len(cal.frames)
    if n_kf <= n_frames:
        chosen = list(range(n_kf))
    else:
        chosen = np.linspace(0, n_kf - 1, n_frames).round().astype(int).tolist()

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.warning("calibration_debug: cannot open %s", clip_path)
        return []

    written: list[Path] = []
    try:
        for ci in chosen:
            cf = cal.frames[ci]
            cap.set(cv2.CAP_PROP_POS_FRAMES, cf.frame)
            ok, frame = cap.read()
            if not ok:
                logger.debug(
                    "calibration_debug: %s frame %d unreadable", shot_id, cf.frame,
                )
                continue
            K = np.asarray(cf.intrinsic_matrix, dtype=np.float64)
            rvec = np.asarray(cf.rotation_vector, dtype=np.float64)
            tvec = np.asarray(cf.translation_vector, dtype=np.float64)
            annotated = draw_overlay(
                frame, K, rvec, tvec,
                line_thickness=line_thickness,
                label=f"{shot_id} f{cf.frame}",
            )
            out_path = debug_dir / f"frame_{cf.frame:05d}.jpg"
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            written.append(out_path)
    finally:
        cap.release()
    return written

"""Detect painted lines on every frame of a shot, persist to
output/camera/<shot_id>_detected_lines.json.

The detector is bootstrapped from the existing camera_track.json. For each
frame:
  1. Read frame.
  2. Use the existing per-frame K, R, t as bootstrap camera.
  3. Iterate detection ↔ camera-refine (line+point hint solve) until line
     RMS stops improving or we converge.
  4. Save detected lines + per-frame refined camera + line RMS.

Output schema:
  {
    "shot_id": "gberch",
    "image_size": [1920, 1080],
    "fps": 30.0,
    "frames": {
      "0": {
        "lines": [
          {"name": "left_18yd_front", "image_segment": [[u1,v1],[u2,v2]],
           "world_segment": [[x1,y1,z1],[x2,y2,z2]], "confidence": 0.97,
           "n_samples": 143}, ...
        ],
        "line_rms_px": 1.18,
        "K": [[fx, 0, cx], [0, fx, cy], [0, 0, 1]],
        "R": [[..], [..], [..]],
        "t": [tx, ty, tz]
      }, ...
    }
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.schemas.anchor import AnchorSet, LineObservation
from src.utils.anchor_solver import (
    _is_rich,
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


def _is_pitch_line(segment) -> bool:
    a, b = segment
    return all(
        0.0 <= p[0] <= 105.0 and 0.0 <= p[1] <= 68.0 and p[2] == 0.0
        for p in (a, b)
    )


def _refine_camera_one_frame(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    point_hint_landmarks: list | None,
    pitch_lines: dict,
    detector_cfg: DetectorConfig,
    max_iters: int = 4,
):
    """Detect-and-solve loop for a single frame.

    Returns (best_K, best_R, best_t, best_dets, best_line_rms) — the
    iteration with lowest line RMS (or the input camera if detection fails).
    """
    cx, cy = float(K[0, 2]), float(K[1, 2])
    best = (float("inf"), K.copy(), R.copy(), t.copy(), [])
    for it in range(max_iters):
        all_dets = detect_painted_lines_in_frame(
            frame_bgr, K, R, t, distortion, pitch_lines, detector_cfg,
        )
        dets = [d for d in all_dets if d.confidence >= 0.5 and d.n_samples >= 40]
        if len(dets) < 2:
            break
        line_obs = [
            LineObservation(name=d.name, image_segment=d.image_segment,
                            world_segment=d.world_segment)
            for d in dets
        ]

        def res(p, lo=line_obs, hint=point_hint_landmarks, dist=distortion):
            rv = p[0:3]; tv = p[3:6]; fx = float(p[6])
            Rm, _ = cv2.Rodrigues(rv); Km = _make_K(fx, cx, cy)
            parts = [_line_residuals(lo, Km, Rm, tv)]
            if hint:
                parts.append(0.3 * _point_residuals_distorted(
                    hint, Km, rv, tv, dist,
                ))
            return np.concatenate(parts)

        rvec_init, _ = cv2.Rodrigues(R)
        p0 = np.concatenate([rvec_init.reshape(3), t, [float(K[0, 0])]])
        fx0 = float(K[0, 0])
        lower = np.array([-np.pi]*3 + [-300]*3 + [fx0 * 0.5])
        upper = np.array([np.pi]*3 + [300]*3 + [fx0 * 2.0])
        try:
            result = least_squares(
                res, p0, bounds=(lower, upper),
                method="trf", max_nfev=2000, loss="huber", f_scale=2.0,
            )
        except Exception as exc:
            logger.warning("frame solve failed: %s", exc)
            break
        R, _ = cv2.Rodrigues(result.x[0:3])
        t = result.x[3:6]
        K = _make_K(float(result.x[6]), cx, cy)
        line_rms = float(np.sqrt((_line_residuals(line_obs, K, R, t) ** 2).mean()))
        if line_rms < best[0]:
            best = (line_rms, K.copy(), R.copy(), t.copy(), list(dets))
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path, help="MP4 clip")
    parser.add_argument("bootstrap_camera", type=Path,
                        help="Existing <shot>_camera_track.json for bootstrap")
    parser.add_argument("anchors", type=Path,
                        help="Anchor JSON (used for point hints on anchor frames "
                             "and for sanity checks)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Write detected_lines.json here")
    parser.add_argument("--strip-px", type=int, default=25)
    parser.add_argument("--min-gradient", type=float, default=10.0)
    parser.add_argument("--every-nth", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with open(args.bootstrap_camera) as f:
        track = json.load(f)
    distortion = tuple(track.get("distortion", [0.0, 0.0]))
    image_size = tuple(track["image_size"])
    fps = float(track.get("fps", 30.0))
    cameras_by_frame = {fr["frame"]: fr for fr in track["frames"]}

    aset = AnchorSet.load(args.anchors)
    anchor_landmarks = {
        a.frame: list(a.landmarks) for a in aset.anchors if a.landmarks
    }

    pitch_lines = {
        n: s for n, s in LINE_CATALOGUE.items() if _is_pitch_line(s)
    }

    cfg = DetectorConfig(
        search_strip_px=args.strip_px,
        min_gradient=args.min_gradient,
    )

    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        raise RuntimeError(f"can't open {args.clip}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("processing %d frames", n_frames)

    out_frames: dict[str, dict] = {}
    sum_rms = 0.0; n_with_lines = 0
    for fi in range(0, n_frames, args.every_nth):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        if fi not in cameras_by_frame:
            continue
        cam = cameras_by_frame[fi]
        K = np.array(cam["K"]); R = np.array(cam["R"]); t = np.array(cam["t"])
        line_rms, K_r, R_r, t_r, dets = _refine_camera_one_frame(
            frame, K, R, t, distortion,
            anchor_landmarks.get(fi),
            pitch_lines, cfg,
        )
        if not dets:
            continue
        out_frames[str(fi)] = {
            "lines": [
                {
                    "name": d.name,
                    "image_segment": [list(d.image_segment[0]), list(d.image_segment[1])],
                    "world_segment": [list(d.world_segment[0]), list(d.world_segment[1])],
                    "confidence": float(d.confidence),
                    "n_samples": int(d.n_samples),
                }
                for d in dets
            ],
            "line_rms_px": float(line_rms),
            "K": K_r.tolist(),
            "R": R_r.tolist(),
            "t": t_r.tolist(),
        }
        sum_rms += line_rms
        n_with_lines += 1
        if fi % 30 == 0:
            print(f"  frame {fi:4d}: {len(dets):2d} lines  line RMS = {line_rms:5.2f} px")
    cap.release()

    args.output.write_text(json.dumps({
        "shot_id": args.clip.stem,
        "image_size": list(image_size),
        "fps": fps,
        "frames": out_frames,
    }))
    mean_rms = sum_rms / max(n_with_lines, 1)
    print(f"\nProcessed {n_with_lines}/{n_frames} frames with detected lines.")
    print(f"Mean line RMS across frames: {mean_rms:.3f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())

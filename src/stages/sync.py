from pathlib import Path

import cv2
import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr

from src.pipeline.base import BaseStage
from src.schemas.calibration import CameraFrame, CalibrationResult
from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import Alignment, SyncMap
from src.utils.ball_detector import BallDetector, YOLOBallDetector
from src.utils.camera import project_to_pitch


def project_ball_to_pitch(
    pixel: np.ndarray, cam_frame: CameraFrame
) -> np.ndarray | None:
    """Project a ball pixel position onto the pitch ground plane using calibration."""
    K = np.array(cam_frame.intrinsic_matrix, dtype=np.float32)
    rvec = np.array(cam_frame.rotation_vector, dtype=np.float32)
    tvec = np.array(cam_frame.translation_vector, dtype=np.float32)
    return project_to_pitch(pixel, K, rvec, tvec)


def cross_correlate_trajectories(
    traj_a: np.ndarray, traj_b: np.ndarray
) -> tuple[int, float]:
    """
    Find the integer frame offset of traj_b relative to traj_a via cross-correlation.

    Returns (offset, confidence) where:
      offset > 0  -> traj_b lags traj_a (traj_b event occurs later)
      offset < 0  -> traj_b leads traj_a (traj_b event occurs earlier)
    Convention: traj_b frame + offset = corresponding traj_a frame.

    Confidence is the Pearson r of the two trajectories at the best lag,
    clamped to [0, 1].  Values near 1.0 indicate a strong, clean match;
    values below ~0.4 suggest the signals are largely uncorrelated.
    """
    norm_a = np.linalg.norm(traj_a)
    norm_b = np.linalg.norm(traj_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0, 0.0

    a = traj_a / norm_a
    b = traj_b / norm_b
    # correlate(b, a): peak at lag L means b[t] ~ a[t-L], so b lags a by L frames
    # For correlate(b, a, "full"), the i-th lag = i - (len(a) - 1)
    corr = correlate(b, a, mode="full")
    lags = np.arange(-(len(a) - 1), len(b))
    peak_idx = int(np.argmax(corr))
    offset = int(lags[peak_idx])

    # Compute Pearson r of the aligned overlap as confidence
    # offset > 0: traj_b lags traj_a -> a[0:] aligns with b[offset:]
    if offset >= 0:
        aligned_a = a[: len(a) - offset] if offset < len(a) else a[:0]
        aligned_b = b[offset: offset + len(aligned_a)]
    else:
        aligned_b = b[: len(b) + offset] if -offset < len(b) else b[:0]
        aligned_a = a[-offset: -offset + len(aligned_b)]

    overlap = min(len(aligned_a), len(aligned_b))
    aligned_a = aligned_a[:overlap]
    aligned_b = aligned_b[:overlap]

    if overlap < 2 or np.std(aligned_a) < 1e-8 or np.std(aligned_b) < 1e-8:
        return offset, 0.0

    r, _ = pearsonr(aligned_a, aligned_b)
    confidence = float(max(0.0, r))
    return offset, min(1.0, confidence)


def _extract_ball_trajectory(
    clip_path: Path,
    calibration: CalibrationResult,
    detector: BallDetector,
) -> np.ndarray:
    """
    Run ball detector on every frame of a clip.
    Returns (N,) array of x-position in pitch coordinates (NaN where undetected).
    """
    cap = cv2.VideoCapture(str(clip_path))
    positions: list[float] = []
    frame_idx = 0

    cal_map = {f.frame: f for f in calibration.frames}
    last_cal: CameraFrame | None = (
        calibration.frames[0] if calibration.frames else None
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in cal_map:
            last_cal = cal_map[frame_idx]

        ball_px = detector.detect(frame)
        if ball_px is not None and last_cal is not None:
            pitch_pos = project_ball_to_pitch(np.array(ball_px), last_cal)
            positions.append(float(pitch_pos[0]) if pitch_pos is not None else float("nan"))
        else:
            positions.append(float("nan"))

        frame_idx += 1

    cap.release()
    return np.array(positions, dtype=float)


class TemporalSyncStage(BaseStage):
    name = "sync"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        ball_detector: BallDetector | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.ball_detector = ball_detector

    def is_complete(self) -> bool:
        return (self.output_dir / "sync" / "sync_map.json").exists()

    def run(self) -> None:
        sync_dir = self.output_dir / "sync"
        sync_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("sync", {})
        min_conf = cfg.get("min_confidence", 0.4)

        shots_dir = self.output_dir / "shots"
        manifest_path = shots_dir / "shots_manifest.json"
        if not manifest_path.exists():
            print("  -> inferred shots_manifest.json from prepared clips")
        manifest = ShotsManifest.load_or_infer(shots_dir, persist=True)
        if len(manifest.shots) < 2:
            ref = manifest.shots[0].id if manifest.shots else ""
            SyncMap(reference_shot=ref).save(sync_dir / "sync_map.json")
            print("  -> only one shot; no sync needed")
            return

        detector = self.ball_detector or YOLOBallDetector(
            model_name=self.config.get("detection", {}).get("ball_model", "yolov8n.pt"),
            confidence=self.config.get("detection", {}).get("confidence_threshold", 0.3),
        )

        cal_dir = self.output_dir / "calibration"
        calibrations: dict[str, CalibrationResult] = {}
        for shot in manifest.shots:
            cal_file = cal_dir / f"{shot.id}_calibration.json"
            if cal_file.exists():
                calibrations[shot.id] = CalibrationResult.load(cal_file)

        trajectories: dict[str, np.ndarray] = {}
        for shot in manifest.shots:
            clip_path = self.output_dir / shot.clip_file
            cal = calibrations.get(
                shot.id,
                CalibrationResult(shot_id=shot.id, camera_type="static", frames=[]),
            )
            trajectories[shot.id] = _extract_ball_trajectory(clip_path, cal, detector)
            detected = int(np.sum(~np.isnan(trajectories[shot.id])))
            print(f"  -> {shot.id}: {detected} ball detections")

        reference = manifest.shots[0].id
        ref_traj = trajectories[reference]
        alignments: list[Alignment] = []

        for shot in manifest.shots[1:]:
            traj = trajectories[shot.id]
            a = np.nan_to_num(ref_traj)
            b = np.nan_to_num(traj)
            offset, confidence = cross_correlate_trajectories(a, b)

            start = max(0, offset)
            end = min(len(ref_traj), offset + len(traj))
            overlap = max(0, end - start)

            method = "ball_trajectory" if confidence >= min_conf else "low_confidence"
            alignments.append(Alignment(
                shot_id=shot.id,
                frame_offset=offset,
                confidence=confidence,
                method=method,
                overlap_frames=[start, end],
            ))
            flag = "" if confidence >= min_conf else " [WARNING] low confidence"
            print(f"  -> {shot.id} offset={offset:+d} frames, confidence={confidence:.2f}{flag}")

        SyncMap(reference_shot=reference, alignments=alignments).save(
            sync_dir / "sync_map.json"
        )

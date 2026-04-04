from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CameraFrame, CalibrationResult
from src.schemas.shots import ShotsManifest
from src.utils.camera import reprojection_error
from src.utils.pitch import FIFA_LANDMARKS


class PitchKeypointDetector(ABC):
    """Detects pitch landmark keypoints in a video frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Returns {landmark_name: (u, v)} for landmarks detected in this frame.
        Only includes landmarks detected with sufficient confidence.
        """
        ...


def calibrate_frame(
    correspondences: dict[str, np.ndarray],
    landmarks_3d: dict[str, np.ndarray],
    image_shape: tuple[int, int],  # (height, width)
    frame_idx: int = 0,
) -> CameraFrame | None:
    """
    Solve camera pose from 2D-3D pitch correspondences.
    Returns None if fewer than 4 common points or solvePnP fails.
    """
    common = [k for k in correspondences if k in landmarks_3d]
    if len(common) < 4:
        return None

    pts_2d = np.array([correspondences[k] for k in common], dtype=np.float32)
    pts_3d = np.array([landmarks_3d[k] for k in common], dtype=np.float32)

    h, w = image_shape
    cx, cy = w / 2.0, h / 2.0

    # Try several focal length candidates; broadcast cameras typically 800-2500 px
    diagonal = float(np.sqrt(h ** 2 + w ** 2))
    focal_candidates = [diagonal * s for s in (0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4)]

    best_K: np.ndarray | None = None
    best_rvec: np.ndarray | None = None
    best_tvec: np.ndarray | None = None
    best_idx: np.ndarray | None = None
    best_err = float("inf")

    for fx in focal_candidates:
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float32)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            K,
            None,
            reprojectionError=8.0,
            confidence=0.99,
            iterationsCount=5000,
        )
        if not success or inliers is None or len(inliers) < 4:
            continue
        idx = inliers.flatten()
        err = reprojection_error(pts_3d[idx], pts_2d[idx], K, rvec, tvec)
        if err < best_err:
            best_err = err
            best_K = K
            best_rvec = rvec
            best_tvec = tvec
            best_idx = idx

    if best_K is None or best_idx is None:
        return None

    K = best_K
    rvec = best_rvec
    tvec = best_tvec
    idx = best_idx
    err = best_err
    confidence = float(max(0.0, 1.0 - err / 15.0))

    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.flatten().tolist(),
        translation_vector=tvec.flatten().tolist(),
        reprojection_error=float(err),
        num_correspondences=int(len(idx)),
        confidence=confidence,
    )


class CameraCalibrationStage(BaseStage):
    name = "calibration"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        detector: PitchKeypointDetector | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.detector = detector  # None = skip per-frame detection (no calibration frames produced)

    def is_complete(self) -> bool:
        cal_dir = self.output_dir / "calibration"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return False
        manifest = ShotsManifest.load(manifest_path)
        return all(
            (cal_dir / f"{shot.id}_calibration.json").exists()
            for shot in manifest.shots
        )

    def run(self) -> None:
        cal_dir = self.output_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("calibration", {})
        keyframe_interval = cfg.get("keyframe_interval", 5)
        max_err = cfg.get("max_reprojection_error", 15.0)

        manifest = ShotsManifest.load(
            self.output_dir / "shots" / "shots_manifest.json"
        )

        for shot in manifest.shots:
            result = self._calibrate_shot(shot.id, shot.clip_file, keyframe_interval, max_err)
            result.save(cal_dir / f"{shot.id}_calibration.json")
            good = sum(1 for f in result.frames if f.reprojection_error <= max_err)
            flag = " (no calibration frames)" if not result.frames else ""
            print(f"  -> {shot.id}: {good}/{len(result.frames)} frames calibrated{flag}")

    def _calibrate_shot(
        self, shot_id: str, clip_file: str, keyframe_interval: int, max_err: float
    ) -> CalibrationResult:
        clip_path = self.output_dir / clip_file
        cap = cv2.VideoCapture(str(clip_path))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frames: list[CameraFrame] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % keyframe_interval == 0 and self.detector is not None:
                correspondences = self.detector.detect(frame)
                cf = calibrate_frame(
                    correspondences, FIFA_LANDMARKS, (h, w), frame_idx
                )
                if cf is not None and cf.reprojection_error <= max_err:
                    frames.append(cf)
            frame_idx += 1

        cap.release()
        camera_type = "tracking" if len(frames) > 1 else "static"
        return CalibrationResult(shot_id=shot_id, camera_type=camera_type, frames=frames)

"""Stage 2: Camera calibration via PnLCalib neural inference.

Broadcast cameras are static within a shot — they sit on a fixed tripod and
only pan, tilt, and zoom.  This stage:

1. Samples keyframes evenly through each shot's clip.
2. Runs PnLCalib on each keyframe to recover ``(K, R, t)``.
3. Computes a robust median camera position across keyframes.
4. Re-anchors each keyframe's translation to the shared median position,
   keeping the per-frame rotation and focal length (which reflect panning
   and zoom within the shot).
5. Writes ``CalibrationResult``.

If PnLCalib fails on every keyframe of a shot (e.g., extreme behind-goal
replay angles), the shot is given an empty calibration and a warning is
logged.  Downstream stages handle empty calibrations gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CalibrationResult, CameraFrame
from src.schemas.shots import ShotsManifest
from src.utils.neural_calibrator import NeuralCalibration, PnLCalibrator

logger = logging.getLogger(__name__)


# Physical plausibility bounds for a broadcast camera, used to filter noisy
# PnLCalib outputs before fusing.  PnLCalib occasionally produces numerically
# valid but physically impossible results (camera underground, fx of 5 or
# 51943, etc.); those samples must not feed into the median.
_DEFAULT_BOUNDS = {
    "x": (-30.0, 135.0),   # 30 m beyond either touchline
    "y": (-60.0, 130.0),   # 60 m behind the near / far touchlines
    "z": (3.0, 80.0),      # typical broadcast elevation range
    "fx": (500.0, 15000.0),
}


def _is_plausible(
    calibration: NeuralCalibration,
    bounds: dict[str, tuple[float, float]],
) -> bool:
    """Return True if a :class:`NeuralCalibration` looks physically plausible.

    Checks camera world position, focal length and optical-axis direction
    against configurable bounds.
    """
    pos = calibration.world_position
    x_lo, x_hi = bounds["x"]
    y_lo, y_hi = bounds["y"]
    z_lo, z_hi = bounds["z"]
    fx_lo, fx_hi = bounds["fx"]

    if not (x_lo <= pos[0] <= x_hi):
        return False
    if not (y_lo <= pos[1] <= y_hi):
        return False
    if not (z_lo <= pos[2] <= z_hi):
        return False

    fx = float(calibration.K[0, 0])
    if not (fx_lo <= fx <= fx_hi):
        return False

    # Optical axis must point downward in world z (camera looks at pitch).
    R, _ = cv2.Rodrigues(np.asarray(calibration.rvec, dtype=np.float64))
    if R[:, 2][2] > 0:
        return False

    return True


def _median_absolute_deviation(arr: np.ndarray) -> np.ndarray:
    """Column-wise median absolute deviation."""
    med = np.median(arr, axis=0)
    return np.median(np.abs(arr - med), axis=0)


def _robust_median_position(
    positions: list[np.ndarray],
) -> np.ndarray | None:
    """Median camera world position across keyframes, filtering MAD outliers.

    Returns ``None`` if the input list is empty.  For ≤ 2 samples the plain
    median is returned.  Otherwise samples more than 3 MADs from the median
    on any axis are dropped before taking a second median.
    """
    if not positions:
        return None
    arr = np.asarray(positions, dtype=np.float64)
    if len(arr) <= 2:
        return np.median(arr, axis=0)

    med = np.median(arr, axis=0)
    mad = _median_absolute_deviation(arr)
    # A MAD of zero means all samples agree on that axis; clip to a tiny
    # positive value so the division below keeps those samples.
    mad_clipped = np.where(mad > 1e-6, mad, 1e-6)
    deviation = np.abs(arr - med) / mad_clipped
    mask = np.all(deviation < 3.0, axis=1)
    if not np.any(mask):
        return med
    return np.median(arr[mask], axis=0)


def _neural_to_cam_frame(
    calibration: NeuralCalibration,
    frame_idx: int,
    override_position: np.ndarray | None = None,
) -> CameraFrame:
    """Convert a :class:`NeuralCalibration` to a :class:`CameraFrame`.

    When ``override_position`` is provided, the translation is recomputed so
    the camera sits exactly at that world position while keeping the
    per-frame rotation and intrinsics from PnLCalib.  This is how the static
    camera fuser enforces a shared position across keyframes.
    """
    K = np.asarray(calibration.K, dtype=np.float64)
    rvec = np.asarray(calibration.rvec, dtype=np.float64).reshape(3)
    if override_position is not None:
        R, _ = cv2.Rodrigues(rvec)
        position = np.asarray(override_position, dtype=np.float64).reshape(3)
        tvec = (-R @ position).astype(np.float64)
    else:
        tvec = np.asarray(calibration.tvec, dtype=np.float64).reshape(3)

    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.tolist(),
        translation_vector=tvec.tolist(),
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
        tracked_landmark_types=[],
    )


def _compute_keyframes(
    total_frames: int,
    keyframe_interval: int,
    max_keyframes: int,
) -> list[int]:
    """Sample keyframes from ``[0, total_frames)``.

    Chooses an interval that is at least ``keyframe_interval`` but never
    produces more than ``max_keyframes`` samples.  Short shots fall back to
    the configured interval.
    """
    if total_frames <= 0:
        return []
    effective = max(keyframe_interval, 1)
    if max_keyframes > 0:
        # Ceil division so (total_frames / max_keyframes) is an upper bound
        # on the resulting sample count.
        interval_from_cap = -(-total_frames // max_keyframes)
        effective = max(effective, interval_from_cap)
    return list(range(0, total_frames, effective))


class CameraCalibrationStage(BaseStage):
    name = "calibration"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        neural_calibrator: PnLCalibrator | None = None,
        **_: object,
    ) -> None:
        super().__init__(config, output_dir)
        self._calibrator_ext = neural_calibrator
        self._calibrator_cache: PnLCalibrator | None = None

    def _calibrator(self) -> PnLCalibrator:
        """Return the :class:`PnLCalibrator` instance, constructing one lazily
        if none was injected.
        """
        if self._calibrator_ext is not None:
            return self._calibrator_ext
        if self._calibrator_cache is None:
            cfg = self.config.get("calibration", {})
            self._calibrator_cache = PnLCalibrator(
                device=str(cfg.get("device", "auto")),
                kp_threshold=float(cfg.get("kp_threshold", 0.3434)),
                line_threshold=float(cfg.get("line_threshold", 0.7867)),
                pnl_refine=bool(cfg.get("pnl_refine", True)),
            )
        return self._calibrator_cache

    def is_complete(self) -> bool:
        cal_dir = self.output_dir / "calibration"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = ShotsManifest.load(manifest_path)
        except Exception:
            return False
        return all(
            (cal_dir / f"{shot.id}_calibration.json").exists()
            for shot in manifest.shots
        )

    def run(self) -> None:
        cal_dir = self.output_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("calibration", {})
        keyframe_interval = int(cfg.get("keyframe_interval", 30))
        max_keyframes = int(cfg.get("max_keyframes_per_shot", 10))
        bounds = self._resolve_bounds(cfg.get("plausibility_bounds"))

        shots_dir = self.output_dir / "shots"
        manifest_path = shots_dir / "shots_manifest.json"
        if not manifest_path.exists():
            print("  -> inferred shots_manifest.json from prepared clips")
        manifest = ShotsManifest.load_or_infer(shots_dir, persist=True)

        for shot in manifest.shots:
            result = self._calibrate_shot(
                shot.id,
                shot.clip_file,
                keyframe_interval=keyframe_interval,
                max_keyframes=max_keyframes,
                bounds=bounds,
            )
            result.save(cal_dir / f"{shot.id}_calibration.json")
            flag = " (no calibration frames — PnLCalib failed)" if not result.frames else ""
            print(f"  -> {shot.id}: {len(result.frames)} frames calibrated{flag}")

    @staticmethod
    def _resolve_bounds(
        override: dict | None,
    ) -> dict[str, tuple[float, float]]:
        """Merge user-provided plausibility bounds on top of the defaults."""
        resolved: dict[str, tuple[float, float]] = {
            k: tuple(v) for k, v in _DEFAULT_BOUNDS.items()
        }
        if override:
            for key, value in override.items():
                if key in resolved and isinstance(value, (list, tuple)) and len(value) == 2:
                    resolved[key] = (float(value[0]), float(value[1]))
        return resolved

    def _calibrate_shot(
        self,
        shot_id: str,
        clip_file: str,
        keyframe_interval: int,
        max_keyframes: int,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> CalibrationResult:
        clip_path = self.output_dir / clip_file
        if bounds is None:
            bounds = self._resolve_bounds(None)

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            logger.warning("%s: failed to open clip %s", shot_id, clip_path)
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            keyframes = _compute_keyframes(total_frames, keyframe_interval, max_keyframes)
            if not keyframes:
                logger.warning("%s: clip has no frames", shot_id)
                return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

            calibrator = self._calibrator()
            per_frame: list[tuple[int, NeuralCalibration]] = []
            rejected_count = 0
            for frame_idx in keyframes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    logger.debug("%s: could not read frame %d", shot_id, frame_idx)
                    continue
                result = calibrator.calibrate(frame)
                if result is None:
                    logger.debug(
                        "%s: PnLCalib returned None for frame %d", shot_id, frame_idx,
                    )
                    continue
                if not _is_plausible(result, bounds):
                    rejected_count += 1
                    logger.debug(
                        "%s: implausible calibration at frame %d "
                        "(pos=%s, fx=%.0f) — dropping",
                        shot_id,
                        frame_idx,
                        result.world_position.tolist(),
                        float(result.K[0, 0]),
                    )
                    continue
                per_frame.append((frame_idx, result))
        finally:
            cap.release()

        if not per_frame:
            logger.warning(
                "%s: PnLCalib produced no plausible calibrations across %d "
                "keyframes (%d rejected on plausibility) — shot will have "
                "an empty calibration",
                shot_id,
                len(keyframes),
                rejected_count,
            )
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        positions = [c.world_position for _, c in per_frame]
        shared_position = _robust_median_position(positions)
        if shared_position is None:
            logger.warning("%s: could not compute median camera position", shot_id)
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        # Keep only frames whose per-frame world position is within the
        # consensus cluster.  PnLCalib sometimes returns plausible-but-wrong
        # results where the rotation + focal length were computed against a
        # different camera location, so naively overriding the position
        # leaves an inconsistent projection.  Drop those frames.
        cluster_tolerance = float(
            self.config.get("calibration", {}).get("cluster_tolerance", 5.0)
        )
        consistent: list[tuple[int, NeuralCalibration]] = []
        for frame_idx, cal in per_frame:
            if np.linalg.norm(cal.world_position - shared_position) <= cluster_tolerance:
                consistent.append((frame_idx, cal))
            else:
                rejected_count += 1
                logger.debug(
                    "%s: frame %d dropped — position %s differs from consensus %s",
                    shot_id,
                    frame_idx,
                    cal.world_position.tolist(),
                    shared_position.tolist(),
                )

        if not consistent:
            logger.warning(
                "%s: no keyframes survived the consensus cluster filter", shot_id,
            )
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        logger.info(
            "%s: static camera at (%.2f, %.2f, %.2f) across %d consensus "
            "keyframes (%d rejected total)",
            shot_id,
            shared_position[0],
            shared_position[1],
            shared_position[2],
            len(consistent),
            rejected_count,
        )

        frames = [
            _neural_to_cam_frame(
                calibration=cal,
                frame_idx=frame_idx,
                override_position=shared_position,
            )
            for frame_idx, cal in consistent
        ]
        return CalibrationResult(shot_id=shot_id, camera_type="static", frames=frames)

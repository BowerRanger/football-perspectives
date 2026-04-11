from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CameraFrame, CalibrationResult
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import TracksResult
from src.utils.camera import reprojection_error, camera_world_position, is_camera_valid
from src.utils.pitch import FIFA_LANDMARKS
from src.utils.player_height import score_player_heights


@dataclass(frozen=True)
class LandmarkDetection:
    uv: np.ndarray
    confidence: float
    source: str | None = None


class PitchKeypointDetector(ABC):
    """Detects pitch landmark keypoints in a video frame."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int | None = None,
        shot_id: str | None = None,
    ) -> dict[str, LandmarkDetection]:
        """
        Returns {landmark_name: LandmarkDetection} for landmarks detected in this frame.
        Only includes landmarks detected with sufficient confidence.
        """
        ...


def _normalize_correspondences(
    correspondences: dict[str, np.ndarray] | dict[str, LandmarkDetection],
) -> dict[str, LandmarkDetection]:
    """Normalize mixed correspondence inputs into validated LandmarkDetection values."""
    normalized: dict[str, LandmarkDetection] = {}
    for name, value in correspondences.items():
        if isinstance(value, LandmarkDetection):
            uv = np.asarray(value.uv, dtype=np.float32)
            conf = float(value.confidence)
            source = value.source
        else:
            uv = np.asarray(value, dtype=np.float32)
            conf = 1.0
            source = None
        if uv.shape != (2,):
            logging.debug("Skipping landmark %s due to invalid uv shape %s", name, uv.shape)
            continue
        if conf < 0.0 or conf > 1.0:
            logging.debug("Skipping landmark %s due to invalid confidence %.3f", name, conf)
            continue
        normalized[name] = LandmarkDetection(uv=uv, confidence=conf, source=source)
    return normalized


def _validate_calibration(tvec: np.ndarray, fx: float) -> bool:
    """Reject calibrations with unreasonable camera distance or focal length."""
    dist = float(np.linalg.norm(tvec))
    if dist < 5.0 or dist > 200.0:
        logging.debug("Rejected calibration: camera distance %.1fm (expected 5-200m)", dist)
        return False
    if fx < 200.0 or fx > 10000.0:
        logging.debug("Rejected calibration: focal length %.0fpx (expected 200-10000)", fx)
        return False
    return True


@dataclass(frozen=True)
class _PnPCandidate:
    rvec: np.ndarray
    tvec: np.ndarray
    K: np.ndarray
    inlier_indices: np.ndarray
    reprojection_error: float
    camera_height: float


def _generate_pnp_candidates(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    focal_candidates: list[float],
    cx: float,
    cy: float,
    ransac_reproj_threshold: float,
    min_height: float,
    max_height: float,
) -> list[_PnPCandidate]:
    """Generate valid PnP candidate solutions using multiple methods and focal lengths."""
    candidates: list[_PnPCandidate] = []
    all_idx = np.arange(len(pts_3d))

    for fx in focal_candidates:
        K32 = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float32)
        K64 = K32.astype(np.float64)
        pts_3d64 = pts_3d.astype(np.float64)
        pts_2d64 = pts_2d.astype(np.float64)

        # Method 1: RANSAC — robust to outliers
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            K32,
            None,
            reprojectionError=ransac_reproj_threshold,
            confidence=0.99,
            iterationsCount=5000,
        )
        if success and inliers is not None and len(inliers) >= 4:
            idx = inliers.flatten()
            if _validate_calibration(tvec, fx) and is_camera_valid(rvec, tvec, min_height, max_height):
                err = reprojection_error(pts_3d[idx], pts_2d[idx], K32, rvec, tvec)
                pos = camera_world_position(rvec, tvec)
                candidates.append(_PnPCandidate(
                    rvec=rvec, tvec=tvec, K=K32, inlier_indices=idx,
                    reprojection_error=err, camera_height=float(pos[2]),
                ))

        # Method 2: IPPE (planar pose estimation — handles coplanar ambiguity; returns 2 solutions)
        try:
            retval, rvecs_ippe, tvecs_ippe, errors_ippe = cv2.solvePnPGeneric(
                pts_3d64,
                pts_2d64,
                K64,
                None,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if retval > 0 and rvecs_ippe is not None:
                for rv, tv in zip(rvecs_ippe, tvecs_ippe):
                    rv = rv.reshape(3, 1)
                    tv = tv.reshape(3, 1)
                    if _validate_calibration(tv, fx) and is_camera_valid(rv, tv, min_height, max_height):
                        err = reprojection_error(pts_3d, pts_2d, K32, rv, tv)
                        pos = camera_world_position(rv, tv)
                        candidates.append(_PnPCandidate(
                            rvec=rv, tvec=tv, K=K32, inlier_indices=all_idx,
                            reprojection_error=err, camera_height=float(pos[2]),
                        ))
        except cv2.error:
            pass

        # Method 3: EPNP — efficient for n>=4 points
        try:
            success, rvec, tvec = cv2.solvePnP(
                pts_3d64,
                pts_2d64,
                K64,
                None,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if success:
                if _validate_calibration(tvec, fx) and is_camera_valid(rvec, tvec, min_height, max_height):
                    err = reprojection_error(pts_3d, pts_2d, K32, rvec, tvec)
                    pos = camera_world_position(rvec, tvec)
                    candidates.append(_PnPCandidate(
                        rvec=rvec, tvec=tvec, K=K32, inlier_indices=all_idx,
                        reprojection_error=err, camera_height=float(pos[2]),
                    ))
        except cv2.error:
            pass

    return candidates


def _score_candidate(c: _PnPCandidate, preferred_height_range: tuple[float, float] = (5.0, 80.0)) -> float:
    """Score a PnP candidate; lower score is better.

    Base score is reprojection error.  A height penalty is added when the camera
    is outside the preferred broadcast-camera height range.
    """
    score = c.reprojection_error
    lo, hi = preferred_height_range
    if c.camera_height < lo:
        score += (lo - c.camera_height) * 10.0
    elif c.camera_height > hi:
        score += (c.camera_height - hi) * 10.0
    return score


def calibrate_frame(
    correspondences: dict[str, np.ndarray] | dict[str, LandmarkDetection],
    landmarks_3d: dict[str, np.ndarray],
    image_shape: tuple[int, int],  # (height, width)
    frame_idx: int = 0,
    max_reprojection_error: float = 15.0,
    ransac_reproj_threshold: float = 40.0,
    min_camera_height: float = 3.0,
    max_camera_height: float = 80.0,
    initial_rvec: np.ndarray | None = None,
    initial_tvec: np.ndarray | None = None,
    initial_fx: float | None = None,
    focal_length_tolerance: float = 0.2,
) -> CameraFrame | None:
    """
    Solve camera pose from 2D-3D pitch correspondences using multiple PnP methods.

    Generates candidates from RANSAC, IPPE, and EPNP, filters by physical constraints
    (camera above pitch, looking downward), and selects the best by reprojection error
    and height plausibility.

    Returns None if fewer than 4 common points or no valid solution is found.
    """
    normalized = _normalize_correspondences(correspondences)
    common = [k for k in normalized if k in landmarks_3d]
    if len(common) < 4:
        return None

    pts_2d = np.array([normalized[k].uv for k in common], dtype=np.float32)
    pts_3d = np.array([landmarks_3d[k] for k in common], dtype=np.float32)

    h, w = image_shape
    cx, cy = w / 2.0, h / 2.0

    # Build focal length candidate list
    diagonal = float(np.sqrt(h ** 2 + w ** 2))
    if initial_fx is not None:
        tol = focal_length_tolerance
        focal_candidates = [
            initial_fx * (1.0 - tol),
            initial_fx,
            initial_fx * (1.0 + tol),
        ]
    else:
        focal_candidates = [diagonal * s for s in (0.25, 0.30, 0.35, 0.40, 0.47, 0.55, 0.65, 0.75, 0.85, 1.0, 1.2, 1.4)]

    candidates = _generate_pnp_candidates(
        pts_3d, pts_2d, focal_candidates, cx, cy,
        ransac_reproj_threshold, min_camera_height, max_camera_height,
    )

    # Optional temporal seeding: try solvePnP with an extrinsic guess
    if initial_rvec is not None and initial_tvec is not None:
        all_idx = np.arange(len(pts_3d))
        for fx in focal_candidates:
            K32 = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float32)
            K64 = K32.astype(np.float64)
            try:
                success, rvec, tvec = cv2.solvePnP(
                    pts_3d.astype(np.float64),
                    pts_2d.astype(np.float64),
                    K64,
                    None,
                    rvec=initial_rvec.reshape(3, 1).astype(np.float64),
                    tvec=initial_tvec.reshape(3, 1).astype(np.float64),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if success:
                    if _validate_calibration(tvec, fx) and is_camera_valid(rvec, tvec, min_camera_height, max_camera_height):
                        err = reprojection_error(pts_3d, pts_2d, K32, rvec, tvec)
                        pos = camera_world_position(rvec, tvec)
                        candidates.append(_PnPCandidate(
                            rvec=rvec, tvec=tvec, K=K32, inlier_indices=all_idx,
                            reprojection_error=err, camera_height=float(pos[2]),
                        ))
            except cv2.error:
                pass

    if not candidates:
        return None

    best = min(candidates, key=_score_candidate)

    idx = best.inlier_indices
    rvec = best.rvec
    tvec = best.tvec
    K = best.K
    err = best.reprojection_error
    tracked_landmark_types = [common[int(i)] for i in idx]
    frame_confidence = float(np.mean([normalized[k].confidence for k in common]))
    confidence = float(max(0.0, 1.0 - err / max_reprojection_error) * frame_confidence)

    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.flatten().tolist(),
        translation_vector=tvec.flatten().tolist(),
        reprojection_error=float(err),
        num_correspondences=int(len(idx)),
        confidence=confidence,
        tracked_landmark_types=tracked_landmark_types,
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
        try:
            manifest = ShotsManifest.load(manifest_path)
            return all(
                (cal_dir / f"{shot.id}_calibration.json").exists()
                for shot in manifest.shots
            )
        except Exception:
            return False

    def run(self) -> None:
        cal_dir = self.output_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("calibration", {})
        keyframe_interval = cfg.get("keyframe_interval", 5)
        max_err = float(cfg.get("max_reprojection_error", 50.0))
        ransac_thresh = float(cfg.get("ransac_reproj_threshold", 40.0))
        require_detector = bool(cfg.get("require_detector", False))

        if self.detector is None:
            if require_detector:
                raise RuntimeError(
                    "calibration.require_detector=true but no pitch keypoint detector is configured"
                )
            logging.warning(
                "No pitch keypoint detector configured; calibration outputs will contain empty frame lists"
            )

        shots_dir = self.output_dir / "shots"
        manifest_path = shots_dir / "shots_manifest.json"
        if not manifest_path.exists():
            print("  -> inferred shots_manifest.json from prepared clips")
        manifest = ShotsManifest.load_or_infer(shots_dir, persist=True)

        for shot in manifest.shots:
            result = self._calibrate_shot(shot.id, shot.clip_file, keyframe_interval, max_err, ransac_thresh)
            result.save(cal_dir / f"{shot.id}_calibration.json")
            good = sum(1 for f in result.frames if f.reprojection_error <= max_err)
            flag = " (no calibration frames)" if not result.frames else ""
            print(f"  -> {shot.id}: {good}/{len(result.frames)} frames calibrated{flag}")

    @staticmethod
    def _merge_manual_landmarks(
        manual_path: Path,
    ) -> dict[str, LandmarkDetection]:
        """Merge manual landmarks across all annotated frames.

        For landmarks annotated on multiple frames, uses the median pixel
        position to reduce annotation noise.  Returns a correspondence dict
        suitable for ``calibrate_frame``.
        """
        import json as _json

        data = _json.loads(manual_path.read_text())
        # Collect all (u, v) per landmark name across frames
        by_name: dict[str, list[tuple[float, float]]] = {}
        for _fid, pts in data.get("frames", {}).items():
            for name, pt in pts.items():
                if name not in FIFA_LANDMARKS:
                    continue
                by_name.setdefault(name, []).append((float(pt["u"]), float(pt["v"])))

        merged: dict[str, LandmarkDetection] = {}
        for name, positions in by_name.items():
            us = [p[0] for p in positions]
            vs = [p[1] for p in positions]
            merged[name] = LandmarkDetection(
                uv=np.array([float(np.median(us)), float(np.median(vs))], dtype=np.float32),
                confidence=1.0,
                source="manual_json_merged",
            )
        return merged

    def _static_calibrate(
        self,
        merged: dict[str, LandmarkDetection],
        image_shape: tuple[int, int],
        max_err: float,
    ) -> CameraFrame | None:
        """Calibrate a static camera using all merged manual landmarks.

        Tries multiple approaches in order of preference:
        1. calibrateCamera with focal length seeded from image diagonal
        2. solvePnP with best focal length candidate
        Returns the first valid result.
        """
        names = [n for n in merged if n in FIFA_LANDMARKS]
        pts_2d_flat = np.array([merged[n].uv for n in names], dtype=np.float32)
        pts_3d = np.array([FIFA_LANDMARKS[n] for n in names], dtype=np.float32)
        h, w = image_shape

        best: CameraFrame | None = None
        best_err = float("inf")

        # Approach 1: calibrateCamera with seeded focal lengths
        for focal_seed in [1200, 1500, 2000]:
            K_init = np.array([[focal_seed, 0, w / 2], [0, focal_seed, h / 2], [0, 0, 1]], dtype=np.float64)
            flags = (
                cv2.CALIB_USE_INTRINSIC_GUESS
                | cv2.CALIB_FIX_PRINCIPAL_POINT
                | cv2.CALIB_FIX_ASPECT_RATIO
                | cv2.CALIB_ZERO_TANGENT_DIST
                | cv2.CALIB_FIX_K2
                | cv2.CALIB_FIX_K3
            )
            try:
                ret, K, dc, rvecs, tvecs = cv2.calibrateCamera(
                    [pts_3d],
                    [pts_2d_flat.reshape(-1, 1, 2)],
                    (w, h),
                    K_init.copy(),
                    None,
                    flags=flags,
                )
            except cv2.error:
                continue
            if ret <= 0 or not rvecs or not tvecs:
                continue
            fx = float(K[0, 0])
            tvec = tvecs[0].flatten()
            if not _validate_calibration(tvec, fx):
                continue
            if ret < best_err:
                best_err = ret
                best = CameraFrame(
                    frame=0,
                    intrinsic_matrix=K.tolist(),
                    rotation_vector=rvecs[0].flatten().tolist(),
                    translation_vector=tvec.tolist(),
                    reprojection_error=float(ret),
                    num_correspondences=len(names),
                    confidence=float(max(0.0, 1.0 - ret / max_err)),
                    tracked_landmark_types=names,
                )

        # Approach 2: solvePnP fallback with fixed focal lengths
        if best is None:
            diagonal = float(np.sqrt(h ** 2 + w ** 2))
            for s in (0.5, 0.7, 0.85, 1.0, 1.2):
                fx = diagonal * s
                K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    pts_3d.astype(np.float64),
                    pts_2d_flat.astype(np.float64),
                    K, None,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    continue
                tvec_f = tvec.flatten()
                if not _validate_calibration(tvec_f, fx):
                    continue
                projected, _ = cv2.projectPoints(pts_3d.astype(np.float64), rvec, tvec, K, None)
                errs = np.linalg.norm(projected.reshape(-1, 2) - pts_2d_flat, axis=1)
                mean_err = float(np.mean(errs))
                if mean_err < best_err:
                    best_err = mean_err
                    best = CameraFrame(
                        frame=0,
                        intrinsic_matrix=K.astype(np.float32).tolist(),
                        rotation_vector=rvec.flatten().tolist(),
                        translation_vector=tvec_f.tolist(),
                        reprojection_error=mean_err,
                        num_correspondences=len(names),
                        confidence=float(max(0.0, 1.0 - mean_err / max_err)),
                        tracked_landmark_types=names,
                    )

        return best

    @staticmethod
    def _is_panning(manual_path: Path, threshold: float = 100.0) -> bool:
        """Detect if camera is panning by checking landmark scatter across frames."""
        import json as _json
        data = _json.loads(manual_path.read_text())
        by_name: dict[str, list[tuple[float, float]]] = {}
        for _fid, pts in data.get("frames", {}).items():
            for name, pt in pts.items():
                by_name.setdefault(name, []).append((float(pt["u"]), float(pt["v"])))
        # Check max scatter of any landmark with 2+ annotations
        for positions in by_name.values():
            if len(positions) < 2:
                continue
            us = [p[0] for p in positions]
            vs = [p[1] for p in positions]
            scatter = max(max(us) - min(us), max(vs) - min(vs))
            if scatter > threshold:
                return True
        return False

    def _per_frame_calibrate(
        self, manual_path: Path, image_shape: tuple[int, int], max_err: float, ransac_thresh: float,
    ) -> list[CameraFrame]:
        """Calibrate each annotated frame independently (for panning cameras)."""
        import json as _json
        data = _json.loads(manual_path.read_text())
        results: list[CameraFrame] = []

        for fid, pts in sorted(data.get("frames", {}).items(), key=lambda x: int(x[0])):
            correspondences: dict[str, LandmarkDetection] = {}
            for name, pt in pts.items():
                if name not in FIFA_LANDMARKS:
                    continue
                correspondences[name] = LandmarkDetection(
                    uv=np.array([float(pt["u"]), float(pt["v"])], dtype=np.float32),
                    confidence=float(pt.get("confidence", 1.0)),
                    source="manual_json",
                )
            if len(correspondences) < 4:
                continue
            cf = calibrate_frame(
                correspondences, FIFA_LANDMARKS, image_shape,
                frame_idx=int(fid),
                max_reprojection_error=max_err,
                ransac_reproj_threshold=ransac_thresh,
            )
            if cf is not None:
                results.append(cf)
        return results

    def _load_track_bboxes(self, shot_id: str, frame_idx: int) -> list[list[float]]:
        """Load player bounding boxes for a given shot and frame from tracks output.

        Returns a list of [x1, y1, x2, y2] bboxes for player/goalkeeper tracks
        visible on ``frame_idx``.  Returns an empty list if the tracks file does
        not exist or the frame has no detections.
        """
        tracks_path = self.output_dir / "tracks" / f"{shot_id}_tracks.json"
        if not tracks_path.exists():
            return []
        try:
            result = TracksResult.load(tracks_path)
        except Exception:
            return []

        bboxes: list[list[float]] = []
        for track in result.tracks:
            if track.class_name not in ("player", "goalkeeper"):
                continue
            for tf in track.frames:
                if tf.frame == frame_idx:
                    bboxes.append(list(tf.bbox))
                    break
        return bboxes

    @staticmethod
    def _passes_temporal_check(
        cf: CameraFrame,
        previous_frames: list[CameraFrame],
        max_jump: float,
    ) -> bool:
        """Return True if the camera world position is within max_jump metres of
        the most recent accepted frame.  Always returns True when previous_frames
        is empty (no prior frame to compare against).
        """
        if not previous_frames:
            return True
        prev = previous_frames[-1]
        prev_pos = camera_world_position(
            np.array(prev.rotation_vector), np.array(prev.translation_vector)
        )
        cur_pos = camera_world_position(
            np.array(cf.rotation_vector), np.array(cf.translation_vector)
        )
        jump = float(np.linalg.norm(cur_pos - prev_pos))
        if jump > max_jump:
            logging.debug(
                "Temporal check failed: camera jumped %.1fm (max %.1fm)", jump, max_jump
            )
            return False
        return True

    def _try_with_player_heights(
        self,
        shot_id: str,
        frame_idx: int,
        correspondences: dict,
        image_shape: tuple[int, int],
        max_err: float,
        ransac_thresh: float,
        min_camera_height: float,
        max_camera_height: float,
        initial_rvec: np.ndarray | None = None,
        initial_tvec: np.ndarray | None = None,
        initial_fx: float | None = None,
        focal_length_tolerance: float = 0.2,
        player_height_weight: float = 0.3,
    ) -> CameraFrame | None:
        """Calibrate a frame and score by player height plausibility.

        Runs ``calibrate_frame`` then checks implied player heights from bounding
        boxes loaded from the tracks output.  Returns the result only if the
        player height score is above a minimum threshold (or no bboxes available).
        """
        cf = calibrate_frame(
            correspondences,
            FIFA_LANDMARKS,
            image_shape,
            frame_idx=frame_idx,
            max_reprojection_error=max_err,
            ransac_reproj_threshold=ransac_thresh,
            min_camera_height=min_camera_height,
            max_camera_height=max_camera_height,
            initial_rvec=initial_rvec,
            initial_tvec=initial_tvec,
            initial_fx=initial_fx,
            focal_length_tolerance=focal_length_tolerance,
        )
        if cf is None:
            return None

        bboxes = self._load_track_bboxes(shot_id, frame_idx)
        if bboxes:
            K = np.array(cf.intrinsic_matrix, dtype=np.float64)
            rv = np.array(cf.rotation_vector, dtype=np.float64)
            tv = np.array(cf.translation_vector, dtype=np.float64)
            height_score = score_player_heights(bboxes, K, rv, tv)
            # Require at least 30% of visible players to have plausible heights
            min_score = self.config.get("calibration", {}).get(
                "player_height_min_score", 0.3
            )
            if height_score < min_score:
                logging.debug(
                    "Frame %d rejected by player height check (score=%.2f < %.2f)",
                    frame_idx,
                    height_score,
                    min_score,
                )
                return None

        return cf

    def _calibrate_frame_with_continuity(
        self,
        shot_id: str,
        frame_idx: int,
        correspondences: dict,
        image_shape: tuple[int, int],
        max_err: float,
        ransac_thresh: float,
        min_camera_height: float,
        max_camera_height: float,
        temporal_max_jump: float,
        accepted_frames: list[CameraFrame],
        focal_length_tolerance: float = 0.2,
    ) -> CameraFrame | None:
        """Calibrate a single frame with temporal seeding and continuity checking.

        Tries calibration with temporal seed first (if prior frames exist), then
        falls back to unconstrained calibration.  The result is checked against
        the temporal continuity constraint.
        """
        initial_rvec: np.ndarray | None = None
        initial_tvec: np.ndarray | None = None
        initial_fx: float | None = None

        if accepted_frames:
            prev = accepted_frames[-1]
            initial_rvec = np.array(prev.rotation_vector, dtype=np.float64)
            initial_tvec = np.array(prev.translation_vector, dtype=np.float64)
            initial_fx = float(np.array(prev.intrinsic_matrix)[0, 0])

        cf = self._try_with_player_heights(
            shot_id=shot_id,
            frame_idx=frame_idx,
            correspondences=correspondences,
            image_shape=image_shape,
            max_err=max_err,
            ransac_thresh=ransac_thresh,
            min_camera_height=min_camera_height,
            max_camera_height=max_camera_height,
            initial_rvec=initial_rvec,
            initial_tvec=initial_tvec,
            initial_fx=initial_fx,
            focal_length_tolerance=focal_length_tolerance,
        )

        if cf is None and initial_rvec is not None:
            # Fall back: unconstrained calibration (no temporal seed)
            cf = self._try_with_player_heights(
                shot_id=shot_id,
                frame_idx=frame_idx,
                correspondences=correspondences,
                image_shape=image_shape,
                max_err=max_err,
                ransac_thresh=ransac_thresh,
                min_camera_height=min_camera_height,
                max_camera_height=max_camera_height,
                initial_rvec=None,
                initial_tvec=None,
                initial_fx=None,
            )

        if cf is None:
            return None

        if not self._passes_temporal_check(cf, accepted_frames, temporal_max_jump):
            return None

        return cf

    def _calibrate_shot_from_manual(
        self,
        shot_id: str,
        manual_path: Path,
        image_shape: tuple[int, int],
        max_err: float,
        ransac_thresh: float,
        min_camera_height: float,
        max_camera_height: float,
        temporal_max_jump: float,
        focal_length_tolerance: float = 0.2,
    ) -> list[CameraFrame]:
        """Calibrate a panning shot from manual landmark annotations with temporal continuity.

        Strategy:
        1. Find the annotated frame with the most correspondences (seed frame).
        2. Calibrate the seed frame without temporal seeding.
        3. Propagate forward from seed, then backward, using each accepted frame
           as the initial guess for the next.

        Returns the calibrated frames sorted by frame index.
        """
        import json as _json

        data = _json.loads(manual_path.read_text())

        # Build per-frame correspondence dicts, ordered by frame index
        frame_corrs: list[tuple[int, dict]] = []
        for fid_str, pts in data.get("frames", {}).items():
            fid = int(fid_str)
            correspondences: dict = {}
            for name, pt in pts.items():
                if name not in FIFA_LANDMARKS:
                    continue
                correspondences[name] = LandmarkDetection(
                    uv=np.array([float(pt["u"]), float(pt["v"])], dtype=np.float32),
                    confidence=float(pt.get("confidence", 1.0)),
                    source="manual_json",
                )
            if len(correspondences) >= 4:
                frame_corrs.append((fid, correspondences))

        if not frame_corrs:
            return []

        frame_corrs.sort(key=lambda x: x[0])

        # Find seed frame (most correspondences)
        seed_idx = max(range(len(frame_corrs)), key=lambda i: len(frame_corrs[i][1]))
        seed_fid, seed_corrs = frame_corrs[seed_idx]

        # Calibrate seed frame (unconstrained)
        seed_cf = calibrate_frame(
            seed_corrs,
            FIFA_LANDMARKS,
            image_shape,
            frame_idx=seed_fid,
            max_reprojection_error=max_err,
            ransac_reproj_threshold=ransac_thresh,
            min_camera_height=min_camera_height,
            max_camera_height=max_camera_height,
        )
        if seed_cf is None:
            logging.warning("%s: seed frame %d calibration failed", shot_id, seed_fid)
            return []

        # Collect all accepted frames indexed by position in frame_corrs
        accepted_by_idx: dict[int, CameraFrame] = {seed_idx: seed_cf}

        # Propagate forward (seed_idx+1 .. end)
        prev_frames: list[CameraFrame] = [seed_cf]
        for i in range(seed_idx + 1, len(frame_corrs)):
            fid, corrs = frame_corrs[i]
            cf = self._calibrate_frame_with_continuity(
                shot_id=shot_id,
                frame_idx=fid,
                correspondences=corrs,
                image_shape=image_shape,
                max_err=max_err,
                ransac_thresh=ransac_thresh,
                min_camera_height=min_camera_height,
                max_camera_height=max_camera_height,
                temporal_max_jump=temporal_max_jump,
                accepted_frames=prev_frames,
                focal_length_tolerance=focal_length_tolerance,
            )
            if cf is not None:
                accepted_by_idx[i] = cf
                prev_frames.append(cf)

        # Propagate backward (seed_idx-1 .. 0)
        prev_frames = [seed_cf]
        for i in range(seed_idx - 1, -1, -1):
            fid, corrs = frame_corrs[i]
            cf = self._calibrate_frame_with_continuity(
                shot_id=shot_id,
                frame_idx=fid,
                correspondences=corrs,
                image_shape=image_shape,
                max_err=max_err,
                ransac_thresh=ransac_thresh,
                min_camera_height=min_camera_height,
                max_camera_height=max_camera_height,
                temporal_max_jump=temporal_max_jump,
                accepted_frames=prev_frames,
                focal_length_tolerance=focal_length_tolerance,
            )
            if cf is not None:
                accepted_by_idx[i] = cf
                prev_frames.append(cf)

        # Return frames sorted by frame index
        sorted_idxs = sorted(accepted_by_idx.keys(), key=lambda i: frame_corrs[i][0])
        return [accepted_by_idx[i] for i in sorted_idxs]

    def _calibrate_shot(
        self, shot_id: str, clip_file: str, keyframe_interval: int, max_err: float, ransac_thresh: float = 40.0,
    ) -> CalibrationResult:
        clip_path = self.output_dir / clip_file
        cfg = self.config.get("calibration", {})
        min_camera_height = float(cfg.get("min_camera_height", 3.0))
        max_camera_height = float(cfg.get("max_camera_height", 80.0))
        temporal_max_jump = float(cfg.get("temporal_max_jump", 5.0))
        focal_length_tolerance = float(cfg.get("focal_length_tolerance", 0.2))

        manual_path = self.output_dir / "calibration" / "manual_landmarks" / f"{shot_id}.json"
        if manual_path.exists():
            merged = self._merge_manual_landmarks(manual_path)
            if len(merged) >= 4:
                cap = cv2.VideoCapture(str(clip_path))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 1080
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 1920
                cap.release()

                panning = self._is_panning(manual_path)

                if panning:
                    # Panning camera: use temporal continuity across annotated frames
                    frames = self._calibrate_shot_from_manual(
                        shot_id=shot_id,
                        manual_path=manual_path,
                        image_shape=(h, w),
                        max_err=max_err,
                        ransac_thresh=ransac_thresh,
                        min_camera_height=min_camera_height,
                        max_camera_height=max_camera_height,
                        temporal_max_jump=temporal_max_jump,
                        focal_length_tolerance=focal_length_tolerance,
                    )
                    if frames:
                        print(f"     panning camera: {len(frames)} frames calibrated with temporal continuity")
                        return CalibrationResult(
                            shot_id=shot_id,
                            camera_type="tracking",
                            frames=frames,
                        )
                    logging.warning("%s: temporal calibration produced 0 frames", shot_id)
                else:
                    # Static camera: merge all landmarks and solve once
                    cf = self._static_calibrate(merged, (h, w), max_err)
                    if cf is not None:
                        print(f"     static calibration from {len(merged)} merged landmarks "
                              f"(reproj err={cf.reprojection_error:.1f}px, {cf.num_correspondences} inliers)")
                        return CalibrationResult(
                            shot_id=shot_id, camera_type="static", frames=[cf]
                        )
                    logging.warning(
                        "%s: static calibration failed with %d merged landmarks",
                        shot_id, len(merged),
                    )

        # ── Fallback: detector-based per-frame calibration with temporal seeding ──
        if self.detector is None:
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        cap = cv2.VideoCapture(str(clip_path))
        try:
            if not cap.isOpened():
                logging.warning("Failed to open clip for calibration: %s", clip_path)
                return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            frames: list[CameraFrame] = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % keyframe_interval == 0:
                    correspondences = self.detector.detect(
                        frame,
                        frame_idx=frame_idx,
                        shot_id=shot_id,
                    )
                    cf = self._calibrate_frame_with_continuity(
                        shot_id=shot_id,
                        frame_idx=frame_idx,
                        correspondences=correspondences,
                        image_shape=(h, w),
                        max_err=max_err,
                        ransac_thresh=ransac_thresh,
                        min_camera_height=min_camera_height,
                        max_camera_height=max_camera_height,
                        temporal_max_jump=temporal_max_jump,
                        accepted_frames=frames,
                        focal_length_tolerance=focal_length_tolerance,
                    )
                    if cf is not None and cf.reprojection_error <= max_err:
                        frames.append(cf)
                frame_idx += 1
        finally:
            cap.release()

        camera_type = "tracking" if len(frames) > 1 else "static"
        return CalibrationResult(shot_id=shot_id, camera_type=camera_type, frames=frames)

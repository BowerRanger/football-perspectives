import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

_VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}

from src.pipeline.base import BaseStage
from src.schemas.poses import Keypoint, PlayerPoseFrame, PlayerPoses, PosesResult
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import TracksResult
from src.utils.pose_estimator import MMPoseEstimator, PoseEstimator

_MIN_PLAYER_HEIGHT_PX = 60  # flag players occupying < 60px height as low-res


def _crop_with_padding(
    frame: np.ndarray, bbox: list[float], pad_ratio: float = 0.2
) -> tuple[np.ndarray, tuple[float, float]]:
    """Crop a player region from frame with proportional padding. Returns (crop, (ox, oy))."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad_ratio))
    cy1 = max(0, int(y1 - bh * pad_ratio))
    cx2 = min(w, int(x2 + bw * pad_ratio))
    cy2 = min(h, int(y2 + bh * pad_ratio))
    return frame[cy1:cy2, cx1:cx2], (float(cx1), float(cy1))


def smooth_keypoints(player_poses: PlayerPoses, sigma: float = 2.0) -> PlayerPoses:
    """
    Apply 1D Gaussian smoothing along the time axis for each keypoint's x and y
    coordinates. Confidence values are left unchanged.
    """
    if len(player_poses.frames) < 3:
        return player_poses
    if not player_poses.frames[0].keypoints:
        return player_poses

    n_kps = len(player_poses.frames[0].keypoints)
    xs = np.array([[kp.x for kp in f.keypoints] for f in player_poses.frames])
    ys = np.array([[kp.y for kp in f.keypoints] for f in player_poses.frames])
    xs_smooth = gaussian_filter1d(xs, sigma=sigma, axis=0)
    ys_smooth = gaussian_filter1d(ys, sigma=sigma, axis=0)

    smoothed_frames = [
        PlayerPoseFrame(
            frame=orig.frame,
            keypoints=[
                Keypoint(
                    name=orig.keypoints[j].name,
                    x=float(xs_smooth[i, j]),
                    y=float(ys_smooth[i, j]),
                    conf=orig.keypoints[j].conf,
                )
                for j in range(n_kps)
            ],
        )
        for i, orig in enumerate(player_poses.frames)
    ]
    return PlayerPoses(
        track_id=player_poses.track_id,
        player_id=player_poses.player_id,
        player_name=player_poses.player_name,
        frames=smoothed_frames,
    )


class PoseEstimationStage(BaseStage):
    name = "pose"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        pose_estimator: PoseEstimator | None = None,
        device: str | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        if device and device not in _VALID_DEVICES and not device.startswith("cuda:"):
            raise ValueError(
                f"Invalid device {device!r}. Expected one of auto, cpu, cuda, mps, or cuda:N."
            )
        self.pose_estimator = pose_estimator
        self.device = device

    def is_complete(self) -> bool:
        poses_dir = self.output_dir / "poses"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = ShotsManifest.load(manifest_path)
            return all(
                (poses_dir / f"{shot.id}_poses.json").exists()
                for shot in manifest.shots
            )
        except Exception:
            return False

    def run(self) -> None:
        poses_dir = self.output_dir / "poses"
        poses_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("pose_estimation", {})
        min_conf = cfg.get("min_confidence", 0.3)
        smooth_sigma = cfg.get("smooth_sigma", 2.0)
        model_config = cfg.get("model_config")
        checkpoint = cfg.get("checkpoint")
        config_device = cfg.get("device", "auto")

        if self.pose_estimator is not None:
            estimator = self.pose_estimator
        else:
            if not model_config:
                raise RuntimeError(
                    "pose_estimation.model_config is required when no custom pose estimator is injected."
                )
            estimator = MMPoseEstimator(
                model_config=model_config,
                checkpoint=checkpoint,
                device=self._resolve_device(config_device),
            )

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        tracks_dir = self.output_dir / "tracks"

        for shot in manifest.shots:
            tracks_path = tracks_dir / f"{shot.id}_tracks.json"
            if not tracks_path.exists():
                logging.info("  [SKIP] %s: no tracks file", shot.id)
                continue
            tracks = TracksResult.load(tracks_path)
            result = self._estimate_shot(
                shot.id, shot.clip_file, tracks, estimator, min_conf, smooth_sigma
            )
            result.save(poses_dir / f"{shot.id}_poses.json")
            logging.info("  -> %s: %d players", shot.id, len(result.players))

    def _resolve_device(self, config_device: str) -> str:
        if self.device and self.device != "auto":
            return self.device
        return config_device

    def _estimate_shot(
        self,
        shot_id: str,
        clip_file: str,
        tracks: TracksResult,
        estimator: PoseEstimator,
        min_conf: float,
        smooth_sigma: float,
    ) -> PosesResult:
        # Pre-build lookup: frame_idx -> [(track_id, TrackFrame)]
        frame_to_tracks: dict[int, list[tuple[str, object]]] = {}
        track_metadata: dict[str, tuple[str, str]] = {}  # track_id -> (player_id, player_name)
        for track in tracks.tracks:
            track_metadata[track.track_id] = (track.player_id, track.player_name)
            for tf in track.frames:
                frame_to_tracks.setdefault(tf.frame, []).append((track.track_id, tf))

        clip_path = self.output_dir / clip_file
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        player_frames: dict[str, list[PlayerPoseFrame]] = {}
        frame_idx = 0
        crops_run = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                for track_id, tf in frame_to_tracks.get(frame_idx, []):
                    x1, y1, x2, y2 = tf.bbox
                    if (y2 - y1) < _MIN_PLAYER_HEIGHT_PX:
                        continue
                    crop, offset = _crop_with_padding(frame, tf.bbox)
                    if crop.size == 0:
                        continue
                    kps = estimator.estimate(crop, offset)
                    crops_run += 1
                    # Zero out confidence below threshold (preserve array length for triangulation alignment)
                    kps_out = [
                        kp if kp.conf >= min_conf
                        else Keypoint(name=kp.name, x=kp.x, y=kp.y, conf=0.0)
                        for kp in kps
                    ]
                    player_frames.setdefault(track_id, []).append(
                        PlayerPoseFrame(frame=frame_idx, keypoints=kps_out)
                    )
                frame_idx += 1
                if frame_idx % 50 == 0:
                    pct = f"{100 * frame_idx / total_frames:.0f}%" if total_frames else f"{frame_idx}f"
                    logging.info(
                        "  [pose] %s  frame %d/%d (%s)  crops so far: %d",
                        shot_id, frame_idx, total_frames, pct, crops_run,
                    )
        finally:
            cap.release()

        players = []
        for track_id, frames in player_frames.items():
            pid, pname = track_metadata.get(track_id, ("", ""))
            pp = PlayerPoses(
                track_id=track_id,
                player_id=pid,
                player_name=pname,
                frames=frames,
            )
            pp = smooth_keypoints(pp, sigma=smooth_sigma)
            players.append(pp)

        return PosesResult(shot_id=shot_id, players=players)

"""Monocular Human Mesh Recovery stage using GVHMR.

Runs GVHMR per-view per-track to produce SMPL parameters directly from
monocular video.  Does NOT require calibration, sync, or matching — only
segmentation (shots) and tracking (bboxes).

Output is written to ``hmr/`` as one ``.npz`` per player per shot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.hmr_result import HmrPlayerTrack, HmrResult
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import TracksResult

logger = logging.getLogger(__name__)

_MIN_PLAYER_HEIGHT_PX = 60


class MonocularHMRStage(BaseStage):
    name = "hmr"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        hmr_estimator: object | None = None,
        device: str | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self._injected_estimator = hmr_estimator
        self._device_override = device

    def is_complete(self) -> bool:
        hmr_dir = self.output_dir / "hmr"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists() or not hmr_dir.exists():
            return False
        try:
            manifest = ShotsManifest.load(manifest_path)
            return all(
                any(hmr_dir.glob(f"{shot.id}_*_hmr.npz"))
                for shot in manifest.shots
            )
        except Exception:
            return False

    def run(self) -> None:
        hmr_dir = self.output_dir / "hmr"
        hmr_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("hmr", {})
        min_track_frames = cfg.get("min_track_frames", 10)
        checkpoint = cfg.get("checkpoint", "checkpoints/gvhmr/gvhmr_siga24_release.ckpt")
        config_device = cfg.get("device", "auto")
        device = self._device_override if self._device_override and self._device_override != "auto" else config_device
        static_cam = bool(cfg.get("static_cam", False))

        estimator = self._injected_estimator
        if estimator is None:
            from src.utils.gvhmr_estimator import GVHMREstimator
            estimator = GVHMREstimator(
                checkpoint=checkpoint, device=device, static_cam=static_cam,
            )

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        tracks_dir = self.output_dir / "tracks"

        for shot in manifest.shots:
            tracks_path = tracks_dir / f"{shot.id}_tracks.json"
            if not tracks_path.exists():
                logger.info("  [SKIP] %s: no tracks file", shot.id)
                continue

            tracks = TracksResult.load(tracks_path)
            clip_path = self.output_dir / shot.clip_file

            fps = _get_video_fps(clip_path)
            logger.info(
                "  [START] %s: %d tracks in input",
                shot.id, len(tracks.tracks),
            )
            result = self._process_shot(
                shot.id, clip_path, tracks, estimator, fps, min_track_frames,
                hmr_dir=hmr_dir,
            )
            logger.info(
                "  -> %s: %d players processed (out of %d tracks)",
                shot.id, len(result.players), len(tracks.tracks),
            )

    def _process_shot(
        self,
        shot_id: str,
        clip_path: Path,
        tracks: TracksResult,
        estimator: object,
        fps: float,
        min_track_frames: int,
        hmr_dir: Path | None = None,
    ) -> HmrResult:
        """Process all tracked players in a single shot.

        If ``hmr_dir`` is provided, each player's .npz is written
        immediately after that player's inference completes — so a crash
        mid-shot doesn't lose previously processed tracks, and a re-run
        skips already-completed tracks (resumable).
        """
        # Read all frames into memory (GVHMR needs full sequences)
        frames = _read_all_frames(clip_path)
        if not frames:
            return HmrResult(shot_id=shot_id, fps=fps, players=[])

        players: list[HmrPlayerTrack] = []

        for track in tracks.tracks:
            # Build per-frame bbox lookup
            frame_bboxes: dict[int, list[float]] = {}
            for tf in track.frames:
                bbox_h = tf.bbox[3] - tf.bbox[1]
                if bbox_h >= _MIN_PLAYER_HEIGHT_PX:
                    frame_bboxes[tf.frame] = tf.bbox

            if len(frame_bboxes) < min_track_frames:
                logger.info(
                    "  [SKIP] %s track %s: only %d frames meet height>=%dpx (min %d)",
                    shot_id, track.track_id, len(frame_bboxes),
                    _MIN_PLAYER_HEIGHT_PX, min_track_frames,
                )
                continue

            # Collect contiguous frames with valid bboxes
            sorted_frame_idxs = sorted(frame_bboxes.keys())
            track_frames = [frames[fi] for fi in sorted_frame_idxs if fi < len(frames)]
            track_bboxes = [frame_bboxes[fi] for fi in sorted_frame_idxs if fi < len(frames)]
            valid_frame_idxs = [fi for fi in sorted_frame_idxs if fi < len(frames)]

            if len(track_frames) < min_track_frames:
                continue

            # Resumable: skip if this track already has an .npz on disk.
            if hmr_dir is not None:
                existing = hmr_dir / f"{shot_id}_{track.track_id}_hmr.npz"
                if existing.exists():
                    logger.info(
                        "  [SKIP] %s track %s: already exists at %s",
                        shot_id, track.track_id, existing.name,
                    )
                    continue

            logger.info(
                "  [hmr] %s track %s (%s): %d frames",
                shot_id, track.track_id, track.player_name or "unnamed",
                len(track_frames),
            )

            try:
                result = estimator.estimate_sequence(
                    track_frames, track_bboxes, fps=fps
                )
            except Exception:
                logger.exception(
                    "  [FAIL] %s track %s: GVHMR inference failed",
                    shot_id, track.track_id,
                )
                continue

            n_result = result["global_orient"].shape[0]
            if n_result == 0:
                continue

            # Confidence: use detection confidence from tracks if available
            confidences = np.ones(n_result, dtype=np.float32)
            for i, fi in enumerate(valid_frame_idxs[:n_result]):
                for tf in track.frames:
                    if tf.frame == fi:
                        confidences[i] = tf.confidence
                        break

            kp2d = result.get("kp2d")
            if kp2d is None:
                kp2d = np.zeros((n_result, 17, 3), dtype=np.float32)
            player_track = HmrPlayerTrack(
                track_id=track.track_id,
                player_id=track.player_id,
                player_name=track.player_name,
                team=track.team,
                frame_indices=np.array(valid_frame_idxs[:n_result], dtype=np.int32),
                global_orient=result["global_orient"],
                body_pose=result["body_pose"],
                betas=result["betas"],
                transl=result["transl"],
                joints_3d=result["joints_3d"],
                pred_cam=result["pred_cam"],
                bbx_xys=result["bbx_xys"],
                confidences=confidences,
                kp2d=kp2d,
            )
            players.append(player_track)

            # Persist immediately so a crash mid-shot doesn't lose this
            # track's work, and a resume skips it.
            if hmr_dir is not None:
                HmrResult(
                    shot_id=shot_id, fps=fps, players=[player_track]
                ).save(hmr_dir)
                logger.info(
                    "  [save] %s track %s -> %s_%s_hmr.npz",
                    shot_id, track.track_id, shot_id, track.track_id,
                )

        return HmrResult(shot_id=shot_id, fps=fps, players=players)


def _read_all_frames(clip_path: Path) -> list[np.ndarray]:
    """Read all frames from a video file into memory."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.error("Cannot open clip: %s", clip_path)
        return []
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _get_video_fps(clip_path: Path) -> float:
    """Get FPS from a video file."""
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps

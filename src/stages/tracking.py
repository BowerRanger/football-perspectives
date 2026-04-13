import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CalibrationResult
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.camera import project_to_pitch
from src.utils.player_detector import Detection, PlayerDetector, YOLOPlayerDetector
from src.utils.team_classifier import CLIPTeamClassifier, FakeTeamClassifier, TeamClassifier

logger = logging.getLogger(__name__)


_ID_TO_CLASS = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}


def _foot_centre(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the bottom-centre pixel of a bounding box (approximate foot position)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


class PlayerTrackingStage(BaseStage):
    name = "tracking"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        player_detector: PlayerDetector | None = None,
        team_classifier: TeamClassifier | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.player_detector = player_detector
        self.team_classifier = team_classifier

    def is_complete(self) -> bool:
        tracks_dir = self.output_dir / "tracks"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = ShotsManifest.load(manifest_path)
            tracks_exist = all(
                (tracks_dir / f"{shot.id}_tracks.json").exists()
                for shot in manifest.shots
            )
            if not tracks_exist:
                return False
            # Matching output is also required when sync data exists
            sync_path = self.output_dir / "sync" / "sync_map.json"
            if sync_path.exists() and len(manifest.shots) > 1:
                return (self.output_dir / "matching" / "player_matches.json").exists()
            return True
        except Exception:
            return False

    def run(self) -> None:
        tracks_dir = self.output_dir / "tracks"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("tracking", {})
        confidence = cfg.get("confidence_threshold", 0.3)
        model_name = cfg.get("player_model", "yolov8x.pt")
        team_classifier_mode = str(cfg.get("team_classifier", "none")).strip().lower()
        default_team_label = str(cfg.get("default_team_label", "unknown")).strip() or "unknown"

        detector = self.player_detector or YOLOPlayerDetector(
            model_name=model_name, confidence=confidence
        )
        if self.team_classifier is not None:
            team_classifier = self.team_classifier
        elif team_classifier_mode == "clip":
            print("  -> team classifier: clip (slow, downloads/loads model)")
            team_classifier = CLIPTeamClassifier()
        else:
            team_classifier = FakeTeamClassifier(default_team_label)

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        cal_dir = self.output_dir / "calibration"

        for shot in manifest.shots:
            cal_path = cal_dir / f"{shot.id}_calibration.json"
            calibration = CalibrationResult.load(cal_path) if cal_path.exists() else None
            print(f"  -> tracking {shot.id}...")
            result = self._track_shot(
                shot.id, shot.clip_file, detector, team_classifier, calibration
            )
            result.save(tracks_dir / f"{shot.id}_tracks.json")
            print(f"  -> {shot.id}: {len(result.tracks)} tracks")

        # --- Cross-view matching (formerly a separate pipeline stage) ---
        self._run_matching(manifest, tracks_dir)

    def _run_matching(self, manifest: ShotsManifest, tracks_dir: Path) -> None:
        """Match player tracks across camera views if sync data is available."""
        sync_path = self.output_dir / "sync" / "sync_map.json"
        if not sync_path.exists():
            logger.info("  [matching] no sync_map.json — skipping cross-view matching")
            return
        if len(manifest.shots) < 2:
            logger.info("  [matching] single shot — skipping cross-view matching")
            return

        from src.schemas.player_matches import MatchedPlayer, PlayerMatches, PlayerView
        from src.schemas.sync_map import SyncMap
        from src.stages.matching import hungarian_match_players

        sync_map = SyncMap.load(sync_path)
        cfg = self.config.get("matching", {})
        max_distance_m = cfg.get("max_distance_m", 5.0)
        n_reference_frames = cfg.get("n_reference_frames", 10)

        tracks_by_shot: dict[str, TracksResult] = {}
        for shot in manifest.shots:
            path = tracks_dir / f"{shot.id}_tracks.json"
            if path.exists():
                tracks_by_shot[shot.id] = TracksResult.load(path)

        if not tracks_by_shot:
            logger.warning("  [matching] no track files found — player_matches will be empty")

        # Assign global player IDs starting from the reference shot
        player_counter = 0
        player_id_map: dict[tuple[str, str], str] = {}  # (shot_id, track_id) -> player_id

        ref_id = sync_map.reference_shot
        if ref_id in tracks_by_shot:
            for track in tracks_by_shot[ref_id].tracks:
                if track.class_name == "ball":
                    continue
                player_counter += 1
                pid = f"P{player_counter:03d}"
                player_id_map[(ref_id, track.track_id)] = pid

        # Match each non-reference shot to the reference
        for alignment in sync_map.alignments:
            other_id = alignment.shot_id
            if ref_id not in tracks_by_shot or other_id not in tracks_by_shot:
                continue
            overlap_start, overlap_end = alignment.overlap_frames
            if overlap_end <= overlap_start:
                continue
            step = max(1, (overlap_end - overlap_start) // n_reference_frames)
            ref_frames = list(range(overlap_start, overlap_end, step))[:n_reference_frames]
            matches = hungarian_match_players(
                tracks_by_shot[ref_id],
                tracks_by_shot[other_id],
                sync_offset=alignment.frame_offset,
                reference_frames=ref_frames,
                max_distance_m=max_distance_m,
            )
            logger.info("  [matching] %s <-> %s: %d matches", ref_id, other_id, len(matches))
            for track_id_ref, track_id_other in matches:
                pid = player_id_map.get((ref_id, track_id_ref))
                if pid is not None:
                    player_id_map[(other_id, track_id_other)] = pid

        # Build PlayerMatches output for downstream compatibility
        pid_to_views: dict[str, list[PlayerView]] = {}
        pid_to_team: dict[str, str] = {}
        for (shot_id, track_id), pid in player_id_map.items():
            pid_to_views.setdefault(pid, []).append(
                PlayerView(shot_id=shot_id, track_id=track_id)
            )
            if pid not in pid_to_team and shot_id in tracks_by_shot:
                for t in tracks_by_shot[shot_id].tracks:
                    if t.track_id == track_id:
                        pid_to_team[pid] = t.team
                        break

        matched_players = [
            MatchedPlayer(
                player_id=pid,
                team=pid_to_team.get(pid, "unknown"),
                views=views,
            )
            for pid, views in sorted(pid_to_views.items())
        ]
        matching_dir = self.output_dir / "matching"
        matching_dir.mkdir(parents=True, exist_ok=True)
        PlayerMatches(matched_players=matched_players).save(
            matching_dir / "player_matches.json"
        )
        logger.info("  [matching] %d matched players", len(matched_players))

        # Write player_id back into the track files so downstream stages can use it
        for (shot_id, track_id), pid in player_id_map.items():
            if shot_id in tracks_by_shot:
                for track in tracks_by_shot[shot_id].tracks:
                    if track.track_id == track_id:
                        track.player_id = pid
                        break

        for shot_id, tracks_result in tracks_by_shot.items():
            tracks_result.save(tracks_dir / f"{shot_id}_tracks.json")

    def _track_shot(
        self,
        shot_id: str,
        clip_file: str,
        detector: PlayerDetector,
        team_classifier: TeamClassifier,
        calibration: CalibrationResult | None,
    ) -> TracksResult:
        try:
            import supervision as sv
        except ImportError:
            raise ImportError(
                "supervision is required for tracking: pip install supervision"
            )

        clip_path = self.output_dir / clip_file
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")

        byte_tracker = sv.ByteTrack()

        cal_map = {f.frame: f for f in calibration.frames} if calibration else {}
        last_cal = (
            calibration.frames[0] if (calibration and calibration.frames) else None
        )
        pending_fit_crops: list[np.ndarray] = []
        max_fit_buffer = int(self.config.get("tracking", {}).get("max_fit_buffer", 200))
        progress_every_frames = max(1, int(self.config.get("tracking", {}).get("progress_every_frames", 150)))
        # Classifiers that do not expose fit() are considered ready by default.
        team_classifier_ready = not hasattr(team_classifier, "fit")
        active_tracks: dict[int, Track] = {}
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in cal_map:
                    last_cal = cal_map[frame_idx]

                if frame_idx > 0 and frame_idx % progress_every_frames == 0:
                    print(f"     processed {frame_idx} frames...")

                detections = detector.detect(frame)
                # Track ball alongside players: ByteTrack handles them
                # uniformly and the ball class is already in the
                # class_id mapping.  Triangulation uses class_name to
                # split ball tracks from player tracks downstream.
                player_dets = list(detections)

                if player_dets:
                    xyxy = np.array([list(d.bbox) for d in player_dets], dtype=np.float32)
                    confs = np.array([d.confidence for d in player_dets], dtype=np.float32)
                    class_ids = np.array([
                        {"player": 0, "goalkeeper": 1, "referee": 2, "ball": 3}.get(d.class_name, 0)
                        for d in player_dets
                    ], dtype=int)
                    sv_dets = sv.Detections(
                        xyxy=xyxy, confidence=confs, class_id=class_ids
                    )
                    tracked = byte_tracker.update_with_detections(sv_dets)

                    crops = []
                    for i in range(len(tracked)):
                        x1, y1, x2, y2 = tracked.xyxy[i]
                        crops.append(
                            frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                        )

                    if crops and not team_classifier_ready and hasattr(team_classifier, "fit"):
                        # Skip empty crops; these can occur near frame edges.
                        pending_fit_crops.extend(c for c in crops if c.size > 0)
                        # Bound memory while waiting for enough samples to fit.
                        if len(pending_fit_crops) > max_fit_buffer:
                            pending_fit_crops = pending_fit_crops[-max_fit_buffer:]
                        try:
                            team_classifier.fit(pending_fit_crops)
                            team_classifier_ready = True
                            pending_fit_crops.clear()
                        except Exception as exc:
                            # Some classifiers need more samples before fitting succeeds.
                            logging.debug(
                                "team classifier fit failed on frame %d (will retry): %s",
                                frame_idx,
                                exc,
                            )
                            team_classifier_ready = False

                    if crops and team_classifier_ready:
                        team_labels = team_classifier.classify(crops)
                    elif crops:
                        team_labels = ["unknown"] * len(crops)
                    else:
                        team_labels = []

                    for i, tid in enumerate(tracked.tracker_id):
                        if tid is None:
                            continue
                        x1, y1, x2, y2 = tracked.xyxy[i]
                        conf = (
                            float(tracked.confidence[i])
                            if tracked.confidence is not None
                            else 0.5
                        )
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        team = team_labels[i] if i < len(team_labels) else "unknown"
                        cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                        class_name = _ID_TO_CLASS.get(cls_id, "player")

                        foot_u, foot_v = _foot_centre((x1, y1, x2, y2))
                        pitch_pos: list[float] | None = None
                        if last_cal is not None:
                            K = np.array(last_cal.intrinsic_matrix, dtype=np.float32)
                            rvec = np.array(last_cal.rotation_vector, dtype=np.float32)
                            tvec = np.array(last_cal.translation_vector, dtype=np.float32)
                            try:
                                pp = project_to_pitch(
                                    np.array([foot_u, foot_v]), K, rvec, tvec
                                )
                                pitch_pos = [float(pp[0]), float(pp[1])]
                            except Exception as exc:
                                logging.warning(
                                    "pitch projection failed for %s frame %d: %s",
                                    shot_id, frame_idx, exc,
                                )

                        track_frame = TrackFrame(
                            frame=frame_idx,
                            bbox=bbox,
                            confidence=conf,
                            pitch_position=pitch_pos,
                        )
                        if tid not in active_tracks:
                            active_tracks[tid] = Track(
                                track_id=f"T{tid:03d}",
                                class_name=class_name,
                                team=team,
                                frames=[],
                            )
                        active_tracks[tid].frames.append(track_frame)

                frame_idx += 1
        finally:
            cap.release()
        return TracksResult(shot_id=shot_id, tracks=list(active_tracks.values()))

from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CalibrationResult
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.camera import project_to_pitch
from src.utils.player_detector import Detection, PlayerDetector, YOLOPlayerDetector
from src.utils.team_classifier import FakeTeamClassifier, TeamClassifier


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
            return all(
                (tracks_dir / f"{shot.id}_tracks.json").exists()
                for shot in manifest.shots
            )
        except Exception:
            return False

    def run(self) -> None:
        tracks_dir = self.output_dir / "tracks"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("tracking", {})
        confidence = cfg.get("confidence_threshold", 0.3)
        model_name = cfg.get("player_model", "yolov8x.pt")

        detector = self.player_detector or YOLOPlayerDetector(
            model_name=model_name, confidence=confidence
        )
        team_classifier = self.team_classifier or FakeTeamClassifier("A")

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        cal_dir = self.output_dir / "calibration"

        for shot in manifest.shots:
            cal_path = cal_dir / f"{shot.id}_calibration.json"
            calibration = CalibrationResult.load(cal_path) if cal_path.exists() else None
            result = self._track_shot(
                shot.id, shot.clip_file, detector, team_classifier, calibration
            )
            result.save(tracks_dir / f"{shot.id}_tracks.json")
            print(f"  -> {shot.id}: {len(result.tracks)} tracks")

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
        byte_tracker = sv.ByteTrack()

        cal_map = {f.frame: f for f in calibration.frames} if calibration else {}
        last_cal = (
            calibration.frames[0] if (calibration and calibration.frames) else None
        )
        active_tracks: dict[int, Track] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in cal_map:
                last_cal = cal_map[frame_idx]

            detections = detector.detect(frame)
            player_dets = [d for d in detections if d.class_name != "ball"]

            if player_dets:
                xyxy = np.array([list(d.bbox) for d in player_dets], dtype=np.float32)
                confs = np.array([d.confidence for d in player_dets], dtype=np.float32)
                class_ids = np.zeros(len(player_dets), dtype=int)
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
                team_labels = team_classifier.classify(crops) if crops else []

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
                        except Exception:
                            pass

                    track_frame = TrackFrame(
                        frame=frame_idx,
                        bbox=bbox,
                        confidence=conf,
                        pitch_position=pitch_pos,
                    )
                    if tid not in active_tracks:
                        active_tracks[tid] = Track(
                            track_id=f"T{tid:03d}",
                            class_name="player",
                            team=team,
                            frames=[],
                        )
                    active_tracks[tid].frames.append(track_frame)

            frame_idx += 1

        cap.release()
        return TracksResult(shot_id=shot_id, tracks=list(active_tracks.values()))

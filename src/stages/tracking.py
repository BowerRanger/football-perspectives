import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CalibrationResult
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.camera import project_to_pitch
from src.utils.player_detector import PlayerDetector, YOLOPlayerDetector
from src.utils.team_classifier import CLIPTeamClassifier, FakeTeamClassifier, TeamClassifier

logger = logging.getLogger(__name__)


_ID_TO_CLASS = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _foot_centre(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the bottom-centre pixel of a bounding box (approximate foot position)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def _resolve_device(spec: str) -> str:
    """Pick a torch device string. ``auto`` -> CUDA > MPS > CPU."""
    requested = (spec or "auto").strip().lower()
    if requested != "auto":
        return "cuda:0" if requested == "cuda" else requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


class _ByteTrackAdapter:
    """Motion-only tracker (supervision's ByteTrack). Fast; swaps IDs
    when bboxes overlap because there's no appearance signal."""

    name = "bytetrack"

    def __init__(self) -> None:
        import supervision as sv

        self._tracker = sv.ByteTrack()

    def update(self, sv_dets, frame):  # noqa: ARG002 (frame unused)
        return self._tracker.update_with_detections(sv_dets)


class _BotSortAdapter:
    """BoxMOT BoT-SORT with OSNet ReID embeddings. Holds IDs through
    occlusions by adding appearance similarity to the association
    cost — the practical fix for the swap problem when two players in
    clashing kits cross."""

    name = "botsort"

    def __init__(
        self,
        reid_weights_path: Path,
        device: str,
        params: dict[str, Any],
    ) -> None:
        try:
            from boxmot import BotSort
        except ImportError as exc:
            raise ImportError(
                "boxmot is required when tracking.tracker = botsort. "
                "Install with `pip install boxmot`."
            ) from exc
        if not reid_weights_path.exists():
            raise FileNotFoundError(
                f"ReID weights not found at {reid_weights_path}. "
                "Run scripts/setup_boxmot.sh to download them, or set "
                "tracking.reid_weights to an absolute path."
            )
        # Half-precision is unsafe on MPS / CPU; only enable for CUDA.
        half = device.startswith("cuda")
        self._tracker = BotSort(
            reid_weights=reid_weights_path,
            device=device,
            half=half,
            track_high_thresh=float(params.get("track_high_thresh", 0.5)),
            track_low_thresh=float(params.get("track_low_thresh", 0.1)),
            new_track_thresh=float(params.get("new_track_thresh", 0.6)),
            match_thresh=float(params.get("match_thresh", 0.8)),
            proximity_thresh=float(params.get("proximity_thresh", 0.5)),
            appearance_thresh=float(params.get("appearance_thresh", 0.25)),
            cmc_method=str(params.get("cmc_method", "ecc")),
            frame_rate=int(params.get("frame_rate", 30)),
            fuse_first_associate=bool(params.get("fuse_first_associate", False)),
            with_reid=bool(params.get("with_reid", True)),
        )

    def update(self, sv_dets, frame):
        import supervision as sv

        n = len(sv_dets)
        if n == 0:
            return sv.Detections.empty()
        # BoxMOT expects (N, 6) ndarray: [x1, y1, x2, y2, conf, cls].
        dets = np.zeros((n, 6), dtype=np.float32)
        dets[:, :4] = sv_dets.xyxy
        if sv_dets.confidence is not None:
            dets[:, 4] = sv_dets.confidence
        else:
            dets[:, 4] = 0.5
        if sv_dets.class_id is not None:
            dets[:, 5] = sv_dets.class_id
        output = self._tracker.update(dets, frame)
        if output is None or len(output) == 0:
            return sv.Detections.empty()
        # BoxMOT returns (M, 8): [x1, y1, x2, y2, track_id, conf, cls, det_idx].
        return sv.Detections(
            xyxy=output[:, :4].astype(np.float32),
            confidence=output[:, 5].astype(np.float32),
            class_id=output[:, 6].astype(int),
            tracker_id=output[:, 4].astype(int),
        )


def _build_tracker(cfg: dict) -> _ByteTrackAdapter | _BotSortAdapter:
    """Construct a per-shot tracker driven by ``tracking.tracker``.

    Per-shot construction is intentional: each shot is independent, so
    track histories must reset at clip boundaries. The BoT-SORT
    constructor reloads the OSNet weights each time — that costs a
    second or two but keeps shot isolation simple.
    """
    track_cfg = cfg.get("tracking", {}) or {}
    name = str(track_cfg.get("tracker", "botsort")).strip().lower()
    if name == "bytetrack":
        return _ByteTrackAdapter()
    if name == "botsort":
        weights = Path(track_cfg.get(
            "reid_weights",
            "third_party/boxmot/osnet_x0_25_msmt17.pt",
        ))
        if not weights.is_absolute():
            weights = _REPO_ROOT / weights
        return _BotSortAdapter(
            reid_weights_path=weights,
            device=_resolve_device(str(track_cfg.get("tracker_device", "auto"))),
            params=track_cfg.get("botsort", {}) or {},
        )
    raise ValueError(
        f"Unknown tracker: {name!r} (expected 'bytetrack' or 'botsort')"
    )


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
        iou_threshold = float(cfg.get("iou_threshold", 0.85))
        imgsz = int(cfg.get("imgsz", 1280))
        sahi_cfg = cfg.get("sahi", {}) or {}
        sahi_enabled = bool(sahi_cfg.get("enabled", False))
        sahi_tile_size = int(sahi_cfg.get("tile_size", 960))
        sahi_overlap_ratio = float(sahi_cfg.get("overlap_ratio", 0.25))
        sahi_nms_iou_threshold = float(sahi_cfg.get("nms_iou_threshold", 0.5))
        model_name = cfg.get("player_model", "yolov8x.pt")
        team_classifier_mode = str(cfg.get("team_classifier", "none")).strip().lower()
        default_team_label = str(cfg.get("default_team_label", "unknown")).strip() or "unknown"

        detector = self.player_detector or YOLOPlayerDetector(
            model_name=model_name,
            confidence=confidence,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            sahi_enabled=sahi_enabled,
            sahi_tile_size=sahi_tile_size,
            sahi_overlap_ratio=sahi_overlap_ratio,
            sahi_nms_iou_threshold=sahi_nms_iou_threshold,
        )
        if sahi_enabled:
            print(
                f"  -> SAHI tiled inference: "
                f"tile={sahi_tile_size}px, overlap={sahi_overlap_ratio}"
            )
        if self.team_classifier is not None:
            team_classifier = self.team_classifier
        elif team_classifier_mode == "clip":
            print("  -> team classifier: clip (slow, downloads/loads model)")
            team_classifier = CLIPTeamClassifier()
        else:
            team_classifier = FakeTeamClassifier(default_team_label)

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")

        for shot in manifest.shots:
            # TODO(Phase 1c): once CameraStage produces camera_track.json,
            # load it here so per-frame pitch_position can be filled in.
            calibration = None
            print(f"  -> tracking {shot.id}...")
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
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")

        tracker = _build_tracker(self.config)
        print(f"  -> tracker: {tracker.name} (shot={shot_id})")

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
                # Ball-class detections continue to flow through the
                # active tracker so the dashboard's class_name=="ball"
                # checks still work against <shot>_tracks.json.
                # BallStage owns its own detection pass and does not
                # consume them.
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
                    tracked = tracker.update(sv_dets, frame)

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

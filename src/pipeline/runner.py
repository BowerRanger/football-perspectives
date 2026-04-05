from pathlib import Path
from src.pipeline.base import BaseStage
from src.stages.segmentation import ShotSegmentationStage
from src.stages.calibration import CameraCalibrationStage
from src.stages.sync import TemporalSyncStage
from src.stages.tracking import PlayerTrackingStage
from src.stages.pose import PoseEstimationStage
from src.stages.matching import CrossViewMatchingStage
from src.stages.calibration import PitchKeypointDetector
from src.utils.ball_detector import YOLOBallDetector
from src.utils.pitch_detector import ManualJsonPitchDetector

STAGE_ORDER: list[tuple[str, type[BaseStage]]] = [
    ("segmentation", ShotSegmentationStage),
    ("calibration", CameraCalibrationStage),
    ("sync", TemporalSyncStage),
    ("tracking", PlayerTrackingStage),
    ("pose", PoseEstimationStage),
    ("matching", CrossViewMatchingStage),
]

_ALIASES: dict[str, str] = {
    "1": "segmentation",
    "2": "calibration",
    "3": "sync",
    "4": "tracking",
    "5": "pose",
    "6": "matching",
}


def _create_pitch_detector(config: dict, output_dir: Path) -> PitchKeypointDetector | None:
    cfg = config.get("calibration", {})
    detector_type = str(cfg.get("detector_type", "none")).strip().lower()
    if detector_type in {"", "none"}:
        return None
    if detector_type == "manual_json":
        landmarks_dir = cfg.get("manual_landmarks_dir")
        if not landmarks_dir:
            raise ValueError(
                "calibration.manual_landmarks_dir is required when detector_type=manual_json"
            )
        min_confidence = float(cfg.get("min_point_confidence", 0.0))
        return ManualJsonPitchDetector(
            annotations_dir=output_dir / landmarks_dir,
            min_confidence=min_confidence,
        )
    raise ValueError(f"Unknown calibration.detector_type: {detector_type!r}")


def resolve_stages(stages: str, from_stage: str | None) -> list[str]:
    all_names = [name for name, _ in STAGE_ORDER]
    if stages == "all":
        selected = all_names
    else:
        selected = []
        for token in stages.split(","):
            token = token.strip()
            name = _ALIASES.get(token, token)
            if name not in all_names:
                raise ValueError(f"Unknown stage: {token!r}")
            selected.append(name)
    if from_stage:
        canonical = _ALIASES.get(from_stage, from_stage)
        idx = all_names.index(canonical)
        selected = [n for n in selected if all_names.index(n) >= idx]
    return selected


def run_pipeline(
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config: dict,
    **stage_kwargs,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    active = resolve_stages(stages, from_stage)
    from_stage_canonical = _ALIASES.get(from_stage, from_stage) if from_stage else None
    shared_ball_detector = None
    pitch_detector = None
    if "segmentation" in active or "sync" in active:
        shot_cfg = config.get("shot_segmentation", {})
        require_ball_in_shot = bool(shot_cfg.get("require_ball_in_shot", True))
        if require_ball_in_shot or "sync" in active:
            detection_cfg = config.get("detection", {})
            ball_model = str(detection_cfg.get("ball_model", "yolov8n.pt")).strip()
            ball_confidence = float(detection_cfg.get("confidence_threshold", 0.3))
            shared_ball_detector = YOLOBallDetector(
                model_name=ball_model,
                confidence=ball_confidence,
            )
    if "calibration" in active:
        pitch_detector = _create_pitch_detector(config=config, output_dir=output_dir)
    for name, StageClass in STAGE_ORDER:
        if name not in active:
            continue
        if StageClass is None:
            print(f"  [SKIP] {name} (not yet implemented)")
            continue
        current_stage_kwargs = dict(stage_kwargs)
        if name in {"segmentation", "sync"} and shared_ball_detector is not None:
            current_stage_kwargs["ball_detector"] = shared_ball_detector
        if name == "calibration" and pitch_detector is not None:
            current_stage_kwargs["detector"] = pitch_detector
        stage = StageClass(config=config, output_dir=output_dir, **current_stage_kwargs)
        if stage.is_complete() and from_stage_canonical != name:
            print(f"  [SKIP] {name} (cached)")
            continue
        print(f"  [RUN]  {name}")
        stage.run()

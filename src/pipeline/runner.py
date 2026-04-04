from pathlib import Path
from src.pipeline.base import BaseStage
from src.stages.segmentation import ShotSegmentationStage
from src.stages.calibration import CameraCalibrationStage
from src.stages.sync import TemporalSyncStage

# Populated as stages are implemented; each entry is (canonical_name, StageClass)
STAGE_ORDER: list[tuple[str, type[BaseStage]]] = [
    ("segmentation", ShotSegmentationStage),
    ("calibration", CameraCalibrationStage),
    ("sync", TemporalSyncStage),
]

_ALIASES: dict[str, str] = {
    "1": "segmentation",
    "2": "calibration",
    "3": "sync",
}


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
    for name, StageClass in STAGE_ORDER:
        if name not in active:
            continue
        if StageClass is None:
            print(f"  [SKIP] {name} (not yet implemented)")
            continue
        stage = StageClass(config=config, output_dir=output_dir, **stage_kwargs)
        if stage.is_complete() and from_stage != _ALIASES.get(name, name):
            print(f"  [SKIP] {name} (cached)")
            continue
        print(f"  [RUN]  {name}")
        stage.run()

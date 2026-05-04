from pathlib import Path

from src.pipeline.base import BaseStage
from src.stages.calibration import CameraCalibrationStage
from src.stages.export import ExportStage
from src.stages.pose import PoseEstimationStage
from src.stages.segmentation import ShotSegmentationStage
from src.stages.smpl_fitting import SmplFittingStage
from src.stages.sync import TemporalSyncStage
from src.stages.tracking import PlayerTrackingStage
from src.stages.triangulation import TriangulationStage
from src.utils.ball_detector import YOLOBallDetector

# Architecture A — classic multi-view triangulation pipeline.
_STAGES_TRIANGULATION: list[tuple[str, type[BaseStage]]] = [
    ("segmentation", ShotSegmentationStage),
    ("tracking", PlayerTrackingStage),
    ("calibration", CameraCalibrationStage),
    ("sync", TemporalSyncStage),
    ("pose", PoseEstimationStage),
    ("triangulation", TriangulationStage),
    ("smpl_fitting", SmplFittingStage),
    ("export", ExportStage),
]

_ALIASES_TRIANGULATION: dict[str, str] = {
    "1": "segmentation",
    "2": "tracking",
    "3": "calibration",
    "4": "sync",
    "5": "pose",
    "6": "triangulation",
    "7": "smpl_fitting",
    "8": "export",
}


def _get_hmr_stages() -> list[tuple[str, type[BaseStage]]]:
    """Architecture B — monocular HMR pipeline (lazy import to avoid hard dep)."""
    from src.stages.hmr import MonocularHMRStage

    return [
        ("segmentation", ShotSegmentationStage),
        ("tracking", PlayerTrackingStage),
        ("hmr", MonocularHMRStage),
        ("export", ExportStage),
    ]


_ALIASES_HMR: dict[str, str] = {
    "1": "segmentation",
    "2": "tracking",
    "3": "hmr",
    "4": "export",
}


def _resolve_mode(config: dict) -> str:
    return config.get("pipeline", {}).get("mode", "triangulation")


def _stage_order_for(config: dict) -> list[tuple[str, type[BaseStage]]]:
    mode = _resolve_mode(config)
    if mode == "hmr":
        return _get_hmr_stages()
    return _STAGES_TRIANGULATION


def _aliases_for(config: dict) -> dict[str, str]:
    mode = _resolve_mode(config)
    if mode == "hmr":
        return _ALIASES_HMR
    return _ALIASES_TRIANGULATION


# Public API: keep STAGE_ORDER and _ALIASES for backward compatibility
# (defaults to triangulation mode).
STAGE_ORDER = _STAGES_TRIANGULATION
_ALIASES = _ALIASES_TRIANGULATION

# Stages that only exist in hmr mode — used to auto-switch mode when
# the caller asks for one of them.
_HMR_ONLY_STAGES = {"hmr"}


def _auto_switch_mode(stages: str, from_stage: str | None, config: dict) -> None:
    """Flip ``config['pipeline']['mode']`` to 'hmr' if any requested stage
    is hmr-only and the config hasn't already chosen a mode.

    Makes it possible to invoke ``--stages hmr`` without first editing the
    config, which matters for the web viewer where the user clicks a stage
    in the sidebar with no way to also toggle the mode.
    """
    if _resolve_mode(config) == "hmr":
        return  # already in hmr mode

    tokens = [from_stage] if from_stage else []
    if stages and stages != "all":
        tokens.extend(t.strip() for t in stages.split(","))

    # Resolve numeric/alias tokens against BOTH tables so "3" in hmr mode maps.
    for token in tokens:
        if token is None:
            continue
        resolved = _ALIASES_TRIANGULATION.get(token, token)
        if resolved in _HMR_ONLY_STAGES:
            config.setdefault("pipeline", {})["mode"] = "hmr"
            return
        resolved_hmr = _ALIASES_HMR.get(token, token)
        if resolved_hmr in _HMR_ONLY_STAGES:
            config.setdefault("pipeline", {})["mode"] = "hmr"
            return


def resolve_stages(
    stages: str,
    from_stage: str | None,
    config: dict | None = None,
) -> list[str]:
    if config is not None:
        _auto_switch_mode(stages, from_stage, config)

    stage_order = _stage_order_for(config) if config else STAGE_ORDER
    aliases = _aliases_for(config) if config else _ALIASES_TRIANGULATION

    all_names = [name for name, _ in stage_order]
    if stages == "all":
        selected = all_names
    else:
        selected = []
        for token in stages.split(","):
            token = token.strip()
            name = aliases.get(token, token)
            if name not in all_names:
                raise ValueError(f"Unknown stage: {token!r}")
            selected.append(name)
    if from_stage:
        canonical = aliases.get(from_stage, from_stage)
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
    # Resolve the active stage set first so the auto-mode-switch (if any)
    # has applied before we lock in stage_order.  Otherwise asking for
    # 'hmr' against a triangulation-mode default config would compute
    # stage_order from the OLD mode (no 'hmr' entry) — the loop would
    # then iterate through triangulation stages, none of which match
    # active=['hmr'], and silently exit.
    active = resolve_stages(stages, from_stage, config=config)
    stage_order = _stage_order_for(config)
    aliases = _aliases_for(config)
    from_stage_canonical = aliases.get(from_stage, from_stage) if from_stage else None

    shared_ball_detector = None
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
    for name, StageClass in stage_order:
        if name not in active:
            continue
        if StageClass is None:
            print(f"  [SKIP] {name} (not yet implemented)")
            continue
        current_stage_kwargs = dict(stage_kwargs)
        if name in {"segmentation", "sync"} and shared_ball_detector is not None:
            current_stage_kwargs["ball_detector"] = shared_ball_detector
        stage = StageClass(config=config, output_dir=output_dir, **current_stage_kwargs)
        if stage.is_complete() and from_stage_canonical != name:
            print(f"  [SKIP] {name} (cached)")
            continue
        print(f"  [RUN]  {name}")
        stage.run()

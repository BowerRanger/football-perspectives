from pathlib import Path

from src.pipeline.base import BaseStage
from src.pipeline.quality_report import write_quality_report

# Stages are imported lazily inside _stage_class() so deleting a not-yet-
# rebuilt stage doesn't break other tooling that imports the runner.

_STAGE_NAMES: list[str] = [
    "prepare_shots",
    "tracking",
    "camera",
    "hmr_world",
    "ball",
    "export",
]


def _stage_class(name: str) -> type[BaseStage] | None:
    """Lazy import so partially-implemented pipelines still load."""
    if name == "prepare_shots":
        from src.stages.prepare_shots import PrepareShotsStage
        return PrepareShotsStage
    if name == "tracking":
        from src.stages.tracking import PlayerTrackingStage
        return PlayerTrackingStage
    if name == "camera":
        from src.stages.camera import CameraStage
        return CameraStage
    if name == "hmr_world":
        from src.stages.hmr_world import HmrWorldStage
        return HmrWorldStage
    if name == "ball":
        from src.stages.ball import BallStage
        return BallStage
    if name == "export":
        from src.stages.export import ExportStage
        return ExportStage
    raise ValueError(f"Unknown stage: {name!r}")


def resolve_stages(stages: str, from_stage: str | None) -> list[str]:
    if stages == "all":
        selected = list(_STAGE_NAMES)
    else:
        selected = []
        for token in stages.split(","):
            name = token.strip()
            if name not in _STAGE_NAMES:
                raise ValueError(f"Unknown stage: {name!r}")
            selected.append(name)
    if from_stage:
        if from_stage not in _STAGE_NAMES:
            raise ValueError(f"Unknown stage: {from_stage!r}")
        idx = _STAGE_NAMES.index(from_stage)
        selected = [n for n in selected if _STAGE_NAMES.index(n) >= idx]
    return selected


def run_pipeline(
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config: dict,
    shot_filter: str | None = None,
    player_filter: str | None = None,
    **stage_kwargs,
) -> None:
    """Run pipeline stages.

    ``shot_filter`` (optional): when set, every stage that iterates the
    shots manifest will only process the named shot. Stages that don't
    use the manifest ignore it. Used by the dashboard's
    /api/run-shot endpoint to re-run a single stage for a single shot
    without re-running everything.

    ``player_filter`` (optional): when set, hmr_world will only fit the
    named ``player_id`` (paired with ``shot_filter`` to disambiguate
    when the same player_id appears in multiple shots). Stages that
    don't iterate per-player ignore it. Used by the dashboard's
    /api/run-shot-player endpoint to iterate quickly on one player.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    active = resolve_stages(stages, from_stage)
    for name in _STAGE_NAMES:
        if name not in active:
            continue
        StageClass = _stage_class(name)
        if StageClass is None:
            print(f"  [SKIP] {name} (not implemented)")
            continue
        stage = StageClass(config=config, output_dir=output_dir, **stage_kwargs)
        if shot_filter is not None:
            stage.shot_filter = shot_filter
        if player_filter is not None:
            stage.player_filter = player_filter
        # Filtered runs (shot or player) always re-enter the stage —
        # is_complete() reflects the unfiltered state and would short-
        # circuit a per-shot or per-player retry otherwise.
        filtered = shot_filter is not None or player_filter is not None
        if stage.is_complete() and from_stage != name and not filtered:
            print(f"  [SKIP] {name} (cached)")
            continue
        print(f"  [RUN]  {name}")
        stage.run()

    # Aggregate per-stage diagnostics into output/quality_report.json.
    # This always runs (each section is independent of stage activation).
    try:
        write_quality_report(output_dir)
    except Exception as exc:  # noqa: BLE001 — diagnostics must never fail the run
        print(f"  [WARN] quality_report aggregation failed: {exc}")

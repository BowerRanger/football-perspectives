#!/usr/bin/env python3
"""Football match reconstruction pipeline CLI."""

from pathlib import Path

import click

from src.pipeline.config import load_config
from src.pipeline.runner import resolve_stages, run_pipeline


@click.group()
def cli() -> None:
    """Football match reconstruction pipeline."""


@cli.command()
@click.option(
    "--input", "input_path", required=False, default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input video file (required when prepare_shots runs).",
)
@click.option(
    "--output", "output_dir", default="./output", show_default=True,
    type=click.Path(path_type=Path), help="Output directory.",
)
@click.option(
    "--stages", default="all", show_default=True,
    help="Stages to run: 'all' or comma-separated stage names "
         "(prepare_shots,tracking,camera,hmr_world,ball,export).",
)
@click.option(
    "--from-stage", default=None,
    help="Resume from this stage (re-runs it even if cached, skips earlier stages).",
)
@click.option(
    "--config", "config_path", default=None,
    type=click.Path(exists=True, path_type=Path),
    help="YAML config file (merged with defaults).",
)
@click.option(
    "--device", default="auto", show_default=True,
    help="Compute device: cuda, cpu, mps, or auto.",
)
@click.option(
    "--clean", is_flag=True, default=False,
    help="Wipe legacy artefact directories (calibration, sync, triangulation, smpl, matching) before running.",
)
@click.option(
    "--shots", "shots", default=None,
    help="Comma-separated shot ids: run the selected stages for only these "
         "shots (one filtered pass per shot).",
)
@click.option(
    "--stale", "stale", is_flag=True, default=False,
    help="Also re-run stages whose code changed, and cascade re-runs "
         "downstream (see `recon.py status`).",
)
def run(
    input_path: Path | None,
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config_path: Path | None,
    device: str,
    clean: bool,
    shots: str | None,
    stale: bool,
) -> None:
    """Run the reconstruction pipeline on a video file."""
    import shutil

    cfg = load_config(config_path)
    if clean:
        for legacy in ("calibration", "sync", "triangulation", "smpl", "matching"):
            target = output_dir / legacy
            if target.exists():
                shutil.rmtree(target)
                click.echo(f"Removed legacy: {target}")

    active_stages = resolve_stages(stages=stages, from_stage=from_stage)
    if "prepare_shots" in active_stages and input_path is None:
        raise click.UsageError(
            "--input is required when prepare_shots is part of the active stages"
        )

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Stages: {stages}")
    shot_ids = [t.strip() for t in shots.split(",") if t.strip()] if shots else None
    if shots is not None and not shot_ids:
        raise click.UsageError("--shots needs at least one shot id")
    try:
        _run(output_dir, stages, from_stage, cfg, input_path, device, shot_ids, stale)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo("Done.")


def _run(output_dir, stages, from_stage, cfg, input_path, device, shot_ids, stale):
    run_pipeline(
        shots=shot_ids,
        stale=stale,
        output_dir=output_dir,
        stages=stages,
        from_stage=from_stage,
        config=cfg,
        video_path=input_path,
        device=device,
    )


@cli.command()
@click.option(
    "--output", "output_dir", default="./output", show_default=True,
    type=click.Path(path_type=Path), help="Output directory to inspect.",
)
@click.option(
    "--config", "config_path", default=None,
    type=click.Path(exists=True, path_type=Path),
    help="YAML config file (merged with defaults).",
)
def status(output_dir: Path, config_path: Path | None) -> None:
    """Show per-stage completeness and freshness (fresh/stale/unknown)."""
    from src.pipeline.runner import stage_status

    rows = stage_status(output_dir, load_config(config_path))
    for row in rows:
        line = f"{row['stage']:<15} {row['completeness']:<9} {row['freshness']}"
        if row["note"]:
            line += f" ({row['note']})"
        click.echo(line)
        for reason in row["reasons"][:5]:
            click.echo(f"    - {reason}")
        if len(row["reasons"]) > 5:
            click.echo(f"    - ... {len(row['reasons']) - 5} more")
        if row["code_drift"] and not row["reasons"]:
            click.echo("    - stage code changed since last run (use run --stale)")


@cli.command()
@click.option(
    "--output",
    "output_dir",
    default="./output",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory to serve.",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port.")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="YAML config file (merged with defaults).",
)
def serve(output_dir: Path, host: str, port: int, config_path: Path | None) -> None:
    """Launch the pipeline dashboard in a browser."""
    import uvicorn
    from src.web.server import create_app

    app = create_app(output_dir=output_dir, config_path=config_path)
    click.echo(f"Dashboard: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()

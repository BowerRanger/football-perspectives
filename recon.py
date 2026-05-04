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
    "--input",
    "input_path",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Input video file (required when segmentation runs).",
)
@click.option(
    "--output",
    "output_dir",
    default="./output",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option(
    "--stages",
    default="all",
    show_default=True,
    help="Stages to run: 'all' or comma-separated (e.g. '1,2,3' or 'segmentation,calibration').",
)
@click.option(
    "--from-stage",
    default=None,
    help="Resume from this stage (re-runs it even if cached, skips earlier stages).",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="YAML config file (merged with defaults).",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="Compute device: cuda, cpu, mps, or auto.",
)
def run(
    input_path: Path | None,
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config_path: Path | None,
    device: str,
) -> None:
    """Run the reconstruction pipeline on a video file."""
    cfg = load_config(config_path)
    active_stages = resolve_stages(stages=stages, from_stage=from_stage, config=cfg)
    if "segmentation" in active_stages and input_path is None:
        raise click.UsageError(
            "--input is required when segmentation is part of the active stages"
        )

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Stages: {stages}")
    run_pipeline(
        output_dir=output_dir,
        stages=stages,
        from_stage=from_stage,
        config=cfg,
        video_path=input_path,
        device=device,
    )
    click.echo("Done.")


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

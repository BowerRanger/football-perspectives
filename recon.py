#!/usr/bin/env python3
"""Football match reconstruction pipeline CLI."""

from pathlib import Path

import click

from src.pipeline.config import load_config
from src.pipeline.runner import run_pipeline


@click.group()
def cli() -> None:
    """Football match reconstruction pipeline."""


@cli.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input video file.",
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
    input_path: Path,
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config_path: Path | None,
    device: str,
) -> None:
    """Run the reconstruction pipeline on a video file."""
    cfg = load_config(config_path)
    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Stages: {stages}")
    run_pipeline(
        output_dir=output_dir,
        stages=stages,
        from_stage=from_stage,
        config=cfg,
        video_path=input_path,
    )
    click.echo("Done.")


if __name__ == "__main__":
    cli()

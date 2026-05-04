"""Tests for the CLI entry point."""

import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch
from src.stages.segmentation import ShotSegmentationStage


# Import the CLI we're about to create
from recon import cli


def test_cli_help():
    """Test that the main CLI group shows help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "reconstruction" in result.output.lower()


def test_run_help_shows_options():
    """Test that run command help shows required options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--stages" in result.output
    assert "--from-stage" in result.output


def test_run_missing_input_fails():
    """Test that run command fails without --input when segmentation is active."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--output", "/tmp/out"])
    assert result.exit_code != 0


def test_run_missing_input_allowed_when_starting_after_segmentation(tmp_path):
    """--input should be optional when segmentation is excluded by --from-stage."""
    runner = CliRunner()
    output_dir = tmp_path / "output"

    with patch("recon.run_pipeline") as run_pipeline:
        result = runner.invoke(
            cli,
            [
                "run",
                "--output",
                str(output_dir),
                "--from-stage",
                "calibration",
            ],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    _, kwargs = run_pipeline.call_args
    assert kwargs["video_path"] is None


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory) -> Path:
    """Synthetic 2-second video with a hard cut at 1 second."""
    import cv2
    import numpy as np

    path = tmp_path_factory.mktemp("fixtures") / "test.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240)
    )
    for _ in range(25):  # blue frames
        writer.write(np.full((240, 320, 3), [200, 50, 50], dtype=np.uint8))
    for _ in range(25):  # green frames (new shot)
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()
    return path


def test_run_produces_manifest(tmp_path, tiny_video):
    """Test that run command produces shots/shots_manifest.json."""
    runner = CliRunner()
    output_dir = tmp_path / "output"
    result = runner.invoke(
        cli,
        [
            "run",
            "--input",
            str(tiny_video),
            "--output",
            str(output_dir),
            "--stages",
            "1",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    manifest_path = output_dir / "shots" / "shots_manifest.json"
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"


def test_run_passes_device_to_runner(tmp_path, tiny_video):
    runner = CliRunner()
    output_dir = tmp_path / "output"

    with patch("recon.run_pipeline") as run_pipeline:
        result = runner.invoke(
            cli,
            [
                "run",
                "--input",
                str(tiny_video),
                "--output",
                str(output_dir),
                "--stages",
                "5",
                "--device",
                "cpu",
            ],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    _, kwargs = run_pipeline.call_args
    assert kwargs["device"] == "cpu"

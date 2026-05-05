"""Tests for the CLI entry point (recon.py)."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from recon import cli


@pytest.mark.unit
def test_cli_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "serve" in result.output


@pytest.mark.unit
def test_run_help_shows_named_stages_and_clean_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--stages" in result.output
    assert "--from-stage" in result.output
    assert "--clean" in result.output
    # Named stages from the new pipeline appear in --stages help text.
    for stage in (
        "prepare_shots",
        "tracking",
        "camera",
        "pose_2d",
        "hmr_world",
        "ball",
        "export",
    ):
        assert stage in result.output


@pytest.mark.unit
def test_run_rejects_prepare_shots_without_input(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--output", str(tmp_path), "--stages", "prepare_shots"],
    )
    assert result.exit_code != 0
    assert "--input" in result.output


@pytest.mark.unit
def test_run_rejects_unknown_stage(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--output", str(tmp_path), "--stages", "calibration"],
    )
    assert result.exit_code != 0


@pytest.mark.unit
def test_serve_help_shows_host_and_port():
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output

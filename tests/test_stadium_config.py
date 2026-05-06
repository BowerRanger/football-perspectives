"""Tests for the per-stadium pitch metadata loader and mow-stripe generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.stadium_config import (
    MowingPattern,
    StadiumConfig,
    StadiumConfigError,
    load_stadiums,
    mow_stripe_lines,
)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "stadiums.yaml"
    path.write_text(content)
    return path


def test_load_stadiums_parses_along_touchline_entry(tmp_path: Path):
    yaml_text = """
stadiums:
  test_stadium:
    display_name: "Test Stadium"
    mowing:
      orientation: along_touchline
      stripe_width_m: 5.0
      origin_y_m: 0.0
"""
    registry = load_stadiums(_write(tmp_path, yaml_text))
    assert "test_stadium" in registry
    cfg = registry["test_stadium"]
    assert cfg.display_name == "Test Stadium"
    assert cfg.mowing is not None
    assert cfg.mowing.orientation == "along_touchline"
    assert cfg.mowing.stripe_width_m == 5.0
    assert cfg.mowing.origin_y_m == 0.0


def test_load_stadiums_supports_grid_orientation(tmp_path: Path):
    yaml_text = """
stadiums:
  grid_stadium:
    display_name: "Grid"
    mowing:
      orientation: grid
      stripe_width_m: 6.0
      origin_x_m: 1.0
      origin_y_m: 2.0
"""
    cfg = load_stadiums(_write(tmp_path, yaml_text))["grid_stadium"]
    assert cfg.mowing == MowingPattern(
        orientation="grid",
        stripe_width_m=6.0,
        origin_x_m=1.0,
        origin_y_m=2.0,
    )


def test_load_stadiums_rejects_unknown_orientation(tmp_path: Path):
    yaml_text = """
stadiums:
  bad:
    display_name: "Bad"
    mowing:
      orientation: diagonal
      stripe_width_m: 5.0
"""
    with pytest.raises(StadiumConfigError, match="invalid mowing.orientation"):
        load_stadiums(_write(tmp_path, yaml_text))


def test_load_stadiums_rejects_non_positive_width(tmp_path: Path):
    yaml_text = """
stadiums:
  bad:
    display_name: "Bad"
    mowing:
      orientation: along_touchline
      stripe_width_m: 0
"""
    with pytest.raises(StadiumConfigError, match="stripe_width_m"):
        load_stadiums(_write(tmp_path, yaml_text))


def test_load_stadiums_missing_file_returns_empty(tmp_path: Path):
    assert load_stadiums(tmp_path / "does_not_exist.yaml") == {}


def test_load_stadiums_empty_file_returns_empty(tmp_path: Path):
    assert load_stadiums(_write(tmp_path, "")) == {}


def test_load_stadiums_handles_stadium_without_mowing(tmp_path: Path):
    """A stadium entry without a ``mowing`` block should load fine —
    just no dynamic mow stripes for the editor."""
    yaml_text = """
stadiums:
  paint_only:
    display_name: "Paint Only"
"""
    cfg = load_stadiums(_write(tmp_path, yaml_text))["paint_only"]
    assert cfg.mowing is None


# ── mow_stripe_lines ────────────────────────────────────────────────────────


def _stadium(orientation: str, **kwargs) -> StadiumConfig:
    return StadiumConfig(
        id="x",
        display_name="X",
        mowing=MowingPattern(orientation=orientation, **kwargs),  # type: ignore[arg-type]
    )


def test_mow_stripe_lines_along_touchline_emits_y_boundaries():
    cfg = _stadium("along_touchline", stripe_width_m=5.5, origin_y_m=0.0)
    lines = mow_stripe_lines(cfg)
    # 5.5, 11.0, ..., up to but not including 68.0
    expected_count = int(68.0 // 5.5)  # → 12 boundaries (5.5..66.0)
    assert len(lines) == expected_count
    assert "mow_y_5.5" in lines
    assert "mow_y_66.0" in lines
    # Endpoints should span pitch length, world y constant
    pa, pb = lines["mow_y_5.5"]
    assert pa == (0.0, 5.5, 0.0)
    assert pb == (105.0, 5.5, 0.0)


def test_mow_stripe_lines_along_goalline_emits_x_boundaries():
    cfg = _stadium("along_goalline", stripe_width_m=10.0, origin_x_m=0.0)
    lines = mow_stripe_lines(cfg)
    assert "mow_x_10.0" in lines
    assert "mow_x_100.0" in lines
    pa, pb = lines["mow_x_10.0"]
    assert pa == (10.0, 0.0, 0.0)
    assert pb == (10.0, 68.0, 0.0)


def test_mow_stripe_lines_grid_emits_both_axes():
    cfg = _stadium("grid", stripe_width_m=10.0)
    lines = mow_stripe_lines(cfg)
    has_x = any(name.startswith("mow_x_") for name in lines)
    has_y = any(name.startswith("mow_y_") for name in lines)
    assert has_x and has_y


def test_mow_stripe_lines_origin_offset_shifts_boundaries():
    cfg = _stadium("along_touchline", stripe_width_m=10.0, origin_y_m=5.0)
    lines = mow_stripe_lines(cfg)
    # Boundaries at 15.0, 25.0, ..., 65.0
    assert "mow_y_15.0" in lines
    assert "mow_y_65.0" in lines
    # 5.0 itself sits inside the pitch and is a valid boundary too
    assert "mow_y_5.0" in lines
    # Should never emit boundaries on the touchlines themselves
    assert "mow_y_0.0" not in lines
    assert "mow_y_68.0" not in lines


def test_mow_stripe_lines_stadium_without_mowing_returns_empty():
    cfg = StadiumConfig(id="paint_only", display_name="Paint Only", mowing=None)
    assert mow_stripe_lines(cfg) == {}


def test_mow_stripe_lines_negative_origin_walks_into_pitch():
    """Origin outside the pitch should still produce the correct interior set."""
    cfg = _stadium("along_touchline", stripe_width_m=10.0, origin_y_m=-3.0)
    lines = mow_stripe_lines(cfg)
    # Boundaries: -3+10=7, 17, 27, 37, 47, 57, 67 — all inside (0, 68)
    assert "mow_y_7.0" in lines
    assert "mow_y_67.0" in lines
    assert len(lines) == 7

"""Quality report surfaces the render stage's per-shot camera outputs."""

from __future__ import annotations

import json

import pytest

from src.pipeline.quality_report import _render_section


@pytest.mark.unit
def test_render_section_lists_outputs(tmp_path):
    d = tmp_path / "render" / "shot01"
    d.mkdir(parents=True)
    (d / "broadcast.mp4").write_bytes(b"x" * 1000)
    (d / "drone.mp4").write_bytes(b"x" * 2000)
    (tmp_path / "render" / "render_timings.json").write_text(
        json.dumps({"shot01": 42.5}))
    section = _render_section(tmp_path)
    assert section["shots"]["shot01"]["cameras"] == ["broadcast", "drone"]
    assert section["shots"]["shot01"]["render_seconds"] == 42.5


@pytest.mark.unit
def test_render_section_none_when_absent(tmp_path):
    assert _render_section(tmp_path) is None

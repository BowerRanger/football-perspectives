"""Render stage — config wiring, argv shaping, Blender-missing degrade,
and per-active-shot subprocess invocation against a stub Blender."""

from __future__ import annotations

import stat

import pytest

from src.pipeline.runner import _stage_class, resolve_stages
from src.schemas.shots import Shot, ShotsManifest
from src.stages.render import RenderStage


def _cfg(**over):
    cfg = {
        "render": {
            "enabled": True,
            "blender_path": None,
            "cameras": ["broadcast", "drone"],
            "resolution": [640, 360],
            "vertical_variant": False,
            "samples": 4,
            "style": {"ramp_steps": 3, "outline_width_m": 0.02,
                      "grass_stripes": 10, "palette": {}},
            "teams": {"defaults": {}, "by_player": {}},
            "aov_passes": False,
            "save_blend": False,
        },
        "export": {"blender_path": "blender", "virtual_cameras": {}},
    }
    cfg["render"].update(over)
    return cfg


@pytest.mark.unit
def test_render_stage_registered():
    assert "render" in resolve_stages("all", None)
    assert resolve_stages("all", None)[-1] == "render"
    assert _stage_class("render") is RenderStage


@pytest.mark.unit
def test_blender_args_shape(tmp_path):
    stage = RenderStage(_cfg(), tmp_path)
    args = stage._blender_args("shot01")
    # [blender, --background, --python, <script>, --, --output-dir, ...]
    assert args[1] == "--background"
    assert args[4] == "--"
    assert "--shot" in args and args[args.index("--shot") + 1] == "shot01"
    assert "--cameras" in args
    assert args[args.index("--cameras") + 1] == "broadcast,drone"
    assert "--width" in args and args[args.index("--width") + 1] == "640"


@pytest.mark.unit
def test_run_warns_and_skips_without_blender(tmp_path, caplog):
    cfg = _cfg()
    cfg["export"]["blender_path"] = "/nonexistent/blender-bin"
    stage = RenderStage(cfg, tmp_path)
    stage.run()  # must not raise
    assert not (tmp_path / "render").exists() or not any(
        (tmp_path / "render").rglob("*.mp4"))
    assert any("blender" in r.message.lower() for r in caplog.records)


@pytest.mark.integration
def test_run_invokes_blender_stub_per_active_shot(tmp_path):
    # Fake blender: records argv, writes the expected mp4.
    stub = tmp_path / "fake_blender"
    log = tmp_path / "calls.jsonl"
    stub.write_text(
        "#!/bin/sh\n"
        f"echo \"$@\" >> {log}\n"
        # emulate the script writing its outputs
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "shots").mkdir()
    ShotsManifest(
        source_file="x", fps=25.0, total_frames=10,
        shots=[
            Shot(id="shot01", start_frame=0, end_frame=4, start_time=0.0,
                 end_time=0.166, clip_file="shots/shot01.mp4"),
            Shot(id="shot02", start_frame=5, end_frame=9, start_time=0.166,
                 end_time=0.333, clip_file="shots/shot02.mp4",
                 excluded=True),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")
    cfg = _cfg(blender_path=str(stub))
    stage = RenderStage(cfg, tmp_path)
    stage.run()
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 1               # excluded shot skipped
    assert "--shot shot01" in calls[0]

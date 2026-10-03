"""render_pass_runner: --vertical-only passthrough, render-root derivation,
render_pass return value / failure behaviour (subprocess stubbed)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.utils import render_pass_runner as rpr

_EXP = {"id": "p1", "camera": "broadcast", "frames": (10, 20), "vertical": True,
        "style": None, "rig": {}, "speed": None, "time_stretch": 1,
        "style_name": None}
_Q = rpr.QUALITY_PRESETS["draft"]


def _cmd(**kw):
    return rpr.build_blender_command(
        blender_bin="blender", output_dir=Path("/o"), shot="s", exp=_EXP,
        quality=_Q, style_payload={}, **kw)


def test_vertical_only_flag_passed_through():
    cmd = _cmd(vertical_only=True)
    assert "--vertical-only" in cmd and "--vertical" in cmd


def test_default_has_no_vertical_only():
    assert "--vertical-only" not in _cmd()


def test_render_root_override():
    cmd = _cmd(render_root="render/shorts")
    assert cmd[cmd.index("--render-root") + 1] == "render/shorts"


def test_render_root_from_out_dir(tmp_path):
    out = tmp_path / "shorts_passes" / "s1"
    assert rpr._render_root_for_out_dir(tmp_path, "s1", "p1", out) == "shorts_passes"
    assert rpr._render_root_for_out_dir(tmp_path, "s1", "p1", None) == "render_experiments/p1"
    assert rpr._render_root_for_out_dir(tmp_path, "s1", "p1", Path("/elsewhere/x")) == "render_experiments/p1"


def _fake_run(tmp_path, rc=0, write=True, seen=None):
    def run(cmd, **kw):
        if seen is not None:
            seen.append(cmd)
        root = cmd[cmd.index("--render-root") + 1]
        d = tmp_path / root / "s1"
        d.mkdir(parents=True, exist_ok=True)
        if write:
            (d / "broadcast.mp4").write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, rc, "", "boom")
    return run


def test_render_pass_returns_path(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(rpr.subprocess, "run", _fake_run(tmp_path, seen=seen))
    out = rpr.render_pass(tmp_path, "s1", _EXP, {}, _Q,
                          out_dir=tmp_path / "shorts_passes" / "s1", vertical_only=False)
    assert out == tmp_path / "shorts_passes" / "s1" / "broadcast.mp4"
    assert "--vertical-only" not in seen[0]


def test_render_pass_vertical_only_default_for_non_broadcast(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(rpr.subprocess, "run", _fake_run(tmp_path, seen=seen))
    monkeypatch.setattr(rpr, "write_camera_track", lambda *a, **k: None)
    exp = {**_EXP, "camera": "orbit"}
    d = tmp_path / "rp" / "s1"
    d.mkdir(parents=True)
    (d / "orbit_9x16.mp4").write_bytes(b"x")
    out = rpr.render_pass(tmp_path, "s1", exp, {}, _Q, out_dir=d)
    assert out.name == "orbit_9x16.mp4"
    assert "--vertical-only" in seen[0]


def test_render_pass_raises_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(rpr.subprocess, "run", _fake_run(tmp_path, rc=1, write=False))
    with pytest.raises(RuntimeError):
        rpr.render_pass(tmp_path, "s1", _EXP, {}, _Q,
                        out_dir=tmp_path / "rp" / "s1", vertical_only=False)

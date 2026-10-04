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

# --- resolve_style_payload (T10) -------------------------------------------

_LIB_CFG = {
    "render": {
        "style": {"ramp_steps": 3, "palette": {"grass_light": "#111111"},
                  "post": {}},
        "teams": {"defaults": {"home": {"shirt": "#c0392b", "shorts": "#ffffff",
                                        "socks": "#c0392b"}}},
    },
    "appearance": {"kits": {"home": {"shirt": "#112233", "shorts": "#ffffff",
                                     "socks": "#112233"}}},
}


def test_resolve_style_payload_precedence_and_sidecar(tmp_path):
    import json as _json
    preset = {"palette": {"grass_light": "#222222"}, "ramp_steps": 5}
    p = rpr.resolve_style_payload(tmp_path, "s1", _LIB_CFG, preset)
    # clip appearance.kits beats render.teams.defaults; preset beats base style
    assert p["teams"]["defaults"]["home"]["shirt"] == "#112233"
    assert p["ramp_steps"] == 5 and p["palette"]["grass_light"] == "#222222"
    # bookkeeping keys never reach Blender, but are recorded
    assert "venue" not in p["stadium"] and "dressing_source" not in p["stadium"]
    side = _json.loads((tmp_path / "render" / "s1_kit_safety.json").read_text())
    assert side["shot"] == "s1" and "dressing_source" in side and side["warnings"] == []


def test_operator_kits_beat_clip_and_preset_wins_over_all(tmp_path):
    import json as _json
    (tmp_path / "appearance").mkdir()
    (tmp_path / "appearance" / "kits_operator.json").write_text(_json.dumps(
        {"kits": {"home": {"shirt": "#abcdef", "shorts": "#ffffff", "socks": "#abcdef"}}}))
    p = rpr.resolve_style_payload(tmp_path, "s1", _LIB_CFG, None)
    assert p["teams"]["defaults"]["home"]["shirt"] == "#abcdef"
    p2 = rpr.resolve_style_payload(
        tmp_path, "s1", _LIB_CFG,
        {"teams": {"defaults": {"home": {"shirt": "#000001"}}}})
    assert p2["teams"]["defaults"]["home"]["shirt"] == "#000001"


def test_resolve_style_payload_lints_and_logs(tmp_path, caplog):
    import json as _json
    cfg = {"render": {"style": {"post": {"duotone": {"shadow": "#101010",
                                                     "highlight": "#e0e0e0"}}},
                      "teams": {"defaults": {
                          "home": {"shirt": "#c0392b", "shorts": "#ffffff", "socks": "#c0392b"},
                          "away": {"shirt": "#2980b9", "shorts": "#ffffff", "socks": "#2980b9"}}}}}
    with caplog.at_level("WARNING"):
        rpr.resolve_style_payload(tmp_path, "s1", cfg, None, safety_dir=tmp_path / "x")
    side = _json.loads((tmp_path / "x" / "s1_kit_safety.json").read_text())
    assert side["warnings"], "duotone should merge the two team kits"
    assert any("kit_safety" in r.message for r in caplog.records)


def test_command_has_python_exit_code_and_capsule_flag():
    cmd = _cmd(allow_capsule_fallback=True)
    assert cmd[cmd.index("--python-exit-code") + 1] == "1"
    assert cmd.index("--python-exit-code") < cmd.index("--python")
    assert "--allow-capsule-fallback" in cmd
    assert "--allow-capsule-fallback" not in _cmd()


def test_exit_zero_but_empty_output_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(rpr.subprocess, "run", _fake_run(tmp_path, rc=0, write=False))
    res = rpr.execute_pass(_EXP, output_dir=tmp_path, shot="s1", quality=_Q, cfg={},
                           out_dir=tmp_path / "rp" / "s1")
    assert res.blender_exit_code != 0 and res.mp4_paths == []
    with pytest.raises(RuntimeError):
        rpr.render_pass(tmp_path, "s1", _EXP, {}, _Q, out_dir=tmp_path / "rp" / "s1",
                        vertical_only=False)


def test_zero_byte_mp4_is_failure(tmp_path, monkeypatch):
    def run(cmd, **kw):
        d = tmp_path / "rp" / "s1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "broadcast.mp4").write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(rpr.subprocess, "run", run)
    res = rpr.execute_pass(_EXP, output_dir=tmp_path, shot="s1", quality=_Q, cfg={},
                           out_dir=tmp_path / "rp" / "s1")
    assert res.blender_exit_code != 0 and res.mp4_paths == []

"""render_experiments.py — matrix YAML parsing/validation, output-path
resolution (everything under render_experiments/<id>/, never render/),
camera-id dispatch to virtual_cameras builders, manifest merge semantics,
and dry-run command assembly against a stubbed subprocess."""

from __future__ import annotations

import json
import subprocess as subprocess_module
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import render_experiments as rex
from tests.conftest import _add_player_fixture, _write_min_fixture

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_MATRIX = _REPO_ROOT / "config" / "render_experiments.yaml"


def _min_exp(**over) -> dict:
    base = {"id": "e1", "camera": "broadcast"}
    base.update(over)
    return base


# --- Matrix YAML: real file -----------------------------------------------

@pytest.mark.unit
def test_real_matrix_yaml_loads_and_covers_the_d1_d2_ids():
    experiments = rex.load_experiments(_REAL_MATRIX)
    ids = {e["id"] for e in experiments}
    # D1
    assert {"goal_left", "goalline_left", "orbit", "chase", "dolly",
            "ots_p003_tight"} <= ids
    # D2
    assert {"night_floodlit", "neon_synthwave", "retro_comic",
            "mono_duotone", "gritty_documentary"} <= ids
    # D3 is a commented template — must NOT parse as a live entry.
    assert "goal_left_clean_vertical" not in ids


@pytest.mark.unit
def test_real_matrix_yaml_d1_camera_ids_and_rig_override():
    by_id = {e["id"]: e for e in rex.load_experiments(_REAL_MATRIX)}
    assert by_id["goal_left"]["camera"] == "goal:left"
    assert by_id["goalline_left"]["camera"] == "goalline:left"
    assert by_id["orbit"]["camera"] == "orbit"
    assert by_id["ots_p003_tight"]["camera"] == "ots:P003"
    assert by_id["ots_p003_tight"]["rig"] == {"ots_fov_deg": 35.0}


@pytest.mark.unit
def test_real_matrix_yaml_d2_styles_populated_and_style_name_fallback():
    by_id = {e["id"]: e for e in rex.load_experiments(_REAL_MATRIX)}
    for exp_id in ("night_floodlit", "neon_synthwave", "retro_comic",
                    "mono_duotone", "gritty_documentary"):
        exp = by_id[exp_id]
        assert exp["camera"] == "broadcast"
        # Each D2 entry carries a concrete non-empty style dict (T2's
        # recommended presets, integrated into the matrix).
        assert isinstance(exp["style"], dict) and exp["style"]
        # style_name isn't set explicitly in the yaml; the loader keeps it
        # None — the id-fallback (style present -> style_name = id) happens
        # at manifest-write time, not load time.
        assert exp["style_name"] is None


# --- validate_experiment / load_experiments -------------------------------

@pytest.mark.unit
def test_validate_experiment_fills_defaults():
    exp = rex.validate_experiment(_min_exp())
    assert exp == {
        "id": "e1", "camera": "broadcast", "rig": {}, "style": None,
        "style_name": None, "frames": None, "vertical": False, "speed": None,
    }


@pytest.mark.unit
@pytest.mark.parametrize("cam", [
    "broadcast", "drone", "orbit", "chase", "dolly",
    "pov:P001", "ots:P099", "goal:left", "goal:right",
    "goalline:left", "goalline:right",
])
def test_validate_experiment_accepts_every_known_camera_id(cam):
    exp = rex.validate_experiment(_min_exp(camera=cam))
    assert exp["camera"] == cam


@pytest.mark.unit
@pytest.mark.parametrize("cam", [
    "", "Broadcast", "goal", "goal:up", "pov", "pov:", "sky:left", "dolly:x",
])
def test_validate_experiment_rejects_bad_camera_id(cam):
    with pytest.raises(ValueError, match="invalid camera id"):
        rex.validate_experiment(_min_exp(camera=cam))


@pytest.mark.unit
def test_validate_experiment_requires_id():
    with pytest.raises(ValueError, match="'id'"):
        rex.validate_experiment({"camera": "broadcast"})


@pytest.mark.unit
def test_validate_experiment_rejects_non_dict_rig():
    with pytest.raises(ValueError, match="'rig' must be a mapping"):
        rex.validate_experiment(_min_exp(rig=["not", "a", "dict"]))


@pytest.mark.unit
def test_validate_experiment_rejects_non_dict_style():
    with pytest.raises(ValueError, match="'style' must be a mapping"):
        rex.validate_experiment(_min_exp(style="oops"))


@pytest.mark.unit
def test_validate_experiment_preserves_empty_style_dict():
    exp = rex.validate_experiment(_min_exp(style={}))
    assert exp["style"] == {}


@pytest.mark.unit
def test_validate_experiment_rejects_frames_wrong_length():
    with pytest.raises(ValueError, match="frames"):
        rex.validate_experiment(_min_exp(frames=[1, 2, 3]))


@pytest.mark.unit
def test_validate_experiment_rejects_frames_start_after_end():
    with pytest.raises(ValueError, match="start > end"):
        rex.validate_experiment(_min_exp(frames=[100, 5]))


@pytest.mark.unit
def test_validate_experiment_normalizes_frames_to_ints():
    exp = rex.validate_experiment(_min_exp(frames=[5, 100]))
    assert exp["frames"] == [5, 100]


@pytest.mark.unit
def test_validate_experiment_rejects_non_bool_vertical():
    with pytest.raises(ValueError, match="'vertical' must be a bool"):
        rex.validate_experiment(_min_exp(vertical="yes"))


@pytest.mark.unit
def test_load_experiments_requires_experiments_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("not_experiments: []\n")
    with pytest.raises(ValueError, match="'experiments' list"):
        rex.load_experiments(path)


@pytest.mark.unit
def test_load_experiments_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dupe.yaml"
    path.write_text(yaml.safe_dump({"experiments": [
        {"id": "dup", "camera": "broadcast"},
        {"id": "dup", "camera": "drone"},
    ]}))
    with pytest.raises(ValueError, match="duplicate experiment id"):
        rex.load_experiments(path)


# --- speed normalization / quality gating ---------------------------------

@pytest.mark.unit
def test_normalize_speed_bare_number():
    assert rex.normalize_speed(2.0) == {"factor": 2.0, "method": "setpts"}


@pytest.mark.unit
def test_normalize_speed_dict_with_method():
    assert rex.normalize_speed({"factor": 3, "method": "minterpolate"}) == {
        "factor": 3.0, "method": "minterpolate"}


@pytest.mark.unit
def test_normalize_speed_none_passthrough():
    assert rex.normalize_speed(None) is None


@pytest.mark.unit
def test_normalize_speed_rejects_non_positive_factor():
    with pytest.raises(ValueError, match="positive"):
        rex.normalize_speed({"factor": 0})


@pytest.mark.unit
def test_normalize_speed_rejects_bad_method():
    with pytest.raises(ValueError, match="method"):
        rex.normalize_speed({"factor": 2.0, "method": "warp"})


@pytest.mark.unit
def test_validate_speed_for_quality_blocks_minterpolate_at_draft():
    speed = rex.normalize_speed({"factor": 2.0, "method": "minterpolate"})
    with pytest.raises(ValueError, match="clean"):
        rex.validate_speed_for_quality(speed, "draft")


@pytest.mark.unit
def test_validate_speed_for_quality_allows_minterpolate_at_clean():
    speed = rex.normalize_speed({"factor": 2.0, "method": "minterpolate"})
    rex.validate_speed_for_quality(speed, "clean")  # must not raise


@pytest.mark.unit
def test_validate_speed_for_quality_allows_setpts_at_draft():
    speed = rex.normalize_speed(2.0)
    rex.validate_speed_for_quality(speed, "draft")  # must not raise


@pytest.mark.unit
def test_validate_speed_for_quality_noop_when_no_speed():
    rex.validate_speed_for_quality(None, "draft")  # must not raise


# --- camera id parsing / dispatch -----------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("cam_id,expected", [
    ("broadcast", ("broadcast", None)),
    ("drone", ("drone", None)),
    ("orbit", ("orbit", None)),
    ("chase", ("chase", None)),
    ("dolly", ("dolly", None)),
    ("pov:P001", ("pov", "P001")),
    ("ots:P099", ("ots", "P099")),
    ("goal:left", ("goal", "left")),
    ("goalline:right", ("goalline", "right")),
])
def test_parse_camera_id(cam_id, expected):
    assert rex.parse_camera_id(cam_id) == expected


@pytest.mark.unit
def test_parse_camera_id_rejects_unrecognised():
    with pytest.raises(ValueError):
        rex.parse_camera_id("sky")


# --- output-path resolution: always under render_experiments/ ------------

@pytest.mark.unit
def test_render_root_for_is_scoped_to_experiment_id():
    assert rex.render_root_for("goal_left") == "render_experiments/goal_left"


@pytest.mark.unit
@pytest.mark.parametrize("shot", ["gberch", ""])
@pytest.mark.parametrize("cam_id", ["drone", "pov:P001", "goal:left"])
def test_camera_track_path_never_under_bare_render_dir(tmp_path, shot, cam_id):
    path = rex.camera_track_path(tmp_path, "exp1", shot, cam_id)
    rel_parts = path.relative_to(tmp_path).parts
    assert rel_parts[0] == "render_experiments"
    assert rel_parts[1] == "exp1"
    assert "render" not in rel_parts[2:]  # shot/cameras/... segments
    assert path.name == f"{cam_id.replace(':', '_')}_camera_track.json"


@pytest.mark.unit
def test_camera_track_dir_matches_render_stage_naming_convention(tmp_path):
    # Mirrors RenderStage._write_virtual_camera_tracks's own
    # output/render/<shot|clip>/cameras/ naming, just rooted under
    # render_experiments/<exp_id>/ instead of render/.
    d = rex.camera_track_dir(tmp_path, "exp1", "")
    assert d == tmp_path / "render_experiments" / "exp1" / "clip" / "cameras"


# --- build_camera_track dispatch ------------------------------------------

@pytest.mark.unit
def test_build_camera_track_dispatch_pov(monkeypatch):
    calls = []

    def fake_build_pov_track(track, cfg, image_size, fps, clip_id):
        calls.append((track, cfg, image_size, fps, clip_id))
        return "POV_TRACK"

    monkeypatch.setattr(rex.vcam, "build_pov_track", fake_build_pov_track)
    cfg = rex.vcam.RigConfig()
    result = rex.build_camera_track(
        "pov:P001", cfg, {"P001": "TRACK"}, None, (640, 360), 25.0, "clip")
    assert result == "POV_TRACK"
    assert calls == [("TRACK", cfg, (640, 360), 25.0, "clip")]


@pytest.mark.unit
def test_build_camera_track_dispatch_ots(monkeypatch):
    calls = []

    def fake_build_ots_track(track, ball_track, cfg, image_size, fps, clip_id):
        calls.append((track, ball_track, cfg, image_size, fps, clip_id))
        return "OTS_TRACK"

    monkeypatch.setattr(rex.vcam, "build_ots_track", fake_build_ots_track)
    cfg = rex.vcam.RigConfig()
    result = rex.build_camera_track(
        "ots:P099", cfg, {"P099": "TRACK"}, "BALL", (640, 360), 25.0, "clip")
    assert result == "OTS_TRACK"
    assert calls == [("TRACK", "BALL", cfg, (640, 360), 25.0, "clip")]


@pytest.mark.unit
def test_build_camera_track_dispatch_drone(monkeypatch):
    calls = []

    def fake_build_drone_track(tracks, ball_track, cfg, image_size, fps, clip_id):
        calls.append((tracks, ball_track, cfg, image_size, fps, clip_id))
        return "DRONE_TRACK"

    monkeypatch.setattr(rex.vcam, "build_drone_track", fake_build_drone_track)
    cfg = rex.vcam.RigConfig()
    result = rex.build_camera_track(
        "drone", cfg, {"P001": "T1", "P002": "T2"}, "BALL",
        (640, 360), 25.0, "clip")
    assert result == "DRONE_TRACK"
    assert calls[0][0] in (["T1", "T2"], ["T2", "T1"])
    assert calls[0][1:] == ("BALL", cfg, (640, 360), 25.0, "clip")


@pytest.mark.unit
def test_build_camera_track_dispatch_new_side_rig(monkeypatch):
    """goal:<side> — instance A's builder hasn't landed on main yet, so
    this simulates it landing with the assumed (side, tracks, ball, cfg,
    image_size, fps, clip_id) signature."""
    calls = []

    def fake_build_goal_track(side, tracks, ball_track, cfg, image_size,
                               fps, clip_id):
        calls.append((side, tracks, ball_track, cfg, image_size, fps, clip_id))
        return "GOAL_TRACK"

    monkeypatch.setattr(rex.vcam, "build_goal_track", fake_build_goal_track,
                         raising=False)
    cfg = rex.vcam.RigConfig()
    result = rex.build_camera_track(
        "goal:left", cfg, {}, None, (640, 360), 25.0, "clip")
    assert result == "GOAL_TRACK"
    assert calls[0][0] == "left"


@pytest.mark.unit
def test_build_camera_track_dispatch_new_noarg_rig(monkeypatch):
    """orbit/chase/dolly — assumed to mirror build_drone_track's shape
    exactly (whole-scene camera, no target argument)."""
    calls = []

    def fake_build_orbit_track(tracks, ball_track, cfg, image_size, fps, clip_id):
        calls.append((tracks, ball_track, cfg, image_size, fps, clip_id))
        return "ORBIT_TRACK"

    monkeypatch.setattr(rex.vcam, "build_orbit_track", fake_build_orbit_track,
                         raising=False)
    cfg = rex.vcam.RigConfig()
    result = rex.build_camera_track(
        "orbit", cfg, {}, None, (640, 360), 25.0, "clip")
    assert result == "ORBIT_TRACK"


@pytest.mark.unit
def test_build_camera_track_missing_builder_raises_clear_runtime_error(monkeypatch):
    monkeypatch.delattr(rex.vcam, "build_goalline_track", raising=False)
    cfg = rex.vcam.RigConfig()
    with pytest.raises(RuntimeError, match="build_goalline_track"):
        rex.build_camera_track(
            "goalline:right", cfg, {}, None, (640, 360), 25.0, "clip")


@pytest.mark.unit
def test_build_camera_track_broadcast_is_rejected():
    cfg = rex.vcam.RigConfig()
    with pytest.raises(ValueError, match="broadcast"):
        rex.build_camera_track(
            "broadcast", cfg, {}, None, (640, 360), 25.0, "clip")


@pytest.mark.unit
def test_build_camera_track_unknown_player_raises():
    cfg = rex.vcam.RigConfig()
    with pytest.raises(ValueError, match="no player track"):
        rex.build_camera_track(
            "pov:P999", cfg, {}, None, (640, 360), 25.0, "clip")


# --- _rig_config overrides --------------------------------------------

@pytest.mark.unit
def test_rig_config_applies_known_override():
    cfg = rex._rig_config({}, {"ots_fov_deg": 35.0})
    assert cfg.ots_fov_deg == 35.0
    assert cfg.pov_fov_deg == 75.0  # untouched default


@pytest.mark.unit
def test_rig_config_rejects_unknown_override_field():
    with pytest.raises(ValueError, match="valid fields are"):
        rex._rig_config({}, {"warp_speed_m_s": 15.0})


@pytest.mark.unit
def test_rig_config_reads_new_rig_fields_generically_from_config():
    # goal_back_m etc. were added to RigConfig after this module's first
    # draft — _rig_config must read them from config/default.yaml's
    # export.virtual_cameras generically (not via a hand-duplicated list
    # that would silently ignore them).
    cfg = {"export": {"virtual_cameras": {"goal_back_m": 12.5}}}
    rig_cfg = rex._rig_config(cfg, {})
    assert rig_cfg.goal_back_m == 12.5


@pytest.mark.unit
def test_rig_config_override_applies_on_top_of_config_value():
    cfg = {"export": {"virtual_cameras": {"orbit_radius_m": 12.0}}}
    rig_cfg = rex._rig_config(cfg, {"orbit_radius_m": 20.0})
    assert rig_cfg.orbit_radius_m == 20.0


# --- style payload ----------------------------------------------------

@pytest.mark.unit
def test_resolve_style_payload_no_override_returns_base_plus_teams():
    base_style = {"ramp_steps": 3, "palette": {"grass_light": "#4d9e46"}}
    payload = rex.resolve_style_payload(base_style, {"defaults": {}}, None)
    assert payload["ramp_steps"] == 3
    assert payload["palette"] == {"grass_light": "#4d9e46"}
    assert payload["teams"] == {"defaults": {}}
    # must not mutate the caller's dict
    assert base_style == {"ramp_steps": 3, "palette": {"grass_light": "#4d9e46"}}


@pytest.mark.unit
def test_resolve_style_payload_deep_merges_partial_palette_override():
    base_style = {"ramp_steps": 3,
                  "palette": {"grass_light": "#4d9e46", "outline": "#1a1a1a"}}
    payload = rex.resolve_style_payload(
        base_style, {}, {"palette": {"grass_light": "#ff0000"}})
    assert payload["palette"]["grass_light"] == "#ff0000"
    assert payload["palette"]["outline"] == "#1a1a1a"  # survives the partial merge
    assert payload["ramp_steps"] == 3


# --- build_blender_command -------------------------------------------

@pytest.mark.unit
def test_build_blender_command_minimal():
    exp = rex.validate_experiment(_min_exp(id="e1", camera="broadcast"))
    cmd = rex.build_blender_command(
        blender_bin="blender-stub", output_dir=Path("/out"), shot="gberch",
        exp=exp, quality=rex._QUALITY_PRESETS["draft"], style_payload={"a": 1})
    assert cmd[0] == "blender-stub"
    assert "--output-dir" in cmd and cmd[cmd.index("--output-dir") + 1] == "/out"
    assert cmd[cmd.index("--shot") + 1] == "gberch"
    assert cmd[cmd.index("--cameras") + 1] == "broadcast"
    assert cmd[cmd.index("--render-root") + 1] == "render_experiments/e1"
    assert cmd[cmd.index("--width") + 1] == "960"
    assert cmd[cmd.index("--height") + 1] == "540"
    assert cmd[cmd.index("--samples") + 1] == "8"
    assert json.loads(cmd[cmd.index("--style-json") + 1]) == {"a": 1}
    assert "--frame-start" not in cmd
    assert "--vertical" not in cmd


@pytest.mark.unit
def test_build_blender_command_with_frames_and_vertical():
    exp = rex.validate_experiment(
        _min_exp(id="e2", camera="orbit", frames=[10, 50], vertical=True))
    cmd = rex.build_blender_command(
        blender_bin="blender-stub", output_dir=Path("/out"), shot="gberch",
        exp=exp, quality=rex._QUALITY_PRESETS["clean"], style_payload={})
    assert cmd[cmd.index("--frame-start") + 1] == "10"
    assert cmd[cmd.index("--frame-end") + 1] == "50"
    assert "--vertical" in cmd
    assert cmd[cmd.index("--width") + 1] == "1920"


# --- manifest merge -----------------------------------------------------

@pytest.mark.unit
def test_merge_manifest_preserves_untouched_and_overwrites_touched(tmp_path):
    manifest_path = tmp_path / "render_experiments" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "a:gberch": {"id": "a", "blender_exit_code": 0},
        "b:gberch": {"id": "b", "blender_exit_code": 1},
    }))

    merged = rex.merge_manifest(tmp_path, {"b:gberch": {"id": "b", "blender_exit_code": 0}})

    assert merged["a:gberch"] == {"id": "a", "blender_exit_code": 0}  # untouched
    assert merged["b:gberch"] == {"id": "b", "blender_exit_code": 0}  # overwritten
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == merged


@pytest.mark.unit
def test_merge_manifest_recovers_from_corrupt_existing_file(tmp_path):
    manifest_path = tmp_path / "render_experiments" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{not valid json")

    merged = rex.merge_manifest(tmp_path, {"a:gberch": {"id": "a"}})

    assert merged == {"a:gberch": {"id": "a"}}


@pytest.mark.unit
def test_merge_manifest_creates_parent_dir(tmp_path):
    merged = rex.merge_manifest(tmp_path, {"a:gberch": {"id": "a"}})
    assert (tmp_path / "render_experiments" / "manifest.json").exists()
    assert merged == {"a:gberch": {"id": "a"}}


# --- resolve_blender_binary ---------------------------------------------

@pytest.mark.unit
def test_resolve_blender_binary_prefers_render_config(monkeypatch):
    monkeypatch.setattr(rex.shutil, "which", lambda name: f"/usr/bin/{name}")
    cfg = {"render": {"blender_path": "blender-render"},
           "export": {"blender_path": "blender-export"}}
    assert rex.resolve_blender_binary(cfg) == "/usr/bin/blender-render"


@pytest.mark.unit
def test_resolve_blender_binary_falls_back_to_export_config(monkeypatch):
    monkeypatch.setattr(rex.shutil, "which", lambda name: f"/usr/bin/{name}")
    cfg = {"render": {}, "export": {"blender_path": "blender-export"}}
    assert rex.resolve_blender_binary(cfg) == "/usr/bin/blender-export"


@pytest.mark.unit
def test_resolve_blender_binary_returns_none_when_not_found(monkeypatch):
    monkeypatch.setattr(rex.shutil, "which", lambda name: None)
    assert rex.resolve_blender_binary({}) is None


# --- CLI: dry-run has zero side effects -----------------------------------

@pytest.mark.unit
def test_dry_run_never_invokes_subprocess_and_touches_no_files(tmp_path, monkeypatch, capsys):
    def _boom(*a, **k):
        raise AssertionError("subprocess.run must never be called in --dry-run")
    monkeypatch.setattr(subprocess_module, "run", _boom)

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"experiments": [
        {"id": "e1", "camera": "broadcast"},
        {"id": "e2", "camera": "goal:left"},  # builder doesn't exist yet — must not matter
    ]}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    rc = rex.main([
        "--output", str(output_dir), "--shot", "gberch",
        "--experiments", str(matrix), "--dry-run",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run] experiment=e1" in out
    assert "[dry-run] experiment=e2" in out
    assert "would write camera track" in out  # for e2 (goal:left), not e1 (broadcast)
    # No files at all should have been written under output/.
    assert list(output_dir.rglob("*")) == []


@pytest.mark.unit
def test_dry_run_command_assembly_matches_build_blender_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess_module, "run",
                         lambda *a, **k: (_ for _ in ()).throw(
                             AssertionError("must not invoke subprocess")))
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"experiments": [
        {"id": "e1", "camera": "dolly", "frames": [1, 20], "vertical": True},
    ]}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    rc = rex.main([
        "--output", str(output_dir), "--shot", "gberch",
        "--experiments", str(matrix), "--dry-run", "--quality", "clean",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--render-root render_experiments/e1" in out
    assert "--frame-start 1" in out
    assert "--frame-end 20" in out
    assert "--vertical" in out
    assert "--width 1920" in out and "--samples 16" in out


# --- CLI: --only filtering / validation errors ----------------------------

@pytest.mark.unit
def test_main_only_unknown_id_errors_without_running_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess_module, "run",
                         lambda *a, **k: (_ for _ in ()).throw(
                             AssertionError("must not invoke subprocess")))
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"experiments": [
        {"id": "e1", "camera": "broadcast"},
    ]}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    rc = rex.main([
        "--output", str(output_dir), "--shot", "gberch",
        "--experiments", str(matrix), "--only", "nope", "--dry-run",
    ])
    assert rc == 2
    assert "unknown experiment id" in capsys.readouterr().err


@pytest.mark.unit
def test_main_blocks_minterpolate_at_draft_before_any_experiment_runs(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess_module, "run",
                         lambda *a, **k: (_ for _ in ()).throw(
                             AssertionError("must not invoke subprocess")))
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"experiments": [
        {"id": "e1", "camera": "broadcast",
         "speed": {"factor": 2.0, "method": "minterpolate"}},
    ]}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    rc = rex.main([
        "--output", str(output_dir), "--shot", "gberch",
        "--experiments", str(matrix), "--quality", "draft",
    ])
    assert rc == 2
    assert "minterpolate" in capsys.readouterr().err


# --- Full (non-dry-run) run, hermetic via a stub subprocess ---------------

class _FakeCompleted:
    def __init__(self, args, returncode=0, stdout="", stderr=""):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_fake_subprocess_run(blender_bin: str):
    def _fake_run(cmd, **kwargs):
        if cmd and cmd[0] == blender_bin:
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            render_root = cmd[cmd.index("--render-root") + 1]
            shot = cmd[cmd.index("--shot") + 1]
            cam = cmd[cmd.index("--cameras") + 1]
            out_dir = output_dir / render_root / (shot or "clip")
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_id = cam.replace(":", "_")
            (out_dir / f"{safe_id}.mp4").write_bytes(b"stub-mp4")
            return _FakeCompleted(cmd, returncode=0)
        if cmd and cmd[0] == "ffprobe":
            return _FakeCompleted(cmd, returncode=0, stdout="10.0\n")
        if cmd and cmd[0] == "ffmpeg":
            # extract_thumbnail / apply_slowmo both pass the destination
            # as the final positional argument.
            Path(cmd[-1]).write_bytes(b"stub")
            return _FakeCompleted(cmd, returncode=0)
        raise AssertionError(f"unexpected subprocess call: {cmd}")
    return _fake_run


@pytest.mark.unit
def test_run_experiment_broadcast_writes_manifest_entry_and_thumbnail(
        tmp_path, monkeypatch):
    _write_min_fixture(tmp_path)  # camera/ball fixture, shot=""
    monkeypatch.setattr(subprocess_module, "run",
                         _make_fake_subprocess_run("blender-stub"))

    exp = rex.validate_experiment({"id": "e1", "camera": "broadcast"})
    entry = rex.run_experiment(
        exp, output_dir=tmp_path, shot="", quality_name="draft",
        cfg={"render": {"style": {}, "teams": {}}, "export": {}},
        blender_bin="blender-stub", dry_run=False,
    )

    assert entry["id"] == "e1"
    assert entry["camera"] == "broadcast"
    assert entry["quality"] == "draft"
    assert entry["blender_exit_code"] == 0
    assert entry["style_name"] is None
    mp4 = tmp_path / "render_experiments" / "e1" / "clip" / "broadcast.mp4"
    thumb = tmp_path / "render_experiments" / "e1" / "clip" / "broadcast_thumb.jpg"
    assert str(mp4) in entry["output_paths"]
    assert str(thumb) in entry["output_paths"]
    assert mp4.exists() and thumb.exists()
    # broadcast never gets a synthesised camera track file.
    assert not (tmp_path / "render_experiments" / "e1" / "clip" / "cameras").exists()


@pytest.mark.unit
@pytest.mark.parametrize("cam_id", ["orbit", "chase", "dolly", "goal:left", "goalline:right"])
def test_write_camera_track_against_real_landed_builders(tmp_path, cam_id):
    """Non-mocked integration check: instance A's goal/goalline/orbit/
    chase/dolly builders have landed in src/utils/virtual_cameras.py —
    exercise write_camera_track against the REAL functions (not a stub)
    for every new rig id this module dispatches to, using the real
    config/default.yaml export.virtual_cameras block."""
    _write_min_fixture(tmp_path)
    _add_player_fixture(tmp_path)
    from src.pipeline.config import load_config
    cfg = load_config()

    dest = rex.camera_track_path(tmp_path, "real1", "", cam_id)
    track = rex.write_camera_track(
        cam_id, dest, output_dir=tmp_path, shot="", rig_overrides={},
        cfg=cfg, image_size=(960, 540),
    )
    assert dest.exists()
    assert track.frames
    loaded = rex.CameraTrack.load(dest)
    assert len(loaded.frames) == len(track.frames)


@pytest.mark.unit
def test_run_experiment_drone_writes_camera_track_under_render_experiments(
        tmp_path, monkeypatch):
    _write_min_fixture(tmp_path)
    _add_player_fixture(tmp_path)
    monkeypatch.setattr(subprocess_module, "run",
                         _make_fake_subprocess_run("blender-stub"))

    exp = rex.validate_experiment({"id": "e2", "camera": "drone"})
    entry = rex.run_experiment(
        exp, output_dir=tmp_path, shot="", quality_name="draft",
        cfg={"render": {"style": {}, "teams": {}}, "export": {}},
        blender_bin="blender-stub", dry_run=False,
    )

    assert entry["blender_exit_code"] == 0
    track_path = (tmp_path / "render_experiments" / "e2" / "clip" / "cameras"
                  / "drone_camera_track.json")
    assert track_path.exists()
    # Never under the protected baseline.
    assert not (tmp_path / "render" / "clip" / "cameras" / "drone_camera_track.json").exists()


@pytest.mark.unit
def test_run_experiment_applies_slowmo_and_records_output(tmp_path, monkeypatch):
    _write_min_fixture(tmp_path)
    monkeypatch.setattr(subprocess_module, "run",
                         _make_fake_subprocess_run("blender-stub"))

    exp = rex.validate_experiment({"id": "e3", "camera": "broadcast", "speed": 2.0})
    entry = rex.run_experiment(
        exp, output_dir=tmp_path, shot="", quality_name="draft",
        cfg={"render": {"style": {}, "teams": {}}, "export": {}},
        blender_bin="blender-stub", dry_run=False,
    )

    slowmo = tmp_path / "render_experiments" / "e3" / "clip" / "broadcast_slowmo.mp4"
    slowmo_thumb = (tmp_path / "render_experiments" / "e3" / "clip"
                     / "broadcast_slowmo_thumb.jpg")
    assert str(slowmo) in entry["output_paths"]
    assert str(slowmo_thumb) in entry["output_paths"]


@pytest.mark.unit
def test_run_experiment_style_name_defaults_to_id_when_style_present(
        tmp_path, monkeypatch):
    _write_min_fixture(tmp_path)
    monkeypatch.setattr(subprocess_module, "run",
                         _make_fake_subprocess_run("blender-stub"))

    exp = rex.validate_experiment(
        {"id": "night_floodlit", "camera": "broadcast", "style": {}})
    entry = rex.run_experiment(
        exp, output_dir=tmp_path, shot="", quality_name="draft",
        cfg={"render": {"style": {}, "teams": {}}, "export": {}},
        blender_bin="blender-stub", dry_run=False,
    )
    assert entry["style_name"] == "night_floodlit"


@pytest.mark.unit
def test_main_end_to_end_writes_manifest(tmp_path, monkeypatch):
    _write_min_fixture(tmp_path)
    monkeypatch.setattr(subprocess_module, "run",
                         _make_fake_subprocess_run("blender-stub"))
    monkeypatch.setattr(rex, "resolve_blender_binary", lambda cfg: "blender-stub")

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"experiments": [
        {"id": "e1", "camera": "broadcast"},
    ]}))

    rc = rex.main([
        "--output", str(tmp_path), "--shot", "",
        "--experiments", str(matrix),
    ])
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "render_experiments" / "manifest.json").read_text())
    assert "e1:" in manifest
    assert manifest["e1:"]["blender_exit_code"] == 0


@pytest.mark.unit
def test_main_isolates_one_bad_experiment_from_the_rest(tmp_path, monkeypatch):
    """goal:left's builder hasn't landed on main — main() must still run
    the other (good) experiment and record the failure, not crash."""
    _write_min_fixture(tmp_path)
    monkeypatch.setattr(subprocess_module, "run",
                         _make_fake_subprocess_run("blender-stub"))
    monkeypatch.setattr(rex, "resolve_blender_binary", lambda cfg: "blender-stub")
    monkeypatch.delattr(rex.vcam, "build_goal_track", raising=False)

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"experiments": [
        {"id": "e1", "camera": "broadcast"},
        {"id": "e2", "camera": "goal:left"},
    ]}))

    rc = rex.main([
        "--output", str(tmp_path), "--shot", "",
        "--experiments", str(matrix),
    ])
    assert rc == 1  # batch reports failure...
    manifest = json.loads(
        (tmp_path / "render_experiments" / "manifest.json").read_text())
    assert manifest["e1:"]["blender_exit_code"] == 0  # ...but e1 still ran
    assert "error" in manifest["e2:"]

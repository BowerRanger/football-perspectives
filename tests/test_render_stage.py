"""Render stage — config wiring, argv shaping, Blender-missing degrade,
and per-active-shot subprocess invocation against a stub Blender."""

from __future__ import annotations

import json
import logging
import stat

import pytest

from src.pipeline.runner import _stage_class, resolve_stages
from src.schemas.render_selection import RenderSelection
from src.schemas.shots import Shot, ShotsManifest
from src.stages.render import RenderStage
from tests.conftest import _add_player_fixture, _write_min_fixture


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
def test_is_complete_requires_every_active_shot(tmp_path):
    """A multi-shot manifest with one rendered + one un-rendered shot must
    not report complete — the pre-fix ``any(render_dir.rglob("*.mp4"))``
    check considered the stage done the moment ANY shot rendered
    anywhere, so a failed second shot cache-skipped forever. Single
    requested camera (broadcast) here, isolating "every shot" totality
    from the camera-granularity checks covered separately below."""
    (tmp_path / "shots").mkdir()
    ShotsManifest(
        source_file="x", fps=25.0, total_frames=10,
        shots=[
            Shot(id="shot01", start_frame=0, end_frame=4, start_time=0.0,
                 end_time=0.166, clip_file="shots/shot01.mp4"),
            Shot(id="shot02", start_frame=5, end_frame=9, start_time=0.166,
                 end_time=0.333, clip_file="shots/shot02.mp4"),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")
    stage = RenderStage(_cfg(cameras=["broadcast"]), tmp_path)
    assert stage.is_complete() is False

    shot01_dir = tmp_path / "render" / "shot01"
    shot01_dir.mkdir(parents=True)
    (shot01_dir / "broadcast.mp4").write_bytes(b"x")
    assert stage.is_complete() is False  # shot02 still missing

    shot02_dir = tmp_path / "render" / "shot02"
    shot02_dir.mkdir(parents=True)
    (shot02_dir / "broadcast.mp4").write_bytes(b"x")
    assert stage.is_complete() is True


@pytest.mark.unit
def test_resolve_blender_double_null_config_returns_none_not_typeerror(tmp_path):
    """render.blender_path and export.blender_path both explicitly null
    (not absent) used to raise TypeError from Path(None) — must instead
    resolve to the "blender" default and warn-and-skip if it's not on
    PATH, same posture as a missing binary."""
    cfg = _cfg(blender_path=None)
    cfg["export"]["blender_path"] = None
    stage = RenderStage(cfg, tmp_path)
    # Must not raise; result depends on whether "blender" happens to be
    # on this machine's PATH, so just assert it doesn't blow up.
    stage._resolve_blender()


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
    style_json = args[args.index("--style-json") + 1]
    assert "teams" in json.loads(style_json)


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


@pytest.mark.integration
def test_virtual_camera_tracks_written(tmp_path):
    _write_min_fixture(tmp_path)          # reuse via a conftest helper
    _add_player_fixture(tmp_path)
    cfg = _cfg(cameras=["broadcast", "drone", "pov:P001"])
    stage = RenderStage(cfg, tmp_path)
    written = stage._write_virtual_camera_tracks("", ["drone", "pov:P001"])
    assert set(written) == {"drone", "pov:P001"}
    cams = tmp_path / "render" / "clip" / "cameras"
    assert (cams / "drone_camera_track.json").exists()
    assert (cams / "pov_P001_camera_track.json").exists()
    track = json.loads((cams / "drone_camera_track.json").read_text())
    assert track["frames"] and "R" in track["frames"][0]


@pytest.mark.unit
def test_unknown_player_camera_skipped_with_warning(tmp_path, caplog):
    _write_min_fixture(tmp_path)
    stage = RenderStage(_cfg(), tmp_path)
    written = stage._write_virtual_camera_tracks("", ["pov:P999"])
    assert written == []
    assert any("P999" in r.message for r in caplog.records)


@pytest.mark.unit
def test_write_virtual_camera_tracks_no_ids_is_noop(tmp_path):
    _write_min_fixture(tmp_path)
    stage = RenderStage(_cfg(), tmp_path)
    assert stage._write_virtual_camera_tracks("", []) == []
    assert not (tmp_path / "render").exists()


@pytest.mark.unit
def test_write_virtual_camera_tracks_no_broadcast_camera_warns(tmp_path, caplog):
    stage = RenderStage(_cfg(), tmp_path)
    written = stage._write_virtual_camera_tracks("", ["drone"])
    assert written == []
    assert any("broadcast" in r.message.lower() for r in caplog.records)


@pytest.mark.integration
def test_run_writes_satisfied_virtual_cameras_and_filters_unsatisfied(tmp_path):
    """run() must write the vcam tracks before shelling out, and only pass
    broadcast + the ids that were actually satisfied in --cameras (an
    unresolvable player ref must never reach the Blender subprocess)."""
    _write_min_fixture(tmp_path)
    _add_player_fixture(tmp_path)
    stub = tmp_path / "fake_blender"
    log = tmp_path / "calls.jsonl"
    stub.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    cfg = _cfg(blender_path=str(stub),
               cameras=["broadcast", "drone", "pov:P001", "pov:P999"])
    stage = RenderStage(cfg, tmp_path)
    stage.run()
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 1
    cams_arg = calls[0].split("--cameras ")[1].split(" ")[0]
    assert set(cams_arg.split(",")) == {"broadcast", "drone", "pov:P001"}
    cams_dir = tmp_path / "render" / "clip" / "cameras"
    assert (cams_dir / "drone_camera_track.json").exists()
    assert (cams_dir / "pov_P001_camera_track.json").exists()
    assert not (cams_dir / "pov_P999_camera_track.json").exists()


# ── RenderSelection sidecar merge (operator input always wins) ─────────
# The sidecar lives at output/render/{shot_id or "clip"}_render_selection
# .json — same "" -> "clip" legacy naming _write_virtual_camera_tracks
# already uses for output/render/<shot|clip>/cameras/.
#
# _resolve_camera_request is tested directly (fast, precise — no shell
# round-trip through a stub Blender's argv) since run()'s per-shot
# --cameras string can't distinguish "empty list" from "no cameras arg"
# once it's gone through `"$@"` echoing. One full run() test below
# still proves the override actually reaches _blender_args end-to-end.

@pytest.mark.unit
def test_resolve_camera_request_sidecar_overrides_cameras(tmp_path):
    RenderSelection(shot_id="", cameras=("broadcast",)).save(
        tmp_path / "render" / "clip_render_selection.json"
    )
    cfg = _cfg(cameras=["broadcast", "drone"])
    stage = RenderStage(cfg, tmp_path)
    cameras, vertical = stage._resolve_camera_request("")
    assert cameras == ["broadcast"]
    assert vertical is None


@pytest.mark.unit
def test_resolve_camera_request_sidecar_overrides_to_empty_list(tmp_path):
    """The operator can explicitly disable every camera for a shot —
    an empty ``cameras`` sidecar list is a real override, not treated
    the same as "sidecar absent"."""
    RenderSelection(shot_id="", cameras=()).save(
        tmp_path / "render" / "clip_render_selection.json"
    )
    cfg = _cfg(cameras=["broadcast", "drone"])
    stage = RenderStage(cfg, tmp_path)
    cameras, _ = stage._resolve_camera_request("")
    assert cameras == []


@pytest.mark.unit
def test_resolve_camera_request_sidecar_overrides_vertical_variant(tmp_path):
    RenderSelection(
        shot_id="", cameras=("broadcast",), vertical_variant=True
    ).save(tmp_path / "render" / "clip_render_selection.json")
    cfg = _cfg(cameras=["broadcast"], vertical_variant=False)
    stage = RenderStage(cfg, tmp_path)
    _, vertical = stage._resolve_camera_request("")
    assert vertical is True


@pytest.mark.unit
def test_resolve_camera_request_absent_sidecar_returns_config_unchanged(tmp_path):
    cfg = _cfg(cameras=["broadcast", "drone"])
    stage = RenderStage(cfg, tmp_path)
    assert not (tmp_path / "render" / "clip_render_selection.json").exists()
    cameras, vertical = stage._resolve_camera_request("")
    assert cameras == ["broadcast", "drone"]
    assert vertical is None


@pytest.mark.unit
def test_resolve_camera_request_malformed_sidecar_warns_and_falls_back(
    tmp_path, caplog
):
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    (render_dir / "clip_render_selection.json").write_text("{not valid json")
    cfg = _cfg(cameras=["broadcast", "drone"])
    stage = RenderStage(cfg, tmp_path)
    cameras, vertical = stage._resolve_camera_request("")  # must not raise
    assert cameras == ["broadcast", "drone"]
    assert vertical is None
    assert any(
        "render" in r.message.lower() and "selection" in r.message.lower()
        for r in caplog.records
    )


@pytest.mark.unit
def test_resolve_camera_request_bad_camera_id_in_sidecar_warns_and_falls_back(
    tmp_path, caplog
):
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    (render_dir / "clip_render_selection.json").write_text(
        '{"shot_id": "", "cameras": ["dolly"]}'
    )
    cfg = _cfg(cameras=["broadcast"])
    stage = RenderStage(cfg, tmp_path)
    cameras, _ = stage._resolve_camera_request("")  # must not raise
    assert cameras == ["broadcast"]
    assert any("selection" in r.message.lower() for r in caplog.records)


@pytest.mark.unit
def test_resolve_camera_request_per_shot_paths_are_independent(tmp_path):
    RenderSelection(shot_id="shot01", cameras=("drone",)).save(
        tmp_path / "render" / "shot01_render_selection.json"
    )
    cfg = _cfg(cameras=["broadcast"])
    stage = RenderStage(cfg, tmp_path)
    cameras01, _ = stage._resolve_camera_request("shot01")
    cameras02, _ = stage._resolve_camera_request("shot02")
    assert cameras01 == ["drone"]
    assert cameras02 == ["broadcast"]  # no sidecar for shot02 -> config


# ── Camera-granularity resume/completeness (_missing_cameras) ──────────
# "Render stage should only skip if ALL shots selected have been
# rendered already; otherwise it should render the missing shots" —
# refined to camera granularity: a shot missing any REQUESTED camera
# output is incomplete, and run() renders only the missing cameras.

def _two_shot_manifest(tmp_path):
    (tmp_path / "shots").mkdir()
    ShotsManifest(
        source_file="x", fps=25.0, total_frames=10,
        shots=[
            Shot(id="shot01", start_frame=0, end_frame=4, start_time=0.0,
                 end_time=0.166, clip_file="shots/shot01.mp4"),
            Shot(id="shot02", start_frame=5, end_frame=9, start_time=0.166,
                 end_time=0.333, clip_file="shots/shot02.mp4"),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")


@pytest.mark.unit
def test_missing_cameras_partial_coverage(tmp_path):
    cfg = _cfg(cameras=["broadcast", "drone"], vertical_variant=False)
    stage = RenderStage(cfg, tmp_path)
    shot_dir = tmp_path / "render" / "clip"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"x")
    assert stage._missing_cameras("") == ["drone"]


@pytest.mark.unit
def test_missing_cameras_vertical_variant_missing(tmp_path):
    """drone.mp4 present but drone_9x16.mp4 absent still counts as
    missing when vertical_variant is on; broadcast never needs a 9x16
    counterpart."""
    cfg = _cfg(cameras=["broadcast", "drone"], vertical_variant=True)
    stage = RenderStage(cfg, tmp_path)
    shot_dir = tmp_path / "render" / "clip"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"x")
    (shot_dir / "drone.mp4").write_bytes(b"x")
    assert stage._missing_cameras("") == ["drone"]


@pytest.mark.unit
def test_missing_cameras_vertical_variant_satisfied(tmp_path):
    cfg = _cfg(cameras=["broadcast", "drone"], vertical_variant=True)
    stage = RenderStage(cfg, tmp_path)
    shot_dir = tmp_path / "render" / "clip"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"x")
    (shot_dir / "drone.mp4").write_bytes(b"x")
    (shot_dir / "drone_9x16.mp4").write_bytes(b"x")
    assert stage._missing_cameras("") == []


@pytest.mark.unit
def test_missing_cameras_empty_request_is_trivially_complete(tmp_path):
    RenderSelection(shot_id="", cameras=()).save(
        tmp_path / "render" / "clip_render_selection.json"
    )
    cfg = _cfg(cameras=["broadcast", "drone"])
    stage = RenderStage(cfg, tmp_path)
    assert stage._missing_cameras("") == []
    assert stage.is_complete() is True


@pytest.mark.unit
def test_is_complete_false_when_one_of_two_shots_partially_rendered(tmp_path):
    _two_shot_manifest(tmp_path)
    cfg = _cfg(cameras=["broadcast"], vertical_variant=False)
    stage = RenderStage(cfg, tmp_path)
    shot01_dir = tmp_path / "render" / "shot01"
    shot01_dir.mkdir(parents=True)
    (shot01_dir / "broadcast.mp4").write_bytes(b"x")
    assert stage.is_complete() is False  # shot02 has nothing at all


@pytest.mark.unit
def test_is_complete_true_when_both_shots_fully_rendered(tmp_path):
    _two_shot_manifest(tmp_path)
    cfg = _cfg(cameras=["broadcast"], vertical_variant=False)
    stage = RenderStage(cfg, tmp_path)
    for sid in ("shot01", "shot02"):
        d = tmp_path / "render" / sid
        d.mkdir(parents=True)
        (d / "broadcast.mp4").write_bytes(b"x")
    assert stage.is_complete() is True


@pytest.mark.integration
def test_run_skips_fully_rendered_shot_and_renders_only_missing_shot(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO)
    _two_shot_manifest(tmp_path)
    stub = tmp_path / "fake_blender"
    log = tmp_path / "calls.jsonl"
    stub.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    shot01_dir = tmp_path / "render" / "shot01"
    shot01_dir.mkdir(parents=True)
    (shot01_dir / "broadcast.mp4").write_bytes(b"x")
    cfg = _cfg(blender_path=str(stub), cameras=["broadcast"],
                vertical_variant=False)
    stage = RenderStage(cfg, tmp_path)
    assert stage.is_complete() is False
    stage.run()
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 1  # shot01 skipped, only shot02 rendered
    assert "--shot shot02" in calls[0]
    assert any(
        "shot01" in r.message and "complete" in r.message.lower()
        for r in caplog.records
    )


@pytest.mark.integration
def test_run_invokes_stub_zero_times_when_all_shots_complete(tmp_path):
    _two_shot_manifest(tmp_path)
    stub = tmp_path / "fake_blender"
    log = tmp_path / "calls.jsonl"
    stub.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    for sid in ("shot01", "shot02"):
        d = tmp_path / "render" / sid
        d.mkdir(parents=True)
        (d / "broadcast.mp4").write_bytes(b"x")
    cfg = _cfg(blender_path=str(stub), cameras=["broadcast"],
                vertical_variant=False)
    stage = RenderStage(cfg, tmp_path)
    assert stage.is_complete() is True
    stage.run()
    assert not log.exists()


@pytest.mark.integration
def test_run_renders_only_missing_camera_for_partial_shot(tmp_path):
    """shot01 has broadcast.mp4 but the request also includes drone ->
    run() must pass --cameras drone only, never re-rendering broadcast."""
    _write_min_fixture(tmp_path)  # legacy "" shot, writes camera_track.json
    _add_player_fixture(tmp_path)  # so build_drone_track has a non-empty track
    stub = tmp_path / "fake_blender"
    log = tmp_path / "calls.jsonl"
    stub.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    shot_dir = tmp_path / "render" / "clip"
    shot_dir.mkdir(parents=True)
    (shot_dir / "broadcast.mp4").write_bytes(b"x")
    cfg = _cfg(blender_path=str(stub), cameras=["broadcast", "drone"],
                vertical_variant=False)
    stage = RenderStage(cfg, tmp_path)
    stage.run()
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 1
    cams_arg = calls[0].split("--cameras ")[1].split(" ")[0]
    assert cams_arg == "drone"


@pytest.mark.integration
def test_run_uses_sidecar_cameras_end_to_end(tmp_path):
    """Full run() wiring: the sidecar's cameras (not config's) reach the
    Blender subprocess argv."""
    _write_min_fixture(tmp_path)
    stub = tmp_path / "fake_blender"
    log = tmp_path / "calls.jsonl"
    stub.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    RenderSelection(shot_id="", cameras=("broadcast",), vertical_variant=True).save(
        tmp_path / "render" / "clip_render_selection.json"
    )
    cfg = _cfg(blender_path=str(stub), cameras=["broadcast", "drone"],
               vertical_variant=False)
    stage = RenderStage(cfg, tmp_path)
    stage.run()
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 1
    assert "--cameras" in calls[0] and "broadcast" in calls[0]
    assert "drone" not in calls[0].split("--cameras")[1].split("--")[0]
    assert "--vertical" in calls[0].split()

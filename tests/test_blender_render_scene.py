"""Tests for scripts/blender_render_scene.py.

Split into pure arg-parsing tests (run everywhere) and a Blender smoke
test (``@pytest.mark.fbx`` — skipped when Blender isn't on PATH, same
posture as tests/test_blender_export_smpl_skeleton.py).
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from scripts.blender_render_scene import _parse_args


@pytest.mark.unit
def test_parse_args_after_double_dash():
    ns = _parse_args([
        "blender", "--background", "--",
        "--output-dir", "/tmp/o", "--shot", "shot01",
        "--cameras", "broadcast,drone", "--width", "640", "--height", "360",
        "--samples", "4", "--style-json", "{}",
    ])
    assert ns.shot == "shot01"
    assert ns.cameras == ["broadcast", "drone"]
    assert (ns.width, ns.height) == (640, 360)
    assert ns.vertical is False and ns.aov is False


@pytest.mark.unit
def test_parse_args_defaults_and_flags():
    ns = _parse_args([
        "--output-dir", "/tmp/o",
        "--vertical", "--aov", "--save-blend",
        "--frame-start", "10", "--frame-end", "20",
    ])
    # No --shot / --cameras / --width / --height / --samples / --style-json
    # supplied: defaults from the brief's skeleton.
    assert ns.shot == ""
    assert ns.cameras == ["broadcast"]
    assert (ns.width, ns.height) == (1920, 1080)
    assert ns.samples == 16
    assert ns.style_json == "{}"
    assert ns.vertical is True and ns.aov is True and ns.save_blend is True
    assert (ns.frame_start, ns.frame_end) == (10, 20)


def _add_player_fixture(root, n=3):
    """One synthetic player refined-pose NPZ, key set mirrored against a
    real ``output/refined_poses/P001_refined.npz`` on disk: player_id,
    frames, betas, thetas, root_R, root_t, confidence, view_count,
    contributing_shots. ``iter_player_fbx_entries`` only reads the first
    six plus contributing_shots (for the sync-offset/shot-id split);
    view_count/betas are along for the ride to match the real file
    exactly. An empty ``contributing_shots`` exercises the "legacy
    single-shot" fallback (shot_id="") that matches ``_write_min_fixture``'s
    unprefixed ``ball_track.json`` layout, so both fixtures agree on the
    render's ``--shot ""`` legacy path.
    """
    (root / "refined_poses").mkdir()
    np.savez(root / "refined_poses" / "P001_refined.npz",
             player_id="P001",
             frames=np.arange(n),
             betas=np.zeros(10, dtype=np.float32),
             thetas=np.zeros((n, 24, 3), dtype=np.float32),
             root_R=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
             root_t=np.tile(np.array([52.5, 30.0, 0.95], dtype=np.float32),
                            (n, 1)),
             confidence=np.ones(n, dtype=np.float32),
             view_count=np.ones(n, dtype=np.int32),
             contributing_shots=np.array([], dtype="<U6"))


_BLENDER = shutil.which("blender")


def _write_min_fixture(root):
    """Minimal single-shot output dir: camera track + ball track, no players.

    Field names mirror the real pipeline artefacts (checked against
    output/camera/gberch_camera_track.json and
    output/ball/gberch_ball_track.json on disk, and against the
    dataclass schemas in src/schemas/camera_track.py and
    src/schemas/ball_track.py): camera frames carry
    frame/K/R/t/confidence/is_anchor, the track carries clip_id/fps/
    image_size/t_world/frames. Ball frames carry frame/world_xyz/
    state/confidence — `state` must be one of BallFrame's documented
    Literal values (grounded/flight/occluded/missing; "rolling" is not
    one of them) and `confidence` is required, not optional; the track
    also carries the required (if empty) `flight_segments` list.
    (prepare_ball_keys additionally reads an optional quat_wxyz — left
    absent here to exercise its identity-quaternion fallback.)
    """
    n = 3
    (root / "camera").mkdir(parents=True)
    K = [[1000.0, 0, 320.0], [0, 1000.0, 180.0], [0, 0, 1.0]]
    frames = []
    for i in range(n):
        # camera 20m up on the near touchline looking at pitch centre
        from src.utils.virtual_cameras import look_at_view
        R, t = look_at_view(np.array([52.5, -20.0, 20.0]),
                             np.array([52.5, 34.0, 0.0]))
        frames.append({"frame": i, "K": K,
                        "R": [list(r) for r in R], "t": list(t),
                        "confidence": 1.0, "is_anchor": False})
    (root / "camera" / "camera_track.json").write_text(json.dumps(
        {"clip_id": "clip", "fps": 25.0, "image_size": [640, 360],
         "t_world": [52.5, -20.0, 20.0], "frames": frames}))
    (root / "ball").mkdir()
    (root / "ball" / "ball_track.json").write_text(json.dumps(
        {"clip_id": "clip", "fps": 25.0, "flight_segments": [],
         "frames": [{"frame": i, "world_xyz": [52.5, 34.0, 0.11],
                      "state": "grounded", "confidence": 1.0}
                     for i in range(n)]}))


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_broadcast_mp4(tmp_path):
    _write_min_fixture(tmp_path)
    script = "scripts/blender_render_scene.py"
    res = subprocess.run(
        [_BLENDER, "--background", "--python", script, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", "{}",
         "--frame-start", "0", "--frame-end", "2"],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "render" / "clip" / "broadcast.mp4"
    assert out.exists() and out.stat().st_size > 0
    assert "RENDER_TIMING broadcast" in res.stdout


@pytest.mark.fbx
@pytest.mark.skipif(_BLENDER is None, reason="blender not on PATH")
def test_smoke_render_with_player(tmp_path):
    """One player fixture on top of the min fixture; 1 frame at 160x90.

    Asserts only the PLAYERS_BUILT marker + render success — never body
    type (SMPL-mesh vs capsule fallback) since that depends on whether
    data/models/smpl_neutral.npz happens to exist on the machine running
    the suite (it's gitignored; absent by default in a fresh worktree).
    """
    _write_min_fixture(tmp_path)
    _add_player_fixture(tmp_path)
    script = "scripts/blender_render_scene.py"
    res = subprocess.run(
        [_BLENDER, "--background", "--python", script, "--",
         "--output-dir", str(tmp_path), "--shot", "",
         "--cameras", "broadcast", "--width", "160", "--height", "90",
         "--samples", "1", "--style-json", "{}",
         "--frame-start", "0", "--frame-end", "0"],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "render" / "clip" / "broadcast.mp4"
    assert out.exists() and out.stat().st_size > 0
    assert "PLAYERS_BUILT 1" in res.stdout

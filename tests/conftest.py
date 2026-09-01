"""Shared test helpers importable by multiple test modules.

``_write_min_fixture`` / ``_add_player_fixture`` originated in
tests/test_blender_render_scene.py (the Blender-render smoke tests) and
moved here (Task 8) so tests/test_render_stage.py's virtual-camera
tests can reuse the same minimal output-dir fixture instead of
duplicating it. Not pytest fixtures (no ``@pytest.fixture``) — plain
helper functions, imported directly (``from tests.conftest import
_write_min_fixture``).
"""

from __future__ import annotations

import json

import numpy as np

from src.utils.virtual_cameras import look_at_view


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

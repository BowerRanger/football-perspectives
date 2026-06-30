"""Stage-seam integration: ``ball.solver: piecewise|global`` with whole-shot
piecewise fallback.

Pins Task 8 of the Phase-2 mode-sequence plan:

* ``solver: global`` runs the beam mode-search end-to-end and produces a valid
  ``BallTrack`` with ``solver: "global"`` + a ``mode_search`` diag block and
  ``mode_search_fallback: False``.
* A forced fit budget of 1 (``mode_search.max_segment_fit_calls: 1``) triggers
  the whole-shot fallback to ``solve_piecewise``: the diag flags
  ``mode_search_fallback: True`` AND the emitted track equals the
  ``solver: piecewise`` result for the SAME input (same kwargs).
* The default (no ``solver`` key) path is byte-identical to an explicit
  ``solver: piecewise`` run — no behaviour change.

Helpers are copied verbatim from ``tests/test_ball_stage_second_pass.py`` so the
two suites stay independent (do NOT import across test modules).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.ball_detector import FakeBallDetector


# ---------------------------------------------------------------------------
# Module-private helpers (copied verbatim from tests/test_ball_stage_second_pass.py).
# ---------------------------------------------------------------------------

def _camera_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _save_camera_track(
    path: Path,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    n: int,
    clip_id: str = "play",
    fps: float = 30.0,
) -> None:
    track = CameraTrack(
        clip_id=clip_id,
        fps=fps,
        image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(
                frame=i,
                K=K.tolist(),
                R=R.tolist(),
                confidence=1.0,
                is_anchor=(i == 0),
            )
            for i in range(n)
        ),
    )
    track.save(path)


def _write_blank_clip(path: Path, n: int, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
    )
    for _ in range(n):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _project(p: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


# ---------------------------------------------------------------------------
# Scene builder: a rolling ball across the pitch, detected every frame.
# ---------------------------------------------------------------------------

def _build_scene(tmp_path: Path, n: int = 40, fps: float = 30.0):
    """Save a camera track + blank clip and return detections for a ball that
    rolls along the ground (z = ball radius). The detector emits a confident hit
    every frame so both solvers have dense pixel evidence."""
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    truth = {i: np.array([30.0 + 0.2 * i, 34.0, 0.11]) for i in range(n)}
    detections = []
    for i in range(n):
        u, v = _project(truth[i], K, R, t)
        detections.append((u, v, 0.9))
    return detections


def _run(tmp_path: Path, detections, *, solver: str | None, extra: dict | None = None):
    """Run the BallStage with the given ball.solver setting and return
    (track, diag)."""
    from src.stages.ball import BallStage

    ball_cfg: dict = {
        "detector": "fake",
        "appearance_bridge": {"enabled": False},
        "second_pass": {"enabled": False},
        "cross_replay": {"enabled": False},
        "auto_anchors": {"enabled": True, "grounded_interval": 8},
    }
    if solver is not None:
        ball_cfg["solver"] = solver
    if extra:
        ball_cfg.update(extra)

    stage = BallStage(
        config={"ball": ball_cfg},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(list(detections)),
    )
    stage.run()

    track = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    return track, diag


def _world_positions(track: BallTrack) -> dict[int, tuple]:
    return {f.frame: f.world_xyz for f in track.frames}


def _assert_track_valid(track: BallTrack, n: int) -> None:
    valid_states = {"grounded", "flight", "occluded", "missing"}
    frames = {f.frame for f in track.frames}
    assert frames == set(range(n)), "every frame must be present in the track"
    for f in track.frames:
        assert f.state in valid_states, f"frame {f.frame} has invalid state {f.state}"
        if f.world_xyz is not None:
            assert all(np.isfinite(x) for x in f.world_xyz)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_global_solver_end_to_end(tmp_path: Path):
    """solver: global produces a valid BallTrack and the expected diag block."""
    n = 40
    detections = _build_scene(tmp_path, n=n)
    track, diag = _run(tmp_path, detections, solver="global")

    _assert_track_valid(track, n)
    assert diag["solver"] == "global"
    assert diag["mode_search_fallback"] is False
    assert "mode_search" in diag and isinstance(diag["mode_search"], dict)
    # The search-effort block carries the expected keys.
    assert "fit_calls" in diag["mode_search"]
    assert "beam_width" in diag["mode_search"]


@pytest.mark.integration
def test_global_solver_budget_falls_back_to_piecewise(tmp_path: Path):
    """A budget of 1 forces BudgetExceeded → whole-shot piecewise fallback.

    The fallback must produce the EXACT piecewise result for the same input
    (same kwargs → same solve_piecewise output), and flag the fallback.
    """
    n = 40
    detections = _build_scene(tmp_path, n=n)

    # Global run with a budget so small the beam cannot complete → fallback.
    fb_track, fb_diag = _run(
        tmp_path, detections,
        solver="global",
        extra={"mode_search": {"max_segment_fit_calls": 1}},
    )
    assert fb_diag["solver"] == "global"
    assert fb_diag["mode_search_fallback"] is True

    # Reference piecewise run on a fresh output dir with identical input.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref_detections = _build_scene(ref_dir, n=n)
    ref_track, ref_diag = _run(ref_dir, ref_detections, solver="piecewise")
    assert ref_diag["solver"] == "piecewise"

    # The fallback track must match the piecewise track world positions tightly.
    fb_pos = _world_positions(fb_track)
    ref_pos = _world_positions(ref_track)
    assert fb_pos.keys() == ref_pos.keys()
    for f in fb_pos:
        a, b = fb_pos[f], ref_pos[f]
        if a is None or b is None:
            assert a is b or (a is None and b is None), f"frame {f}: {a} vs {b}"
            continue
        np.testing.assert_allclose(
            np.asarray(a, float), np.asarray(b, float), atol=1e-6,
            err_msg=f"frame {f} world mismatch fallback vs piecewise",
        )


@pytest.mark.integration
def test_default_solver_is_events(tmp_path: Path):
    """No ``solver`` key (default) resolves to the new ``events`` mode and
    flags the dense track as derived; explicit ``solver: piecewise`` still
    runs the old dense solve unchanged."""
    n = 40

    default_dir = tmp_path / "default"
    default_dir.mkdir()
    det_default = _build_scene(default_dir, n=n)
    default_track, default_diag = _run(default_dir, det_default, solver=None)

    assert default_diag["solver"] == "events"
    assert default_diag["derived"] is True
    _assert_track_valid(default_track, n)

    piecewise_dir = tmp_path / "piecewise"
    piecewise_dir.mkdir()
    det_pw = _build_scene(piecewise_dir, n=n)
    pw_track, pw_diag = _run(piecewise_dir, det_pw, solver="piecewise")
    assert pw_diag["solver"] == "piecewise"
    assert pw_diag.get("derived", False) is False


@pytest.mark.integration
def test_events_mode_writes_keyframes_with_segments(tmp_path: Path):
    """Events mode emits a sparse keyframes sidecar carrying interpolation
    segments (the authoritative product UE consumes)."""
    n = 40
    detections = _build_scene(tmp_path, n=n)
    _track, diag = _run(tmp_path, detections, solver="events")
    assert diag["solver"] == "events"

    from src.schemas.ball_keyframes import BallKeyframeSet
    kf = BallKeyframeSet.load(tmp_path / "ball" / "ball_keyframes.json")
    # A rolling ball with grounded sampling yields at least one keyframe and
    # at least one derived segment between / around them.
    assert len(kf.keyframes) >= 1
    assert len(kf.segments) >= 1
    assert all(s.kind in {"ballistic", "roll", "carry", "rest", "free_flight"}
               for s in kf.segments)


@pytest.mark.integration
def test_unknown_solver_raises(tmp_path: Path):
    n = 10
    detections = _build_scene(tmp_path, n=n)
    with pytest.raises(ValueError, match="Unknown ball.solver"):
        _run(tmp_path, detections, solver="banana")

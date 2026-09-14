"""Integration tests: CameraStage's tri-state ``static_camera`` wiring
(auto-gate + moving-centre fallback), gap surfacing, and click triage.

See docs/superpowers/specs/2026-09-09-moving-camera-support.md for the
gberch-2 diagnosis this feature fixes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.schemas.camera_track import CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.camera import CameraStage

IMAGE_SIZE: tuple[int, int] = (1920, 1080)
CX_TRUE = IMAGE_SIZE[0] / 2.0
CY_TRUE = IMAGE_SIZE[1] / 2.0
# The actual video file is tiny/blank — nothing in the default (no
# line_extraction, no auto_anchors) code path decodes frame content, so
# a small resolution keeps these tests fast. The anchors' own image_size
# uses the realistic 1920x1080 scale independently.
VIDEO_SIZE = (64, 64)
FPS = 25.0


def _K(fx: float) -> np.ndarray:
    return np.array([[fx, 0.0, CX_TRUE], [0.0, fx, CY_TRUE], [0.0, 0.0, 1.0]])


def _yaw(angle_deg: float) -> np.ndarray:
    look = np.array([0.0, 64.0, -30.0])
    look = look / np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    base = np.array([right, down, look], dtype=float)
    a = np.deg2rad(angle_deg)
    Ry = np.array(
        [[np.cos(a), -np.sin(a), 0.0],
         [np.sin(a), np.cos(a), 0.0],
         [0.0, 0.0, 1.0]],
    )
    return base @ Ry.T


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, world: np.ndarray) -> tuple[float, float]:
    cam = R @ world + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


_LANDMARK_WORLD: list[tuple[str, tuple[float, float, float]]] = [
    ("near_left_corner", (0.0, 0.0, 0.0)),
    ("near_right_corner", (105.0, 0.0, 0.0)),
    ("far_left_corner", (0.0, 68.0, 0.0)),
    ("far_right_corner", (105.0, 68.0, 0.0)),
    ("halfway_near", (52.5, 0.0, 0.0)),
    ("near_left_corner_flag_top", (0.0, 0.0, 1.5)),
    ("left_goal_crossbar_left", (0.0, 30.34, 2.44)),
    ("left_goal_crossbar_right", (0.0, 37.66, 2.44)),
]

# A spidercam hovering 12-15m above midfield (gberch-2's diagnosed
# geometry) sees a LOCAL patch around the halfway line/centre circle,
# not the whole 105x68m pitch corner-to-corner the way a behind-the-goal
# broadcast camera 30m back does — reusing _LANDMARK_WORLD's full-pitch
# corners for a close overhead camera puts several of them behind the
# camera or wildly off-axis. Numerically verified (all z>0, finite
# pixels) for the c_start/c_end/fx range _moving_anchor_set uses.
_SPIDERCAM_LANDMARK_WORLD: list[tuple[str, tuple[float, float, float]]] = [
    ("halfway_near", (52.5, 0.0, 0.0)),
    ("halfway_far_ish", (52.5, 40.0, 0.0)),
    ("quarter_left", (40.0, 20.0, 0.0)),
    ("quarter_right", (65.0, 20.0, 0.0)),
    ("centre_spot", (52.5, 25.0, 0.0)),
    ("circle_top", (52.5, 25.0 + 9.15, 0.0)),
    ("circle_bottom", (52.5, 25.0 - 9.15, 0.0)),
    ("pole_marker", (52.5, 25.0, 3.0)),   # non-coplanar point
]


def _look_at_R(
    C: np.ndarray,
    target: np.ndarray = np.array([52.5, 25.0, 0.0]),
    up_hint: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """World->camera rotation for a camera AT ``C`` looking at ``target``
    — see the identical helper (and its rationale) in
    tests/test_camera_mode_gate.py."""
    look = target - C
    look = look / np.linalg.norm(look)
    right = np.cross(look, up_hint)
    right = right / np.linalg.norm(right)
    down = np.cross(look, right)
    return np.array([right, down, look], dtype=float)


def _anchor_at(
    C: np.ndarray, R: np.ndarray, fx: float, frame: int,
    landmark_world: list[tuple[str, tuple[float, float, float]]] = _LANDMARK_WORLD,
) -> Anchor:
    K = _K(fx)
    t = -R @ C
    return Anchor(
        frame=frame,
        landmarks=tuple(
            LandmarkObservation(
                name=name, image_xy=_project(K, R, t, np.asarray(xyz, dtype=float)),
                world_xyz=xyz,
            )
            for name, xyz in landmark_world
        ),
    )


def _write_blank_clip(path: Path, n_frames: int, size: tuple[int, int] = VIDEO_SIZE, fps: float = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, size)
    blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for _ in range(n_frames):
        vw.write(blank)
    vw.release()


def _write_manifest(output_dir: Path, shot_id: str, n_frames: int, fps: float = FPS) -> None:
    end_frame = max(0, n_frames - 1)
    ShotsManifest(
        source_file="test", fps=fps, total_frames=n_frames,
        shots=[Shot(
            id=shot_id, start_frame=0, end_frame=end_frame, start_time=0.0,
            end_time=(end_frame + 1) / fps, clip_file=f"shots/{shot_id}.mp4",
        )],
    ).save(output_dir / "shots" / "shots_manifest.json")


def _setup_shot(tmp_path: Path, shot_id: str, n_frames: int, anchor_set: AnchorSet) -> None:
    _write_blank_clip(tmp_path / "shots" / f"{shot_id}.mp4", n_frames)
    _write_manifest(tmp_path, shot_id, n_frames)
    anchor_set.save(tmp_path / "camera" / f"{shot_id}_anchors.json")


def _moving_anchor_set(shot_id: str, n_frames: int = 180) -> tuple[AnchorSet, dict[int, np.ndarray]]:
    c_start = np.array([50.0, 20.0, 12.0])
    c_end = np.array([44.0, 15.0, 15.0])
    fx_start, fx_end = 1000.0, 2600.0
    anchor_frames = (0, 90, 179)
    anchors = []
    truth: dict[int, np.ndarray] = {}
    for af in anchor_frames:
        w = af / (n_frames - 1)
        C = (1.0 - w) * c_start + w * c_end
        fx = (1.0 - w) * fx_start + w * fx_end
        R = _look_at_R(C)
        anchors.append(_anchor_at(C, R, fx, af, _SPIDERCAM_LANDMARK_WORLD))
        truth[af] = C
    return AnchorSet(clip_id=shot_id, image_size=IMAGE_SIZE, anchors=tuple(anchors)), truth


def _static_anchor_set(shot_id: str, n_frames: int = 180) -> AnchorSet:
    C = np.array([52.5, -30.0, 30.0])
    anchor_frames = (0, 90, 179)
    anchors = [
        _anchor_at(C, _yaw((af / (n_frames - 1)) * 16.0 - 8.0), 1500.0, af)
        for af in anchor_frames
    ]
    return AnchorSet(clip_id=shot_id, image_size=IMAGE_SIZE, anchors=tuple(anchors))


@pytest.mark.integration
def test_auto_mode_moving_camera_produces_no_shared_centre(tmp_path: Path) -> None:
    shot_id = "spidercam"
    anchor_set, truth = _moving_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    stage = CameraStage(config={"camera": {"static_camera": "auto"}}, output_dir=tmp_path)
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / f"{shot_id}_camera_track.json")
    assert track.camera_centre is None, (
        "moving-camera track must not claim a single shared camera centre"
    )
    by_frame = {f.frame: f for f in track.frames}
    for af, C_true in truth.items():
        cf = by_frame[af]
        R = np.asarray(cf.R)
        t = np.asarray(cf.t)
        C_hat = -R.T @ t
        err = float(np.linalg.norm(C_hat - C_true))
        assert err <= 1.0, f"anchor frame {af}: centre error {err:.2f}m > 1.0m"


@pytest.mark.integration
def test_auto_mode_static_camera_matches_todays_static_behaviour(tmp_path: Path) -> None:
    shot_id = "broadcast"
    anchor_set = _static_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    stage = CameraStage(config={"camera": {"static_camera": "auto"}}, output_dir=tmp_path)
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / f"{shot_id}_camera_track.json")
    assert track.camera_centre is not None
    C = np.asarray(track.camera_centre)
    for f in track.frames:
        R = np.asarray(f.R)
        t = np.asarray(f.t)
        recovered = -R.T @ t
        assert np.allclose(recovered, C, atol=1e-3), (
            f"frame {f.frame}: -R^T @ t = {recovered} != C = {C}"
        )


@pytest.mark.integration
def test_legacy_bool_true_forces_static_path_even_when_moving(tmp_path: Path) -> None:
    """Back-compat escape hatch: an operator who explicitly pins
    static_camera: true must get today's unconditional static path, with
    no gating, even on a clip that would otherwise fail the gate."""
    shot_id = "spidercam"
    anchor_set, _truth = _moving_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    stage = CameraStage(config={"camera": {"static_camera": True}}, output_dir=tmp_path)
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / f"{shot_id}_camera_track.json")
    assert track.camera_centre is not None


@pytest.mark.integration
def test_legacy_bool_false_forces_moving_path(tmp_path: Path) -> None:
    """Back-compat: static_camera: false must take the moving path
    unconditionally, even on geometry that would pass the gate."""
    shot_id = "broadcast"
    anchor_set = _static_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    stage = CameraStage(config={"camera": {"static_camera": False}}, output_dir=tmp_path)
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / f"{shot_id}_camera_track.json")
    assert track.camera_centre is None


@pytest.mark.integration
def test_moving_mode_interior_frame_has_honest_lower_confidence(tmp_path: Path) -> None:
    shot_id = "spidercam"
    anchor_set, _truth = _moving_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    stage = CameraStage(config={"camera": {"static_camera": "auto"}}, output_dir=tmp_path)
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / f"{shot_id}_camera_track.json")
    by_frame = {f.frame: f for f in track.frames}
    anchor_conf = by_frame[0].confidence
    midpoint_conf = by_frame[45].confidence  # midway between anchors 0 and 90
    assert midpoint_conf < anchor_conf, (
        f"interior gap frame confidence {midpoint_conf} should be lower "
        f"than anchor confidence {anchor_conf} — no independent support there"
    )
    # And it must lie smoothly between the two bracketing anchors' own
    # recovered centres (not some wild extrapolation).
    R0, t0 = np.asarray(by_frame[0].R), np.asarray(by_frame[0].t)
    R90, t90 = np.asarray(by_frame[90].R), np.asarray(by_frame[90].t)
    C0, C90 = -R0.T @ t0, -R90.T @ t90
    Rm, tm = np.asarray(by_frame[45].R), np.asarray(by_frame[45].t)
    Cm = -Rm.T @ tm
    lo = np.minimum(C0, C90) - 0.5
    hi = np.maximum(C0, C90) + 0.5
    assert np.all(Cm >= lo) and np.all(Cm <= hi), (
        f"interior centre {Cm} not between anchor centres {C0} and {C90}"
    )


@pytest.mark.integration
def test_gap_surfacing_warns_on_wide_low_support_span(tmp_path: Path, caplog) -> None:
    """Task D: a moving-camera shot whose anchors leave a wide,
    unsupported span should name that span and suggest a midpoint."""
    shot_id = "spidercam_sparse"
    n_frames = 121
    c_a = np.array([50.0, 20.0, 12.0])
    c_b = np.array([44.0, 15.0, 15.0])
    anchors = (
        _anchor_at(c_a, _look_at_R(c_a), 1200.0, 0, _SPIDERCAM_LANDMARK_WORLD),
        _anchor_at(c_b, _look_at_R(c_b), 1800.0, 120, _SPIDERCAM_LANDMARK_WORLD),
    )
    anchor_set = AnchorSet(clip_id=shot_id, image_size=IMAGE_SIZE, anchors=anchors)
    _setup_shot(tmp_path, shot_id, n_frames, anchor_set)

    stage = CameraStage(
        config={"camera": {
            "static_camera": "auto",
            "moving_gap": {"max_gap_frames": 30},
        }},
        output_dir=tmp_path,
    )
    with caplog.at_level(logging.WARNING):
        stage.run()

    messages = "\n".join(r.message for r in caplog.records)
    assert "0" in messages and "120" in messages
    assert "60" in messages  # suggested midpoint


@pytest.mark.integration
def test_click_triage_warning_names_the_culprit_landmark(tmp_path: Path, caplog) -> None:
    """Task C: an anchor with one badly mislabeled click (gberch-2's
    frame 162 / pnl_kp_24 pattern) must be named by label in the stage
    warnings once its residual stands out from its neighbours'."""
    shot_id = "spidercam_badclick"
    n_frames = 181
    c_start = np.array([50.0, 20.0, 12.0])
    c_end = np.array([44.0, 15.0, 15.0])
    anchor_frames = (0, 90, 180)
    anchors = []
    for af in anchor_frames:
        w = af / (n_frames - 1)
        C = (1.0 - w) * c_start + w * c_end
        fx = (1.0 - w) * 1000.0 + w * 2000.0
        anchors.append(
            _anchor_at(C, _look_at_R(C), fx, af, _SPIDERCAM_LANDMARK_WORLD)
        )
    # Corrupt one landmark on the middle anchor (mirrors "pnl_kp_24").
    culprit_name = "circle_top"
    mid = anchors[1]
    lms = list(mid.landmarks)
    for i, lm in enumerate(lms):
        if lm.name == culprit_name:
            u, v = lm.image_xy
            lms[i] = LandmarkObservation(
                name=culprit_name, image_xy=(u + 500.0, v - 400.0),
                world_xyz=lm.world_xyz,
            )
    anchors[1] = Anchor(frame=mid.frame, landmarks=tuple(lms))
    anchor_set = AnchorSet(clip_id=shot_id, image_size=IMAGE_SIZE, anchors=tuple(anchors))
    _setup_shot(tmp_path, shot_id, n_frames, anchor_set)

    stage = CameraStage(config={"camera": {"static_camera": "auto"}}, output_dir=tmp_path)
    with caplog.at_level(logging.WARNING):
        stage.run()

    messages = "\n".join(r.message for r in caplog.records)
    assert culprit_name in messages, (
        f"expected click-triage warning naming '{culprit_name}'; got:\n{messages}"
    )


@pytest.mark.integration
def test_camera_summary_sidecar_records_mode_and_gate(tmp_path: Path) -> None:
    """Task B: the shot's chosen model + why must be persisted so a
    downstream reader can see which path was taken without re-deriving it."""
    shot_id = "spidercam"
    anchor_set, _truth = _moving_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    stage = CameraStage(config={"camera": {"static_camera": "auto"}}, output_dir=tmp_path)
    stage.run()

    summary_path = tmp_path / "camera" / f"{shot_id}_camera_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["mode"] == "moving"
    assert "gate" in summary
    assert summary["gate"]["holds"] is False


@pytest.mark.integration
def test_camera_summary_sidecar_for_static_shot(tmp_path: Path) -> None:
    shot_id = "broadcast"
    anchor_set = _static_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    stage = CameraStage(config={"camera": {"static_camera": "auto"}}, output_dir=tmp_path)
    stage.run()

    summary_path = tmp_path / "camera" / f"{shot_id}_camera_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["mode"] == "static"
    assert summary["gate"]["holds"] is True


@pytest.mark.integration
def test_line_extraction_speed_clamp_also_applies_to_a_failed_detection_fallback(
    tmp_path: Path, monkeypatch,
) -> None:
    """Real-clip regression found on gberch-2 after the corridor fix
    alone: a run of successfully-refined frames can legitimately drift
    a few metres from the anchor-interpolated LERP baseline (each step
    individually respecting the speed budget, within the corridor
    bound) — but the NEXT frame, if its own line detection fails
    entirely, used to silently revert to the untouched, un-drifted LERP
    position, creating exactly the frame-to-frame jump this feature
    exists to prevent (observed: 1.3-2.6m in a single 1/30s frame,
    tens of times the 0.2m budget). The fallback must ALSO respect the
    speed budget relative to the predecessor's actual final position."""
    import src.utils.line_camera_refine as line_camera_refine
    from src.utils.line_camera_refine import FrameRefinement

    shot_id = "spidercam"
    anchor_set, _truth = _moving_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    call_n = [-1]

    def _fake_refine(frame_bgr, K_init, R_init, t_init, distortion, **kwargs):
        call_n[0] += 1
        if call_n[0] == 5:
            # Legitimate corridor-bounded drift: 3m from this frame's
            # own LERP seed (within the 5m corridor budget).
            C_seed = -np.asarray(R_init, dtype=np.float64).T @ np.asarray(t_init, dtype=np.float64)
            C_drifted = C_seed + np.array([3.0, 0.0, 0.0])
            t_drifted = -np.asarray(R_init, dtype=np.float64) @ C_drifted
            return FrameRefinement(
                line_rms_px=1.0, K=K_init, R=R_init, t=t_drifted,
                detected_lines=[], n_detections=2, corridor_deviation_m=3.0,
            )
        if call_n[0] == 6:
            # The very next frame's own detection fails outright.
            return FrameRefinement(
                line_rms_px=float("nan"), K=K_init, R=R_init, t=t_init,
                detected_lines=[], n_detections=0,
            )
        return FrameRefinement(
            line_rms_px=1.0, K=K_init, R=R_init, t=t_init,
            detected_lines=[], n_detections=2,
        )

    monkeypatch.setattr(line_camera_refine, "refine_camera_from_lines", _fake_refine)
    stage = CameraStage(
        config={"camera": {
            "static_camera": "auto", "line_extraction": True,
            "motion": {"max_speed_m_s": 6.0, "max_corridor_deviation_m": 5.0},
        }},
        output_dir=tmp_path,
    )
    stage.run()

    track = CameraTrack.load(tmp_path / "camera" / f"{shot_id}_camera_track.json")
    by_frame = {f.frame: f for f in track.frames}

    def centre(f):
        R = np.asarray(f.R, dtype=np.float64)
        t = np.asarray(f.t, dtype=np.float64)
        return -R.T @ t

    step = float(np.linalg.norm(centre(by_frame[6]) - centre(by_frame[5])))
    max_step = 6.0 / track.fps
    assert step <= max_step + 1e-3, (
        f"frame 5->6 step {step:.3f}m exceeds the speed budget {max_step:.3f}m "
        f"— a failed-detection fallback must also respect it"
    )


@pytest.mark.integration
def test_line_extraction_threads_corridor_and_speed_bounds_for_non_anchor_frames(
    tmp_path: Path, monkeypatch,
) -> None:
    """Coordinator-reported defect: a mid-span frame with only 1-2
    detected lines is under-determined, and refine_camera_from_lines's
    per-frame LM used to bound raw OpenCV tvec to +/-300m (vacuous for
    the camera CENTRE) — 10/181 real gberch-2 frames wandered up to
    212m off the anchor-interpolated corridor at confidence ~1.0.

    Wiring check: _refine_with_line_extraction must pass a real
    per-frame corridor centre (from whatever per_frame_R/t the Step-2
    LERP already put there) plus a growing prev_centre/max_step_m into
    refine_camera_from_lines for every NON-anchor frame in moving mode
    — and must NOT corridor/speed-constrain the anchor frames
    themselves (their own click-based pose is authoritative, matching
    "anchor-frame eval numbers stay at current quality")."""
    import src.utils.line_camera_refine as line_camera_refine
    from src.utils.line_camera_refine import FrameRefinement

    shot_id = "spidercam"
    anchor_set, _truth = _moving_anchor_set(shot_id)
    _setup_shot(tmp_path, shot_id, 180, anchor_set)

    calls: list[dict] = []

    def _fake_refine(frame_bgr, K_init, R_init, t_init, distortion, **kwargs):
        calls.append(dict(kwargs))
        return FrameRefinement(
            line_rms_px=float("nan"), K=K_init, R=R_init, t=t_init,
            detected_lines=[], n_detections=0,
        )

    monkeypatch.setattr(line_camera_refine, "refine_camera_from_lines", _fake_refine)

    stage = CameraStage(
        config={"camera": {
            "static_camera": "auto",
            "line_extraction": True,
            "motion": {"max_speed_m_s": 6.0, "max_corridor_deviation_m": 4.0},
        }},
        output_dir=tmp_path,
    )
    stage.run()

    assert calls, "refine_camera_from_lines was never called"
    anchor_calls = [c for c in calls if c.get("corridor_centre") is None]
    non_anchor_calls = [c for c in calls if c.get("corridor_centre") is not None]
    assert anchor_calls, "expected the anchor frames to skip corridor constraints"
    assert non_anchor_calls, "expected interior frames to be corridor-constrained"
    for c in non_anchor_calls:
        assert c.get("max_corridor_deviation_m") == 4.0
    # At least one non-anchor call (after the first accepted frame)
    # should carry a prev_centre/max_step_m pair for the speed clamp.
    speed_clamped_calls = [c for c in non_anchor_calls if c.get("prev_centre") is not None]
    assert speed_clamped_calls, "expected the speed clamp to engage for later frames"
    for c in speed_clamped_calls:
        assert c.get("max_step_m") == pytest.approx(6.0 / 25.0)

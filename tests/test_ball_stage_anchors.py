"""Integration tests for Layer 5 ball anchor injection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0]); look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _write_blank_clip(path: Path, n: int, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    for _ in range(n):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _save_camera_track(path: Path, K, R, t, n: int, fps: float = 30.0):
    track = CameraTrack(
        clip_id="play", fps=fps, image_size=(1280, 720), t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(n)
        ),
    )
    track.save(path)


def _save_manifest(path: Path, n: int):
    ShotsManifest(
        source_file="fake.mp4", fps=30.0, total_frames=n,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=n - 1,
                    start_time=0.0, end_time=(n - 1) / 30.0)],
    ).save(path)


def _minimal_cfg() -> dict:
    return {
        "ball": {
            "detector": "fake",
            "ball_radius_m": 0.11,
            "max_gap_frames": 6,
            "flight_max_residual_px": 5.0,
            "tracker": {
                "process_noise_grounded_px": 4.0,
                "process_noise_flight_px": 12.0,
                "measurement_noise_px": 2.0,
                "gating_sigma": 4.0,
                "min_flight_frames": 6,
                "max_flight_frames": 90,
            },
            "spin": {"enabled": False, "min_flight_seconds": 0.5,
                     "min_residual_improvement": 0.2,
                     "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
            "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
            "flight_promotion": {"enabled": False, "min_run_frames": 6,
                                 "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
            "kick_anchor": {"enabled": False, "max_pixel_distance_px": 30.0, "lookahead_frames": 4,
                            "min_pixel_acceleration_px_per_frame": 6.0, "foot_anchor_z_m": 0.11},
            "appearance_bridge": {"enabled": False, "max_gap_frames": 8, "template_size_px": 32,
                                  "search_radius_px": 64, "min_ncc": 0.6,
                                  "template_max_age_frames": 30, "template_update_confidence": 0.5},
        },
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


@pytest.mark.integration
def test_anchor_overrides_wasb_detection(tmp_path: Path):
    """When an anchor exists at a frame, its pixel position is used and
    WASB is ignored on that frame."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # WASB will return one (wrong) pixel for every frame.
    wasb_uv = (100.0, 100.0)
    wasb_detections = [(wasb_uv[0], wasb_uv[1], 0.85) for _ in range(n_frames)]

    # Anchor at frame 5 says the ball is at a different pixel position.
    anchor_uv = (640.0, 360.0)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=5, image_xy=anchor_uv, state="grounded"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(wasb_detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    f5 = next(f for f in track.frames if f.frame == 5)
    # Recover the pixel by reprojection: ground-projection of the
    # anchor pixel at z=0.11 should equal f5.world_xyz exactly.
    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected_world = ankle_ray_to_pitch(anchor_uv, K=K, R=R, t=t, plane_z=0.11)
    assert f5.state == "grounded"
    assert np.allclose(f5.world_xyz, expected_world, atol=1e-3)


@pytest.mark.integration
def test_no_anchor_file_runs_normally(tmp_path: Path):
    """When no anchor file exists, ball stage runs exactly as before."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # No anchors → WASB pixel used → at least one frame is grounded.
    assert any(f.state == "grounded" for f in track.frames)


@pytest.mark.integration
def test_off_screen_flight_anchor_skips_pixel(tmp_path: Path):
    """An off_screen_flight anchor produces no pixel detection but
    forces the frame into a flight run."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # WASB returns None for every frame.
    detections: list[tuple[float, float, float] | None] = [None] * n_frames

    anchors = [
        BallAnchor(frame=fi, image_xy=None, state="off_screen_flight")
        for fi in range(10, 20)
    ]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=tuple(anchors),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frames 10..19 should not be state="missing".
    forced = [f for f in track.frames if 10 <= f.frame <= 19]
    assert all(f.state != "missing" for f in forced), (
        f"off-screen-flight anchors must not emit missing: {[f.state for f in forced]}"
    )


@pytest.mark.integration
def test_kick_event_anchored_pins_p0(tmp_path: Path, monkeypatch):
    """A 'kick' anchor at the start of a flight segment pins p0 to the
    pixel ray-cast at z=0.11.

    The IMM is bypassed (via monkeypatch) so the test isolates Layer 5
    knot-frame logic, not the flight-detection heuristic.
    """
    from src.stages.ball import BallStage as _BallStage

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 40
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    p0_true = np.array([52.5, 34.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    g_vec = np.array([0.0, 0.0, -9.81])

    def _proj(p):
        cam = R @ p + t; pix = K @ cam
        return (float(pix[0] / pix[2]), float(pix[1] / pix[2]))

    detections: list[tuple[float, float, float] | None] = [None] * 5
    for i in range(25):
        dt = i / 30.0
        pt = p0_true + v0_true * dt + 0.5 * g_vec * dt ** 2
        u, v = _proj(pt)
        detections.append((u, v, 0.85))
    while len(detections) < n_frames:
        detections.append(None)

    # Anchor the kick at frame 5 (the flight segment start).
    kick_uv = _proj(p0_true)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=5, image_xy=kick_uv, state="kick"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    # Force a single flight run covering frames 5-29 so the IMM is bypassed.
    monkeypatch.setattr(
        _BallStage, "_flight_runs",
        lambda self_arg, steps, min_flight, max_flight: [(5, 29)],
    )

    cfg = _minimal_cfg()
    BallStage(
        config=cfg, output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    assert len(track.flight_segments) >= 1
    # Find the flight segment that includes frame 5 (the kick).
    seg = next((s for s in track.flight_segments if s.frame_range[0] <= 5 <= s.frame_range[1]), None)
    assert seg is not None, "expected a flight segment covering the kick at frame 5"
    p0_fit = np.array(seg.parabola["p0"])
    assert np.linalg.norm(p0_fit - p0_true) < 0.5, (
        f"expected anchored p0 ≈ {p0_true.tolist()}, got {p0_fit.tolist()}"
    )


@pytest.mark.integration
def test_bounce_event_splits_flight_run(tmp_path: Path, monkeypatch):
    """A 'bounce' anchor mid-run should terminate one flight segment and
    start a new one (segment(s) on both sides of the bounce frame).

    The IMM is bypassed (via monkeypatch) so the test isolates the
    event-splitting logic in Layer 5, not flight-detection heuristics.
    """
    from src.stages.ball import BallStage as _BallStage

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 60
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    def _proj(p):
        cam = R @ p + t; pix = K @ cam
        return (float(pix[0] / pix[2]), float(pix[1] / pix[2]))

    g_vec = np.array([0.0, 0.0, -9.81])

    # Arc A is designed so the ball naturally returns to z=0.11 at exactly
    # bounce_frame (v0z = 0.5 * 9.81 → 1-second symmetric arc).  This keeps
    # the bounce anchor pixel within ~3 px of the previous frame so the
    # Kalman tracker doesn't jump, avoiding corrupted post-bounce tracker UVs.
    bounce_frame = 30
    v0z_a = 0.5 * 9.81  # ≈ 4.905 m/s
    p0_a = np.array([52.5, 34.0, 0.11]); v0_a = np.array([3.0, 0.5, v0z_a])
    p0_b = p0_a + v0_a * (bounce_frame / 30.0) + 0.5 * g_vec * (bounce_frame / 30.0) ** 2
    p0_b[2] = 0.11  # pin to ground after bounce
    v0_b = np.array([2.0, 0.5, v0z_a])  # same upward kick for arc B

    detections: list[tuple[float, float, float] | None] = [None] * 5
    for i in range(25):
        dt = i / 30.0
        pt = p0_a + v0_a * dt + 0.5 * g_vec * dt ** 2
        u, v = _proj(pt); detections.append((u, v, 0.85))
    detections.append(None)  # bounce frame placeholder
    for i in range(20):
        dt = i / 30.0
        pt = p0_b + v0_b * dt + 0.5 * g_vec * dt ** 2
        u, v = _proj(pt); detections.append((u, v, 0.85))
    while len(detections) < n_frames:
        detections.append(None)

    bounce_uv = _proj(np.array([p0_b[0], p0_b[1], 0.11]))
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=bounce_frame, image_xy=bounce_uv, state="bounce"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    # Force a single large flight run (frames 5-54) that the bounce anchor
    # must split into two sub-runs: [5-29] and [31-54].
    monkeypatch.setattr(
        _BallStage, "_flight_runs",
        lambda self_arg, steps, min_flight, max_flight: [(5, 54)],
    )

    cfg = _minimal_cfg()
    BallStage(
        config=cfg, output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    pre_segs = [s for s in track.flight_segments if s.frame_range[1] < bounce_frame]
    post_segs = [s for s in track.flight_segments if s.frame_range[0] > bounce_frame]
    assert pre_segs, f"expected a flight segment ending before frame {bounce_frame}"
    assert post_segs, f"expected a flight segment starting after frame {bounce_frame}"


@pytest.mark.integration
def test_consecutive_grounded_anchors_interpolate_world(tmp_path: Path):
    """Frames between two grounded anchors (no anchors in between) are
    overridden with linearly-interpolated world XY at z=ball_radius,
    regardless of what WASB or the IMM produced."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Anchor at frame 5 and frame 15 with deliberately different pixel
    # positions (so the world positions clearly differ). In between,
    # WASB returns wildly different (off-pitch) pixels.
    anchor_a_uv = (640.0, 360.0)
    anchor_b_uv = (700.0, 380.0)
    detections: list[tuple[float, float, float] | None] = []
    for i in range(n_frames):
        # Wild WASB pixels in the gap to make sure they get overridden.
        detections.append((50.0, 50.0, 0.85))

    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5, image_xy=anchor_a_uv, state="grounded"),
            BallAnchor(frame=15, image_xy=anchor_b_uv, state="grounded"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    from src.utils.foot_anchor import ankle_ray_to_pitch
    pa = ankle_ray_to_pitch(anchor_a_uv, K=K, R=R, t=t, plane_z=0.11)
    pb = ankle_ray_to_pitch(anchor_b_uv, K=K, R=R, t=t, plane_z=0.11)

    # Sample frame 10 — exactly halfway between anchors. Should equal
    # the midpoint of (pa, pb).
    f10 = next(f for f in track.frames if f.frame == 10)
    expected_mid = (pa + pb) / 2.0
    assert f10.state == "grounded"
    assert f10.world_xyz is not None
    assert np.allclose(f10.world_xyz, expected_mid, atol=0.05), (
        f"expected linear-interp midpoint {expected_mid.tolist()}, got {list(f10.world_xyz)}"
    )

    # Sample frame 12 — 70% of the way from a to b.
    f12 = next(f for f in track.frames if f.frame == 12)
    t12 = (12 - 5) / (15 - 5)
    expected_12 = pa * (1 - t12) + pb * t12
    assert np.allclose(f12.world_xyz, expected_12, atol=0.05)


@pytest.mark.integration
def test_grounded_to_event_anchor_no_interpolation(tmp_path: Path):
    """A grounded anchor followed by a non-grounded anchor (e.g. kick)
    should NOT interpolate between them — the user marked a state change
    so we leave WASB alone in the gap."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections: list[tuple[float, float, float] | None] = [
        (640.0, 360.0, 0.85) for _ in range(n_frames)
    ]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5, image_xy=(640.0, 360.0), state="grounded"),
            BallAnchor(frame=15, image_xy=(700.0, 380.0), state="kick"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frames in (5, 15) should use the WASB-driven world position
    # (which lands at the WASB pixel ground-projected), NOT a linear
    # interpolation between the two anchors. The simplest check: the
    # midpoint frame's world should NOT be near the linear midpoint.
    from src.utils.foot_anchor import ankle_ray_to_pitch
    pa = ankle_ray_to_pitch((640.0, 360.0), K=K, R=R, t=t, plane_z=0.11)
    pb_kick = ankle_ray_to_pitch((700.0, 380.0), K=K, R=R, t=t, plane_z=0.11)
    expected_interp_mid = (pa + pb_kick) / 2.0
    f10 = next(f for f in track.frames if f.frame == 10)
    # f10 should equal pa (since WASB returns (640, 360) every frame =
    # the same as anchor A), not the interpolated midpoint.
    assert f10.world_xyz is not None
    actual = np.array(f10.world_xyz)
    # Distinguish by checking against WASB pixel rather than interp.
    wasb_world = pa  # WASB returns (640, 360) → same projection as anchor A
    assert np.linalg.norm(actual - wasb_world) < 0.1, (
        f"expected WASB-driven world {wasb_world}, got {actual}"
    )


@pytest.mark.integration
def test_isolated_airborne_anchor_no_placeholder_segment(tmp_path: Path):
    """A single airborne_mid anchor should mark the frame state=flight
    but must NOT add a placeholder FlightSegment with zero p0/v0."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=10, image_xy=(700.0, 200.0), state="airborne_mid"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    f10 = next(f for f in track.frames if f.frame == 10)
    assert f10.state == "flight", "airborne anchor should mark state=flight"
    # No flight segment should have been created — segments table is
    # for real parabola fits, not single-frame placeholders.
    junk = [
        s for s in track.flight_segments
        if s.parabola.get("p0") == [0.0, 0.0, 0.0]
    ]
    assert not junk, f"expected no zero-parabola placeholder segments, got {junk}"


@pytest.mark.integration
def test_consecutive_airborne_anchors_pin_p0_at_start(tmp_path: Path):
    """Two airborne anchors (no kick, no hard-knot end) — Phase 2 pins
    p0 to the first anchor's bucket-midpoint ray-cast (airborne-start
    promotion) and fits v0 from the remaining pixel obs + bucket-range
    hinge. The result is a smooth parabola filling every frame in the
    span, not the old depth-zigzag from bucket-midpoint linear interp.
    """
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="airborne_low"),
            BallAnchor(frame=10, image_xy=(720.0, 300.0), state="airborne_mid"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    segs = [s for s in track.flight_segments if s.frame_range == (5, 10)]
    assert segs, f"expected one Phase 2 segment 5-10, got {[s.frame_range for s in track.flight_segments]}"
    seg = segs[0]

    # p0 should equal airborne_low ray-cast at z=1 m (bucket midpoint).
    from src.utils.ball_anchor_heights import state_to_height
    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected_p0 = ankle_ray_to_pitch(
        (640.0, 360.0), K=K, R=R, t=t,
        plane_z=state_to_height("airborne_low"),
    )
    p0 = np.array(seg.parabola["p0"])
    assert np.allclose(p0, expected_p0, atol=0.05)

    # All in-span frames are state=flight with a real world position
    # (no more bucket-midpoint zigzag — the parabola fills every frame).
    for fi in range(5, 11):
        f = next(x for x in track.frames if x.frame == fi)
        assert f.state == "flight", f"frame {fi} expected flight, got {f.state}"
        assert f.world_xyz is not None, f"frame {fi}: expected filled by parabola"


@pytest.mark.integration
def test_grounded_then_airborne_does_not_interpolate_world(tmp_path: Path):
    """A grounded anchor adjacent to an airborne anchor must NOT
    span-interpolate (state boundary). The in-between frames go through
    WASB as usual."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="grounded"),
            BallAnchor(frame=10, image_xy=(700.0, 300.0), state="airborne_mid"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frame 7: NOT in any flight (no airborne→airborne or grounded→grounded
    # contiguous pair covering it). Should be state=grounded from WASB.
    f7 = next(f for f in track.frames if f.frame == 7)
    assert f7.state == "grounded", (
        f"grounded→airborne boundary should leave in-between frames alone, got {f7.state}"
    )


@pytest.mark.integration
def test_phase2_parabola_fit_through_anchor_bracketed_flight(tmp_path: Path):
    """A kick + several airborne anchors + bounce should produce a
    single FlightSegment whose parabola matches the synthesised
    trajectory within 1 m at every frame in the span."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 70
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Synthesise a true parabolic flight from frame 5 (kick) back to
    # ground at frame 54. v0_z = 8 m/s gives a full cycle of
    # 2*8/9.81 = 1.631 s ≈ 49 frames at 30 fps. Apex ≈ frame 29 at z ≈ 3.37 m.
    p0_true = np.array([30.0, 30.0, 0.11])
    v0_true = np.array([5.0, 1.0, 8.0])
    g_vec = np.array([0.0, 0.0, -9.81])
    kick_frame = 5
    bounce_frame = 54

    def project(p):
        cam = R @ p + t; pix = K @ cam
        return (float(pix[0] / pix[2]), float(pix[1] / pix[2]))

    def world_at(fi):
        dt = (fi - kick_frame) / 30.0
        return p0_true + v0_true * dt + 0.5 * g_vec * dt ** 2

    detections: list[tuple[float, float, float] | None] = []
    for i in range(n_frames):
        if kick_frame <= i <= bounce_frame:
            u, v = project(world_at(i))
            detections.append((u, v, 0.85))
        else:
            detections.append((640.0, 360.0, 0.85))

    anchor_states = [
        (kick_frame,         "kick"),
        (kick_frame +  6,    "airborne_low"),
        (kick_frame + 14,    "airborne_mid"),
        (kick_frame + 24,    "airborne_high"),
        (kick_frame + 34,    "airborne_mid"),
        (kick_frame + 43,    "airborne_low"),
        (bounce_frame,       "bounce"),
    ]
    anchors = []
    for fi, state in anchor_states:
        wp = world_at(fi)
        if state == "bounce":
            wp = np.array([wp[0], wp[1], 0.11])
        anchors.append(BallAnchor(frame=fi, image_xy=project(wp), state=state))
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720), anchors=tuple(anchors),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    segs = [s for s in track.flight_segments if s.frame_range == (kick_frame, bounce_frame)]
    assert len(segs) == 1, (
        f"expected one Phase-2 FlightSegment covering {kick_frame}-{bounce_frame}, "
        f"got {[(s.id, s.frame_range) for s in track.flight_segments]}"
    )
    seg = segs[0]
    p0_fit = np.array(seg.parabola["p0"])
    assert np.linalg.norm(p0_fit - p0_true) < 0.5

    for fi in range(kick_frame, bounce_frame + 1):
        f = next(x for x in track.frames if x.frame == fi)
        assert f.state == "flight", f"frame {fi} expected flight, got {f.state}"

    # Per-frame world matches the synthesised trajectory within 1 m.
    for fi in range(kick_frame, bounce_frame + 1):
        f = next(x for x in track.frames if x.frame == fi)
        assert f.world_xyz is not None
        expected = world_at(fi)
        err = float(np.linalg.norm(np.array(f.world_xyz) - expected))
        assert err < 1.0, (
            f"frame {fi}: world {list(f.world_xyz)} vs truth {expected.tolist()}, err={err:.2f} m"
        )

    span_zs = [
        f.world_xyz[2] for f in track.frames
        if kick_frame <= f.frame <= bounce_frame and f.world_xyz is not None
    ]
    assert max(span_zs) >= 3.0, f"expected apex z >= 3 m, got max={max(span_zs):.2f}"


@pytest.mark.integration
def test_phase2_fits_two_anchor_kick_to_bounce_span(tmp_path: Path):
    """A 2-anchor span where both endpoints are hard knots (kick at
    z=0.11 + bounce at z=0.11) fully determines v0 — Phase 2 should
    accept it. Previously the >= 3 obs threshold rejected these short
    flights, leaving the operator with state=flight gaps for short
    kick→bounce sequences."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)
    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
            BallAnchor(frame=15, image_xy=(700.0, 360.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    segs = [s for s in track.flight_segments if s.frame_range == (5, 15)]
    assert segs, (
        f"two-anchor kick→bounce span should produce a Phase 2 FlightSegment; "
        f"got {[s.frame_range for s in track.flight_segments]}"
    )
    # Every frame in 5..15 should have a world position from the parabola
    # (no gaps, no None values).
    for fi in range(5, 16):
        f = next(x for x in track.frames if x.frame == fi)
        assert f.state == "flight", f"frame {fi} expected flight, got {f.state}"
        assert f.world_xyz is not None, (
            f"frame {fi}: 2-anchor span should fill world_xyz from parabola, got None"
        )


@pytest.mark.integration
def test_phase2_splits_span_at_kick_and_bounce(tmp_path: Path):
    """A run of non-grounded anchors that contains multiple kicks and
    a bounce must split into separate flights, not lump into one giant
    parabola fit covering all events."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 80
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    # Anchor pattern mirrors the user's real clip shape:
    #   kick → airborne → kick → bounce → kick → airborne → bounce
    # Phase 2 should build THREE spans: (k1..before_k2), (k2..bounce1),
    # (k3..bounce2). NOT one merged 5-event span.
    anchors = (
        BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
        BallAnchor(frame=10, image_xy=(650.0, 320.0), state="airborne_low"),
        BallAnchor(frame=15, image_xy=(660.0, 300.0), state="airborne_mid"),
        BallAnchor(frame=20, image_xy=(670.0, 340.0), state="airborne_low"),
        BallAnchor(frame=25, image_xy=(680.0, 360.0), state="kick"),
        BallAnchor(frame=30, image_xy=(690.0, 360.0), state="bounce"),
        BallAnchor(frame=40, image_xy=(700.0, 360.0), state="kick"),
        BallAnchor(frame=45, image_xy=(710.0, 320.0), state="airborne_low"),
        BallAnchor(frame=50, image_xy=(720.0, 360.0), state="bounce"),
    )
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720), anchors=anchors,
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    seg_ranges = sorted(s.frame_range for s in track.flight_segments)
    # We expect three distinct flight ranges, each covering ONE physical
    # flight (kick to next event). At minimum: no single segment covering
    # both kicks.
    assert not any(
        a <= 5 and b >= 40 for (a, b) in seg_ranges
    ), (
        f"a single FlightSegment must not span multiple kicks; got {seg_ranges}"
    )
    # Each kick frame should be the START of its own span (or absent if
    # the fit was rejected by plausibility — but the test mainly asserts
    # there's no monster span).
    starts = {a for (a, b) in seg_ranges}
    # If Phase 2 succeeded for any of the three, the start frames must
    # be a subset of {5, 25, 40} — the kick frames.
    assert starts <= {5, 25, 40, 457}, f"unexpected span start: {seg_ranges}"


@pytest.mark.integration
def test_phase2_splits_at_header_and_pins_endpoint(tmp_path: Path):
    """A header anchor mid-flight should split the span into two
    parabolas. The pre-header parabola must END at the header's
    z=2.5 ray-cast; the post-header parabola must START from it."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 60
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Anchor sequence: kick at 5, three airborne_low rising to header at
    # 20, three airborne_low descending to bounce at 35. Header must be
    # the boundary between the two sub-spans.
    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    anchors = (
        BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
        BallAnchor(frame=10, image_xy=(650.0, 340.0), state="airborne_low"),
        BallAnchor(frame=15, image_xy=(660.0, 320.0), state="airborne_low"),
        BallAnchor(frame=20, image_xy=(670.0, 310.0), state="header"),
        BallAnchor(frame=25, image_xy=(680.0, 320.0), state="airborne_low"),
        BallAnchor(frame=30, image_xy=(690.0, 340.0), state="airborne_low"),
        BallAnchor(frame=35, image_xy=(700.0, 360.0), state="bounce"),
    )
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720), anchors=anchors,
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    seg_ranges = sorted(s.frame_range for s in track.flight_segments)

    # Expect two segments: kick→header (5–20) and header→bounce (20–35).
    # The header frame 20 is in BOTH spans.
    pre = [s for s in track.flight_segments if s.frame_range == (5, 20)]
    post = [s for s in track.flight_segments if s.frame_range == (20, 35)]
    assert pre, f"expected pre-header segment 5-20, got {seg_ranges}"
    assert post, f"expected post-header segment 20-35, got {seg_ranges}"

    # The header anchor's frame should evaluate to z ≈ 2.5 m on BOTH
    # parabolas (the hard-knot pin), and the world XY should be the
    # ray-cast of the header pixel at z=2.5.
    from src.utils.ball_anchor_heights import state_to_height
    from src.utils.foot_anchor import ankle_ray_to_pitch
    header_world = ankle_ray_to_pitch(
        (670.0, 310.0), K=K, R=R, t=t,
        plane_z=state_to_height("header"),
    )
    f20 = next(f for f in track.frames if f.frame == 20)
    assert f20.world_xyz is not None
    assert np.allclose(f20.world_xyz, header_world, atol=0.3), (
        f"header anchor frame should pin world to ray-cast at z=2.5; "
        f"expected {header_world}, got {list(f20.world_xyz)}"
    )


@pytest.mark.integration
def test_phase2_pins_p0_at_airborne_start_when_no_kick_present(tmp_path: Path):
    """When a flight span starts with an airborne_* anchor (the user
    didn't place a kick), the parabola's p0 should be pinned at the
    airborne anchor's bucket-midpoint ray-cast. Without this, the LM
    can drift p0 several metres along the camera ray."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 60
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    a_start_uv = (640.0, 360.0)
    anchors = (
        BallAnchor(frame=5,  image_xy=a_start_uv,        state="airborne_low"),
        BallAnchor(frame=10, image_xy=(650.0, 330.0),    state="airborne_mid"),
        BallAnchor(frame=15, image_xy=(660.0, 320.0),    state="airborne_mid"),
        BallAnchor(frame=20, image_xy=(670.0, 340.0),    state="airborne_low"),
        BallAnchor(frame=25, image_xy=(680.0, 360.0),    state="bounce"),
    )
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720), anchors=anchors,
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    segs = [s for s in track.flight_segments if s.frame_range == (5, 25)]
    assert segs, f"expected one Phase 2 segment 5-25, got {[s.frame_range for s in track.flight_segments]}"
    seg = segs[0]

    # The parabola's p0 should equal the airborne_low ray-cast at z=1 m.
    from src.utils.ball_anchor_heights import state_to_height
    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected_p0 = ankle_ray_to_pitch(
        a_start_uv, K=K, R=R, t=t,
        plane_z=state_to_height("airborne_low"),
    )
    p0 = np.array(seg.parabola["p0"])
    assert np.allclose(p0, expected_p0, atol=0.05), (
        f"airborne-start parabola should be pinned to bucket-midpoint ray-cast; "
        f"expected {expected_p0}, got {p0.tolist()}"
    )

    # And the kick frame's emitted world position should match p0 exactly.
    f5 = next(f for f in track.frames if f.frame == 5)
    assert f5.world_xyz is not None
    assert np.allclose(f5.world_xyz, expected_p0, atol=0.05)

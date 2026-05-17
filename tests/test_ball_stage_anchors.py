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
def test_grounded_to_kick_interpolates_at_ground_level(tmp_path: Path):
    """A grounded anchor followed by a kick anchor — both at z=0.11 m
    — should linearly interpolate world XY between them. The ball is
    on the pitch surface throughout (rolling toward the kicker), so
    the smooth interp at ground level is correct."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # WASB returns wild off-pitch pixels to prove the interp overrides it.
    detections: list[tuple[float, float, float] | None] = [
        (50.0, 50.0, 0.85) for _ in range(n_frames)
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
    from src.utils.foot_anchor import ankle_ray_to_pitch
    pa = ankle_ray_to_pitch((640.0, 360.0), K=K, R=R, t=t, plane_z=0.11)
    pb_kick = ankle_ray_to_pitch((700.0, 380.0), K=K, R=R, t=t, plane_z=0.11)
    expected_mid = (pa + pb_kick) / 2.0
    f10 = next(f for f in track.frames if f.frame == 10)
    assert f10.world_xyz is not None
    actual = np.array(f10.world_xyz)
    assert np.allclose(actual, expected_mid, atol=0.05), (
        f"expected ground-level interp midpoint {expected_mid}, got {actual}"
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


@pytest.mark.parametrize("event_state,plane_z", [("volley", 1.0), ("chest", 1.3)])
@pytest.mark.integration
def test_volley_and_chest_split_spans_and_pin_endpoint(tmp_path: Path, event_state, plane_z):
    """volley (z=1.0 m) and chest (z=1.3 m) split flight spans the same
    way header does — incoming parabola lands at the contact, outgoing
    parabola starts from it. The contact frame is in both sub-spans and
    its world position is pinned via the state's height ray-cast."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 60
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    anchors = (
        BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
        BallAnchor(frame=10, image_xy=(650.0, 340.0), state="airborne_low"),
        BallAnchor(frame=15, image_xy=(660.0, 330.0), state="airborne_low"),
        BallAnchor(frame=20, image_xy=(670.0, 340.0), state=event_state),
        BallAnchor(frame=25, image_xy=(680.0, 340.0), state="airborne_low"),
        BallAnchor(frame=30, image_xy=(690.0, 350.0), state="airborne_low"),
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
    pre = [s for s in track.flight_segments if s.frame_range == (5, 20)]
    post = [s for s in track.flight_segments if s.frame_range == (20, 35)]
    assert pre, f"expected pre-event segment 5-20, got {[s.frame_range for s in track.flight_segments]}"
    assert post, f"expected post-event segment 20-35, got {[s.frame_range for s in track.flight_segments]}"

    # The contact frame's world position should pin to the ray-cast of
    # the anchor pixel at the state's height plane.
    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected_world = ankle_ray_to_pitch(
        (670.0, 340.0), K=K, R=R, t=t, plane_z=plane_z,
    )
    f20 = next(f for f in track.frames if f.frame == 20)
    assert f20.world_xyz is not None
    assert np.allclose(f20.world_xyz, expected_world, atol=0.3), (
        f"{event_state} should pin world to ray-cast at z={plane_z}; "
        f"expected {expected_world}, got {list(f20.world_xyz)}"
    )


@pytest.mark.integration
def test_bounce_to_kick_interpolates_at_ground_level(tmp_path: Path):
    """A bounce → kick pair (both at z=0.11) should interpolate world
    XY along the pitch surface — the ball rolled from the bounce to the
    kicker."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 25
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(50.0, 50.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="bounce"),
            BallAnchor(frame=15, image_xy=(700.0, 380.0), state="kick"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")
    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    from src.utils.foot_anchor import ankle_ray_to_pitch
    pa = ankle_ray_to_pitch((640.0, 360.0), K=K, R=R, t=t, plane_z=0.11)
    pb = ankle_ray_to_pitch((700.0, 380.0), K=K, R=R, t=t, plane_z=0.11)
    expected_mid = (pa + pb) / 2.0
    f10 = next(f for f in track.frames if f.frame == 10)
    assert f10.world_xyz is not None
    assert np.allclose(np.array(f10.world_xyz), expected_mid, atol=0.05)


@pytest.mark.integration
def test_bounce_to_volley_fits_rising_parabola(tmp_path: Path):
    """A bounce followed by a volley represents a flight that rises
    from the ground (z=0.11) to the volley apex (z=1.0). The span
    [bounce, volley] should be built and Phase 2 should fit the
    rising arc — frames between bounce and volley get parabola world
    positions, not gaps."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    bounce_uv = (640.0, 360.0)
    volley_uv = (680.0, 320.0)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=bounce_uv, state="bounce"),
            BallAnchor(frame=15, image_xy=volley_uv, state="volley"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    segs = [s for s in track.flight_segments if s.frame_range == (5, 15)]
    assert segs, (
        f"expected one Phase 2 segment 5-15 (bounce→volley apex), "
        f"got {[s.frame_range for s in track.flight_segments]}"
    )

    # Every frame in the bounce→volley span should be state=flight with a
    # filled world position from the parabola — no gaps.
    for fi in range(5, 16):
        f = next(x for x in track.frames if x.frame == fi)
        assert f.state == "flight", f"frame {fi}: expected flight, got {f.state}"
        assert f.world_xyz is not None, (
            f"frame {fi}: parabola should fill world_xyz; got None"
        )

    # Bounce frame's world position is at z=0.11 (its hard-knot height).
    # Volley frame's world position is at z=1.0 (its hard-knot height).
    # The midpoint should sit somewhere in between.
    f5 = next(f for f in track.frames if f.frame == 5)
    f15 = next(f for f in track.frames if f.frame == 15)
    assert f5.world_xyz[2] < 0.5, f"bounce frame z should be ~0.11, got {f5.world_xyz[2]}"
    assert 0.5 < f15.world_xyz[2] < 1.5, f"volley frame z should be ~1.0, got {f15.world_xyz[2]}"


@pytest.mark.integration
def test_player_touch_drives_trajectory_through_bone_world(tmp_path: Path):
    """A 'player_touch' anchor pins the parabola at the named bone's
    actual world position (SMPL FK), not at a fixed state-height
    bucket. The pixel ray-cast is ignored — the bone's XYZ wins."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Synthesise a player whose r_foot is at world (40, 30, 1.0) at the
    # anchor frame — clearly airborne (above the 0.4 m ground-touch
    # threshold), modelling a foot volley. We achieve this by placing
    # root_t such that the canonical-y-up r_foot offset transforms to
    # that world position.
    from src.utils.smpl_skeleton import (
        SMPL_JOINT_NAMES, SMPL_REST_JOINTS_YUP, compute_joint_world,
    )
    from src.schemas.smpl_world import SmplWorldTrack
    r_foot_idx = SMPL_JOINT_NAMES.index("r_foot")
    # With thetas=0 and root_R=I, joint world = canonical_yup + root_t.
    # We want r_foot world = (40, 30, 1.0) → root_t = (40, 30, 1.0) - rest.
    rest_r_foot = SMPL_REST_JOINTS_YUP[r_foot_idx]
    target_world = np.array([40.0, 30.0, 1.0])
    root_t_one = target_world - rest_r_foot
    smpl_dir = out / "hmr_world"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    SmplWorldTrack(
        player_id="P007",
        frames=np.arange(n_frames, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.zeros((n_frames, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        root_t=np.tile(root_t_one.astype(np.float32), (n_frames, 1)),
        confidence=np.ones(n_frames, dtype=np.float32),
        shot_id="play",
    ).save(smpl_dir / "play__P007_smpl_world.npz")

    # Build anchors: kick at frame 5, player_touch at frame 15 (the volley
    # apex via the player's r_foot), bounce at frame 25.
    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
            # The image_xy here is deliberately wrong — the bone world
            # should win, not the pixel ray-cast.
            BallAnchor(
                frame=15, image_xy=(50.0, 50.0),
                state="player_touch", player_id="P007", bone="r_foot",
            ),
            BallAnchor(frame=25, image_xy=(700.0, 380.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frame 15 should be at the bone's world position, NOT the ray-cast
    # of (50, 50).
    expected = compute_joint_world(
        np.zeros((24, 3)), np.eye(3), root_t_one, r_foot_idx,
    )
    f15 = next(f for f in track.frames if f.frame == 15)
    assert f15.world_xyz is not None
    assert np.allclose(f15.world_xyz, expected, atol=1e-3), (
        f"player_touch should pin world to bone position {expected.tolist()}, "
        f"got {list(f15.world_xyz)}"
    )
    assert f15.state == "flight"


@pytest.mark.integration
def test_player_touch_requires_player_and_bone(tmp_path: Path):
    """Loading an anchors file with state='player_touch' but missing
    player_id or bone raises ValueError at load time."""
    out = tmp_path / "out"
    (out / "ball").mkdir(parents=True)
    (out / "ball" / "play_ball_anchors.json").write_text(
        json.dumps({
            "clip_id": "play", "image_size": [1280, 720],
            "anchors": [
                {"frame": 5, "image_xy": [640.0, 360.0], "state": "player_touch"},
            ],
        })
    )
    with pytest.raises(ValueError, match="player_id is required"):
        BallAnchorSet.load(out / "ball" / "play_ball_anchors.json")


@pytest.mark.integration
def test_player_touch_prefers_refined_pose_over_hmr_world(tmp_path: Path):
    """When both ``refined_poses/{pid}_refined.npz`` and
    ``hmr_world/{shot}__{pid}_smpl_world.npz`` exist, the ball stage
    looks up bone positions from refined_poses (post-cleanup track).
    The lookup also applies the sync-map offset to translate the
    shot-local anchor frame into the refined timeline's reference
    frame indices."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    from src.utils.smpl_skeleton import (
        SMPL_JOINT_NAMES, SMPL_REST_JOINTS_YUP, compute_joint_world,
    )
    from src.schemas.smpl_world import SmplWorldTrack
    from src.schemas.refined_pose import RefinedPose
    from src.schemas.sync_map import Alignment, SyncMap

    r_foot_idx = SMPL_JOINT_NAMES.index("r_foot")
    rest_r_foot = SMPL_REST_JOINTS_YUP[r_foot_idx]

    # Refined pose says r_foot is at world (40, 30, 1.2). Reference
    # timeline frames are [0..n_frames). With sync offset = 3, the
    # shot-local anchor at frame 15 looks up ref frame 12.
    refined_target = np.array([40.0, 30.0, 1.2])
    refined_root_t = (refined_target - rest_r_foot).astype(np.float32)
    refined_dir = out / "refined_poses"
    refined_dir.mkdir(parents=True, exist_ok=True)
    RefinedPose(
        player_id="P007",
        frames=np.arange(n_frames, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.zeros((n_frames, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        root_t=np.tile(refined_root_t, (n_frames, 1)),
        confidence=np.ones(n_frames, dtype=np.float32),
        view_count=np.ones(n_frames, dtype=np.int32),
        contributing_shots=("play",),
    ).save(refined_dir / "P007_refined.npz")

    # hmr_world has a DIFFERENT r_foot world position. If the ball
    # stage reads from hmr_world by mistake, the assertion below will
    # land on this value instead of the refined one.
    hmr_target = np.array([10.0, 10.0, 2.0])
    hmr_root_t = (hmr_target - rest_r_foot).astype(np.float32)
    smpl_dir = out / "hmr_world"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    SmplWorldTrack(
        player_id="P007",
        frames=np.arange(n_frames, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.zeros((n_frames, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        root_t=np.tile(hmr_root_t, (n_frames, 1)),
        confidence=np.ones(n_frames, dtype=np.float32),
        shot_id="play",
    ).save(smpl_dir / "play__P007_smpl_world.npz")

    # Sync map: play has offset 3, so the anchor's local frame 15
    # corresponds to reference frame 12 in the refined track.
    SyncMap(
        reference_shot="play",
        alignments=[
            Alignment(shot_id="play", frame_offset=3, method="manual", confidence=1.0),
        ],
    ).save(out / "shots" / "sync_map.json")

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
            BallAnchor(
                frame=15, image_xy=(50.0, 50.0),
                state="player_touch", player_id="P007", bone="r_foot",
            ),
            BallAnchor(frame=25, image_xy=(700.0, 380.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    expected = compute_joint_world(
        np.zeros((24, 3)), np.eye(3), refined_root_t, r_foot_idx,
    )
    f15 = next(f for f in track.frames if f.frame == 15)
    assert f15.world_xyz is not None
    assert np.allclose(f15.world_xyz, expected, atol=1e-3), (
        f"ball stage should have used refined_poses (target {expected.tolist()}), "
        f"got {list(f15.world_xyz)}"
    )


@pytest.mark.integration
def test_player_touch_falls_back_when_smpl_track_missing(tmp_path: Path):
    """If the named player has no SmplWorldTrack on disk, the ball
    stage falls back to ray-casting the anchor pixel at the
    player_touch fallback height (1.0 m). Doesn't crash."""
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
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
            BallAnchor(
                frame=10, image_xy=(660.0, 360.0),
                state="player_touch", player_id="P999_missing", bone="r_foot",
            ),
            BallAnchor(frame=15, image_xy=(680.0, 360.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")
    # Stage runs without crashing — fallback ray-cast is used.
    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    f10 = next(f for f in track.frames if f.frame == 10)
    assert f10.world_xyz is not None  # something was emitted
    assert f10.state == "flight"


def _save_ground_player_track(
    out: Path, n_frames: int, *, player_id: str, target_xy: tuple[float, float],
    foot_z: float,
) -> np.ndarray:
    """Helper: save a SmplWorldTrack whose r_foot stays at a given XY/Z
    for the full clip. Returns the per-frame root_t (constant)."""
    from src.utils.smpl_skeleton import (
        SMPL_JOINT_NAMES, SMPL_REST_JOINTS_YUP,
    )
    from src.schemas.smpl_world import SmplWorldTrack
    r_foot_idx = SMPL_JOINT_NAMES.index("r_foot")
    rest_r_foot = SMPL_REST_JOINTS_YUP[r_foot_idx]
    target_world = np.array([target_xy[0], target_xy[1], foot_z])
    root_t_one = target_world - rest_r_foot
    smpl_dir = out / "hmr_world"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    SmplWorldTrack(
        player_id=player_id,
        frames=np.arange(n_frames, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.zeros((n_frames, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        root_t=np.tile(root_t_one.astype(np.float32), (n_frames, 1)),
        confidence=np.ones(n_frames, dtype=np.float32),
        shot_id="play",
    ).save(smpl_dir / f"play__{player_id}_smpl_world.npz")
    return root_t_one


@pytest.mark.integration
def test_ground_touch_player_touches_skip_parabola(tmp_path: Path):
    """Two consecutive player_touch anchors whose bone Z is below 0.4 m
    are treated as a ground pass / dribble: NO parabola flight segment
    is emitted between them, and the linear ground-interp fills the
    gap at z ~= ball radius. Models the user's stated case of using
    `player_touch` for short ground touches."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # r_foot at z=0.1 (well below the 0.4 m ground threshold).
    _save_ground_player_track(
        out, n_frames, player_id="P007",
        target_xy=(50.0, 30.0), foot_z=0.1,
    )

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(
                frame=5, image_xy=(640.0, 360.0),
                state="player_touch", player_id="P007", bone="r_foot",
            ),
            BallAnchor(
                frame=20, image_xy=(660.0, 360.0),
                state="player_touch", player_id="P007", bone="r_foot",
            ),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # No FlightSegment should span the gap — both touches are
    # classified as ground.
    parabola_segs = [
        s for s in track.flight_segments
        if not (s.frame_range[1] < 5 or s.frame_range[0] > 20)
    ]
    assert parabola_segs == [], (
        f"Expected no flight segments around ground touches, got: "
        f"{[(s.start_frame, s.end_frame) for s in parabola_segs]}"
    )
    # The anchored frames themselves stay at bone Z (~0.1).
    f5 = next(f for f in track.frames if f.frame == 5)
    f20 = next(f for f in track.frames if f.frame == 20)
    assert f5.world_xyz is not None and f5.world_xyz[2] < 0.4
    assert f20.world_xyz is not None and f20.world_xyz[2] < 0.4
    assert f5.state == "grounded"
    assert f20.state == "grounded"


@pytest.mark.integration
def test_grounded_pt_grounded_keeps_continuous_ground_track(tmp_path: Path):
    """The user's real ball-rolling scenario: a player_touch sandwiched
    between grounded anchors. The next-anchor rule classifies the
    touch as ground (next anchor is grounded), so it does NOT block
    the ground-level interp between the surrounding grounded
    anchors. Models the bug where HMR puts the foot at ~1.5 m world
    Z and an earlier bone-Z threshold misclassified every touch as
    airborne, leaving a gap after every contact."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # HMR puts the foot at z=1.5 m — the realistic over-elevation
    # case. A bone-Z threshold would misclassify this as airborne,
    # but the next-anchor rule correctly classifies it as ground.
    _save_ground_player_track(
        out, n_frames, player_id="P007",
        target_xy=(50.0, 30.0), foot_z=1.5,
    )

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=0, image_xy=(620.0, 360.0), state="grounded"),
            BallAnchor(
                frame=10, image_xy=(640.0, 360.0),
                state="player_touch", player_id="P007", bone="r_foot",
            ),
            BallAnchor(frame=20, image_xy=(660.0, 360.0), state="grounded"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Every frame between the grounded anchors must have a world
    # position at ground level (z ≈ ball_radius).
    for fi in range(0, 21):
        f = next(f for f in track.frames if f.frame == fi)
        assert f.world_xyz is not None, f"frame {fi} has no world_xyz"
        assert f.world_xyz[2] < 0.5, (
            f"frame {fi} z={f.world_xyz[2]:.3f} is not at ground level — "
            f"HMR foot Z leaked into the ball trajectory"
        )
    # The touch frame itself must not be marked as a flight.
    f10 = next(f for f in track.frames if f.frame == 10)
    assert f10.state == "grounded", (
        f"frame 10 state={f10.state} — ground touch must not be a flight"
    )
    # No flight segments between the grounded anchors.
    overlapping = [
        s for s in track.flight_segments
        if not (s.frame_range[1] < 0 or s.frame_range[0] > 20)
    ]
    assert overlapping == [], (
        f"expected no flight segments across the grounded span, got: "
        f"{[s.frame_range for s in overlapping]}"
    )


@pytest.mark.integration
def test_ground_touch_then_airborne_anchor_spans_continuously(tmp_path: Path):
    """A ground-touch player_touch followed by an airborne anchor and
    another ground-touch must be parabola-fit across the full
    sequence (ground-touch → flight → ground-touch). Regression
    guard: an earlier version closed the span at every ground-touch
    without starting a new one, leaving the airborne segment
    uncovered."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 80
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # r_foot at z=0.1 — ground touch.
    _save_ground_player_track(
        out, n_frames, player_id="P007",
        target_xy=(50.0, 30.0), foot_z=0.1,
    )

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(
                frame=10, image_xy=(640.0, 360.0),
                state="player_touch", player_id="P007", bone="r_foot",
            ),
            BallAnchor(frame=30, image_xy=(660.0, 360.0), state="airborne_mid"),
            BallAnchor(
                frame=60, image_xy=(680.0, 360.0),
                state="player_touch", player_id="P007", bone="r_foot",
            ),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Every frame between the two ground touches must have a world
    # position — gaps after either touch are the bug we're guarding
    # against.
    for fi in range(10, 61):
        f = next(f for f in track.frames if f.frame == fi)
        assert f.world_xyz is not None, (
            f"frame {fi} has no world_xyz — gap after ground touch"
        )


@pytest.mark.integration
def test_airborne_player_touch_still_fits_parabola(tmp_path: Path):
    """A player_touch anchor whose bone Z is above 0.4 m (e.g. a header
    or a chest control) is still treated as airborne — the parabola
    fit is unchanged. Sanity-checks that the ground-touch carve-out
    doesn't accidentally swallow all player_touch anchors."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # r_foot at z=2.0 — well above the 0.4 m threshold (modelling a
    # raised foot for a volley).
    _save_ground_player_track(
        out, n_frames, player_id="P007",
        target_xy=(50.0, 30.0), foot_z=2.0,
    )

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
            BallAnchor(
                frame=15, image_xy=(660.0, 360.0),
                state="player_touch", player_id="P007", bone="r_foot",
            ),
            BallAnchor(frame=25, image_xy=(680.0, 360.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # At least one parabola flight segment should cover the kick→touch
    # or touch→bounce span.
    parabola_segs = [
        s for s in track.flight_segments
        if not (s.frame_range[1] < 5 or s.frame_range[0] > 25)
    ]
    assert parabola_segs, "expected a flight segment around the airborne touch"
    f15 = next(f for f in track.frames if f.frame == 15)
    assert f15.state == "flight"
    assert f15.world_xyz is not None and f15.world_xyz[2] > 0.4


@pytest.mark.integration
def test_grounded_anchor_inside_flight_run_stays_at_ground(tmp_path: Path, monkeypatch):
    """A 'grounded' anchor placed inside an IMM-detected flight run must
    keep z = ball_radius. Anchors are the user's ground truth and must
    not be lifted into the air by a parabola fit that happens to span
    them."""
    from src.stages.ball import BallStage as _BallStage
    from src.stages import ball as _ball

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 40
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    def _proj(p):
        cam = R @ p + t
        pix = K @ cam
        return (float(pix[0] / pix[2]), float(pix[1] / pix[2]))

    detections = [(640.0 + i * 3.0, 360.0 - i * 2.0, 0.85) for i in range(n_frames)]
    grounded_uv = _proj(np.array([55.0, 35.0, 0.11]))
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=15, image_xy=grounded_uv, state="grounded"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    # Force a flight run spanning the anchor, and force the parabola fit
    # to return a clean low-residual airborne parabola — together these
    # guarantee the IMM-side write loop covers f=15 with z > ball_radius,
    # which is exactly the production bug we're testing for.
    monkeypatch.setattr(
        _BallStage, "_flight_runs",
        lambda self_arg, steps, min_flight, max_flight: [(5, 34)],
    )

    p0_fake = np.array([50.0, 34.0, 6.0])
    v0_fake = np.array([3.0, 0.5, 0.0])

    def fake_fit(obs, *, Ks, Rs, t_world, fps, distortion, **kwargs):
        return p0_fake.copy(), v0_fake.copy(), 1.0
    monkeypatch.setattr(_ball, "fit_parabola_to_image_observations", fake_fit)

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    overlapping = [
        s for s in track.flight_segments
        if s.frame_range[0] <= 15 <= s.frame_range[1]
    ]
    assert overlapping, (
        f"test setup did not produce a flight segment over f=15; "
        f"segments={[s.frame_range for s in track.flight_segments]}"
    )

    f15 = next(f for f in track.frames if f.frame == 15)
    assert f15.world_xyz is not None
    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected = ankle_ray_to_pitch(grounded_uv, K=K, R=R, t=t, plane_z=0.11)
    assert abs(f15.world_xyz[2] - 0.11) < 0.02, (
        f"grounded anchor lifted by parabola: z={f15.world_xyz[2]:.3f}"
    )
    assert np.allclose(f15.world_xyz, expected, atol=0.05), (
        f"grounded anchor world XY should match its pixel ray-cast at z=0.11; "
        f"expected {expected.tolist()}, got {list(f15.world_xyz)}"
    )


@pytest.mark.integration
def test_promotion_skips_runs_containing_grounded_anchors(tmp_path: Path, monkeypatch):
    """The grounded-to-flight promotion stage must not promote a run
    that overlaps a user-clicked 'grounded' anchor. Promoting it would
    contradict the user's explicit non-flight intent."""
    from src.stages import ball as _ball
    from src.utils.ball_plausibility import GroundedRun

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Smooth horizontal pixel motion — the IMM stays in the grounded branch.
    detections = [(640.0 + i * 5.0, 360.0, 0.85) for i in range(n_frames)]
    grounded_uv = (640.0 + 15 * 5.0, 360.0)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=15, image_xy=grounded_uv, state="grounded"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    # Force the promotion detector to flag [10, 25] as implausible.
    monkeypatch.setattr(
        _ball, "find_implausible_grounded_runs",
        lambda **kwargs: [GroundedRun(start=10, end=25)],
    )
    # And force the refit to return a clean airborne parabola so the
    # only way the run can be rejected is the anchor veto.
    p0_fake = np.array([50.0, 34.0, 6.0])
    v0_fake = np.array([3.0, 0.5, 0.0])

    def fake_fit(obs, *, Ks, Rs, t_world, fps, distortion, **kwargs):
        return p0_fake.copy(), v0_fake.copy(), 1.0
    monkeypatch.setattr(_ball, "fit_parabola_to_image_observations", fake_fit)

    cfg = _minimal_cfg()
    cfg["ball"]["flight_promotion"] = {
        "enabled": True,
        "min_run_frames": 6,
        "off_pitch_margin_m": 5.0,
        "max_ground_speed_m_s": 15.0,
    }

    BallStage(
        config=cfg, output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    for seg in track.flight_segments:
        a, b = seg.frame_range
        assert not (a <= 15 <= b), (
            f"promotion lifted grounded anchor f=15 into flight segment "
            f"{seg.frame_range}: parabola apex would override the user's ground anchor"
        )


@pytest.mark.integration
def test_promotion_rejects_high_residual_refits(tmp_path: Path, monkeypatch):
    """The promotion stage must enforce the same residual cap as the
    IMM-side parabola fit. A 60-px residual refit is not a real flight
    arc — it's tracker confusion that should leave the run as
    grounded."""
    from src.stages import ball as _ball
    from src.utils.ball_plausibility import GroundedRun

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0 + i * 5.0, 360.0, 0.85) for i in range(n_frames)]

    monkeypatch.setattr(
        _ball, "find_implausible_grounded_runs",
        lambda **kwargs: [GroundedRun(start=10, end=25)],
    )

    p0_fake = np.array([50.0, 34.0, 6.0])
    v0_fake = np.array([3.0, 0.5, 0.0])
    # High residual — should be rejected by the promotion residual cap.
    def fake_fit(obs, *, Ks, Rs, t_world, fps, distortion, **kwargs):
        return p0_fake.copy(), v0_fake.copy(), 60.0
    monkeypatch.setattr(_ball, "fit_parabola_to_image_observations", fake_fit)

    cfg = _minimal_cfg()
    cfg["ball"]["flight_max_residual_px"] = 5.0
    cfg["ball"]["flight_promotion"] = {
        "enabled": True,
        "min_run_frames": 6,
        "off_pitch_margin_m": 5.0,
        "max_ground_speed_m_s": 15.0,
    }

    BallStage(
        config=cfg, output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    for seg in track.flight_segments:
        a, b = seg.frame_range
        assert (b < 10) or (a > 25), (
            f"promotion accepted high-residual refit as flight: {seg.frame_range} "
            f"resid={seg.fit_residual_px}"
        )


@pytest.mark.integration
def test_airborne_player_touch_without_smpl_falls_back_to_airborne_height(tmp_path: Path):
    """An airborne player_touch (e.g. mid-flight knee or chest contact)
    whose SMPL track is missing or doesn't cover that frame must NOT
    fall back to ball_radius (which would teleport the ball to the
    ground for one frame). The fallback plane is the player_touch
    state_to_height value of 1.0 m, matching the airborne intent."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # No SMPL track written for P999 — bone_world will return None.
    # Bracketing the touch with airborne anchors makes it airborne.
    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=(640.0, 360.0), state="kick"),
            BallAnchor(frame=10, image_xy=(650.0, 350.0), state="airborne_low"),
            BallAnchor(
                frame=15, image_xy=(660.0, 340.0),
                state="player_touch", player_id="P999", bone="r_knee",
            ),
            BallAnchor(frame=20, image_xy=(670.0, 350.0), state="airborne_low"),
            BallAnchor(frame=25, image_xy=(680.0, 360.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    f15 = next(f for f in track.frames if f.frame == 15)
    assert f15.world_xyz is not None
    # Must land at the 1.0 m ray-cast, NOT the 0.11 m ray-cast.
    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected_air = ankle_ray_to_pitch((660.0, 340.0), K=K, R=R, t=t, plane_z=1.0)
    expected_ground = ankle_ray_to_pitch((660.0, 340.0), K=K, R=R, t=t, plane_z=0.11)
    err_air = float(np.linalg.norm(np.array(f15.world_xyz) - expected_air))
    err_ground = float(np.linalg.norm(np.array(f15.world_xyz) - expected_ground))
    assert err_air < err_ground, (
        f"airborne player_touch without SMPL fell back to ground level: "
        f"got {list(f15.world_xyz)}, expected near {expected_air.tolist()}"
    )
    assert f15.world_xyz[2] > 0.5, (
        f"airborne player_touch should be at airborne fallback height, "
        f"not ground level z={f15.world_xyz[2]:.3f}"
    )


@pytest.mark.integration
def test_phase2_ground_touch_player_touch_uses_clicked_pixel_for_p0(tmp_path: Path):
    """A Phase 2 parabola fit for a span starting with a ground-touch
    player_touch (a kick from a foot at ground level) must pin p0 to
    the clicked-pixel ray-cast at z=ball_radius, NOT to the SMPL bone
    XY. SMPL bone XY drifts 1–2 m due to monocular HMR depth ambiguity,
    and that offset propagates into the LM's v0 estimate — producing a
    wildly off-pitch parabola when the airborne pixels alone don't
    tightly determine it. Locks in the consistency between the Phase 2
    knot resolver and the end-of-run hard-knot override."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 40
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # SMPL r_foot deliberately placed 3 m off the clicked anchor pixel
    # location. The Phase 2 fit must ignore the bone XY and use the
    # clicked pixel.
    clicked_uv = (640.0, 360.0)
    from src.utils.foot_anchor import ankle_ray_to_pitch
    clicked_world = ankle_ray_to_pitch(clicked_uv, K=K, R=R, t=t, plane_z=0.11)
    bone_target_xy = (float(clicked_world[0]) + 3.0, float(clicked_world[1]) + 3.0)
    _save_ground_player_track(
        out, n_frames, player_id="P007",
        target_xy=bone_target_xy, foot_z=0.1,
    )

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=0, image_xy=(620.0, 360.0), state="grounded"),
            BallAnchor(
                frame=5, image_xy=clicked_uv,
                state="player_touch", player_id="P007", bone="r_foot",
            ),
            BallAnchor(frame=10, image_xy=(650.0, 340.0), state="airborne_low"),
            BallAnchor(frame=15, image_xy=(660.0, 330.0), state="airborne_mid"),
            BallAnchor(frame=20, image_xy=(670.0, 340.0), state="airborne_low"),
            BallAnchor(frame=25, image_xy=(680.0, 360.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    seg = next(
        (s for s in track.flight_segments if s.frame_range[0] == 5),
        None,
    )
    assert seg is not None, (
        f"expected Phase 2 segment starting at the kick (f=5); "
        f"got segments {[s.frame_range for s in track.flight_segments]}"
    )
    p0_fit = np.array(seg.parabola["p0"])
    # The fit's p0 must be at the clicked-pixel ray-cast, not the bone
    # location (3 m offset by construction).
    assert np.linalg.norm(p0_fit[:2] - clicked_world[:2]) < 0.5, (
        f"Phase 2 p0_pin should match clicked-pixel ray-cast "
        f"{clicked_world.tolist()}, got {p0_fit.tolist()} (bone was at "
        f"{bone_target_xy})"
    )


@pytest.mark.integration
def test_ground_level_interp_bulge_does_not_overshoot_anchors(tmp_path: Path):
    """When WASB-derived per-frame positions consistently land on one
    side of the straight line between two close ground-level anchors,
    the line+bulge interp's LSQ produces |D| ≫ |AB| — and the in-
    between frames swoop past the next anchor and back. This is the
    user-visible rubber-band at every dribble/short-pass run. The
    bulge magnitude must be capped relative to the anchor-to-anchor
    distance to prevent the overshoot."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Anchor pixels close in u; ground-truth world line ≈ 1 m long.
    anchor_a_uv = (600.0, 360.0)
    anchor_b_uv = (620.0, 360.0)

    # WASB pixel chosen so it ray-casts ~1.5 m off the AB line in +y —
    # inside the 2 m offline tolerance, but enough to pull the LSQ bulge
    # to 5–7 m without the cap.
    wasb_uv = (608.6, 346.7)
    detections = [(wasb_uv[0], wasb_uv[1], 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5,  image_xy=anchor_a_uv, state="grounded"),
            BallAnchor(frame=15, image_xy=anchor_b_uv, state="grounded"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    from src.utils.foot_anchor import ankle_ray_to_pitch
    a_world = ankle_ray_to_pitch(anchor_a_uv, K=K, R=R, t=t, plane_z=0.11)
    b_world = ankle_ray_to_pitch(anchor_b_uv, K=K, R=R, t=t, plane_z=0.11)
    ab = np.asarray(b_world[:2] - a_world[:2])
    ab_len = float(np.linalg.norm(ab))
    ab_unit = ab / max(ab_len, 1e-6)

    # For every in-between frame, the projection onto AB must be inside
    # [0, ab_len] (no overshoot past either anchor) and the perpendicular
    # offset must be bounded by the anchor-pair scale (no wild sideways
    # swing). The cap |D| ≤ |AB| produces a max perpendicular apex of
    # |AB|/4 and zero overshoot — well within these envelopes.
    for fi in range(6, 15):
        f = next(x for x in track.frames if x.frame == fi)
        assert f.world_xyz is not None
        p_rel = np.asarray(f.world_xyz[:2]) - np.asarray(a_world[:2])
        par = float(np.dot(p_rel, ab_unit))
        perp = float(np.linalg.norm(p_rel - par * ab_unit))
        assert -0.05 * ab_len <= par <= 1.05 * ab_len, (
            f"frame {fi} overshoots anchor pair: par={par:.2f}m, |AB|={ab_len:.2f}m"
        )
        assert perp <= 0.5 * ab_len + 0.05, (
            f"frame {fi} swings wildly off the AB line: perp={perp:.2f}m, |AB|={ab_len:.2f}m"
        )


@pytest.mark.integration
def test_ground_touch_player_touch_uses_clicked_pixel_not_bone_xy(tmp_path: Path):
    """A ground-touch player_touch's world XY must come from the
    clicked-pixel ray-cast at z=ball_radius, NOT from the SMPL bone
    position. SMPL bone XY drifts 0.5–2 m due to HMR depth ambiguity,
    which produces a one-frame rubber-band jump at every dribble/short-
    pass touch when the surrounding ground-level interp is on a
    different line."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # SMPL r_foot intentionally placed AWAY from where the user clicked.
    # Bone is at (50, 30); the user clicked a pixel that projects to a
    # different XY — bone-driven code would emit (50, 30) at the touch
    # frame, snapping the ball away from the clicked location.
    _save_ground_player_track(
        out, n_frames, player_id="P007",
        target_xy=(50.0, 30.0), foot_z=0.1,
    )

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    anchor_uv = (700.0, 380.0)  # a pixel that ray-casts to a different XY than (50,30)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=0,  image_xy=(620.0, 360.0), state="grounded"),
            BallAnchor(
                frame=10, image_xy=anchor_uv,
                state="player_touch", player_id="P007", bone="r_foot",
            ),
            BallAnchor(frame=20, image_xy=(660.0, 360.0), state="grounded"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    f10 = next(f for f in track.frames if f.frame == 10)
    assert f10.world_xyz is not None

    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected = ankle_ray_to_pitch(anchor_uv, K=K, R=R, t=t, plane_z=0.11)

    # The touch frame must land at the clicked-pixel ray-cast, not the
    # SMPL bone position. A bone-driven implementation would put the
    # ball at (50, 30); this test fails in that case.
    assert np.allclose(f10.world_xyz, expected, atol=0.05), (
        f"ground-touch player_touch should use clicked-pixel ray-cast, "
        f"not SMPL bone XY; expected {expected.tolist()}, got {list(f10.world_xyz)}"
    )

    # Rubber-banding shows up as a frame-to-frame XY delta at the touch
    # that is much larger than the surrounding deltas. With clicked-
    # pixel ray-cast at the touch, f=10 lies on the same interp curve
    # as its neighbours, so |Δ(9→10)| ≈ |Δ(8→9)| and |Δ(10→11)| ≈
    # |Δ(11→12)|. Bone-driven code violated this with a ~1.5 m spike
    # at f=10.
    def _xy(fi: int) -> np.ndarray:
        return np.asarray(
            next(f for f in track.frames if f.frame == fi).world_xyz[:2]
        )
    d_pre = float(np.linalg.norm(_xy(9) - _xy(8)))
    d_at_in = float(np.linalg.norm(_xy(10) - _xy(9)))
    d_at_out = float(np.linalg.norm(_xy(11) - _xy(10)))
    d_post = float(np.linalg.norm(_xy(12) - _xy(11)))
    assert d_at_in < max(0.2, 2.0 * d_pre), (
        f"rubber-band into ground-touch: |Δ(9→10)|={d_at_in:.2f}m vs "
        f"baseline |Δ(8→9)|={d_pre:.2f}m"
    )
    assert d_at_out < max(0.2, 2.0 * d_post), (
        f"rubber-band out of ground-touch: |Δ(10→11)|={d_at_out:.2f}m vs "
        f"baseline |Δ(11→12)|={d_post:.2f}m"
    )


@pytest.mark.integration
def test_groundprojection_does_not_emit_runaway_world_positions(tmp_path: Path, monkeypatch):
    """When the IMM-smoothed pixel produces a ground-projection wildly
    off the pitch (near-horizon ray-to-plane blow-up), the ball stage
    must drop it rather than writing a hundred-metre teleport into the
    track."""
    from src.stages import ball as ball_mod

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 10
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)
    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]

    # Simulate near-horizon ray-cast: every projection returns a wildly
    # off-pitch world position. The stage must not pass these through.
    def _runaway(uv, *, K, R, t, plane_z=0.11, distortion=(0.0, 0.0)):
        return np.array([-500.0, 2000.0, plane_z])
    monkeypatch.setattr(ball_mod, "ankle_ray_to_pitch", _runaway)

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    for f in track.frames:
        if f.world_xyz is None:
            continue
        assert abs(f.world_xyz[0]) < 200 and abs(f.world_xyz[1]) < 200, (
            f"f={f.frame} emitted runaway world position {list(f.world_xyz)}"
        )


def _goal_facing_camera():
    """Camera at (-10, 34, 1.22) world, looking +x toward the near goal.

    World axes per CLAUDE.md: x along touchline, y across (0..68), z up.
    Goal at x=0, posts at y=34 ± 3.66, crossbar z=2.44. With this pose
    a goal_impact pixel ray cleanly hits the goal-frame geometry —
    perfect for exercising the resolver wired into BallStage.
    """
    K = np.array([[1500.0, 0.0, 640.0],
                  [0.0, 1500.0, 360.0],
                  [0.0, 0.0, 1.0]])
    R = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0]])
    C = np.array([-10.0, 34.0, 1.22])
    t = -R @ C
    return K, R, t


@pytest.mark.integration
def test_goal_impact_pins_world_to_goal_geometry(tmp_path: Path):
    """A 'goal_impact' anchor with goal_element='crossbar' pins the
    trajectory at the crossbar's known 3D position via the
    pixel-ray-to-goal-element resolver, not at a state-height
    ray-cast.
    """
    K, R, t = _goal_facing_camera()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Pixel (640, 177) maps cleanly to the crossbar at (0, 34, 2.44)
    # for this camera (see test_goal_geometry.py).
    crossbar_pixel = (640.0, 177.0)
    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5, image_xy=(640.0, 360.0), state="kick"),
            BallAnchor(
                frame=15, image_xy=crossbar_pixel,
                state="goal_impact", goal_element="crossbar",
            ),
            BallAnchor(frame=25, image_xy=(700.0, 360.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    f15 = next(f for f in track.frames if f.frame == 15)
    assert f15.world_xyz is not None
    np.testing.assert_allclose(
        f15.world_xyz, [0.0, 34.0, 2.44], atol=0.05,
    )
    # goal_impact ∈ AIRBORNE_STATES so the impact frame is in flight.
    assert f15.state == "flight"


@pytest.mark.integration
def test_phase2_splits_at_goal_impact_and_pins_endpoint(tmp_path: Path):
    """A goal_impact anchor mid-flight must split the Phase 2 span
    into two parabolas, both with the impact frame as a hard-knot
    endpoint. Without this, the surrounding fit ignores the impact
    and the visualised trajectory teleports to/from the pinned frame.
    Mirrors test_phase2_splits_at_header_and_pins_endpoint for the
    new goal_impact event state.
    """
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 60
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    def _proj(p):
        cam = R @ p + t
        pix = K @ cam
        return (float(pix[0] / pix[2]), float(pix[1] / pix[2]))

    # Reproject anchors from realistic world positions so the parabola
    # fit stays within the plausibility envelope (the test camera at
    # (52.5, -30, 30) means pixel-only anchors near image centre would
    # ground-project ~50 m from the far goal, blowing up velocity).
    kick_world      = np.array([10.0, 34.0, 0.11])
    rise_a_world    = np.array([7.5,  34.0, 1.5])
    rise_b_world    = np.array([5.0,  34.0, 2.2])
    impact_world    = np.array([0.0,  34.0, 2.44])
    fall_a_world    = np.array([4.0,  34.0, 1.8])
    fall_b_world    = np.array([8.0,  34.0, 0.9])
    bounce_world    = np.array([12.0, 34.0, 0.11])

    detections = [_proj(kick_world) + (0.85,) for _ in range(n_frames)]
    anchors = (
        BallAnchor(frame=5,  image_xy=_proj(kick_world),   state="kick"),
        BallAnchor(frame=10, image_xy=_proj(rise_a_world), state="airborne_low"),
        BallAnchor(frame=15, image_xy=_proj(rise_b_world), state="airborne_mid"),
        BallAnchor(
            frame=20, image_xy=_proj(impact_world),
            state="goal_impact", goal_element="crossbar",
        ),
        BallAnchor(frame=25, image_xy=_proj(fall_a_world), state="airborne_low"),
        BallAnchor(frame=30, image_xy=_proj(fall_b_world), state="airborne_low"),
        BallAnchor(frame=35, image_xy=_proj(bounce_world), state="bounce"),
    )
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720), anchors=anchors,
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    pre = [s for s in track.flight_segments if s.frame_range == (5, 20)]
    post = [s for s in track.flight_segments if s.frame_range == (20, 35)]
    assert pre, (
        f"expected pre-impact segment 5-20, got "
        f"{[s.frame_range for s in track.flight_segments]}"
    )
    assert post, (
        f"expected post-impact segment 20-35, got "
        f"{[s.frame_range for s in track.flight_segments]}"
    )
    # The goal_impact frame's world is pinned to the crossbar (0, 34, 2.44).
    f20 = next(f for f in track.frames if f.frame == 20)
    assert f20.world_xyz is not None
    np.testing.assert_allclose(f20.world_xyz, [0.0, 34.0, 2.44], atol=0.1)


def _integrate_magnus_truth(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray,
    duration_s: float,
    *,
    g: float = -9.81,
    drag_k_over_m: float = 0.005,
    substep: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray]:
    """Fine-grained RK4 integration of a Magnus-augmented ball flight,
    used to synthesise ground-truth curving trajectories for the spin
    integration tests. Mirrors the helper in tests/test_ball_spin_fit.py.
    """
    g_vec = np.array([0.0, 0.0, g])

    def accel(v: np.ndarray) -> np.ndarray:
        return g_vec + drag_k_over_m * np.cross(omega, v)

    n_steps = int(round(duration_s / substep)) + 1
    times = np.arange(n_steps) * substep
    pos = np.zeros((n_steps, 3))
    pos[0] = p0
    p, v = p0.copy(), v0.copy()
    for i in range(1, n_steps):
        h = substep
        k1v = accel(v); k1p = v
        k2v = accel(v + 0.5 * h * k1v); k2p = v + 0.5 * h * k1v
        k3v = accel(v + 0.5 * h * k2v); k3p = v + 0.5 * h * k2v
        k4v = accel(v + h * k3v); k4p = v + h * k3v
        p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        pos[i] = p
    return times, pos


def test_spin_seed_for_segment_extracts_player_touch_spin():
    """``_spin_seed_for_segment`` finds a player_touch anchor inside
    the [a, b] segment, translates its spin preset to an omega seed
    via ``omega_seed_from_preset``, and reports hint_provided=True.

    Anchors outside the segment, anchors with no spin, and anchors
    with ``spin="none"`` all leave the segment unhinted.
    """
    from src.stages.ball import _spin_seed_for_segment

    # Helper to build a player_touch anchor with the right shape.
    def _pt(frame: int, *, spin: str | None, touch_type: str | None = None):
        return BallAnchor(
            frame=frame, image_xy=(640.0, 360.0),
            state="player_touch", player_id="P1", bone="r_foot",
            touch_type=touch_type, spin=spin,
        )

    v0 = np.array([10.0, 0.0, 5.0])

    # Case 1: instep_curl_right inside the segment → vertical seed.
    anchors = {2: _pt(2, spin="instep_curl_right", touch_type="shot")}
    seed, hint = _spin_seed_for_segment(anchors, a=2, b=28, v0=v0)
    assert hint is True
    assert seed[2] > 0 and seed[0] == 0.0 and seed[1] == 0.0

    # Case 2: matching anchor outside [a, b] → no hint.
    seed, hint = _spin_seed_for_segment(anchors, a=10, b=28, v0=v0)
    assert hint is False
    np.testing.assert_array_equal(seed, np.zeros(3))

    # Case 3: spin == "none" → no hint (user explicitly opted out).
    anchors = {2: _pt(2, spin="none", touch_type="shot")}
    seed, hint = _spin_seed_for_segment(anchors, a=2, b=28, v0=v0)
    assert hint is False
    np.testing.assert_array_equal(seed, np.zeros(3))

    # Case 4: no spin attribute on the anchor → no hint.
    anchors = {2: _pt(2, spin=None)}
    seed, hint = _spin_seed_for_segment(anchors, a=2, b=28, v0=v0)
    assert hint is False
    np.testing.assert_array_equal(seed, np.zeros(3))

    # Case 5: only non-player_touch anchors → no hint even with spin
    # (defense-in-depth; the schema rejects this combination at load).
    anchors = {2: BallAnchor(
        frame=2, image_xy=(640.0, 360.0), state="bounce",
    )}
    seed, hint = _spin_seed_for_segment(anchors, a=2, b=28, v0=v0)
    assert hint is False
    np.testing.assert_array_equal(seed, np.zeros(3))

    # Case 6: topspin preset → horizontal axis perpendicular to v0,
    # not zeros (orientation check is in test_ball_spin_presets).
    anchors = {2: _pt(2, spin="topspin", touch_type="volley")}
    seed, hint = _spin_seed_for_segment(anchors, a=2, b=28, v0=v0)
    assert hint is True
    assert seed[2] == 0.0
    assert float(np.linalg.norm(seed)) > 0


@pytest.mark.integration
def test_goal_impact_falls_back_when_resolver_fails(tmp_path: Path):
    """If the chosen pixel ray cannot intersect the requested goal
    element (e.g. a horizontal ray + side_net normal), the stage
    catches the resolver error and falls through to the generic
    ray-to-plane projection instead of crashing the whole shot."""
    K, R, t = _camera_pose()  # the standard test camera at the touchline
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # Pixel (640, 360) maps to a pitch-centre ray that won't intersect
    # the side_net y-planes within their x-range — the resolver raises,
    # the stage catches it, and the fallback ray-to-plane at the
    # goal_impact fallback height (2.44 m) supplies a position.
    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(
            BallAnchor(frame=5, image_xy=(640.0, 360.0), state="kick"),
            BallAnchor(
                frame=10, image_xy=(640.0, 360.0),
                state="goal_impact", goal_element="side_net",
            ),
            BallAnchor(frame=15, image_xy=(700.0, 360.0), state="bounce"),
        ),
    ).save(out / "ball" / "play_ball_anchors.json")

    # The stage must not crash even when the goal_impact resolver fails.
    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frame 10 emitted *some* output (either fallback position or
    # nothing — the assertion is only that the pipeline survived).
    assert any(f.frame == 10 for f in track.frames)

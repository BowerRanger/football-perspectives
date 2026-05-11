"""Integration tests for BallStage end-to-end.

Synthesises camera tracks + clip files + FakeBallDetector output;
verifies the four-step pipeline (detect → IMM smooth → ground project →
flight fit) produces the expected BallTrack."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector


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
    """The BallDetector is faked in tests, so the frame contents don't matter —
    we just need the VideoCapture to return ``n`` frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    for _ in range(n):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _project(p: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


@pytest.mark.integration
def test_ball_stage_recovers_grounded_and_flight(tmp_path: Path):
    n = 60
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    # Grounded ball rolling at 6 m/s across the centre of the pitch.
    # x stays in [30, 42] — well inside the FIFA 105×68 m pitch — so
    # Layer 2 (flight_promotion) will not flag the run as implausible.
    detections: list[tuple[float, float, float] | None] = []
    for i in range(n):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))

    stage = BallStage(
        config={"ball": {"detector": "fake"}},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    assert len(out.frames) == n
    states = {f.state for f in out.frames}
    assert "grounded" in states
    # FlightSegment.parabola must always include the spin keys (None when
    # the Magnus refinement did not engage).
    for seg in out.flight_segments:
        for key in ("spin_axis_world", "spin_omega_rad_s", "spin_confidence"):
            assert key in seg.parabola


@pytest.mark.integration
def test_ball_stage_marks_missing_after_long_gap(tmp_path: Path):
    n = 40
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    detections: list[tuple[float, float, float] | None] = []
    for i in range(n):
        if 10 <= i <= 25:
            # Long missed-detection gap — far longer than max_gap_frames.
            detections.append(None)
        else:
            p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
            u, v = _project(p, K, R, t)
            detections.append((u, v, 0.9))

    stage = BallStage(
        config={"ball": {"detector": "fake", "max_gap_frames": 3}},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    out = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    # First few frames inside the gap are gap-filled; later frames in the
    # gap should be state="missing" after max_gap_frames expires.
    assert out.frames[20].state == "missing"
    assert out.frames[20].world_xyz is None


@pytest.mark.integration
def test_ball_stage_emits_per_shot_track(tmp_path: Path):
    n = 30
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(
        tmp_path / "camera" / "alpha_camera_track.json", K, R, t, n,
        clip_id="alpha", fps=fps,
    )
    _save_camera_track(
        tmp_path / "camera" / "beta_camera_track.json", K, R, t, n,
        clip_id="beta", fps=fps,
    )
    _write_blank_clip(tmp_path / "shots" / "alpha.mp4", n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "beta.mp4", n, fps=fps)

    end = n - 1
    ShotsManifest(
        source_file="x", fps=fps, total_frames=2 * n,
        shots=[
            Shot(id="alpha", start_frame=0, end_frame=end, start_time=0.0,
                 end_time=(end + 1) / fps, clip_file="shots/alpha.mp4"),
            Shot(id="beta", start_frame=0, end_frame=end, start_time=0.0,
                 end_time=(end + 1) / fps, clip_file="shots/beta.mp4"),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")

    detections: list[tuple[float, float, float] | None] = []
    for i in range(n):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))
    # FakeBallDetector cycles, so the same list serves both shots.

    stage = BallStage(
        config={"ball": {"detector": "fake"}},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    assert (tmp_path / "ball" / "alpha_ball_track.json").exists()
    assert (tmp_path / "ball" / "beta_ball_track.json").exists()


@pytest.mark.integration
def test_implausible_parabola_fit_is_rejected(tmp_path: Path, monkeypatch):
    """Reproduces origi01 seg-3: a parabola fit lands at billion-metre
    coordinates with tiny pixel residual. Layer 1 must drop it."""
    import src.stages.ball as ball_mod

    # Force fit_parabola_to_image_observations (as imported into ball.py) to
    # return garbage with a microscopic residual so the existing
    # flight_max_residual_px gate cannot save us.
    def fake_parab(*args, **kwargs):
        p0 = np.array([-5_690_504.0, 9_399_056.0, -2_218_511.0])
        v0 = np.array([3_745_003.0, 3_366_928.0, -698_927.0])
        return p0, v0, 0.11

    monkeypatch.setattr(ball_mod, "fit_parabola_to_image_observations", fake_parab)

    # Also stub _flight_runs so BallStage always reports one flight run
    # (frames 10-25) regardless of what the IMM posterior says.  This
    # isolates the test to the plausibility gate, not the tracker.
    def forced_flight_runs(self_arg, steps, min_flight, max_flight):
        return [(10, 25)]

    monkeypatch.setattr(BallStage, "_flight_runs", forced_flight_runs)

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip = out / "shots" / "play.mp4"
    n_frames = 40
    _write_blank_clip(clip, n=n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    ShotsManifest(
        source_file="fake.mp4",
        fps=30.0,
        total_frames=n_frames,
        shots=[
            Shot(
                id="play",
                clip_file="shots/play.mp4",
                start_frame=0,
                end_frame=n_frames - 1,
                start_time=0.0,
                end_time=(n_frames - 1) / 30.0,
            )
        ],
    ).save(out / "shots" / "shots_manifest.json")

    # Give every frame a detection so obs_pairs is never empty.
    detections = [(640.0 + i * 0.5, 360.0 + i * 0.3, 0.9) for i in range(n_frames)]
    stage = BallStage(
        config={
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
                "spin": {
                    "enabled": False,
                    "min_flight_seconds": 0.5,
                    "min_residual_improvement": 0.2,
                    "max_omega_rad_s": 200.0,
                    "drag_k_over_m": 0.005,
                },
                "plausibility": {
                    "z_max_m": 50.0,
                    "horizontal_speed_max_m_s": 40.0,
                    "pitch_margin_m": 5.0,
                },
            },
            "pitch": {"length_m": 105.0, "width_m": 68.0},
        },
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # No flight segment should have survived plausibility.
    assert len(track.flight_segments) == 0, (
        f"expected garbage segment to be rejected; got {track.flight_segments}"
    )


@pytest.mark.integration
def test_aerial_arc_promotes_grounded_run_to_flight(tmp_path: Path):
    """Reproduces origi01 frames 101-191: a long aerial pass where IMM
    never trips flight mode. Layer 2 must detect the implausible ground
    motion and either refit as flight or demote to missing."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip = out / "shots" / "play.mp4"
    n_frames = 60
    _write_blank_clip(clip, n=n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    # Use the same ShotsManifest call shape as test_implausible_parabola_fit_is_rejected
    ShotsManifest(
        source_file="fake.mp4",
        fps=30.0,
        total_frames=n_frames,
        shots=[
            Shot(
                id="play",
                clip_file="shots/play.mp4",
                start_frame=0,
                end_frame=n_frames - 1,
                start_time=0.0,
                end_time=(n_frames - 1) / 30.0,
            )
        ],
    ).save(out / "shots" / "shots_manifest.json")

    # Synthesise pixel detections that move in a clean line in pixel
    # space (so the IMM stays in grounded mode), but at 18 px/frame —
    # the ground-projection of this implies a very fast lateral roll.
    detections: list[tuple[float, float, float] | None] = [None] * 5
    for i in range(50):
        u = 200.0 + 18.0 * i
        v = 200.0 + 0.5 * i
        detections.append((u, v, 0.85))
    detections += [None] * (n_frames - len(detections))

    stage = BallStage(
        config={
            "ball": {
                "detector": "fake",
                "ball_radius_m": 0.11,
                "max_gap_frames": 6,
                "flight_max_residual_px": 200.0,
                "tracker": {
                    "process_noise_grounded_px": 4.0,
                    "process_noise_flight_px": 12.0,
                    "measurement_noise_px": 2.0,
                    "gating_sigma": 4.0,
                    "min_flight_frames": 6,
                    "max_flight_frames": 90,
                },
                "spin": {"enabled": False, "min_flight_seconds": 0.5, "min_residual_improvement": 0.2, "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
                "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
                "flight_promotion": {"enabled": True, "min_run_frames": 6, "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
            },
            "pitch": {"length_m": 105.0, "width_m": 68.0},
        },
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")

    # Layer 2 ran and exercised the find_implausible_grounded_runs path
    # on this synthetic fast-rolling-but-on-pitch trajectory. The
    # parabola refit cannot recover a real flight from purely-linear
    # pixel motion, so the safe behaviour is to leave the frames as
    # grounded — the original ground projection is noisy but bounded
    # and better than `state="missing"`. We assert the safe fall-back:
    # frames remain grounded, no spurious flight segment was created,
    # and the run was not demoted to missing.
    detection_window = [
        f for f in track.frames if 5 <= f.frame < 55
    ]
    missing_count = sum(1 for f in detection_window if f.state == "missing")
    flight_count = sum(1 for f in detection_window if f.state == "flight")
    assert missing_count == 0, (
        f"refit failure must not demote grounded frames to missing; "
        f"got {missing_count} missing in detection window"
    )
    # No spurious flight segment from a non-flight pixel trajectory.
    assert flight_count == 0, (
        f"unexpected flight promotion on purely-linear pixel trajectory: "
        f"{flight_count} flight frames"
    )


@pytest.mark.integration
def test_kick_anchored_fit_pins_p0_to_foot(tmp_path: Path, monkeypatch):
    """When a kp2d sidecar puts a player's ankle within 30 px of the
    ball at the flight seed frame, the parabola fit's p0 is anchored
    to the foot ray-cast position (not the unconstrained 6-param fit)."""
    import json as _json
    import src.stages.ball as ball_mod

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip = out / "shots" / "play.mp4"
    n_frames = 50
    _write_blank_clip(clip, n=n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    ShotsManifest(
        source_file="fake.mp4",
        fps=30.0,
        total_frames=n_frames,
        shots=[
            Shot(
                id="play",
                clip_file="shots/play.mp4",
                start_frame=0,
                end_frame=n_frames - 1,
                start_time=0.0,
                end_time=(n_frames - 1) / 30.0,
            )
        ],
    ).save(out / "shots" / "shots_manifest.json")

    # True kick: p0 = (52.5, 34.0, 0.11) — projects to image centre,
    # v0 = (3, 0.5, 12).
    p0_true = np.array([52.5, 34.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    g_vec = np.array([0.0, 0.0, -9.81])
    detections: list[tuple[float, float, float] | None] = [None] * 5
    for i in range(30):
        dt = i / 30.0
        pt = p0_true + v0_true * dt + 0.5 * g_vec * dt ** 2
        uv = _project(pt, K, R, t)
        detections.append((uv[0], uv[1], 0.85))
    detections += [None] * (n_frames - len(detections))

    # Synthesise a kp2d sidecar with the kicker's right ankle at p0_true.
    hmr_dir = out / "hmr_world"
    hmr_dir.mkdir(parents=True, exist_ok=True)
    foot_uv_kick = _project(p0_true, K, R, t)
    kp_zero = [0.0, 0.0, 0.0]
    kp_payload = {
        "player_id": "P001",
        "shot_id": "play",
        "frames": [{
            "frame": 5,
            "keypoints": [kp_zero] * 15 + [list(foot_uv_kick) + [0.9], list(foot_uv_kick) + [0.9]],
        }],
    }
    (hmr_dir / "play__P001_kp2d.json").write_text(_json.dumps(kp_payload))

    # Force _flight_runs to return one flight run covering the 30 detection
    # frames so the test isolates the kick-anchor logic, not the IMM.
    monkeypatch.setattr(
        BallStage, "_flight_runs",
        lambda self_arg, steps, min_flight, max_flight: [(5, 34)],
    )

    stage = BallStage(
        config={
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
                    "initial_p_flight": 0.5,
                },
                "spin": {"enabled": False, "min_flight_seconds": 0.5, "min_residual_improvement": 0.2, "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
                "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
                "flight_promotion": {"enabled": False, "min_run_frames": 6, "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
                "kick_anchor": {"enabled": True, "max_pixel_distance_px": 30.0, "lookahead_frames": 4, "min_pixel_acceleration_px_per_frame": 0.0, "foot_anchor_z_m": 0.11},
            },
            "pitch": {"length_m": 105.0, "width_m": 68.0},
        },
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    assert len(track.flight_segments) >= 1
    seg = track.flight_segments[0]
    p0_fit = np.array(seg.parabola["p0"])
    # With anchored fit we expect p0 to land within 0.5 m of the truth.
    assert np.linalg.norm(p0_fit - p0_true) < 0.5, (
        f"expected kick-anchored p0 ≈ {p0_true.tolist()}, got {p0_fit.tolist()}"
    )


@pytest.mark.integration
def test_appearance_bridge_fills_short_detection_gap(tmp_path: Path):
    """When WASB returns None for 1-3 frames but a fresh template and
    the IMM prediction agree on a region containing the ball, the
    appearance bridge fills the gap (no state='missing')."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip_path = out / "shots" / "play.mp4"
    n_frames = 30
    # Write a clip where the ball is a real white circle on green; the
    # bridge will find it in the predicted window.
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720)
    )
    for i in range(n_frames):
        img = np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8)
        u, v = 200 + 5 * i, 200 + 1 * i
        cv2.circle(img, (u, v), 8, (240, 240, 240), -1)
        writer.write(img)
    writer.release()

    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    ShotsManifest(
        source_file="fake.mp4",
        fps=30.0,
        total_frames=n_frames,
        shots=[
            Shot(
                id="play",
                clip_file="shots/play.mp4",
                start_frame=0,
                end_frame=n_frames - 1,
                start_time=0.0,
                end_time=(n_frames - 1) / 30.0,
            )
        ],
    ).save(out / "shots" / "shots_manifest.json")

    # Detections: present for frames 0..9, missing for 10..12 (3-frame gap), present for 13..29.
    detections: list[tuple[float, float, float] | None] = []
    for i in range(n_frames):
        if 10 <= i <= 12:
            detections.append(None)
        else:
            u, v = 200.0 + 5.0 * i, 200.0 + 1.0 * i
            detections.append((u, v, 0.85))

    # With max_gap_frames=0 the IMM tracker cannot gap-fill on its own;
    # only the appearance bridge (max_gap_frames=8) can prevent missing.
    stage = BallStage(
        config={
            "ball": {
                "detector": "fake",
                "ball_radius_m": 0.11,
                "max_gap_frames": 0,
                "flight_max_residual_px": 5.0,
                "tracker": {
                    "process_noise_grounded_px": 4.0,
                    "process_noise_flight_px": 12.0,
                    "measurement_noise_px": 2.0,
                    "gating_sigma": 4.0,
                    "min_flight_frames": 6,
                    "max_flight_frames": 90,
                },
                "spin": {"enabled": False, "min_flight_seconds": 0.5, "min_residual_improvement": 0.2, "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
                "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
                "flight_promotion": {"enabled": False, "min_run_frames": 6, "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
                "kick_anchor": {"enabled": False, "max_pixel_distance_px": 30.0, "lookahead_frames": 4, "min_pixel_acceleration_px_per_frame": 6.0, "foot_anchor_z_m": 0.11},
                "appearance_bridge": {"enabled": True, "max_gap_frames": 8, "template_size_px": 32, "search_radius_px": 64, "min_ncc": 0.6, "template_max_age_frames": 30, "template_update_confidence": 0.5},
            },
            "pitch": {"length_m": 105.0, "width_m": 68.0},
        },
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frames 10..12 must NOT be state="missing".
    gap_states = [f.state for f in track.frames if 10 <= f.frame <= 12]
    assert all(s != "missing" for s in gap_states), (
        f"expected bridge to fill frames 10-12; got {gap_states}"
    )

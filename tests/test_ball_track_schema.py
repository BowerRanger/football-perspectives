"""Round-trip tests for the BallTrack / BallFrame / FlightSegment schema."""

from pathlib import Path

from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment


def _track(frames: tuple[BallFrame, ...]) -> BallTrack:
    return BallTrack(
        clip_id="clipA",
        fps=25.0,
        frames=frames,
        flight_segments=(
            FlightSegment(
                id=0,
                frame_range=(1, 5),
                parabola={
                    "p0": [0.0, 0.0, 0.5],
                    "v0": [10.0, 0.0, 5.0],
                    "g": -9.81,
                    "spin_axis_world": [0.0, 0.0, 1.0],
                    "spin_omega_rad_s": 40.0,
                },
                fit_residual_px=1.2,
            ),
        ),
    )


def test_round_trip_without_quat_defaults_none(tmp_path: Path):
    frames = (
        BallFrame(
            frame=1,
            world_xyz=(0.0, 0.0, 0.5),
            state="flight",
            confidence=0.9,
            flight_segment_id=0,
        ),
        BallFrame(
            frame=2,
            world_xyz=None,
            state="missing",
            confidence=0.0,
        ),
    )
    src = _track(frames)
    path = tmp_path / "ball_track.json"
    src.save(path)
    back = BallTrack.load(path)
    assert back == src
    assert all(f.quat_wxyz is None for f in back.frames)


def test_round_trip_with_quat_exact(tmp_path: Path):
    q = (0.92387953, 0.0, 0.0, 0.38268343)
    frames = (
        BallFrame(
            frame=1,
            world_xyz=(0.0, 0.0, 0.5),
            state="flight",
            confidence=0.9,
            flight_segment_id=0,
            quat_wxyz=q,
        ),
        BallFrame(
            frame=2,
            world_xyz=(1.0, 0.0, 0.5),
            state="grounded",
            confidence=0.8,
            flight_segment_id=None,
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    src = _track(frames)
    path = tmp_path / "ball_track.json"
    src.save(path)
    back = BallTrack.load(path)
    assert back == src
    assert back.frames[0].quat_wxyz == q
    assert back.frames[1].quat_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_load_tolerates_mixed_frames(tmp_path: Path):
    """A track whose JSON has some frames with quat and some without."""
    frames = (
        BallFrame(
            frame=1,
            world_xyz=(0.0, 0.0, 0.5),
            state="flight",
            confidence=0.9,
            flight_segment_id=0,
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        BallFrame(
            frame=2,
            world_xyz=(1.0, 0.0, 0.5),
            state="grounded",
            confidence=0.8,
        ),
    )
    src = _track(frames)
    path = tmp_path / "ball_track.json"
    src.save(path)
    # Simulate a legacy file: strip the quat key from the second frame.
    import json

    raw = json.loads(path.read_text())
    raw["frames"][1].pop("quat_wxyz", None)
    path.write_text(json.dumps(raw))

    back = BallTrack.load(path)
    assert back.frames[0].quat_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert back.frames[1].quat_wxyz is None

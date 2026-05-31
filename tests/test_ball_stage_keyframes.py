"""Integration: the ball stage writes a sparse ball_keyframes.json sidecar
next to the dense ball_track.json, with one entry per manual anchor."""
from pathlib import Path

from src.schemas.ball_keyframes import BallKeyframeSet
from src.schemas.ball_track import BallFrame


def test_run_shot_writes_keyframes_sidecar(tmp_path: Path, monkeypatch):
    from src.stages.ball import _emit_ball_keyframes
    from src.schemas.ball_anchor import BallAnchor
    import numpy as np

    ball_out = tmp_path / "ball" / "clipX_ball_track.json"
    ball_out.parent.mkdir(parents=True)
    per_frame_out = [
        BallFrame(frame=4, world_xyz=(1.0, 2.0, 0.11), state="grounded", confidence=1.0),
        BallFrame(frame=5, world_xyz=(1.5, 2.5, 0.11), state="grounded", confidence=1.0),
        BallFrame(frame=6, world_xyz=(2.0, 3.0, 4.0), state="flight", confidence=0.5),
    ]
    anchor_by_frame = {
        4: BallAnchor(frame=4, image_xy=(800.0, 600.0), state="grounded"),
        6: BallAnchor(frame=6, image_xy=(820.0, 300.0), state="airborne_high"),
    }
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    _emit_ball_keyframes(
        ball_out_path=ball_out,
        clip_id="clipX",
        fps=25.0,
        image_size=(1920, 1080),
        per_frame_out=per_frame_out,
        anchor_by_frame=anchor_by_frame,
        per_frame_K={4: K, 6: K},
        per_frame_R={4: np.eye(3), 6: np.eye(3)},
        per_frame_t={4: np.array([0.0, 0, 10.0]), 6: np.array([0.0, 0, 10.0])},
        distortion=(0.0, 0.0),
        ground_touch_frames=set(),
    )

    kf_path = tmp_path / "ball" / "clipX_ball_keyframes.json"
    assert kf_path.exists()
    kfset = BallKeyframeSet.load(kf_path)
    assert [k.frame for k in kfset.keyframes] == [4, 6]
    by_frame = {k.frame: k for k in kfset.keyframes}
    assert by_frame[4].world_xyz == (1.0, 2.0, 0.11)
    assert by_frame[4].ray is None
    assert by_frame[6].world_xyz == (2.0, 3.0, 4.0)  # matches dense track
    assert by_frame[6].ray is not None
    assert by_frame[6].depth_source == "ray_physics"

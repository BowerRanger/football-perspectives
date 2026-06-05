import numpy as np

from src.schemas.anchor import AnchorSet
from src.utils.neural_calibrator import NeuralCalibration
from src.utils.auto_anchor import generate


class _FakeCalibrator:
    def __init__(self, good_frames, flipped_frames=()):
        self.good = set(good_frames)
        self.flipped = set(flipped_frames)

    def calibrate(self, frame_bgr):
        f = int(frame_bgr[0, 0, 0])  # frame index smuggled in pixel 0
        if f in self.flipped:
            pos = np.array([52.5, 200.0, 15.0])  # far-side flip -> MAD outlier
        elif f in self.good:
            pos = np.array([52.5, -30.0, 15.0])
        else:
            return None
        K = np.array([[3000.0, 0, 960], [0, 3000.0, 540], [0, 0, 1]])
        return NeuralCalibration(K=K, rvec=np.zeros(3), tvec=np.zeros(3),
                                 world_position=pos)

    def extract_keypoints_pixels(self, frame_bgr):
        f = int(frame_bgr[0, 0, 0])
        if f not in (self.good | self.flipped):
            return {}
        return {i: (float(100 + i), float(900 - i)) for i in (1, 4, 5, 7, 9, 16)}


def _frames_reader(indices, image_size=(1920, 1080)):
    out = {}
    for idx in indices:
        fr = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        fr[0, 0, 0] = idx
        out[idx] = fr
    return out


def _cfg(**over):
    base = dict(
        keyframe_interval=30, max_keyframes=12, min_points_per_anchor=4,
        plausibility_bounds={"x": (-30, 135), "y": (-60, 130), "z": (3, 80)},
        consensus_max_position_mad=3.0,
    )
    base.update(over)
    return base


def test_generate_builds_anchorset_and_drops_flips():
    cal = _FakeCalibrator(good_frames=[0, 30, 60], flipped_frames=[90])
    anchor_set = generate(
        calibrator=cal, clip_id="gberch", n_frames=120, image_size=(1920, 1080),
        cfg=_cfg(), frames_reader=_frames_reader,
    )
    assert isinstance(anchor_set, AnchorSet)
    assert {a.frame for a in anchor_set.anchors} == {0, 30, 60}  # 90 flip dropped


def test_generate_returns_none_when_nothing_plausible():
    cal = _FakeCalibrator(good_frames=[])
    assert generate(
        calibrator=cal, clip_id="x", n_frames=120, image_size=(1920, 1080),
        cfg=_cfg(), frames_reader=_frames_reader,
    ) is None

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_PNLCALIB_E2E"),
    reason="set RUN_PNLCALIB_E2E=1 and provide PNLCALIB_E2E_CLIP to run",
)


def test_zero_click_generates_rich_anchorset():
    """With zero manual clicks, the auto pipeline yields an AnchorSet with at
    least one rich (>=6 non-coplanar pts) anchor on a real clip."""
    import cv2
    from src.utils.auto_anchor import generate
    from src.utils.neural_calibrator import PnLCalibrator
    from src.utils.anchor_solver import _is_rich

    clip = os.environ["PNLCALIB_E2E_CLIP"]
    cap = cv2.VideoCapture(clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    def reader(indices, image_size):
        cap = cv2.VideoCapture(clip)
        out = {}
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                out[i] = fr
        cap.release()
        return out

    cal = PnLCalibrator(device=os.environ.get("PNLCALIB_DEVICE", "auto"))
    cfg = dict(keyframe_interval=30, max_keyframes=12, min_points_per_anchor=4,
               consensus_max_position_mad=3.0)
    anchor_set = generate(calibrator=cal, clip_id="gberch", n_frames=n,
                          image_size=(w, h), cfg=cfg, frames_reader=reader)
    assert anchor_set is not None
    assert any(_is_rich(a) for a in anchor_set.anchors), "no rich anchor"

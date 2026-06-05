"""Generate camera-stage anchors automatically from PnLCalib keypoints.

The learned model replaces the human clicker: per keyframe, PnLCalib's
keypoint detections become point LandmarkObservations (world coords in our
frame), an Anchor per keyframe, and the existing camera solver does the rest.
"""

from __future__ import annotations

from src.schemas.anchor import Anchor, LandmarkObservation
from src.utils.pnlcalib_pitch_map import keypoint_world_xyz_ours


def keypoints_to_anchor(
    pixels: dict[int, tuple[float, float]],
    frame: int,
    *,
    min_points: int,
) -> Anchor | None:
    """Build a point-only Anchor from PnLCalib keypoint pixels.

    Names are synthesised (``pnl_kp_<id>``); the solver consumes world_xyz,
    not the name. Returns None if fewer than ``min_points`` known keypoints.
    """
    landmarks = []
    for kp_id, image_xy in pixels.items():
        world = keypoint_world_xyz_ours(kp_id)
        if world is None:
            continue
        landmarks.append(
            LandmarkObservation(
                name=f"pnl_kp_{kp_id}",
                image_xy=(float(image_xy[0]), float(image_xy[1])),
                world_xyz=world,
            )
        )
    if len(landmarks) < min_points:
        return None
    return Anchor(frame=frame, landmarks=tuple(landmarks), lines=())

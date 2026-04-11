"""Score PnP candidates by checking implied player heights from bounding boxes."""

import numpy as np
import cv2

from src.utils.camera import project_to_pitch


def score_player_heights(
    bboxes: list[list[float]],
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    height_range: tuple[float, float] = (1.5, 2.1),
) -> float:
    """Score a PnP solution by how many player bboxes imply plausible heights.

    For each bounding box, projects the foot (bottom-centre) and head (top-centre)
    onto the pitch plane and computes the implied 3D height. Returns the fraction
    of players with heights in the plausible range.

    Args:
        bboxes: list of [x1, y1, x2, y2] bounding boxes in pixel space
        K: 3x3 intrinsic matrix
        rvec: rotation vector (3,)
        tvec: translation vector (3,)
        height_range: (min_height, max_height) in metres

    Returns:
        fraction of players with plausible heights (0.0 to 1.0)
    """
    if not bboxes:
        return 0.0

    K_f64 = np.asarray(K, dtype=np.float64)
    rv = np.asarray(rvec, dtype=np.float64).reshape(3)
    tv = np.asarray(tvec, dtype=np.float64).reshape(3)

    min_h, max_h = height_range
    plausible = 0

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        foot_px = np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)

        # Project foot onto pitch plane (z=0) to get pitch position
        foot_pitch = project_to_pitch(foot_px, K_f64, rv, tv)
        foot_x, foot_y = float(foot_pitch[0]), float(foot_pitch[1])

        # Project the foot position at two heights to find pixels-per-metre
        pt_ground = np.array([[foot_x, foot_y, 0.0]], dtype=np.float64)
        pt_high = np.array([[foot_x, foot_y, 3.0]], dtype=np.float64)

        proj_ground, _ = cv2.projectPoints(pt_ground, rv, tv, K_f64, None)
        proj_high, _ = cv2.projectPoints(pt_high, rv, tv, K_f64, None)

        ground_v = float(proj_ground[0, 0, 1])
        high_v = float(proj_high[0, 0, 1])

        dv = ground_v - high_v  # pixels per 3.0m of height
        if abs(dv) < 1.0:
            continue

        head_v = float(y1)
        implied_height = 3.0 * (ground_v - head_v) / dv

        if min_h <= implied_height <= max_h:
            plausible += 1

    return float(plausible) / float(len(bboxes))

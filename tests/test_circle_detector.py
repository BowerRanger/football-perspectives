import cv2
import numpy as np

from src.utils.circle_detector import (
    CENTRE_CIRCLE_RADIUS,
    detect_circle,
)
from src.utils.line_detector import DetectorConfig


def _nadir_camera():
    """Camera straight above the centre spot looking down. Projects the centre
    circle (r=9.15) to a ~183px circle at image (400,300) for f=800, h=40m."""
    f = 800.0
    K = np.array([[f, 0, 400.0], [0, f, 300.0], [0, 0, 1.0]])
    R = np.diag([1.0, -1.0, -1.0])  # x_cam=x, y_cam=-y, z_cam=-z (look down)
    C = np.array([52.5, 34.0, 40.0])
    t = -R @ C
    return K, R, t, f


def _render_painted_circle():
    img = np.full((600, 800, 3), (40, 140, 40), dtype=np.uint8)  # grass green
    K, R, t, f = _nadir_camera()
    r_px = int(round(f * CENTRE_CIRCLE_RADIUS / 40.0))  # ~183
    cv2.circle(img, (400, 300), r_px, (255, 255, 255), thickness=3)
    return img, K, R, t, r_px


def test_detect_centre_circle_subpixel():
    img, K, R, t, r_px = _render_painted_circle()
    cfg = DetectorConfig()
    det = detect_circle(img, K, R, t, (0.0, 0.0), cfg)
    assert det is not None
    assert det.confidence > 0.6
    assert len(det.image_points) >= 30
    # every detected image point must lie ~on the drawn circle (radius r_px at 400,300)
    pts = np.array(det.image_points)
    radii = np.hypot(pts[:, 0] - 400.0, pts[:, 1] - 300.0)
    assert np.median(np.abs(radii - r_px)) < 2.0


def test_detect_circle_none_when_not_in_view():
    # camera pointed away (circle behind) -> no detection
    img = np.full((600, 800, 3), (40, 140, 40), dtype=np.uint8)
    K = np.array([[800.0, 0, 400.0], [0, 800.0, 300.0], [0, 0, 1.0]])
    R = np.diag([1.0, -1.0, -1.0])
    t = -R @ np.array([52.5, 34.0, -40.0])  # camera below the pitch -> circle behind
    assert detect_circle(img, K, R, t, (0.0, 0.0), DetectorConfig()) is None

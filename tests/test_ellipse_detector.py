import cv2
import numpy as np

from src.utils.ellipse_detector import detect_circle_ellipse
from src.utils.circle_detector import CENTRE_CIRCLE_RADIUS
from src.utils.line_detector import DetectorConfig


def _nadir():
    f = 800.0
    K = np.array([[f, 0, 400.0], [0, f, 300.0], [0, 0, 1.0]])
    R = np.diag([1.0, -1.0, -1.0])
    t = -R @ np.array([52.5, 34.0, 40.0])
    return K, R, t, f


def test_detect_centre_circle_ellipse():
    img = np.full((600, 800, 3), (40, 140, 40), dtype=np.uint8)
    K, R, t, f = _nadir()
    r_px = int(round(f * CENTRE_CIRCLE_RADIUS / 40.0))  # ~183
    cv2.circle(img, (400, 300), r_px, (255, 255, 255), thickness=3)
    # a crossing white line (halfway) through the centre — must be rejected
    cv2.line(img, (400 - r_px - 40, 300), (400 + r_px + 40, 300), (255, 255, 255), 3)
    det = detect_circle_ellipse(img, K, R, t, (0.0, 0.0), DetectorConfig())
    assert det is not None
    assert det.confidence > 0.6
    (cx, cy), (MA, ma), _ = det.ellipse
    assert abs(cx - 400) < 6 and abs(cy - 300) < 6      # centred on the drawn circle
    assert abs(max(MA, ma) - 2 * r_px) < 12             # right diameter


def test_none_when_not_in_view():
    img = np.full((600, 800, 3), (40, 140, 40), dtype=np.uint8)
    K = np.array([[800.0, 0, 400.0], [0, 800.0, 300.0], [0, 0, 1.0]])
    R = np.diag([1.0, -1.0, -1.0])
    t = -R @ np.array([52.5, 34.0, -40.0])  # below pitch -> behind camera
    assert detect_circle_ellipse(img, K, R, t, (0.0, 0.0), DetectorConfig()) is None

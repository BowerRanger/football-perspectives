"""Camera-compensation homography + motion-blob ball candidates (Phase B1).
Pure, torch-free."""

from __future__ import annotations

import numpy as np

from src.utils.ball_motion_flow import frame_homography


def _roty(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def test_homography_maps_static_direction_consistently():
    """A static world ray that projects to x1 in frame 1 must map (via H)
    to its frame-0 pixel x0 — i.e. H cancels the camera rotation."""
    K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
    R0 = np.eye(3)
    R1 = _roty(3.0)
    d = np.array([0.12, 0.05, 1.0])
    d /= np.linalg.norm(d)
    x0 = K @ (R0 @ d); x0 /= x0[2]
    x1 = K @ (R1 @ d); x1 /= x1[2]
    H = frame_homography(K, R0, K, R1)
    p = H @ np.array([x1[0], x1[1], 1.0]); p /= p[2]
    assert np.allclose(p[:2], x0[:2], atol=1e-6)


def test_homography_identity_for_same_pose():
    K = np.array([[900.0, 0, 600.0], [0, 900.0, 340.0], [0, 0, 1.0]])
    R = _roty(5.0)
    H = frame_homography(K, R, K, R)
    H /= H[2, 2]
    assert np.allclose(H, np.eye(3), atol=1e-9)


def _frame(shapes):
    import cv2
    img = np.zeros((200, 200, 3), np.uint8)
    for kind, *a in shapes:
        if kind == "dot":
            cv2.circle(img, (a[0], a[1]), 3, (255, 255, 255), -1)
        elif kind == "rect":
            cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (255, 255, 255), -1)
    return img


def test_motion_candidates_picks_small_fast_blob_not_big_one():
    from src.utils.ball_motion_flow import motion_candidates
    prev = _frame([("dot", 90, 100), ("rect", 10, 10, 70, 70)])
    cur = _frame([("dot", 120, 100), ("rect", 20, 10, 80, 70)])  # both moved
    cands = motion_candidates(prev, cur, None, max_ball_px=20)
    # the small ball blob (near its new position) is surfaced
    assert any(abs(u - 120) < 10 and abs(v - 100) < 10 for u, v, _ in cands)
    # the big moving block (>20px) is excluded
    assert not any(u < 90 and v < 90 for u, v, _ in cands)


def test_motion_candidates_respects_exclude_boxes():
    from src.utils.ball_motion_flow import motion_candidates
    prev = _frame([("dot", 90, 100)])
    cur = _frame([("dot", 120, 100)])
    cands = motion_candidates(
        prev, cur, None, max_ball_px=20,
        exclude_boxes=[(110.0, 90.0, 130.0, 110.0)],
    )
    assert not any(110 <= u <= 130 and 90 <= v <= 110 for u, v, _ in cands)


def test_motion_candidates_empty_when_no_motion():
    from src.utils.ball_motion_flow import motion_candidates
    f = _frame([("dot", 100, 100), ("rect", 10, 10, 70, 70)])
    assert motion_candidates(f, f.copy(), None, max_ball_px=20) == []

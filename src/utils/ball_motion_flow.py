"""Camera-compensated motion detection for the ball (Phase B1/B2).

The broadcast rig is a fixed-translation PTZ (``t`` constant; ``R`` and focal
vary), so two frames of the static background differ by a pure homography
induced by the camera rotation + intrinsics:

    H_{1->0} = K0 @ R0 @ R1^T @ K1^-1

Warping frame 1 by ``H`` aligns its background to frame 0; the abs-difference
then isolates *real* motion (ball + players). The ball is the small, fast,
roughly-round blob that moves distinctly from the larger, slower player
blobs — and the flow vector gives its velocity directly, so it fires on the
blurred turn frames the appearance detector drops.

Pure numpy + opencv; torch-free and unit-testable. See
docs/superpowers/specs/2026-06-15-ball-detection-direction-changes-design.md §5.1.
"""

from __future__ import annotations

import cv2
import numpy as np

Candidate = tuple[float, float, float]  # (u, v, score) in full-frame pixels
Box = tuple[float, float, float, float]  # (x0, y0, x1, y1)


def frame_homography(
    K0: np.ndarray, R0: np.ndarray, K1: np.ndarray, R1: np.ndarray
) -> np.ndarray:
    """Homography mapping a frame-1 pixel to its frame-0 pixel for the same
    world ray (pure-rotation model): ``K0 R0 R1^T K1^-1``."""
    K0 = np.asarray(K0, float)
    K1 = np.asarray(K1, float)
    R0 = np.asarray(R0, float)
    R1 = np.asarray(R1, float)
    return K0 @ R0 @ R1.T @ np.linalg.inv(K1)


def warp_to_reference(
    img: np.ndarray, H: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Warp ``img`` by ``H`` into a ``size=(w, h)`` canvas."""
    return cv2.warpPerspective(img, H, size, flags=cv2.INTER_LINEAR)


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _in_any_box(u: float, v: float, boxes: list[Box]) -> bool:
    return any(x0 <= u <= x1 and y0 <= v <= y1 for x0, y0, x1, y1 in boxes)


def motion_candidates(
    prev: np.ndarray,
    cur: np.ndarray,
    H: np.ndarray | None,
    *,
    max_ball_px: float = 40.0,
    min_area_px: float = 2.0,
    diff_thresh: int = 18,
    exclude_boxes: list[Box] | None = None,
    top_k: int = 5,
) -> list[Candidate]:
    """Small fast moving blobs in ``cur`` after camera-compensating ``prev``.

    ``H`` warps ``prev`` into ``cur``'s frame (``frame_homography(Kc,Rc,Kp,Rp)``);
    pass ``None`` for a static camera. Returns ``(u, v, score)`` candidates,
    ``score`` in ``[0,1]`` rising with blob compactness and difference mass,
    strongest first. Blobs larger than ``max_ball_px`` (players, crowd
    motion) or inside ``exclude_boxes`` (player bboxes) are dropped.
    """
    exclude_boxes = exclude_boxes or []
    h, w = cur.shape[:2]
    gp = _gray(prev)
    gc = _gray(cur)
    if H is not None:
        gp = warp_to_reference(gp, H, (w, h))
    diff = cv2.absdiff(gc, gp)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[Candidate] = []
    for i in range(1, n):  # skip background label 0
        area = float(stats[i, cv2.CC_STAT_AREA])
        bw = float(stats[i, cv2.CC_STAT_WIDTH])
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area_px:
            continue
        if max(bw, bh) > max_ball_px:
            continue  # too big to be the ball (player limb, crowd)
        cu, cv_ = float(centroids[i][0]), float(centroids[i][1])
        if _in_any_box(cu, cv_, exclude_boxes):
            continue
        # Compactness: area / bbox area (round blob -> ~0.78; sparse -> low).
        fill = area / max(1.0, bw * bh)
        size_score = 1.0 - min(1.0, max(bw, bh) / max_ball_px)
        score = float(np.clip(0.5 * fill + 0.5 * size_score, 0.0, 1.0))
        out.append((cu, cv_, score))
    out.sort(key=lambda c: c[2], reverse=True)
    return out[:top_k]


__all__ = ["frame_homography", "warp_to_reference", "motion_candidates",
           "Candidate", "Box"]

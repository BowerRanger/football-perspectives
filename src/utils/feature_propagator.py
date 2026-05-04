"""Frame-to-frame propagator for fixed-position broadcast cameras.

Given two consecutive frames and the prior frame's (K, R), recover
the next frame's (K, R) from a feature-tracking homography.

For a camera with fixed t and a far scene, frame-to-frame motion is
approximately pure rotation + zoom: H ~ K_next * dR * K_prev^-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PropagatorResult:
    K: np.ndarray
    R: np.ndarray
    inlier_ratio: float
    feature_count: int


def decompose_homography_to_R_zoom(
    H: np.ndarray, K_prev: np.ndarray
) -> tuple[np.ndarray, float]:
    """Decompose H = K_next * dR * K_prev^-1 assuming K_next = zoom * K_prev
    in fx/fy and the same principal point (broadcast PTZ assumption).

    Returns (dR, zoom_ratio).
    """
    K_inv = np.linalg.inv(K_prev)
    M = H @ K_prev  # = K_next * dR
    # K_next is zoom * K_prev structure (same principal point, same fx/fy ratio).
    # Estimate zoom from the ratio of M[:, :2] columns to dR columns.
    # Trick: extract dR by removing the K_next from M, where K_next = K_prev with fx scaled.
    # We solve for zoom s such that K_prev_scaled^-1 @ M is closest to a rotation.

    cx = K_prev[0, 2]
    cy = K_prev[1, 2]

    def _try_zoom(s: float) -> tuple[np.ndarray, float]:
        K_next = np.array(
            [[s * K_prev[0, 0], 0, cx], [0, s * K_prev[1, 1], cy], [0, 0, 1.0]]
        )
        dR_candidate = np.linalg.inv(K_next) @ H @ K_prev
        # Project onto SO(3) via SVD
        U, _, Vt = np.linalg.svd(dR_candidate)
        dR_proj = U @ Vt
        if np.linalg.det(dR_proj) < 0:
            U[:, -1] *= -1
            dR_proj = U @ Vt
        residual = float(np.linalg.norm(dR_candidate - dR_proj, ord="fro"))
        return dR_proj, residual

    # Golden-section search over zoom in [0.5, 2.0]
    lo, hi = 0.5, 2.0
    phi = (1 + 5 ** 0.5) / 2
    for _ in range(60):
        a = hi - (hi - lo) / phi
        b = lo + (hi - lo) / phi
        _, ra = _try_zoom(a)
        _, rb = _try_zoom(b)
        if ra < rb:
            hi = b
        else:
            lo = a
    zoom = (lo + hi) / 2
    dR, _ = _try_zoom(zoom)
    return dR, zoom


def propagate_one_frame(
    img_prev: np.ndarray,
    img_next: np.ndarray,
    K_prev: np.ndarray,
    R_prev: np.ndarray,
    *,
    detector: str = "orb",
    max_features: int = 1000,
    ransac_inlier_min_ratio: float = 0.4,
    mask_prev: np.ndarray | None = None,
) -> PropagatorResult | None:
    """Track features prev -> next, fit homography, decompose to (K, R).

    Returns None when feature count or RANSAC inlier ratio fall below
    thresholds — caller treats as low confidence.
    """
    if detector == "orb":
        det = cv2.ORB_create(nfeatures=max_features)
    else:
        raise NotImplementedError(f"detector {detector!r} not supported")

    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY) if img_prev.ndim == 3 else img_prev
    gray_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY) if img_next.ndim == 3 else img_next

    kp_prev, desc_prev = det.detectAndCompute(gray_prev, mask_prev)
    kp_next, desc_next = det.detectAndCompute(gray_next, None)
    if desc_prev is None or desc_next is None or len(kp_prev) < 50 or len(kp_next) < 50:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_prev, desc_next)
    if len(matches) < 30:
        return None

    src_pts = np.array([kp_prev[m.queryIdx].pt for m in matches])
    dst_pts = np.array([kp_next[m.trainIdx].pt for m in matches])

    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if H is None or inlier_mask is None:
        return None
    inlier_ratio = float(inlier_mask.sum()) / float(len(inlier_mask))
    if inlier_ratio < ransac_inlier_min_ratio:
        return None

    dR, zoom = decompose_homography_to_R_zoom(H, K_prev)
    R_next = dR @ R_prev
    K_next = K_prev.copy()
    K_next[0, 0] *= zoom
    K_next[1, 1] *= zoom
    return PropagatorResult(
        K=K_next, R=R_next, inlier_ratio=inlier_ratio, feature_count=len(matches),
    )

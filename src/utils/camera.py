import cv2
import numpy as np


def build_projection_matrix(
    K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    """Return 3x4 projection matrix P = K @ [R | t]."""
    R, _ = cv2.Rodrigues(rvec)
    return K @ np.hstack([R, tvec.reshape(3, 1)])


def project_to_pitch(
    pixel: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    """
    Un-project a pixel (u, v) onto the pitch ground plane (z=0).
    Returns (x, y) in pitch coordinates (metres).
    """
    R, _ = cv2.Rodrigues(rvec)
    # Homography H maps pitch plane (z=0) -> image:  H = K @ [r1 | r2 | t]
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec.reshape(3)])
    H_inv = np.linalg.inv(H)
    pt_h = H_inv @ np.array([pixel[0], pixel[1], 1.0])
    return (pt_h[:2] / pt_h[2]).astype(np.float32)


def reprojection_error(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """Mean pixel distance between projected 3D points and observed 2D points."""
    projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - pts_2d, axis=1)))

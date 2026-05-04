"""Anchor-frame camera solver.

First anchor: full 3x4 projection matrix DLT, then RQ decomposition
into (K, R, t). Subsequent anchors: t inherited; solve only fx and R
for the camera-body-fixed assumption.
"""

from __future__ import annotations

import numpy as np

from src.schemas.anchor import LandmarkObservation
from src.utils.pitch_landmarks import has_non_coplanar


class AnchorSolveError(RuntimeError):
    pass


def _build_dlt_matrix(landmarks: list[LandmarkObservation]) -> np.ndarray:
    rows: list[list[float]] = []
    for lm in landmarks:
        X, Y, Z = lm.world_xyz
        u, v = lm.image_xy
        rows.append([ X,  Y,  Z, 1.0,   0,  0,  0,  0,  -u * X, -u * Y, -u * Z, -u])
        rows.append([ 0,  0,  0,   0,   X,  Y,  Z, 1.0, -v * X, -v * Y, -v * Z, -v])
    return np.asarray(rows, dtype=float)


def _rq_decomposition(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """RQ decomposition of a 3x3 matrix into (K, R) with K upper-triangular,
    R orthogonal, K[2,2] == 1.

    Returns ``(K, R, scale)`` where ``scale`` is the value of ``K[2, 2]`` before
    being normalised to 1. Callers use ``scale`` to keep the translation column
    of the projection matrix consistent with the (now-rescaled) K — see
    ``solve_first_anchor``.
    """
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = np.flipud(R.T)
    R = np.fliplr(R)
    Q = Q.T
    Q = np.flipud(Q)
    K = R
    Rmat = Q
    # Force K diagonals positive
    sign = np.diag(np.sign(np.diag(K)))
    K = K @ sign
    Rmat = sign @ Rmat
    scale = float(K[2, 2])
    return K / scale, Rmat, scale


def solve_first_anchor(
    landmarks: tuple[LandmarkObservation, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (K, R, t) where R, t are world->camera (OpenCV convention)."""
    if len(landmarks) < 6:
        raise AnchorSolveError(
            f"first anchor needs >=6 landmarks, got {len(landmarks)}"
        )
    if not has_non_coplanar(landmarks):
        raise AnchorSolveError(
            "first anchor needs at least one non-coplanar landmark "
            "(crossbar or corner flag top)"
        )
    A = _build_dlt_matrix(list(landmarks))
    _, _, vh = np.linalg.svd(A)
    p = vh[-1]
    P = p.reshape(3, 4)

    M = P[:, :3]
    K, R, M_scale = _rq_decomposition(M)
    # P is recovered from SVD up to an overall scale; M and P[:, 3] inherit
    # that same scale. ``_rq_decomposition`` divides K by ``M_scale`` so K
    # equals the calibration matrix in canonical form (K[2,2] == 1). To keep
    # the translation column consistent we apply the same divisor to P[:, 3].
    t = np.linalg.solve(K, P[:, 3] / M_scale)
    # Disambiguate sign: enforce that the first landmark projects with positive
    # depth (cam_z > 0).
    Xw = np.array(landmarks[0].world_xyz)
    if (R @ Xw + t)[2] < 0:
        R = -R
        t = -t
    return K, R, t


def solve_subsequent_anchor(
    landmarks: tuple[LandmarkObservation, ...],
    t_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve (K, R) given a fixed t. Iterative LM minimising reprojection
    residual; K parameterised as fx=fy with principal point at image centre.

    Returns (K, R).
    """
    from scipy.optimize import least_squares

    if len(landmarks) < 4:
        raise AnchorSolveError(
            f"subsequent anchor needs >=4 landmarks, got {len(landmarks)}"
        )

    # Image centre: take from first landmark image_xy max as a proxy for size
    # (caller passes image_size separately; we expose a helper if needed).
    # Here we infer from the landmark spread — caller can also override.
    us = np.array([lm.image_xy[0] for lm in landmarks])
    vs = np.array([lm.image_xy[1] for lm in landmarks])
    cx = float((us.min() + us.max()) / 2)
    cy = float((vs.min() + vs.max()) / 2)

    world_pts = np.array([lm.world_xyz for lm in landmarks])
    image_pts = np.array([lm.image_xy for lm in landmarks])

    def _params_to_KR(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fx = p[0]
        rvec = p[1:4]
        theta = np.linalg.norm(rvec)
        if theta < 1e-9:
            R = np.eye(3)
        else:
            k = rvec / theta
            K_skew = np.array(
                [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
            )
            R = (
                np.eye(3)
                + np.sin(theta) * K_skew
                + (1 - np.cos(theta)) * (K_skew @ K_skew)
            )
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
        return K, R

    def _residuals(p: np.ndarray) -> np.ndarray:
        K, R = _params_to_KR(p)
        cam = world_pts @ R.T + t_world
        pix = cam @ K.T
        proj = pix[:, :2] / pix[:, 2:3]
        return (proj - image_pts).reshape(-1)

    p0 = np.array([1500.0, 0.01, 0.01, 0.01])
    result = least_squares(_residuals, p0, method="lm", max_nfev=200)
    K, R = _params_to_KR(result.x)
    return K, R

"""SMPL 24-joint skeleton constants and helpers.

Pure-Python module — no Blender or torch deps. Imported by the export
stage, the FBX export script, and tests.

Joint names, parent map, and rest-pose joint positions match the
canonical SMPL skeleton (mean shape, neutral betas) in y-up canonical
space, identical to the table baked into ``src/web/static/viewer.html``
at ``SMPL_J_REST``.
"""

from __future__ import annotations

import numpy as np

SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "l_hip", "r_hip", "spine1",
    "l_knee", "r_knee", "spine2",
    "l_ankle", "r_ankle", "spine3",
    "l_foot", "r_foot", "neck",
    "l_collar", "r_collar", "head",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hand", "r_hand",
)

SMPL_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
    12, 13, 14, 16, 17, 18, 19, 20, 21,
)

SMPL_REST_JOINTS_YUP: np.ndarray = np.array(
    [
        [0.000,  0.000,  0.000],
        [0.060, -0.087, -0.013],
        [-0.060, -0.087, -0.013],
        [0.001,  0.108, -0.027],
        [0.099, -0.494, -0.001],
        [-0.099, -0.494, -0.001],
        [0.002,  0.246,  0.018],
        [0.087, -0.882, -0.038],
        [-0.087, -0.882, -0.038],
        [0.000,  0.300,  0.060],
        [0.117, -0.939,  0.071],
        [-0.117, -0.939,  0.071],
        [0.000,  0.518,  0.013],
        [0.084,  0.474,  0.008],
        [-0.084,  0.474,  0.008],
        [0.000,  0.609,  0.052],
        [0.184,  0.427, -0.012],
        [-0.184,  0.427, -0.012],
        [0.448,  0.426, -0.039],
        [-0.448,  0.426, -0.039],
        [0.711,  0.420, -0.046],
        [-0.711,  0.420, -0.046],
        [0.789,  0.418, -0.034],
        [-0.789,  0.418, -0.034],
    ],
    dtype=np.float64,
)


def parent_relative_offsets_yup() -> np.ndarray:
    """Per-joint rest offset from its parent, in y-up canonical metres.

    Pelvis is (0,0,0). Used by the FBX exporter to place each child bone
    relative to its parent at rest.
    """
    offsets = np.zeros_like(SMPL_REST_JOINTS_YUP)
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        offsets[j] = SMPL_REST_JOINTS_YUP[j] - SMPL_REST_JOINTS_YUP[p]
    return offsets


def axis_angle_to_quaternion(aa: np.ndarray) -> np.ndarray:
    """Convert a 3-vector axis-angle to a (w, x, y, z) quaternion.

    Identity for near-zero magnitudes.
    """
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / theta
    half = theta * 0.5
    s = float(np.sin(half))
    c = float(np.cos(half))
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])

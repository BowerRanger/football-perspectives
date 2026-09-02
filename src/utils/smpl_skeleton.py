"""SMPL 24-joint skeleton constants and helpers.

Pure-Python module — no Blender or torch deps. Imported by the export
stage, the FBX export script, and tests.

Joint names, parent map, and rest-pose joint positions match the
canonical SMPL skeleton (mean shape, neutral betas) in y-up canonical
space, identical to the table baked into ``src/web/static/viewer.html``
at ``SMPL_J_REST``.
"""

from __future__ import annotations

from pathlib import Path

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


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert a 3-vector axis-angle to a 3x3 rotation matrix.

    Identity for near-zero magnitudes.
    """
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def compute_joint_world_pose(
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    joint_idx: int,
    rest_joints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematics: world position **and** world rotation of a joint.

    Returns ``(pos, R_world)`` where ``pos`` is the joint centre in pitch
    metres and ``R_world`` is the joint's global rotation expressed in the
    pitch frame (``root_R`` composed onto the canonical joint rotation).

    Inputs:
        thetas: (24, 3) axis-angle, one row per joint (including pelvis
            at index 0). ``thetas[0]`` (the global orient) is
            **intentionally ignored** — ``root_R`` already carries the root
            joint's world orientation (the same convention the web viewer's
            ``smplFK`` uses). Only ``thetas[1:]`` drive the articulated pose.
        root_R: (3, 3) world rotation of the root joint in the pitch frame
            (combines the canonical-y-up → z-up map with the body's yaw).
        root_t: (3,) translation of the pelvis in pitch world (metres).
        joint_idx: index into ``SMPL_JOINT_NAMES`` (0–23).
        rest_joints: optional (24, 3) override of the rest-pose joint
            table. Defaults to :data:`SMPL_REST_JOINTS_YUP` (mean betas).

    Pure numpy; no torch / no SMPL body model. Accurate to ~5 cm per
    joint for typical players — beta-conditioned regression would
    tighten this further but isn't needed for ball-anchor purposes.
    """
    rest = (
        np.asarray(rest_joints, dtype=np.float64)
        if rest_joints is not None else SMPL_REST_JOINTS_YUP
    )
    thetas = np.asarray(thetas, dtype=np.float64).reshape(24, 3)
    # Local rotations per joint.
    local_rot = np.empty((24, 3, 3))
    for j in range(24):
        local_rot[j] = axis_angle_to_matrix(thetas[j])
    # Walk hierarchy. The root joint's global rotation is identity in the
    # canonical frame — ``thetas[0]`` is NOT applied here, because ``root_R``
    # (applied below) already carries the body's world orientation. Applying
    # both double-counts the orientation and flips the body upside down for
    # any non-trivial ``thetas[0]`` (matches the viewer's ``smplFK``).
    global_rot = np.empty((24, 3, 3))
    global_pos = np.empty((24, 3))
    global_rot[0] = np.eye(3)
    global_pos[0] = rest[0]  # canonical pelvis at origin
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        global_rot[j] = global_rot[p] @ local_rot[j]
        global_pos[j] = global_pos[p] + global_rot[p] @ (rest[j] - rest[p])
    root_R = np.asarray(root_R, dtype=np.float64)
    root_t = np.asarray(root_t, dtype=np.float64)
    j = int(joint_idx)
    # Canonical y-up → pitch world.
    pos = root_R @ global_pos[j] + root_t
    R_world = root_R @ global_rot[j]
    return pos, R_world


def compute_all_joint_worlds(
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    rest_joints: np.ndarray | None = None,
) -> np.ndarray:
    """World positions of all 24 SMPL joints for one frame, shape (24, 3).

    Same conventions as :func:`compute_joint_world_pose` (``thetas[0]``
    ignored; ``root_R`` carries the world orientation). One chain walk
    instead of one per queried joint — use this when a caller needs
    several joints of the same frame (e.g. ball contact detection).
    """
    rest = (
        np.asarray(rest_joints, dtype=np.float64)
        if rest_joints is not None else SMPL_REST_JOINTS_YUP
    )
    thetas = np.asarray(thetas, dtype=np.float64).reshape(24, 3)
    local_rot = np.empty((24, 3, 3))
    for j in range(24):
        local_rot[j] = axis_angle_to_matrix(thetas[j])
    global_rot = np.empty((24, 3, 3))
    global_pos = np.empty((24, 3))
    global_rot[0] = np.eye(3)
    global_pos[0] = rest[0]
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        global_rot[j] = global_rot[p] @ local_rot[j]
        global_pos[j] = global_pos[p] + global_rot[p] @ (rest[j] - rest[p])
    root_R = np.asarray(root_R, dtype=np.float64)
    root_t = np.asarray(root_t, dtype=np.float64)
    return global_pos @ root_R.T + root_t


def _axis_angle_to_matrix_batch(aa: np.ndarray) -> np.ndarray:
    """Convert (N, 3) axis-angle vectors to (N, 3, 3) rotation matrices.

    Vectorized Rodrigues' formula (same shape as the one in
    ``gvhmr_estimator._axis_angle_to_matrix``, duplicated locally so this
    module keeps its own zero-torch-dependency contract). The near-zero
    cutoff intentionally matches :func:`axis_angle_to_matrix` exactly
    (``theta < 1e-12`` -> identity) so batched and per-frame FK agree to
    float64 precision — a looser cutoff here would diverge from the
    single-frame path for tiny-but-nonzero rotations.
    """
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    n = aa.shape[0]
    out = np.tile(np.eye(3), (n, 1, 1))
    if n == 0:
        return out

    theta = np.linalg.norm(aa, axis=1)  # (N,)
    nonzero = theta >= 1e-12
    if not np.any(nonzero):
        return out

    axis = aa[nonzero] / theta[nonzero, np.newaxis]  # (M, 3)
    th = theta[nonzero]
    sin_t = np.sin(th)[:, np.newaxis, np.newaxis]
    cos_t = np.cos(th)[:, np.newaxis, np.newaxis]

    M = axis.shape[0]
    K = np.zeros((M, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    K2 = K @ K
    R = np.eye(3) + sin_t * K + (1.0 - cos_t) * K2
    out[nonzero] = R
    return out


def compute_canonical_joints_batch(
    thetas: np.ndarray,
    rest_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Batched FK: root-relative canonical (y-up) joint positions, before
    the world transform. Shape (F, 24, 3), pelvis at the origin.

    This is exactly the ``global_pos`` intermediate that
    :func:`compute_all_joint_worlds_batch` computes internally before
    applying ``root_R``/``root_t`` — extracted here so callers that need
    the posed skeleton in the body's own frame (e.g. foot-contact
    anchoring, which needs a per-frame root->ankle offset before it knows
    the root translation) don't have to duplicate the FK chain walk.
    ``compute_all_joint_worlds_batch`` calls this function and then
    applies the world transform, so the two are provably identical for
    the same ``thetas``/``rest_joints``.

    Same conventions as the other batched FK helpers: ``thetas[:, 0]``
    (per-frame global orient) is ignored — the canonical root is always
    at the origin with identity rotation; only ``thetas[:, 1:]`` drive
    the articulated pose.

    Inputs:
        thetas: (F, 24, 3) axis-angle, one row per frame per joint.
        rest_joints: optional (24, 3) override of the rest-pose joint
            table, as in the other FK helpers.
    """
    rest = (
        np.asarray(rest_joints, dtype=np.float64)
        if rest_joints is not None else SMPL_REST_JOINTS_YUP
    )
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.ndim != 3 or thetas.shape[1:] != (24, 3):
        raise ValueError(f"thetas must have shape (F, 24, 3), got {thetas.shape}")
    n_frames = thetas.shape[0]

    local_rot = _axis_angle_to_matrix_batch(thetas.reshape(-1, 3)).reshape(
        n_frames, 24, 3, 3
    )

    global_rot = np.empty((n_frames, 24, 3, 3), dtype=np.float64)
    global_pos = np.empty((n_frames, 24, 3), dtype=np.float64)
    global_rot[:, 0] = np.eye(3)
    global_pos[:, 0] = rest[0]
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        global_rot[:, j] = np.einsum(
            "fik,fkl->fil", global_rot[:, p], local_rot[:, j]
        )
        offset = rest[j] - rest[p]
        global_pos[:, j] = global_pos[:, p] + np.einsum(
            "fik,k->fi", global_rot[:, p], offset
        )
    return global_pos


def compute_all_joint_worlds_batch(
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    rest_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Batched forward-kinematics: world positions of all 24 SMPL joints,
    vectorized over frames. Shape (F, 24, 3).

    Vectorized-over-frames counterpart to :func:`compute_all_joint_worlds`,
    for callers that need FK on a whole track at once (export, refined
    poses) instead of one Python-level call per frame. Same conventions:
    ``thetas[:, 0]`` (per-frame global orient) is ignored — ``root_R``
    already carries each frame's root world orientation. Output is
    numerically identical (to float64 precision) to stacking
    ``compute_all_joint_worlds`` over frames.

    Inputs:
        thetas: (F, 24, 3) axis-angle, one row per frame per joint.
        root_R: (F, 3, 3) per-frame world rotation of the root joint.
        root_t: (F, 3) per-frame pelvis translation in pitch world (metres).
        rest_joints: optional (24, 3) override of the rest-pose joint
            table, as in the single-frame functions.

    Implemented as :func:`compute_canonical_joints_batch` followed by the
    world transform — the two are kept in lockstep by construction.
    """
    thetas_arr = np.asarray(thetas, dtype=np.float64)
    root_R = np.asarray(root_R, dtype=np.float64)
    root_t = np.asarray(root_t, dtype=np.float64)

    if thetas_arr.ndim != 3 or thetas_arr.shape[1:] != (24, 3):
        raise ValueError(f"thetas must have shape (F, 24, 3), got {thetas_arr.shape}")
    n_frames = thetas_arr.shape[0]
    if root_R.shape != (n_frames, 3, 3):
        raise ValueError(
            f"root_R must have shape ({n_frames}, 3, 3), got {root_R.shape}"
        )
    if root_t.shape != (n_frames, 3):
        raise ValueError(
            f"root_t must have shape ({n_frames}, 3), got {root_t.shape}"
        )

    global_pos = compute_canonical_joints_batch(thetas_arr, rest_joints)

    # Canonical y-up -> pitch world, per frame: pos = root_R @ global_pos[j] + root_t.
    world_pos = np.einsum("fba,fja->fjb", root_R, global_pos) + root_t[:, np.newaxis, :]
    return world_pos


def load_smpl_neutral_model() -> dict | None:
    """Load the SMPL neutral shape data so callers can beta-adjust the
    rest joint table per player. Returns ``None`` when the file is
    absent (e.g. CI without ``data/models/smpl_neutral.npz``) — callers
    must fall back to the constant ``SMPL_REST_JOINTS_YUP`` in that case.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "data" / "models" / "smpl_neutral.npz"
    )
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return None
    out: dict = {"joint_positions": np.asarray(z["joint_positions"])}
    if "joint_shapedirs" in z.files:
        out["joint_shapedirs"] = np.asarray(z["joint_shapedirs"])
    return out


def beta_adjusted_rest_joints(
    betas: np.ndarray | None, smpl_model: dict | None,
) -> np.ndarray:
    """Build a (24, 3) pelvis-relative rest joint table for one player.

    Without ``smpl_model`` (file missing), returns the constant
    ``SMPL_REST_JOINTS_YUP`` so callers still get something usable.

    With ``smpl_model`` and ``betas``, applies the per-shape
    ``joint_shapedirs`` delta on top of the neutral joint positions,
    then shifts the whole table so the pelvis joint sits at the
    origin. This matches the canonical convention used by the FK
    routines and yields the player's actual leg length, fixing the
    ~8-10 cm gap between mean-betas canonical feet and beta-shaped
    mesh feet that left players floating above the pitch.
    """
    if smpl_model is None:
        return np.asarray(SMPL_REST_JOINTS_YUP, dtype=float)
    jp = np.asarray(smpl_model["joint_positions"], dtype=float).copy()
    jsd = smpl_model.get("joint_shapedirs")
    if jsd is not None and betas is not None:
        betas = np.asarray(betas, dtype=float).reshape(-1)
        K = min(jsd.shape[2], len(betas))
        if K > 0:
            jp = jp + jsd[:, :, :K] @ betas[:K]
    # Shift so pelvis is at origin (matches src table convention).
    return jp - jp[0]


def compute_joint_world(
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    joint_idx: int,
    rest_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Forward-kinematics: world position of ``joint_idx`` (single frame).

    Thin wrapper over :func:`compute_joint_world_pose` kept for existing
    callers (ball-anchor in ``src/stages/ball.py``). See that function for
    the full input/convention docs.
    """
    pos, _ = compute_joint_world_pose(thetas, root_R, root_t, joint_idx, rest_joints)
    return pos

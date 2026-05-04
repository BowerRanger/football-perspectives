"""SMPL body model fitting utilities.

Fits SMPL pose (θ), shape (β), and translation (t) parameters to
triangulated 3D joint targets using PyTorch optimization.

When `smplx` is not installed, a lightweight fallback produces
approximate SMPL-like parameters from the raw 3D joints so the
pipeline can still run without the full body model.
"""

import logging
from typing import Protocol

import numpy as np
import torch

# COCO 17 → SMPL 24 joint mapping
# Maps COCO joint index to the nearest SMPL joint index.
# Joints without a clear SMPL counterpart get weight 0 in the loss.
COCO_TO_SMPL: dict[int, int] = {
    0: 15,   # nose → head
    5: 16,   # left_shoulder → left_shoulder
    6: 17,   # right_shoulder → right_shoulder
    7: 18,   # left_elbow → left_elbow
    8: 19,   # right_elbow → right_elbow
    9: 20,   # left_wrist → left_wrist
    10: 21,  # right_wrist → right_wrist
    11: 1,   # left_hip → left_hip
    12: 2,   # right_hip → right_hip
    13: 4,   # left_knee → left_knee
    14: 5,   # right_knee → right_knee
    15: 7,   # left_ankle → left_ankle
    16: 8,   # right_ankle → right_ankle
}

# COCO joints that have no reliable SMPL mapping (eyes, ears)
_UNMAPPED_COCO = {1, 2, 3, 4}

_N_SMPL_JOINTS = 24
_N_SMPL_BETAS = 10
_N_SMPL_POSE_PARAMS = 72  # 24 joints × 3 axis-angle


class SmplBodyModel(Protocol):
    """Protocol for an SMPL-like body model forward pass."""

    def __call__(
        self,
        betas: torch.Tensor,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ) -> object: ...


def _try_load_smplx(model_path: str, device: str) -> SmplBodyModel | None:
    """Try to load the SMPL model via smplx. Returns None if unavailable."""
    try:
        import smplx

        model = smplx.create(
            model_path,
            model_type="smpl",
            gender="neutral",
            num_betas=_N_SMPL_BETAS,
            batch_size=1,
        ).to(device)
        return model
    except ImportError:
        logging.info("smplx not installed — using fallback SMPL fitting")
        return None
    except Exception as exc:
        logging.warning("Failed to load SMPL model from %s: %s", model_path, exc)
        return None


def fit_smpl_sequence(
    positions_3d: np.ndarray,
    confidences: np.ndarray,
    model_path: str = "data/smpl/SMPL_NEUTRAL.pkl",
    device: str = "cpu",
    lr: float = 0.01,
    n_iterations: int = 100,
    lambda_joint: float = 1.0,
    lambda_prior: float = 0.01,
    lambda_shape: float = 0.1,
    lambda_smooth: float = 0.5,
    lambda_ground: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit SMPL parameters to a sequence of 3D joint targets.

    Args:
        positions_3d: (N, 17, 3) triangulated COCO joint positions.
        confidences: (N, 17) per-joint confidence weights.
        model_path: path to SMPL model file (for smplx).
        device: torch device string.
        lr: Adam learning rate.
        n_iterations: optimization iterations per frame.
        lambda_*: loss term weights.

    Returns:
        (betas, poses, transl) — shapes (10,), (N, 72), (N, 3).
    """
    n_frames = positions_3d.shape[0]

    smpl_model = _try_load_smplx(model_path, device)

    if smpl_model is not None:
        return _fit_with_smplx(
            smpl_model, positions_3d, confidences, device,
            lr, n_iterations,
            lambda_joint, lambda_prior, lambda_shape, lambda_smooth, lambda_ground,
        )

    return _fit_fallback(positions_3d, confidences)


def _fit_fallback(
    positions_3d: np.ndarray,
    confidences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lightweight fallback: derive approximate SMPL params from 3D joints.

    Produces zero shape, identity rotations, and translation from hip midpoint.
    This allows downstream stages to work without the SMPL model file.
    """
    n_frames = positions_3d.shape[0]
    betas = np.zeros(_N_SMPL_BETAS, dtype=np.float32)
    poses = np.zeros((n_frames, _N_SMPL_POSE_PARAMS), dtype=np.float32)
    transl = np.zeros((n_frames, 3), dtype=np.float32)

    for f in range(n_frames):
        # Translation from hip midpoint (COCO joints 11, 12)
        left_hip = positions_3d[f, 11]
        right_hip = positions_3d[f, 12]
        if not np.any(np.isnan(left_hip)) and not np.any(np.isnan(right_hip)):
            transl[f] = (left_hip + right_hip) / 2.0
        elif not np.any(np.isnan(left_hip)):
            transl[f] = left_hip
        elif not np.any(np.isnan(right_hip)):
            transl[f] = right_hip

    return betas, poses, transl


def _fit_with_smplx(
    model: SmplBodyModel,
    positions_3d: np.ndarray,
    confidences: np.ndarray,
    device: str,
    lr: float,
    n_iterations: int,
    lambda_joint: float,
    lambda_prior: float,
    lambda_shape: float,
    lambda_smooth: float,
    lambda_ground: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full SMPLify-style optimization using the smplx body model."""
    n_frames = positions_3d.shape[0]

    # Build target joints in SMPL joint order
    target_joints = torch.zeros((n_frames, _N_SMPL_JOINTS, 3), device=device)
    joint_weights = torch.zeros((n_frames, _N_SMPL_JOINTS), device=device)

    for coco_idx, smpl_idx in COCO_TO_SMPL.items():
        pos = torch.from_numpy(positions_3d[:, coco_idx]).float().to(device)
        conf = torch.from_numpy(confidences[:, coco_idx]).float().to(device)
        valid = ~torch.isnan(pos[:, 0])
        target_joints[valid, smpl_idx] = pos[valid]
        joint_weights[valid, smpl_idx] = conf[valid]

    # Optimizable parameters
    betas = torch.zeros(1, _N_SMPL_BETAS, device=device, requires_grad=True)
    all_poses = torch.zeros(n_frames, _N_SMPL_POSE_PARAMS, device=device)
    all_transl = torch.zeros(n_frames, 3, device=device)

    # Initialize translation from target hip midpoint
    for f in range(n_frames):
        hip_mid = (target_joints[f, 1] + target_joints[f, 2]) / 2
        if joint_weights[f, 1] > 0 and joint_weights[f, 2] > 0:
            all_transl[f] = hip_mid

    # Per-frame optimization with warm-starting
    for f in range(n_frames):
        if joint_weights[f].sum() < 1e-6:
            continue

        pose_f = all_poses[f:f + 1].clone().detach().requires_grad_(True)
        transl_f = all_transl[f:f + 1].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pose_f, transl_f, betas], lr=lr)

        for _ in range(n_iterations):
            optimizer.zero_grad()

            output = model(
                betas=betas,
                body_pose=pose_f[:, 3:],
                global_orient=pose_f[:, :3],
                transl=transl_f,
            )
            pred_joints = output.joints[:, :_N_SMPL_JOINTS]

            # Joint position loss
            diff = pred_joints[0] - target_joints[f]
            weighted_diff = diff * joint_weights[f].unsqueeze(-1)
            loss_joint = lambda_joint * (weighted_diff ** 2).sum()

            # Pose prior (L2 regularization toward rest pose)
            loss_prior = lambda_prior * (pose_f ** 2).sum()

            # Shape regularization
            loss_shape = lambda_shape * (betas ** 2).sum()

            # Temporal smoothness (penalize pose change from previous frame)
            loss_smooth = torch.tensor(0.0, device=device)
            if f > 0:
                loss_smooth = lambda_smooth * ((pose_f - all_poses[f - 1:f].detach()) ** 2).sum()

            # Ground contact penalty
            loss_ground = torch.tensor(0.0, device=device)
            ankle_z = pred_joints[0, [7, 8], 2]
            loss_ground = lambda_ground * torch.clamp(-ankle_z, min=0).sum()

            loss = loss_joint + loss_prior + loss_shape + loss_smooth + loss_ground
            loss.backward()
            optimizer.step()

        all_poses[f] = pose_f.detach()[0]
        all_transl[f] = transl_f.detach()[0]

    return (
        betas.detach().cpu().numpy().flatten(),
        all_poses.detach().cpu().numpy(),
        all_transl.detach().cpu().numpy(),
    )

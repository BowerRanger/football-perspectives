"""Pure-PyTorch reimplementation of pytorch3d.transforms functions used by GVHMR.

This shim avoids the full pytorch3d build (which requires CUDA or complex C++
compilation on macOS).  All functions match the pytorch3d API signatures and
semantics exactly.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert rotations given as axis/angle to rotation matrices (Rodrigues).

    Args:
        axis_angle: (*, 3) tensor of axis-angle vectors.
    Returns:
        (*, 3, 3) rotation matrices.
    """
    angles = torch.norm(axis_angle, dim=-1, keepdim=True)  # (*, 1)
    half = angles * 0.5
    small = (angles.squeeze(-1) < 1e-6)

    k = axis_angle / (angles + 1e-8)  # unit axis
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]

    cos_a = torch.cos(angles.squeeze(-1))
    sin_a = torch.sin(angles.squeeze(-1))
    one_minus_cos = 1.0 - cos_a

    # Rodrigues formula
    R = torch.zeros(axis_angle.shape[:-1] + (3, 3), device=axis_angle.device, dtype=axis_angle.dtype)
    R[..., 0, 0] = cos_a + kx * kx * one_minus_cos
    R[..., 0, 1] = kx * ky * one_minus_cos - kz * sin_a
    R[..., 0, 2] = kx * kz * one_minus_cos + ky * sin_a
    R[..., 1, 0] = ky * kx * one_minus_cos + kz * sin_a
    R[..., 1, 1] = cos_a + ky * ky * one_minus_cos
    R[..., 1, 2] = ky * kz * one_minus_cos - kx * sin_a
    R[..., 2, 0] = kz * kx * one_minus_cos - ky * sin_a
    R[..., 2, 1] = kz * ky * one_minus_cos + kx * sin_a
    R[..., 2, 2] = cos_a + kz * kz * one_minus_cos

    # Near-zero angles → identity
    R[small] = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return R


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to axis-angle representation.

    Args:
        matrix: (*, 3, 3) rotation matrices.
    Returns:
        (*, 3) axis-angle vectors.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to quaternions (w, x, y, z).

    Args:
        matrix: (*, 3, 3) rotation matrices.
    Returns:
        (*, 4) quaternions in (w, x, y, z) format.
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    quat = torch.zeros(m.shape[0], 4, device=matrix.device, dtype=matrix.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0
    mask1 = trace > 0
    quat[mask1, 0] = 0.25 * s[mask1]
    quat[mask1, 1] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s[mask1]
    quat[mask1, 2] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s[mask1]
    quat[mask1, 3] = (m[mask1, 1, 0] - m[mask1, 0, 1]) / s[mask1]

    # Case 2: m00 > m11 and m00 > m22
    mask2 = (~mask1) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-10)) * 2.0
    quat[mask2, 0] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s2[mask2]
    quat[mask2, 1] = 0.25 * s2[mask2]
    quat[mask2, 2] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s2[mask2]
    quat[mask2, 3] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s2[mask2]

    # Case 3: m11 > m22
    mask3 = (~mask1) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    s3 = torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-10)) * 2.0
    quat[mask3, 0] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s3[mask3]
    quat[mask3, 1] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s3[mask3]
    quat[mask3, 2] = 0.25 * s3[mask3]
    quat[mask3, 3] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s3[mask3]

    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-10)) * 2.0
    quat[mask4, 0] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s4[mask4]
    quat[mask4, 1] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s4[mask4]
    quat[mask4, 2] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s4[mask4]
    quat[mask4, 3] = 0.25 * s4[mask4]

    return quat.reshape(batch_shape + (4,))


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert quaternions (w, x, y, z) to rotation matrices.

    Args:
        quaternions: (*, 4) quaternions.
    Returns:
        (*, 3, 3) rotation matrices.
    """
    q = F.normalize(quaternions, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.zeros(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quaternion_to_axis_angle(quaternions: Tensor) -> Tensor:
    """Convert quaternions (w, x, y, z) to axis-angle.

    Args:
        quaternions: (*, 4) quaternions.
    Returns:
        (*, 3) axis-angle vectors.
    """
    q = F.normalize(quaternions, dim=-1)
    # Ensure w >= 0
    q = q * (q[..., :1] >= 0).float() * 2 - q
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    sin_half = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    axis = xyz / (sin_half + 1e-8)
    return axis * angle


def so3_exp_map(log_rot: Tensor) -> Tensor:
    """Exponential map from so(3) to SO(3). Same as axis_angle_to_matrix."""
    return axis_angle_to_matrix(log_rot)


def so3_log_map(R: Tensor) -> Tensor:
    """Logarithmic map from SO(3) to so(3). Same as matrix_to_axis_angle."""
    return matrix_to_axis_angle(R)


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Convert 6D rotation representation to rotation matrix.

    Args:
        d6: (*, 6) 6D rotation vectors (first two columns of rotation matrix).
    Returns:
        (*, 3, 3) rotation matrices.
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Convert rotation matrix to 6D representation (first two columns).

    Args:
        matrix: (*, 3, 3) rotation matrices.
    Returns:
        (*, 6) 6D vectors.
    """
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))


def euler_angles_to_matrix(euler_angles: Tensor, convention: str = "XYZ") -> Tensor:
    """Convert Euler angles to rotation matrix.

    Args:
        euler_angles: (*, 3) Euler angles.
        convention: axis ordering string (e.g., "XYZ").
    Returns:
        (*, 3, 3) rotation matrices.
    """
    def _axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        if axis == "X":
            return torch.stack([
                torch.stack([one, zero, zero], dim=-1),
                torch.stack([zero, cos_a, -sin_a], dim=-1),
                torch.stack([zero, sin_a, cos_a], dim=-1),
            ], dim=-2)
        elif axis == "Y":
            return torch.stack([
                torch.stack([cos_a, zero, sin_a], dim=-1),
                torch.stack([zero, one, zero], dim=-1),
                torch.stack([-sin_a, zero, cos_a], dim=-1),
            ], dim=-2)
        elif axis == "Z":
            return torch.stack([
                torch.stack([cos_a, -sin_a, zero], dim=-1),
                torch.stack([sin_a, cos_a, zero], dim=-1),
                torch.stack([zero, zero, one], dim=-1),
            ], dim=-2)
        raise ValueError(f"Unknown axis: {axis}")

    matrices = [
        _axis_angle_rotation(c, euler_angles[..., i])
        for i, c in enumerate(convention)
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

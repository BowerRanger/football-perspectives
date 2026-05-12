"""Tests for SMPL skeleton constants and helpers."""

from __future__ import annotations

import numpy as np

from src.utils.smpl_skeleton import (
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    SMPL_REST_JOINTS_YUP,
    axis_angle_to_quaternion,
    parent_relative_offsets_yup,
)


def test_joint_count_is_24() -> None:
    assert len(SMPL_JOINT_NAMES) == 24
    assert len(SMPL_PARENTS) == 24
    assert SMPL_REST_JOINTS_YUP.shape == (24, 3)


def test_pelvis_is_root() -> None:
    assert SMPL_JOINT_NAMES[0] == "pelvis"
    assert SMPL_PARENTS[0] == -1


def test_parents_are_lower_indices() -> None:
    for j, p in enumerate(SMPL_PARENTS):
        if p == -1:
            assert j == 0
        else:
            assert 0 <= p < j, f"joint {j} parent {p} not topologically before"


def test_pelvis_at_origin() -> None:
    np.testing.assert_allclose(SMPL_REST_JOINTS_YUP[0], np.zeros(3), atol=1e-9)


def test_parent_relative_offsets_pelvis_zero() -> None:
    offsets = parent_relative_offsets_yup()
    np.testing.assert_allclose(offsets[0], np.zeros(3), atol=1e-9)


def test_parent_relative_offsets_match_diff() -> None:
    offsets = parent_relative_offsets_yup()
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        expected = SMPL_REST_JOINTS_YUP[j] - SMPL_REST_JOINTS_YUP[p]
        np.testing.assert_allclose(offsets[j], expected, atol=1e-9)


def test_axis_angle_to_quaternion_identity() -> None:
    q = axis_angle_to_quaternion(np.zeros(3))
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_axis_angle_to_quaternion_90deg_x() -> None:
    aa = np.array([np.pi / 2, 0.0, 0.0])
    q = axis_angle_to_quaternion(aa)
    expected = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-9)


from src.utils.smpl_skeleton import (
    SMPL_JOINT_NAMES,
    SMPL_REST_JOINTS_YUP,
    axis_angle_to_matrix,
    compute_joint_world,
)


def test_axis_angle_to_matrix_identity() -> None:
    R = axis_angle_to_matrix(np.zeros(3))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_axis_angle_to_matrix_90deg_z() -> None:
    R = axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2]))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(R, expected, atol=1e-9)


def test_compute_joint_world_rest_pose_pelvis_at_root_t() -> None:
    """Zero thetas + identity root_R: the pelvis (joint 0) ends up at
    root_t, and every other joint sits at its canonical rest offset."""
    thetas = np.zeros((24, 3))
    root_R = np.eye(3)
    root_t = np.array([10.0, 5.0, 0.0])
    pelvis = compute_joint_world(thetas, root_R, root_t, 0)
    np.testing.assert_allclose(pelvis, root_t, atol=1e-9)
    # Head (joint 15) at rest is above pelvis.
    head_idx = SMPL_JOINT_NAMES.index("head")
    head = compute_joint_world(thetas, root_R, root_t, head_idx)
    expected_head = SMPL_REST_JOINTS_YUP[head_idx] + root_t
    np.testing.assert_allclose(head, expected_head, atol=1e-9)


def test_compute_joint_world_applies_root_rotation() -> None:
    """A 90° root_R about the world z-axis rotates the head sideways."""
    thetas = np.zeros((24, 3))
    root_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    root_t = np.zeros(3)
    head_idx = SMPL_JOINT_NAMES.index("head")
    head = compute_joint_world(thetas, root_R, root_t, head_idx)
    expected = root_R @ SMPL_REST_JOINTS_YUP[head_idx]
    np.testing.assert_allclose(head, expected, atol=1e-9)


def test_compute_joint_world_propagates_parent_rotation() -> None:
    """Rotating the pelvis 180° in canonical y-up flips the head's
    position to the opposite side."""
    thetas = np.zeros((24, 3))
    thetas[0] = np.array([0.0, 0.0, np.pi])  # 180° about canonical z
    root_R = np.eye(3)
    root_t = np.zeros(3)
    head_idx = SMPL_JOINT_NAMES.index("head")
    head = compute_joint_world(thetas, root_R, root_t, head_idx)
    expected = np.array([0.0, 0.0, np.pi])
    R180 = axis_angle_to_matrix(expected)
    expected_head = R180 @ SMPL_REST_JOINTS_YUP[head_idx]
    np.testing.assert_allclose(head, expected_head, atol=1e-9)

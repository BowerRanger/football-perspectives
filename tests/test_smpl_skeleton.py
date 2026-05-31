"""Tests for SMPL skeleton constants and helpers."""

from __future__ import annotations

import numpy as np

from src.utils.smpl_skeleton import (
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    SMPL_REST_JOINTS_YUP,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    compute_joint_world,
    compute_joint_world_pose,
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


def test_compute_joint_world_ignores_theta0() -> None:
    """``thetas[0]`` (global orient) is ignored — ``root_R`` carries the
    root orientation. Rotating only the pelvis leaves every joint at its
    rest world position (otherwise the body double-rotates and flips)."""
    head_idx = SMPL_JOINT_NAMES.index("head")
    root_R = np.eye(3)
    root_t = np.zeros(3)
    rest_head = compute_joint_world(np.zeros((24, 3)), root_R, root_t, head_idx)

    thetas = np.zeros((24, 3))
    thetas[0] = np.array([0.0, 0.0, np.pi])  # 180° pelvis — must be ignored
    head = compute_joint_world(thetas, root_R, root_t, head_idx)
    np.testing.assert_allclose(head, rest_head, atol=1e-9)
    np.testing.assert_allclose(head, SMPL_REST_JOINTS_YUP[head_idx], atol=1e-9)


def test_compute_joint_world_propagates_non_root_parent_rotation() -> None:
    """An articulated (non-root) parent rotation propagates to its
    descendants. Rotating spine1 (joint 3, an ancestor of the head) moves
    the head off its rest position."""
    head_idx = SMPL_JOINT_NAMES.index("head")
    root_R = np.eye(3)
    root_t = np.zeros(3)
    thetas = np.zeros((24, 3))
    thetas[3] = np.array([0.0, 0.0, np.pi / 2])  # spine1 yaw
    head = compute_joint_world(thetas, root_R, root_t, head_idx)
    rest_head = SMPL_REST_JOINTS_YUP[head_idx]
    assert not np.allclose(head, rest_head, atol=1e-3)


def test_compute_joint_world_pose_returns_position_and_rotation() -> None:
    thetas = np.zeros((24, 3))
    root_R = np.eye(3)
    root_t = np.array([1.0, 2.0, 0.0])
    head_idx = 15

    pos, R_world = compute_joint_world_pose(thetas, root_R, root_t, head_idx)

    np.testing.assert_allclose(pos, compute_joint_world(thetas, root_R, root_t, head_idx))
    np.testing.assert_allclose(R_world, np.eye(3), atol=1e-9)


def test_compute_joint_world_pose_applies_root_rotation_to_orientation() -> None:
    thetas = np.zeros((24, 3))
    root_R = axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2]))
    root_t = np.zeros(3)

    _, R_world = compute_joint_world_pose(thetas, root_R, root_t, 15)

    np.testing.assert_allclose(R_world, root_R, atol=1e-9)


def test_compute_joint_world_pose_composes_joint_and_root_rotation() -> None:
    # l_knee (joint 4) chain is 4<-1<-0; with pelvis/hip thetas zero its
    # global rotation equals its own local rotation, so R_world must equal
    # root_R @ R(theta_knee).
    thetas = np.zeros((24, 3))
    thetas[4] = np.array([0.0, np.pi / 2, 0.0])
    root_R = axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2]))
    root_t = np.zeros(3)

    _, R_world = compute_joint_world_pose(thetas, root_R, root_t, 4)

    expected = root_R @ axis_angle_to_matrix(np.array([0.0, np.pi / 2, 0.0]))
    np.testing.assert_allclose(R_world, expected, atol=1e-9)

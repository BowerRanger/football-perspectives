"""Tests for SMPL skeleton constants and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.smpl_skeleton import (
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    SMPL_REST_JOINTS_YUP,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    compute_all_joint_worlds,
    compute_all_joint_worlds_batch,
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


# --- compute_all_joint_worlds_batch --------------------------------------


def test_compute_all_joint_worlds_batch_matches_loop_random_poses() -> None:
    """Batched FK must exactly match calling compute_all_joint_worlds
    per-frame, for randomized poses (root rotation + translation)."""
    rng = np.random.default_rng(42)
    F = 37
    thetas = rng.normal(scale=0.6, size=(F, 24, 3))
    root_axis_angle = rng.normal(scale=np.pi, size=(F, 3))
    root_R = np.stack([axis_angle_to_matrix(a) for a in root_axis_angle])
    root_t = rng.normal(scale=5.0, size=(F, 3))

    batch = compute_all_joint_worlds_batch(thetas, root_R, root_t)
    expected = np.stack(
        [
            compute_all_joint_worlds(thetas[f], root_R[f], root_t[f])
            for f in range(F)
        ]
    )

    assert batch.shape == (F, 24, 3)
    np.testing.assert_allclose(batch, expected, atol=1e-9)


def test_compute_all_joint_worlds_batch_near_zero_axis_angle() -> None:
    """Small-magnitude axis-angle rows exercise Rodrigues' small-angle
    branch — batch must agree with the per-frame path to <=1e-9, including
    exact zeros and magnitudes far below the 1e-12 identity cutoff."""
    rng = np.random.default_rng(7)
    F = 10
    thetas = rng.normal(scale=1e-10, size=(F, 24, 3))
    thetas[0] = 0.0  # exact zero row
    thetas[1, 5] = np.array([0.0, 0.0, np.pi / 3])  # one real rotation mixed in
    root_R = np.stack([np.eye(3) for _ in range(F)])
    root_t = np.zeros((F, 3))

    batch = compute_all_joint_worlds_batch(thetas, root_R, root_t)
    expected = np.stack(
        [
            compute_all_joint_worlds(thetas[f], root_R[f], root_t[f])
            for f in range(F)
        ]
    )

    np.testing.assert_allclose(batch, expected, atol=1e-9)


def test_compute_all_joint_worlds_batch_single_frame() -> None:
    """F=1 edge case: a batch of one frame matches the single-frame call."""
    rng = np.random.default_rng(3)
    thetas = rng.normal(scale=0.4, size=(1, 24, 3))
    root_R = axis_angle_to_matrix(np.array([0.1, -0.2, 0.3]))[np.newaxis, ...]
    root_t = np.array([[1.5, -2.5, 0.0]])

    batch = compute_all_joint_worlds_batch(thetas, root_R, root_t)
    expected = compute_all_joint_worlds(thetas[0], root_R[0], root_t[0])

    assert batch.shape == (1, 24, 3)
    np.testing.assert_allclose(batch[0], expected, atol=1e-9)


def test_compute_all_joint_worlds_batch_custom_rest_joints() -> None:
    """``rest_joints`` override passes through unchanged, matching the
    single-frame function's optional parameter."""
    rng = np.random.default_rng(11)
    F = 4
    thetas = rng.normal(scale=0.3, size=(F, 24, 3))
    root_R = np.stack([np.eye(3) for _ in range(F)])
    root_t = np.zeros((F, 3))
    rest = SMPL_REST_JOINTS_YUP * 1.05  # different bone lengths

    batch = compute_all_joint_worlds_batch(thetas, root_R, root_t, rest_joints=rest)
    expected = np.stack(
        [
            compute_all_joint_worlds(thetas[f], root_R[f], root_t[f], rest_joints=rest)
            for f in range(F)
        ]
    )

    np.testing.assert_allclose(batch, expected, atol=1e-9)


def test_compute_all_joint_worlds_batch_rejects_shape_mismatch() -> None:
    thetas = np.zeros((3, 24, 3))
    root_R = np.stack([np.eye(3)] * 2)  # wrong F (2 instead of 3)
    root_t = np.zeros((3, 3))
    with pytest.raises(ValueError):
        compute_all_joint_worlds_batch(thetas, root_R, root_t)


def test_compute_all_joint_worlds_batch_ignores_theta0() -> None:
    """Same convention as the single-frame path: thetas[:, 0] (global
    orient) must be ignored in the batched FK too."""
    F = 5
    thetas = np.zeros((F, 24, 3))
    thetas[:, 0] = np.array([0.0, 0.0, np.pi])  # 180 deg pelvis, must be ignored
    root_R = np.stack([np.eye(3) for _ in range(F)])
    root_t = np.zeros((F, 3))

    batch = compute_all_joint_worlds_batch(thetas, root_R, root_t)
    head_idx = SMPL_JOINT_NAMES.index("head")
    for f in range(F):
        np.testing.assert_allclose(
            batch[f, head_idx], SMPL_REST_JOINTS_YUP[head_idx], atol=1e-9
        )


def test_compute_all_joint_worlds_batch_does_not_mutate_single_frame_functions() -> None:
    """Sanity check that adding the batch path left the existing
    single-frame API and its outputs untouched."""
    thetas = np.zeros((24, 3))
    root_R = np.eye(3)
    root_t = np.array([1.0, 2.0, 3.0])
    result = compute_all_joint_worlds(thetas, root_R, root_t)
    assert result.shape == (24, 3)
    np.testing.assert_allclose(result[0], root_t, atol=1e-9)

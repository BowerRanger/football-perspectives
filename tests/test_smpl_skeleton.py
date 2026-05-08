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

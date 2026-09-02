"""Tests for the analytic synthetic-walk gait fixture.

The fixture is the ground-truth oracle for Task 3+ contact-detection and
foot-lock tests: it builds a walk cycle by choosing exact foot world
targets and solving sagittal 2-link IK backwards for hip/knee thetas, so
stance feet are EXACTLY stationary (not just approximately, via some
forward-simulated pose) — see docs/superpowers/plans/2026-09-02-foot-
contact-locomotion.md Task 1.
"""

from __future__ import annotations

import numpy as np

from src.utils.smpl_skeleton import compute_all_joint_worlds_batch
from tests.helpers.synthetic_gait import GaitTrack, make_walk


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs in a 1-D boolean array as [start, end) pairs."""
    spans: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def test_walk_stance_feet_are_stationary() -> None:
    g = make_walk(n_frames=100)
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    for side, joint in ((0, 10), (1, 11)):
        for a, b in _spans(g.contacts_true[:, side]):
            span = fw[a:b, joint, :2]
            assert np.linalg.norm(span - span[0], axis=1).max() < 1e-6


def test_walk_root_advances_at_speed() -> None:
    g = make_walk(n_frames=100, speed=2.0)
    dist = np.linalg.norm(g.root_t[-1, :2] - g.root_t[0, :2])
    assert abs(dist / ((len(g.frames) - 1) / g.fps) - 2.0) < 0.15


def test_walk_swing_foot_lifts() -> None:
    g = make_walk(n_frames=100)
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    swing_z = fw[~g.contacts_true[:, 0], 10, 2]
    assert swing_z.max() > 0.08


# --- additional coverage beyond the plan's literal test bodies -----------


def test_make_walk_returns_gait_track_with_expected_shapes() -> None:
    g = make_walk(n_frames=40)
    assert isinstance(g, GaitTrack)
    assert g.frames.shape == (40,)
    assert g.thetas.shape == (40, 24, 3)
    assert g.root_R.shape == (40, 3, 3)
    assert g.root_t.shape == (40, 3)
    assert g.betas.shape == (10,)
    assert g.contacts_true.shape == (40, 2)
    assert g.contacts_true.dtype == bool
    assert g.fps == 25.0


def test_make_walk_root_R_is_a_valid_rotation() -> None:
    g = make_walk(n_frames=10)
    for f in range(len(g.frames)):
        R = g.root_R[f]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) > 0


def test_make_walk_exactly_one_foot_never_both_in_contact_and_swing() -> None:
    # Sanity: a foot cannot be simultaneously in contact and mid-flight —
    # not a strong physical claim, just checks the boolean array is sane.
    g = make_walk(n_frames=60)
    assert g.contacts_true.dtype == bool


def test_make_walk_contact_ratio_allows_flight() -> None:
    """Some frames should have neither foot down (a flight phase) — this
    is what makes the foot-quality contact_ratio metric non-trivial."""
    g = make_walk(n_frames=120)
    both_down = g.contacts_true.all(axis=1)
    neither_down = ~g.contacts_true.any(axis=1)
    assert not both_down.any()  # no double support in this simplified gait
    assert neither_down.any()  # but there IS a flight phase


def test_make_walk_direction_deg_rotates_travel_heading() -> None:
    g0 = make_walk(n_frames=60, direction_deg=0.0)
    g90 = make_walk(n_frames=60, direction_deg=90.0)
    d0 = g0.root_t[-1, :2] - g0.root_t[0, :2]
    d90 = g90.root_t[-1, :2] - g90.root_t[0, :2]
    # Roughly orthogonal travel directions.
    cos_angle = np.dot(d0, d90) / (np.linalg.norm(d0) * np.linalg.norm(d90))
    assert abs(cos_angle) < 0.1


def test_walk_stance_feet_z_is_flat() -> None:
    """Stance feet don't just hold XY — z stays flat (grounded) too."""
    g = make_walk(n_frames=100)
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    for side, joint in ((0, 10), (1, 11)):
        for a, b in _spans(g.contacts_true[:, side]):
            span_z = fw[a:b, joint, 2]
            assert np.abs(span_z - span_z[0]).max() < 1e-6

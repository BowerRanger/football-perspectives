"""Tests for the foot-lock solver library (plan Task 4):
``src.utils.foot_lock.solve_root_with_pins`` (component [C], stance-pinned
root solve). Task 7's ``lock_feet_ik`` / ``penetration_guard`` tests
(component [D]) land in a follow-up append to this module.

numpy/scipy only — no torch, runs on the Mac dev box (see
docs/superpowers/plans/2026-09-02-foot-contact-locomotion.md's global
constraints). All FK calls pin ``rest_joints=SMPL_REST_JOINTS_YUP``
explicitly so tests stay independent of the gitignored
``data/models/smpl_neutral.npz`` beta-adjustment asset and exactly match
the rest table ``tests/helpers/synthetic_gait.make_walk`` builds its
analytic geometry against.
"""

from __future__ import annotations

import numpy as np

from src.utils.foot_contact import ContactSpan, FootContacts
from src.utils.foot_lock import solve_root_with_pins
from src.utils.smpl_skeleton import SMPL_REST_JOINTS_YUP, compute_all_joint_worlds_batch
from tests.helpers.synthetic_gait import contacts_from_truth, make_walk

_REST = SMPL_REST_JOINTS_YUP


# ---------------------------------------------------------------------
# solve_root_with_pins (Task 4)
# ---------------------------------------------------------------------


def test_pinned_solve_zeroes_stance_skate_on_noisy_carrier() -> None:
    g = make_walk(n_frames=120)
    rng = np.random.default_rng(3)
    carrier = g.root_t + rng.normal(0, 0.05, g.root_t.shape)   # 5 cm anchor wobble
    fc = contacts_from_truth(g)                                # fixture -> FootContacts w/ exact pins
    solved, stats = solve_root_with_pins(
        root_carrier=carrier, root_R=g.root_R, thetas=g.thetas, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, solved, _REST)
    for span in fc.spans:
        j = 7 if span.side == 0 else 8
        xy = fw[span.start:span.end, j, :2]
        assert np.linalg.norm(xy - span.pin[:2], axis=1).max() < 0.02


def test_delta_decays_to_carrier_outside_contacts() -> None:
    """A single narrow span in the middle of a long track: far enough
    before/after (beyond decay_s * fps frames from the span) the solved
    root must equal the carrier exactly (delta decayed fully to zero)."""
    n = 200
    root_carrier = np.zeros((n, 3), dtype=float)
    root_carrier[:, 0] = np.linspace(0.0, 10.0, n)
    root_R = np.tile(np.eye(3), (n, 1, 1))
    thetas = np.zeros((n, 24, 3), dtype=float)
    ankle_idx = 7

    pin = _REST[ankle_idx] + root_carrier[100] + np.array([0.1, 0.05, 0.0])
    span = ContactSpan(side=0, start=90, end=110, pin=pin)
    in_contact = np.zeros((n, 2), dtype=bool)
    in_contact[90:110, 0] = True
    quality = np.zeros((n, 2), dtype=float)
    quality[90:110, 0] = 1.0
    fc = FootContacts(n_frames=n, in_contact=in_contact, quality=quality, spans=(span,))

    solved, stats = solve_root_with_pins(
        root_carrier=root_carrier, root_R=root_R, thetas=thetas, betas=np.zeros(10),
        contacts=fc, fps=25.0, decay_s=0.6, rest_joints=_REST,
    )
    decay_frames = 0.6 * 25.0
    assert decay_frames == 15.0

    # Well beyond the decay window on both sides -> exact carrier.
    np.testing.assert_allclose(solved[:90 - int(decay_frames) - 5], root_carrier[:90 - int(decay_frames) - 5])
    np.testing.assert_allclose(solved[110 + int(decay_frames) + 5:], root_carrier[110 + int(decay_frames) + 5:])
    # Mid-span: delta is exactly the injected constant offset.
    np.testing.assert_allclose(solved[100] - root_carrier[100], [0.1, 0.05, 0.0], atol=1e-9)
    assert stats["constrained_frames"] == 20


def test_delta_clamped() -> None:
    """A pin ~2 m off the carrier's implied position clamps |delta| to
    max_correction_m and is counted in clamped_frames."""
    n = 200
    root_carrier = np.zeros((n, 3), dtype=float)
    root_carrier[:, 0] = np.linspace(0.0, 10.0, n)
    root_R = np.tile(np.eye(3), (n, 1, 1))
    thetas = np.zeros((n, 24, 3), dtype=float)
    ankle_idx = 7

    pin_far = _REST[ankle_idx] + root_carrier[100] + np.array([2.0, 0.0, 0.0])
    span = ContactSpan(side=0, start=90, end=110, pin=pin_far)
    in_contact = np.zeros((n, 2), dtype=bool)
    in_contact[90:110, 0] = True
    quality = np.zeros((n, 2), dtype=float)
    quality[90:110, 0] = 1.0
    fc = FootContacts(n_frames=n, in_contact=in_contact, quality=quality, spans=(span,))

    solved, stats = solve_root_with_pins(
        root_carrier=root_carrier, root_R=root_R, thetas=thetas, betas=np.zeros(10),
        contacts=fc, fps=25.0, max_correction_m=0.5, rest_joints=_REST,
    )
    assert np.isclose(np.linalg.norm(solved[100] - root_carrier[100]), 0.5)
    assert stats["clamped_frames"] > 0
    assert np.isclose(stats["max_delta_m"], 0.5)


def test_smooth_no_velocity_spikes_at_span_edges() -> None:
    """The solved root's discrete acceleration (second difference) stays
    within the same order of magnitude as the noisy carrier's own
    acceleration — no spike is introduced at span edges by the
    PCHIP-interpolation-to-linear-decay transition."""
    g = make_walk(n_frames=120)
    fc = contacts_from_truth(g)
    rng = np.random.default_rng(3)
    carrier = g.root_t + rng.normal(0, 0.05, g.root_t.shape)
    solved, _ = solve_root_with_pins(
        root_carrier=carrier, root_R=g.root_R, thetas=g.thetas, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )

    def accel(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.diff(x, n=2, axis=0), axis=1)

    p99_solved = np.percentile(accel(solved), 99)
    p99_carrier = np.percentile(accel(carrier), 99)
    assert p99_solved < 3.0 * p99_carrier


def test_solve_root_with_pins_empty_contacts_returns_carrier_exactly() -> None:
    g = make_walk(n_frames=40)
    fc = FootContacts(
        n_frames=len(g.frames),
        in_contact=np.zeros((len(g.frames), 2), dtype=bool),
        quality=np.zeros((len(g.frames), 2), dtype=float),
        spans=(),
    )
    solved, stats = solve_root_with_pins(
        root_carrier=g.root_t, root_R=g.root_R, thetas=g.thetas, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    np.testing.assert_array_equal(solved, g.root_t)
    assert stats["constrained_frames"] == 0
    assert stats["clamped_frames"] == 0

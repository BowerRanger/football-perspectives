"""Tests for the foot-lock solver library (plan Tasks 4 + 7):
``src.utils.foot_lock.solve_root_with_pins`` (component [C], stance-pinned
root solve) and ``lock_feet_ik`` / ``penetration_guard`` (component [D],
the foot-lock IK finale + penetration guard).

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
import pytest
from scipy.spatial.transform import Rotation

from src.utils.foot_contact import ContactSpan, FootContacts
from src.utils.foot_lock import (
    lock_feet_ik,
    penetration_guard,
    solve_root_with_pins,
)
from src.utils.smpl_skeleton import (
    SMPL_REST_JOINTS_YUP,
    compute_all_joint_worlds_batch,
    compute_joint_world_pose,
)
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
    # solve_root_with_pins now implies the root from the FOOT/toe (10/11)
    # offset, not the ankle (7/8) — see that function's docstring for why
    # (only the toe is exactly stationary during real stance).
    # contacts_from_truth's pin IS the foot/toe position (see its own
    # docstring), so this checks the same joint the pin represents.
    for span in fc.spans:
        j = 10 if span.side == 0 else 11
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
    foot_idx = 10  # foot/toe, not ankle — see solve_root_with_pins's docstring

    pin = _REST[foot_idx] + root_carrier[100] + np.array([0.1, 0.05, 0.0])
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
    foot_idx = 10  # foot/toe, not ankle — see solve_root_with_pins's docstring

    pin_far = _REST[foot_idx] + root_carrier[100] + np.array([2.0, 0.0, 0.0])
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


# ---------------------------------------------------------------------
# lock_feet_ik (Task 7)
# ---------------------------------------------------------------------


def test_lock_feet_ik_lands_feet_on_pins() -> None:
    g = make_walk(n_frames=120)
    noisy = g.root_t + np.random.default_rng(5).normal(0, 0.03, g.root_t.shape)
    fc = contacts_from_truth(g)
    th2, rt2, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=noisy, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    fw = compute_all_joint_worlds_batch(th2, g.root_R, rt2, _REST)
    for span in fc.spans:
        j = 10 if span.side == 0 else 11
        core = slice(span.start + 3, max(span.start + 3, span.end - 3))
        err = np.linalg.norm(fw[core, j, :2] - span.pin[:2], axis=1)
        if err.size:
            assert err.max() < 0.03


def test_lock_feet_ik_lands_feet_on_pins_with_long_stance_spans() -> None:
    """Same scenario as above but with a slower stride so each stance
    span is long enough to have a non-trivial edge-eased "core" (the
    default-parameter plan test's 6-frame spans collapse the core to an
    empty slice with edge_ease_frames=3) — this is the meaningful check
    that the IK genuinely converges once the ease ramp has finished.

    A span whose thetas end up bit-identical to the input was skipped
    (joint-clamp overflow — the separate, deliberate scenario
    ``test_lock_feet_ik_respects_joint_clamp`` covers that path) and is
    excluded from the tight-tolerance check here: with 3 cm/axis noise
    over a long (~12-frame) span, an unlucky single-frame noise draw can
    legitimately need more correction than ik_max_joint_delta_deg allows
    and get skipped-and-restored by design, not by bug."""
    g = make_walk(n_frames=200, stride_s=1.2)
    noisy = g.root_t + np.random.default_rng(5).normal(0, 0.03, g.root_t.shape)
    fc = contacts_from_truth(g)
    th2, rt2, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=noisy, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    assert stats["spans_locked"] > 0
    fw = compute_all_joint_worlds_batch(th2, g.root_R, rt2, _REST)
    max_core_err = 0.0
    n_core_frames = 0
    n_locked_spans_checked = 0
    for span in fc.spans:
        was_skipped = np.array_equal(
            th2[span.start:span.end], g.thetas[span.start:span.end],
        )
        if was_skipped:
            continue
        n_locked_spans_checked += 1
        j = 10 if span.side == 0 else 11
        core = slice(span.start + 3, max(span.start + 3, span.end - 3))
        err = np.linalg.norm(fw[core, j, :2] - span.pin[:2], axis=1)
        if err.size:
            n_core_frames += err.size
            max_core_err = max(max_core_err, float(err.max()))
    assert n_locked_spans_checked >= len(fc.spans) - 1, "expected at most the one deliberately-hard span skipped"
    assert n_core_frames > 0, "expected at least one locked span with a non-empty core"
    assert max_core_err < 0.03


def test_lock_feet_ik_respects_joint_clamp() -> None:
    """Injecting 0.5 m-scale per-frame noise into just ONE span's root
    translation forces a correction beyond ik_max_joint_delta_deg's
    reach; the clamped solve should still leave the foot > 4 cm off, so
    that span is skipped and its thetas are restored bit-exactly, while
    the OTHER (clean) spans still lock normally."""
    g = make_walk(n_frames=120)
    fc = contacts_from_truth(g)
    bad_root = g.root_t.copy()
    target_span = fc.spans[0]
    rng = np.random.default_rng(7)
    span_len = target_span.end - target_span.start
    bad_root[target_span.start:target_span.end] += rng.normal(0, 0.5, (span_len, 3))

    th2, rt2, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=bad_root, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    assert stats["spans_skipped"] >= 1
    assert stats["spans_locked"] >= 1
    np.testing.assert_array_equal(
        th2[target_span.start:target_span.end], g.thetas[target_span.start:target_span.end],
    )


def test_lock_feet_ik_skip_pin_err_m_is_configurable() -> None:
    """``skip_pin_err_m`` controls the pin-landing tolerance beyond which
    a clamped IK solve's span is skipped (default 4 cm, matching
    ``config/default.yaml``'s commented ``refined_poses.foot_lock.
    skip_pin_err_m``). A much larger tolerance should let the same
    deliberately-hard span (see ``test_lock_feet_ik_respects_joint_clamp``)
    through that the default rejects."""
    g = make_walk(n_frames=120)
    fc = contacts_from_truth(g)
    bad_root = g.root_t.copy()
    target_span = fc.spans[0]
    rng = np.random.default_rng(7)
    span_len = target_span.end - target_span.start
    bad_root[target_span.start:target_span.end] += rng.normal(0, 0.5, (span_len, 3))

    _, _, stats_default = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=bad_root, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    assert stats_default["spans_skipped"] >= 1

    th_loose, rt_loose, stats_loose = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=bad_root, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST, skip_pin_err_m=10.0,
    )
    assert stats_loose["spans_skipped"] == 0
    assert stats_loose["spans_locked"] == len(fc.spans)


def test_lock_feet_ik_preserves_foot_global_orientation() -> None:
    g = make_walk(n_frames=120)
    noisy = g.root_t + np.random.default_rng(5).normal(0, 0.03, g.root_t.shape)
    fc = contacts_from_truth(g)
    th2, rt2, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=noisy, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    assert stats["spans_locked"] > 0

    max_angle_deg = 0.0
    for span in fc.spans:
        foot_idx = 10 if span.side == 0 else 11
        for f in range(span.start, span.end):
            _, R_old = compute_joint_world_pose(g.thetas[f], g.root_R[f], noisy[f], foot_idx, _REST)
            _, R_new = compute_joint_world_pose(th2[f], g.root_R[f], rt2[f], foot_idx, _REST)
            angle = Rotation.from_matrix(R_old.T @ R_new).magnitude()
            max_angle_deg = max(max_angle_deg, float(np.degrees(angle)))
    assert max_angle_deg <= 2.0


def test_lock_feet_ik_empty_contacts_returns_input_exactly() -> None:
    g = make_walk(n_frames=40)
    fc = FootContacts(
        n_frames=len(g.frames),
        in_contact=np.zeros((len(g.frames), 2), dtype=bool),
        quality=np.zeros((len(g.frames), 2), dtype=float),
        spans=(),
    )
    th2, rt2, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=g.root_t, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    np.testing.assert_array_equal(th2, g.thetas)
    np.testing.assert_array_equal(rt2, g.root_t)
    assert stats["spans_locked"] == 0
    assert stats["spans_skipped"] == 0


# ---------------------------------------------------------------------
# resolved_pin_err_m / honest unresolved-span handling (Wave 5)
# ---------------------------------------------------------------------


def test_resolved_pin_err_m_defaults_to_skip_pin_err_m_when_omitted() -> None:
    """Not passing resolved_pin_err_m must be a complete no-op: every
    span that locks under the OLD single-threshold logic is reported as
    resolved, spans_unresolved stays 0, and resolved_spans exactly
    matches the locked-span positions -- byte-identical behaviour to
    before this kwarg existed."""
    g = make_walk(n_frames=120)
    noisy = g.root_t + np.random.default_rng(5).normal(0, 0.03, g.root_t.shape)
    fc = contacts_from_truth(g)
    _, _, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=noisy, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    assert stats["spans_unresolved"] == 0
    assert len(stats["resolved_spans"]) == stats["spans_locked"]
    assert stats["spans_locked"] > 0
    locked_positions = {
        (s.side, s.start, s.end) for s in stats["resolved_spans"]
    }
    # Every resolved span position is one of the fixture's true spans
    # (side/start/end pulled straight from contacts.spans).
    true_positions = {(s.side, s.start, s.end) for s in fc.spans}
    assert locked_positions <= true_positions


def test_lock_feet_ik_marks_span_unresolved_when_residual_exceeds_tighter_threshold() -> None:
    """A span whose clamped IK solve lands within the (loose)
    skip_pin_err_m tolerance but NOT within a tighter resolved_pin_err_m
    is reported as unresolved: its theta edits are discarded (matches
    the input exactly, same mechanism as a skip) and it is excluded
    from resolved_spans, while OTHER (clean) spans still resolve
    normally."""
    g = make_walk(n_frames=120)
    fc = contacts_from_truth(g)
    bad_root = g.root_t.copy()
    target_span = fc.spans[0]
    rng = np.random.default_rng(9)
    span_len = target_span.end - target_span.start
    # Small enough noise that the loosened skip_pin_err_m below still
    # lets the clamped IK land within tolerance, but large enough that
    # a tight resolved_pin_err_m rejects it.
    bad_root[target_span.start:target_span.end] += rng.normal(0, 0.08, (span_len, 3))

    th2, rt2, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=bad_root, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
        skip_pin_err_m=0.5, resolved_pin_err_m=0.01,
    )
    assert stats["spans_skipped"] == 0, "sanity: the loose skip tolerance still locks it"
    assert stats["spans_unresolved"] >= 1
    assert stats["spans_locked"] == len(fc.spans) - stats["spans_unresolved"]
    np.testing.assert_array_equal(
        th2[target_span.start:target_span.end], g.thetas[target_span.start:target_span.end],
    )
    resolved_positions = {(s.side, s.start, s.end) for s in stats["resolved_spans"]}
    assert (target_span.side, target_span.start, target_span.end) not in resolved_positions


def test_resolved_pin_err_m_cannot_loosen_past_skip_pin_err_m() -> None:
    """Passing a resolved_pin_err_m LOOSER than skip_pin_err_m is
    clamped to skip_pin_err_m internally -- it can only ever tighten
    the effective bar, never loosen it below what step 4 already
    enforces."""
    g = make_walk(n_frames=120)
    noisy = g.root_t + np.random.default_rng(5).normal(0, 0.03, g.root_t.shape)
    fc = contacts_from_truth(g)
    _, _, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=noisy, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
        skip_pin_err_m=0.04, resolved_pin_err_m=10.0,
    )
    assert stats["spans_unresolved"] == 0
    assert len(stats["resolved_spans"]) == stats["spans_locked"]


def test_lock_feet_ik_marks_span_unresolved_on_large_consecutive_step() -> None:
    """A span whose worst-frame error from the PIN stays under
    resolved_pin_err_m can still fail the honesty check via
    resolved_max_step_m: a big-enough single-frame root_t perturbation
    (mid-span, not at a span edge -- edge_ease_frames=0 here so every
    frame is fully weighted, but the point still generalises) hits the
    IK's joint-angle clamp, landing that one frame CLOSE to the pin
    (within resolved_pin_err_m) but not close enough to its now-clean
    neighbours to avoid a real consecutive-frame jump -- exactly the
    "clamped-but-not-quite-converged frame reads as a skate spike"
    failure mode resolved_max_step_m exists to catch. This is the same
    mechanism that produced gberch P019's real p95 skate regression
    during Wave 5 tuning (see the plan's Wave 5 report)."""
    g = make_walk(n_frames=120, stride_s=1.2)  # longer spans, room for a mid-span outlier
    fc = contacts_from_truth(g)
    target_span = next(s for s in fc.spans if s.end - s.start >= 6)
    noisy_root = g.root_t.copy()
    noisy_root[target_span.start + 2] += np.array([0.15, 0.0, 0.0])

    th2, rt2, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=noisy_root, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST, edge_ease_frames=0,
        skip_pin_err_m=0.5, resolved_pin_err_m=0.06, resolved_max_step_m=None,
    )
    resolved_without_step_check = {
        (s.side, s.start, s.end) for s in stats["resolved_spans"]
    }
    assert (target_span.side, target_span.start, target_span.end) in resolved_without_step_check, (
        "sanity: max_err alone lets this span through"
    )

    th3, rt3, stats2 = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=noisy_root, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST, edge_ease_frames=0,
        skip_pin_err_m=0.5, resolved_pin_err_m=0.06, resolved_max_step_m=0.02,
    )
    resolved_with_step_check = {
        (s.side, s.start, s.end) for s in stats2["resolved_spans"]
    }
    assert (target_span.side, target_span.start, target_span.end) not in resolved_with_step_check
    assert stats2["spans_unresolved"] >= 1


def test_lock_feet_ik_empty_contacts_has_empty_resolved_spans() -> None:
    g = make_walk(n_frames=40)
    fc = FootContacts(
        n_frames=len(g.frames),
        in_contact=np.zeros((len(g.frames), 2), dtype=bool),
        quality=np.zeros((len(g.frames), 2), dtype=float),
        spans=(),
    )
    _, _, stats = lock_feet_ik(
        thetas=g.thetas, root_R=g.root_R, root_t=g.root_t, betas=g.betas,
        contacts=fc, fps=g.fps, rest_joints=_REST,
    )
    assert stats["resolved_spans"] == ()
    assert stats["spans_unresolved"] == 0


# ---------------------------------------------------------------------
# penetration_guard (Task 7)
# ---------------------------------------------------------------------


def test_penetration_guard_raises_only_and_clears_ground() -> None:
    g = make_walk(n_frames=60)
    sunk = g.root_t.copy()
    sunk[:, 2] -= 0.05
    rt2, stats = penetration_guard(
        thetas=g.thetas, root_R=g.root_R, root_t=sunk, betas=g.betas, rest_joints=_REST,
    )
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, rt2, _REST)
    assert fw[:, [10, 11], 2].min() >= 0.025 - 1e-6
    assert (rt2[:, 2] >= sunk[:, 2] - 1e-9).all()
    assert stats["frames_raised"] > 0
    assert stats["max_raise_cm"] > 0.0


def test_penetration_guard_noop_when_clear() -> None:
    g = make_walk(n_frames=60)
    rt2, stats = penetration_guard(
        thetas=g.thetas, root_R=g.root_R, root_t=g.root_t, betas=g.betas, rest_joints=_REST,
    )
    np.testing.assert_array_equal(rt2, g.root_t)
    assert stats["frames_raised"] == 0
    assert stats["max_raise_cm"] == 0.0


def test_penetration_guard_empty_track_is_a_no_op() -> None:
    empty = np.zeros((0, 3), dtype=float)
    rt2, stats = penetration_guard(
        thetas=np.zeros((0, 24, 3)), root_R=np.zeros((0, 3, 3)), root_t=empty,
        betas=np.zeros(10), rest_joints=_REST,
    )
    assert rt2.shape == (0, 3)
    assert stats == {"frames_raised": 0, "max_raise_cm": 0.0}

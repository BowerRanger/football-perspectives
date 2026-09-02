"""Tests for src/utils/foot_contact.py.

Task 1 tests (dataclass shape + JSON round-trip + ``shifted()``) come
first. Task 3 appends tests for the detection algorithms
(``detect_contacts`` / ``derive_contacts_from_fk``) below, exercised
against the analytic synthetic walk (``tests/helpers/synthetic_gait.py``)
so exact ground-truth stance spans are known. Camera/projection helpers
are kept local to this file per the plan (mirrors the equivalent helper
in ``tests/test_foot_quality.py``, duplicated rather than shared so this
file stays independently owned).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.utils.foot_contact import (
    ContactSpan,
    FootContacts,
    derive_contacts_from_fk,
    detect_contacts,
)
from src.utils.smpl_skeleton import (
    beta_adjusted_rest_joints,
    compute_all_joint_worlds_batch,
    load_smpl_neutral_model,
)
from tests.helpers.synthetic_gait import make_walk


def _sample_contacts() -> FootContacts:
    n = 10
    in_contact = np.zeros((n, 2), dtype=bool)
    in_contact[2:6, 0] = True
    in_contact[5:9, 1] = True
    quality = np.zeros((n, 2), dtype=float)
    quality[2:6, 0] = 0.9
    quality[5:9, 1] = 0.8
    spans = (
        ContactSpan(side=0, start=2, end=6, pin=np.array([1.0, 2.0, 0.05])),
        ContactSpan(side=1, start=5, end=9, pin=np.array([3.0, -1.0, 0.05])),
    )
    return FootContacts(n_frames=n, in_contact=in_contact, quality=quality, spans=spans)


def test_contact_span_to_json_from_json_round_trip() -> None:
    span = ContactSpan(side=1, start=5, end=9, pin=np.array([3.0, -1.0, 0.05]))
    d = span.to_json()
    restored = ContactSpan.from_json(d)
    assert restored.side == span.side
    assert restored.start == span.start
    assert restored.end == span.end
    np.testing.assert_allclose(restored.pin, span.pin)


def test_contact_span_to_json_is_plain_json_types() -> None:
    span = ContactSpan(side=0, start=2, end=6, pin=np.array([1.0, 2.0, 0.05]))
    d = span.to_json()
    assert isinstance(d["side"], int)
    assert isinstance(d["start"], int)
    assert isinstance(d["end"], int)
    assert isinstance(d["pin"], list)
    assert all(isinstance(x, float) for x in d["pin"])


def test_foot_contacts_to_json_from_json_round_trip() -> None:
    fc = _sample_contacts()
    d = fc.to_json()
    restored = FootContacts.from_json(d)
    assert restored.n_frames == fc.n_frames
    np.testing.assert_array_equal(restored.in_contact, fc.in_contact)
    np.testing.assert_allclose(restored.quality, fc.quality)
    assert len(restored.spans) == len(fc.spans)
    for a, b in zip(restored.spans, fc.spans):
        assert a.side == b.side
        assert a.start == b.start
        assert a.end == b.end
        np.testing.assert_allclose(a.pin, b.pin)


def test_foot_contacts_to_json_round_trips_through_real_json_module() -> None:
    import json

    fc = _sample_contacts()
    text = json.dumps(fc.to_json())
    restored = FootContacts.from_json(json.loads(text))
    np.testing.assert_array_equal(restored.in_contact, fc.in_contact)


def test_foot_contacts_empty_spans_round_trip() -> None:
    fc = FootContacts(
        n_frames=5,
        in_contact=np.zeros((5, 2), dtype=bool),
        quality=np.zeros((5, 2), dtype=float),
        spans=(),
    )
    restored = FootContacts.from_json(fc.to_json())
    assert restored.spans == ()
    assert restored.n_frames == 5


def test_foot_contacts_shifted_offsets_span_frame_indices() -> None:
    fc = _sample_contacts()
    shifted = fc.shifted(-2)
    assert len(shifted.spans) == len(fc.spans)
    for orig, new in zip(fc.spans, shifted.spans):
        assert new.start == orig.start - 2
        assert new.end == orig.end - 2
        assert new.side == orig.side
        np.testing.assert_allclose(new.pin, orig.pin)


def test_foot_contacts_shifted_preserves_dense_arrays_and_n_frames() -> None:
    """shifted() re-labels span frame numbers (e.g. for sync_map offset
    application); it does not resample the dense per-position arrays."""
    fc = _sample_contacts()
    shifted = fc.shifted(7)
    assert shifted.n_frames == fc.n_frames
    np.testing.assert_array_equal(shifted.in_contact, fc.in_contact)
    np.testing.assert_allclose(shifted.quality, fc.quality)


def test_foot_contacts_shifted_zero_is_a_no_op_copy() -> None:
    fc = _sample_contacts()
    shifted = fc.shifted(0)
    assert shifted is not fc
    for orig, new in zip(fc.spans, shifted.spans):
        assert new.start == orig.start
        assert new.end == orig.end


def test_foot_contacts_is_frozen() -> None:
    fc = _sample_contacts()
    with pytest.raises(Exception):
        fc.n_frames = 99  # type: ignore[misc]


def test_contact_span_is_frozen() -> None:
    span = ContactSpan(side=0, start=0, end=1, pin=np.zeros(3))
    with pytest.raises(Exception):
        span.start = 5  # type: ignore[misc]


# ===========================================================================
# Task 3: detect_contacts / derive_contacts_from_fk
# ===========================================================================


def _default_cfg() -> dict:
    """Mirrors ``config/default.yaml``'s ``hmr_world.contact`` block."""
    return {
        "speed_enter_m_s": 0.6,
        "speed_exit_m_s": 1.2,
        "min_span_frames": 4,
        "max_pin_spread_m": 0.25,
        "px_noise": 2.0,
        "max_correction_m": 0.5,
        "decay_s": 0.6,
    }


def _lookat_camera(
    back: float, up: float, look_at: tuple[float, float, float] = (5.0, 0.0, 0.1),
    fx: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pinhole camera positioned ``back`` metres behind and ``up``
    metres above the pitch origin (along -y/+z), aimed at ``look_at``,
    in the OpenCV world->camera convention (y-down, z-into-scene) used
    elsewhere in the pipeline.

    Steep enough (``up`` well above ``back``, ~60-65 degrees of
    declination at the walk) that ray-casting an ankle pixel onto the
    fixed z=0.05 ground plane isn't dominated by the *inherent*
    plane-height-assumption bias this method always carries (a real
    ankle keypoint sits ~10-15 cm above the ground, and a shallow
    broadcast-camera angle amplifies that into tens of centimetres of
    ground-plane XY error) — this fixture wants a geometry where that
    unavoidable bias is small, so what it actually tests is the
    hysteresis+span DETECTION logic, not the well-known monocular
    ray-cast approximation.
    """
    K = np.array([[fx, 0.0, 960.0], [0.0, fx, 540.0], [0.0, 0.0, 1.0]])
    C = np.array([0.0, -back, up])
    fwd = np.array(look_at) - C
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    R = np.stack([right, -true_up, fwd])  # world->camera rotation
    t = -R @ C
    return K, R, t


def _make_broadcast_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """10 m back, 20 m up, aimed at the walk — see ``_lookat_camera``."""
    return _lookat_camera(back=10.0, up=20.0)


def _make_far_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same declination as ``_make_broadcast_camera`` but 10x further
    out — a tiny, far player whose ankle pixels move only a couple of
    pixels for a real footstep, so pixel noise should swamp the signal
    rather than produce a confident (and correctly-shaped) stance
    detection."""
    return _lookat_camera(back=100.0, up=200.0)


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    cam = pts @ R.T + t
    uv = cam[:, :2] / cam[:, 2:3]
    return uv @ K[:2, :2].T + K[:2, 2]


def _kp2d_from_walk(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, fw: np.ndarray, conf: float = 0.9,
) -> np.ndarray:
    """Build a (F, 17, 3) COCO kp2d array whose ankle rows are the true
    FK ankle joints (7, 8) projected through the given camera."""
    n = fw.shape[0]
    kp2d = np.zeros((n, 17, 3))
    kp2d[..., 2] = conf
    kp2d[:, 15, :2] = _project(K, R, t, fw[:, 7])
    kp2d[:, 16, :2] = _project(K, R, t, fw[:, 8])
    return kp2d


def _true_pin_for_span(fw: np.ndarray, span: ContactSpan) -> np.ndarray:
    """The true (noise-free) FK ankle XY at the midpoint of a detected
    span — valid because the synthetic walk holds the stance ankle
    exactly stationary for the whole true stance window, so any frame
    inside it (and a >85%-agreeing detected span's midpoint should be)
    gives the same answer."""
    ankle_idx = 7 if span.side == 0 else 8
    mid = min(max((span.start + span.end) // 2, 0), fw.shape[0] - 1)
    return fw[mid, ankle_idx, :2]


def test_detect_contacts_recovers_true_stance_spans() -> None:
    g = make_walk(n_frames=120)
    K, R, t = _make_broadcast_camera()
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    kp2d = _kp2d_from_walk(K, R, t, fw)

    fc = detect_contacts(
        kp2d=kp2d, frame_indices=g.frames,
        per_frame_K={int(f): K for f in g.frames},
        per_frame_R={int(f): R for f in g.frames},
        per_frame_t={int(f): t for f in g.frames},
        distortion=(0.0, 0.0),
        thetas=g.thetas, root_R=g.root_R, betas=g.betas, fps=g.fps,
        cfg=_default_cfg(),
    )

    assert fc.n_frames == 120
    assert fc.in_contact.shape == (120, 2)
    agree = (fc.in_contact == g.contacts_true).mean()
    assert agree > 0.85  # edges may differ by a frame or two

    assert len(fc.spans) > 0
    for span in fc.spans:
        true_xy = _true_pin_for_span(fw, span)
        assert np.linalg.norm(span.pin[:2] - true_xy) < 0.08
        assert span.pin[2] == pytest.approx(0.05)


def test_detect_contacts_pixel_noise_no_false_stance_when_far() -> None:
    g = make_walk(n_frames=120)
    K, R, t = _make_far_camera()
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    kp2d = _kp2d_from_walk(K, R, t, fw)

    rng = np.random.default_rng(42)
    kp2d[:, 15, :2] += rng.normal(0.0, 2.0, size=(120, 2))
    kp2d[:, 16, :2] += rng.normal(0.0, 2.0, size=(120, 2))

    fc = detect_contacts(
        kp2d=kp2d, frame_indices=g.frames,
        per_frame_K={int(f): K for f in g.frames},
        per_frame_R={int(f): R for f in g.frames},
        per_frame_t={int(f): t for f in g.frames},
        distortion=(0.0, 0.0),
        thetas=g.thetas, root_R=g.root_R, betas=g.betas, fps=g.fps,
        cfg=_default_cfg(),
    )

    # Every surviving span already satisfies detect_contacts's OWN
    # max_pin_spread_m gate by construction (that's what "surviving"
    # means) — re-asserting that here would be vacuous. What actually
    # matters for "no false stance when far" is that any span which
    # DOES survive the noise isn't a confidently-wrong read of genuine
    # swing motion: a majority of its frames must be real ground-truth
    # stance, not just internally self-consistent under noise.
    for span in fc.spans:
        true_frac = g.contacts_true[span.start:span.end, span.side].mean()
        assert true_frac > 0.5


def test_low_confidence_frames_never_in_contact() -> None:
    g = make_walk(n_frames=60)
    K, R, t = _make_broadcast_camera()
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    kp2d = _kp2d_from_walk(K, R, t, fw, conf=0.1)  # below _ANKLE_CONF_MIN

    fc = detect_contacts(
        kp2d=kp2d, frame_indices=g.frames,
        per_frame_K={int(f): K for f in g.frames},
        per_frame_R={int(f): R for f in g.frames},
        per_frame_t={int(f): t for f in g.frames},
        distortion=(0.0, 0.0),
        thetas=g.thetas, root_R=g.root_R, betas=g.betas, fps=g.fps,
        cfg=_default_cfg(),
    )

    assert not fc.in_contact.any()
    assert fc.spans == ()


# Constant root rotation mapping canonical y-up (x=lateral, y=up, z=fwd)
# into a pitch z-up world where canonical y (up) maps to world z, and
# canonical x maps to world x — used by the kick-rejection test below,
# which hand-crafts a world/pixel trajectory directly rather than going
# through the leg-IK gait fixture. Constructed via cross product
# (image(x) x image(y) = image(z)) to guarantee a proper rotation.
_ROOT_R_UP_Z = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def test_min_span_and_spread_gates_reject_kick() -> None:
    n = 20
    fps = 25.0
    K, R, t = _make_broadcast_camera()
    thetas = np.zeros((n, 24, 3))
    root_R = np.tile(_ROOT_R_UP_Z, (n, 1, 1))
    betas = np.zeros(10)
    frame_indices = np.arange(n)

    # Left-ankle world path: fast (~3.75 m/s) linear motion, except a
    # 5-frame freeze in the middle (kick-like deceleration) that only
    # produces 3 consecutive near-zero-velocity frames via central
    # differencing — one short of min_span_frames (4).
    world = np.zeros((n, 3))
    world[:, 0] = np.arange(n) * 0.15
    world[:, 2] = 0.05
    world[8:13] = world[8].copy()

    uv = _project(K, R, t, world)
    kp2d = np.zeros((n, 17, 3))
    kp2d[:, 15, :2] = uv
    kp2d[:, 15, 2] = 0.9
    kp2d[:, 16, 2] = 0.0  # right ankle never confident -- no interference

    fc = detect_contacts(
        kp2d=kp2d, frame_indices=frame_indices,
        per_frame_K={int(f): K for f in frame_indices},
        per_frame_R={int(f): R for f in frame_indices},
        per_frame_t={int(f): t for f in frame_indices},
        distortion=(0.0, 0.0),
        thetas=thetas, root_R=root_R, betas=betas, fps=fps,
        cfg=_default_cfg(),
    )

    assert not fc.in_contact[9:12, 0].any()
    assert not any(s.side == 0 and s.start <= 10 < s.end for s in fc.spans)


def test_json_round_trip_and_shift() -> None:
    g = make_walk(n_frames=80)
    K, R, t = _make_broadcast_camera()
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    kp2d = _kp2d_from_walk(K, R, t, fw)

    fc = detect_contacts(
        kp2d=kp2d, frame_indices=g.frames,
        per_frame_K={int(f): K for f in g.frames},
        per_frame_R={int(f): R for f in g.frames},
        per_frame_t={int(f): t for f in g.frames},
        distortion=(0.0, 0.0),
        thetas=g.thetas, root_R=g.root_R, betas=g.betas, fps=g.fps,
        cfg=_default_cfg(),
    )
    assert len(fc.spans) > 0

    restored = FootContacts.from_json(json.loads(json.dumps(fc.to_json())))
    np.testing.assert_array_equal(restored.in_contact, fc.in_contact)
    np.testing.assert_allclose(restored.quality, fc.quality)
    assert len(restored.spans) == len(fc.spans)
    for a, b in zip(restored.spans, fc.spans):
        assert a.side == b.side and a.start == b.start and a.end == b.end
        np.testing.assert_allclose(a.pin, b.pin)

    shifted = fc.shifted(-5)
    assert len(shifted.spans) == len(fc.spans)
    for orig, new in zip(fc.spans, shifted.spans):
        assert new.start == orig.start - 5
        assert new.end == orig.end - 5


def test_derive_contacts_from_fk_matches_truth_on_walk() -> None:
    g = make_walk(n_frames=120)

    fc = derive_contacts_from_fk(
        thetas=g.thetas, root_R=g.root_R, root_t=g.root_t, betas=g.betas, fps=g.fps,
    )

    assert fc.n_frames == 120
    agree = (fc.in_contact == g.contacts_true).mean()
    assert agree > 0.9

    # derive_contacts_from_fk tracks the SMPL *foot* joint (10/11), which
    # is what the synthetic walk fixture holds exactly stationary during
    # true stance (see src.utils.foot_contact.derive_contacts_from_fk's
    # docstring for why the ankle, 7/8, is NOT similarly invariant here).
    # Ground truth is computed with the SAME beta-adjusted rest joints
    # derive_contacts_from_fk uses internally (rather than the bare
    # SMPL_REST_JOINTS_YUP constant the fixture's own IK was solved
    # against) so this comparison isn't sensitive to whether this
    # machine happens to have data/models/smpl_neutral.npz — that table
    # differs from the constant by a centimetre or two, which would
    # otherwise leak into this assertion as spurious "error".
    rest_joints = beta_adjusted_rest_joints(g.betas, load_smpl_neutral_model())
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t, rest_joints)
    assert len(fc.spans) > 0
    for span in fc.spans:
        foot_idx = 10 if span.side == 0 else 11
        mid = (span.start + span.end) // 2
        true_xy = fw[mid, foot_idx, :2]
        assert np.linalg.norm(span.pin[:2] - true_xy) < 0.02


def test_derive_contacts_from_fk_empty_track() -> None:
    fc = derive_contacts_from_fk(
        thetas=np.zeros((0, 24, 3)), root_R=np.zeros((0, 3, 3)),
        root_t=np.zeros((0, 3)), betas=np.zeros(10), fps=25.0,
    )
    assert fc.n_frames == 0
    assert fc.spans == ()
    assert fc.in_contact.shape == (0, 2)

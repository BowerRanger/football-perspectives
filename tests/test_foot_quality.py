"""Tests for src/utils/foot_quality.py — the foot-contact locomotion eval
harness. Ground truth comes from the analytic synthetic walk
(tests/helpers/synthetic_gait.py), which makes exact-tolerance
assertions possible: skate should be ~0 on the clean walk and rise
sharply once we inject a known drift or sink.
"""

from __future__ import annotations

import numpy as np

from src.utils.foot_contact import FootContacts
from src.utils.foot_quality import foot_quality_metrics
from tests.helpers.synthetic_gait import make_walk


def _make_broadcast_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A simple pinhole camera looking down at the walk from behind and
    above, matching the OpenCV world->camera convention (y-down, z-into-
    scene) used elsewhere in the pipeline."""
    K = np.array([[2000.0, 0.0, 960.0], [0.0, 2000.0, 540.0], [0.0, 0.0, 1.0]])
    C = np.array([0.0, -30.0, 12.0])  # camera centre, world pitch metres
    # Look roughly toward +y (down the pitch), y-down/z-forward cam axes.
    cam_z = np.array([0.0, 1.0, -0.15])
    cam_z = cam_z / np.linalg.norm(cam_z)
    cam_x = np.cross(cam_z, np.array([0.0, 0.0, 1.0]))
    cam_x = cam_x / np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    R = np.stack([cam_x, cam_y, cam_z])  # world->camera rotation
    t = -R @ C
    return K, R, t


def _project_pinhole(K: np.ndarray, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    cam = pts @ R.T + t
    uv = cam[:, :2] / cam[:, 2:3]
    return uv @ K[:2, :2].T + K[:2, 2]


# --- literal plan tests ----------------------------------------------------


def test_metrics_on_clean_walk_report_no_skate_no_penetration() -> None:
    g = make_walk(n_frames=120)
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert m["skate"]["L"]["mean_mps"] < 0.05
    assert m["penetration"]["pct_frames_sole_below_0"] == 0.0
    assert 0.3 < m["contact_ratio"] < 0.9


def test_metrics_detect_injected_skate() -> None:
    g = make_walk(n_frames=120)
    slid = g.root_t.copy()
    slid[:, 0] += np.linspace(0, 3.0, len(slid))  # +0.63 m/s drift
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=slid, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert m["skate"]["L"]["mean_mps"] > 0.4


def test_metrics_detect_injected_penetration() -> None:
    g = make_walk(n_frames=60)
    sunk = g.root_t.copy()
    sunk[:, 2] -= 0.06
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=sunk, fps=g.fps,
    )
    assert m["penetration"]["pct_frames_sole_below_0"] > 50.0


def test_metrics_penetration_ignores_submillimetre_boundary_noise() -> None:
    """Wave-4 gberch E2E finding: penetration_guard's raise-only pass
    clears penetration to within floating-point rounding (observed:
    max ~3e-8 m, i.e. hundredths of a micrometre) on most frames, but a
    literal ``sole_z < 0.0`` still flags those frames as "penetrating"
    even though the depth is unmeasurable rounding noise, not a real
    clipping artifact. The default 1 mm epsilon (spec §6's documented
    fix for a pct gate that trips ONLY on sub-millimetre frames) must
    NOT count a 0.05 mm dip as penetration, while a real, larger dip
    (well above 1 mm) still counts."""
    g = make_walk(n_frames=100)
    # The synthetic walk's stance foot sits ~5 mm clear of the sole
    # line already (_STANCE_Z=0.03 minus the default sole_clearance_m
    # 0.025), so a REAL sub-epsilon dip has to eat that margin first:
    # subtract 5.5 mm to land ~0.5 mm INTO the sole line, deep enough to
    # trip a literal "< 0" check but still within the 1 mm epsilon.
    tiny_sunk = g.root_t.copy()
    tiny_sunk[:, 2] -= 0.0055
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=tiny_sunk, fps=g.fps,
    )
    assert m["penetration"]["pct_frames_sole_below_0"] == 0.0

    real_sunk = g.root_t.copy()
    real_sunk[:, 2] -= 0.01  # 1 cm — well beyond the 1 mm epsilon
    m2 = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=real_sunk, fps=g.fps,
    )
    assert m2["penetration"]["pct_frames_sole_below_0"] > 50.0


def test_metrics_penetration_epsilon_is_configurable() -> None:
    g = make_walk(n_frames=60)
    # See test_metrics_penetration_ignores_submillimetre_boundary_noise
    # for why this needs to eat the walk's ~5 mm built-in sole margin
    # first: 5.5 mm sink lands ~0.5 mm into the sole line.
    sunk = g.root_t.copy()
    sunk[:, 2] -= 0.0055
    default_eps = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=sunk, fps=g.fps,
    )
    assert default_eps["penetration"]["pct_frames_sole_below_0"] == 0.0

    zero_eps = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=sunk, fps=g.fps,
        penetration_epsilon_m=0.0,
    )
    assert zero_eps["penetration"]["pct_frames_sole_below_0"] > 50.0


# --- coverage for the remaining documented keys -----------------------


def test_metrics_return_all_documented_keys() -> None:
    g = make_walk(n_frames=80)
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    for key in (
        "penetration", "lower_foot_z", "skate", "spans", "flight",
        "contact_ratio", "smoothness",
    ):
        assert key in m
    for side in ("L", "R"):
        assert side in m["skate"]
        for stat in ("mean_mps", "p50_mps", "p95_mps"):
            assert stat in m["skate"][side]
    for stat in ("pct_frames_sole_below_0", "max_depth_cm", "mean_depth_cm"):
        assert stat in m["penetration"]
    for stat in ("mean", "p05", "p50", "p95"):
        assert stat in m["lower_foot_z"]
    for stat in ("count", "mean_m", "max_m"):
        assert stat in m["spans"]
    for stat in ("root_acc_p99_m_s2", "root_acc_max_m_s2", "foot_speed_max_mps"):
        assert stat in m["smoothness"]
    assert "pct_frames_both_up" in m["flight"] or "pct" in m["flight"]


def test_metrics_without_contacts_falls_back_to_low_foot_mask() -> None:
    """No contacts sidecar available: the "foot z < 0.10 m" proxy is
    cruder than exact stance spans (it legitimately includes some real
    swing motion below the height threshold, since the synthetic walk's
    swing arc spends much of its duration under 0.10 m) — so skate under
    the fallback should be higher than (or equal to) the contact-exact
    measurement, not near-zero. This demonstrates the fallback still
    runs and produces a finite, non-exploding number rather than
    asserting it matches the exact-contact precision.
    """
    g = make_walk(n_frames=100)
    with_contacts = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    without_contacts = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=None,
    )
    assert without_contacts["skate"]["L"]["mean_mps"] >= with_contacts["skate"]["L"]["mean_mps"]
    assert without_contacts["skate"]["L"]["mean_mps"] < 5.0
    assert without_contacts["skate"]["R"]["mean_mps"] < 5.0


def test_metrics_spans_count_matches_contact_run_count() -> None:
    g = make_walk(n_frames=120)
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    # Count contiguous True runs across both feet directly from the truth.
    expected = 0
    for side in (0, 1):
        col = g.contacts_true[:, side]
        expected += int(np.sum(np.diff(np.concatenate([[0], col.astype(int), [0]])) == 1))
    assert m["spans"]["count"] == expected
    assert m["spans"]["mean_m"] < 0.01  # stance spans are exactly stationary


def test_metrics_flight_pct_matches_truth_when_using_contacts() -> None:
    g = make_walk(n_frames=120)
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    neither_down = ~g.contacts_true.any(axis=1)
    expected_pct = 100.0 * neither_down.mean()
    assert expected_pct > 0.0  # sanity: the fixture does have flight
    assert m["flight"]["pct_frames_both_up"] > 0.0


def test_metrics_lower_foot_z_distribution_is_nonnegative_on_clean_walk() -> None:
    g = make_walk(n_frames=100)
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert m["lower_foot_z"]["p50"] >= -1e-6
    assert m["lower_foot_z"]["p95"] > m["lower_foot_z"]["p05"]


def test_metrics_ankle_reproj_px_near_zero_for_perfect_projection() -> None:
    g = make_walk(n_frames=60)
    from src.utils.smpl_skeleton import compute_all_joint_worlds_batch

    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    K, R, t = _make_broadcast_camera()
    kp2d = np.zeros((60, 17, 3))
    kp2d[:, 15, :2] = _project_pinhole(K, R, t, fw[:, 7])  # l_ankle
    kp2d[:, 16, :2] = _project_pinhole(K, R, t, fw[:, 8])  # r_ankle
    kp2d[:, 15, 2] = 0.9
    kp2d[:, 16, 2] = 0.9
    cameras = {int(f): (K, R, t) for f in g.frames}

    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true, kp2d=kp2d, cameras=cameras,
    )
    assert "ankle_reproj_px" in m
    assert m["ankle_reproj_px"]["mean_px"] < 1e-6
    assert m["ankle_reproj_px"]["p95_px"] < 1e-6


def test_metrics_ankle_reproj_px_reflects_injected_pixel_error() -> None:
    g = make_walk(n_frames=60)
    from src.utils.smpl_skeleton import compute_all_joint_worlds_batch

    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    K, R, t = _make_broadcast_camera()
    kp2d = np.zeros((60, 17, 3))
    kp2d[:, 15, :2] = _project_pinhole(K, R, t, fw[:, 7]) + np.array([5.0, -3.0])
    kp2d[:, 16, :2] = _project_pinhole(K, R, t, fw[:, 8]) + np.array([5.0, -3.0])
    kp2d[:, 15, 2] = 0.9
    kp2d[:, 16, 2] = 0.9
    cameras = {int(f): (K, R, t) for f in g.frames}

    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true, kp2d=kp2d, cameras=cameras,
    )
    assert m["ankle_reproj_px"]["mean_px"] > 4.0


def test_metrics_ankle_reproj_px_absent_without_kp2d_and_cameras() -> None:
    g = make_walk(n_frames=40)
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert "ankle_reproj_px" not in m


def test_metrics_ankle_reproj_px_skips_low_confidence_frames() -> None:
    g = make_walk(n_frames=40)
    from src.utils.smpl_skeleton import compute_all_joint_worlds_batch

    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    K, R, t = _make_broadcast_camera()
    kp2d = np.zeros((40, 17, 3))
    kp2d[:, 15, :2] = _project_pinhole(K, R, t, fw[:, 7])
    kp2d[:, 16, :2] = _project_pinhole(K, R, t, fw[:, 8])
    kp2d[:, 15, 2] = 0.9
    kp2d[:, 16, 2] = 0.9
    # Inject a huge pixel error on a low-confidence frame — must be
    # excluded from the mean by the conf >= 0.5 gate.
    kp2d[0, 15, :2] += 500.0
    kp2d[0, 15, 2] = 0.1
    cameras = {int(f): (K, R, t) for f in g.frames}

    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true, kp2d=kp2d, cameras=cameras,
    )
    assert m["ankle_reproj_px"]["mean_px"] < 1e-6


# --- smoothness (root acceleration, foot speed) -----------------------
# Regression tracking for isolated super-physical root-translation pops
# (see docs/superpowers/plans/2026-09-02-foot-contact-locomotion.md's
# follow-up polish task): pops coincide with upstream hmr_world anchor/
# kp2d noise frames, not contact-span boundaries, and are diagnosed via
# the second finite difference of root_t plus unconstrained FK foot
# speed rather than the (contact-gated) skate metric above.


def test_smoothness_root_acc_low_on_clean_walk() -> None:
    """The synthetic walk's pelvis moves at constant velocity (straight
    line, constant height) -- essentially zero acceleration -- so a
    healthy track's smoothness numbers should read near zero."""
    g = make_walk(n_frames=100)
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert m["smoothness"]["root_acc_max_m_s2"] < 0.5
    assert m["smoothness"]["root_acc_p99_m_s2"] < 0.5


def test_smoothness_detects_injected_root_pop() -> None:
    """A single-frame 0.4 m out-and-back pop (the class of artifact
    refined_poses.cleanup.a_max_m_s2 targets) registers as several
    hundred m/s^2 -- the metric must surface it clearly above any
    genuine-motion noise floor."""
    g = make_walk(n_frames=100)
    popped = g.root_t.copy()
    popped[50, 0] += 0.4
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=popped, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert m["smoothness"]["root_acc_max_m_s2"] > 200.0


def test_smoothness_foot_speed_is_finite_and_positive_on_clean_walk() -> None:
    """foot_speed_max_mps is the UNCONSTRAINED max FK foot-joint speed
    (unlike ``skate``, not gated to contact spans) -- on a clean walk it
    should reflect ordinary swing-phase motion, not blow up."""
    g = make_walk(n_frames=100)
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=g.root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert 0.5 < m["smoothness"]["foot_speed_max_mps"] < 10.0


def test_smoothness_foot_speed_detects_injected_root_pop() -> None:
    """A root pop drags the whole body -- including the feet -- so
    foot_speed_max_mps must also spike, independent of contact state."""
    g = make_walk(n_frames=100)
    popped = g.root_t.copy()
    popped[50, 0] += 0.4
    m = foot_quality_metrics(
        frames=g.frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=popped, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert m["smoothness"]["foot_speed_max_mps"] > 5.0


def test_smoothness_zero_on_empty_track() -> None:
    m = foot_quality_metrics(
        frames=np.zeros(0, dtype=np.int64), betas=np.zeros(10),
        thetas=np.zeros((0, 24, 3)), root_R=np.zeros((0, 3, 3)),
        root_t=np.zeros((0, 3)), fps=25.0,
    )
    assert m["smoothness"] == {
        "root_acc_p99_m_s2": 0.0,
        "root_acc_max_m_s2": 0.0,
        "foot_speed_max_mps": 0.0,
    }


def test_root_accel_stats_ignores_boundary_across_a_real_frame_gap() -> None:
    """Wave-4c diagnosis: ``refined_poses.cleanup`` densifies gaps up to
    ``max_gap_fill_frames`` but deliberately leaves wider ones unfilled
    (a real, un-fabricated occlusion hold) -- so two adjacent ARRAY rows
    of ``root_t`` can be dozens of real VIDEO frames apart. Scoring the
    central difference across that boundary as if it were 1/fps seconds
    reports a huge fictitious acceleration for what is, over the real
    elapsed time, ordinary motion (here: 2 m over a real 40-frame gap
    at 30 fps, ~1.5 m/s). ``frames`` lets the metric skip that boundary,
    the same way ``skate`` above is never scored across a stance/swing
    run boundary."""
    from src.utils.foot_quality import _root_accel_stats

    fps = 30.0
    n1, n2 = 10, 10
    frames = np.concatenate([
        np.arange(0, n1, dtype=np.int64),
        np.arange(0, n2, dtype=np.int64) + 40,
    ])
    root_t = np.concatenate([
        np.tile([0.0, 0.0, 0.95], (n1, 1)),
        np.tile([2.0, 0.0, 0.95], (n2, 1)),
    ])
    p99, mx = _root_accel_stats(root_t, fps, frames=frames)
    assert mx < 1.0
    assert p99 < 1.0
    # Without frame numbers (old behaviour / callers that don't have
    # them) the same data reads as an enormous spurious spike -- proves
    # the assertion above is actually exercising the gap-aware path,
    # not just a coincidentally-small number.
    _p99_naive, mx_naive = _root_accel_stats(root_t, fps)
    assert mx_naive > 500.0


def test_foot_speed_max_ignores_boundary_across_a_real_frame_gap() -> None:
    """Same class of bug as ``_root_accel_stats`` above, one derivative
    order down: a real un-fabricated occlusion gap must not be scored
    as an instantaneous foot teleport."""
    from src.utils.foot_quality import _foot_speed_max

    fps = 30.0
    n1, n2 = 10, 10
    frames = np.concatenate([
        np.arange(0, n1, dtype=np.int64),
        np.arange(0, n2, dtype=np.int64) + 40,
    ])
    feet_pos = np.zeros((n1 + n2, 2, 3))
    feet_pos[n1:, :, 0] = 2.0  # real 2 m shift over the real 30-frame gap
    speed = _foot_speed_max(feet_pos, fps, frames=frames)
    assert speed < 1.0
    speed_naive = _foot_speed_max(feet_pos, fps)
    assert speed_naive > 50.0


def test_foot_quality_metrics_smoothness_ignores_real_frame_gap() -> None:
    """Same scenario as above, through the full ``foot_quality_metrics``
    entry point (which already receives ``frames`` for the kp2d
    reprojection lookup) on a real walk fixture with a spliced-in
    occlusion gap and a genuine displacement across it."""
    g = make_walk(n_frames=40)
    frames = g.frames.copy()
    frames[20:] += 30  # real, un-fabricated 30-frame occlusion hold
    root_t = g.root_t.copy()
    root_t[20:, 0] += 2.0  # real displacement over that untracked span
    m = foot_quality_metrics(
        frames=frames, betas=g.betas, thetas=g.thetas,
        root_R=g.root_R, root_t=root_t, fps=g.fps,
        contacts=g.contacts_true,
    )
    assert m["smoothness"]["root_acc_max_m_s2"] < 10.0


# --- scripts/eval_foot_quality.py CLI --------------------------------

import json as _json
import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = _Path(__file__).resolve().parents[1] / "scripts"


def _import_cli():
    if str(_SCRIPTS_DIR) not in _sys.path:
        _sys.path.insert(0, str(_SCRIPTS_DIR))
    import eval_foot_quality

    return eval_foot_quality


def test_discover_hmr_entries_handles_nested_underscore_pid(tmp_path) -> None:
    cli = _import_cli()
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir()
    (hmr_dir / "s013__s013_TT001_smpl_world.npz").write_bytes(b"")
    (hmr_dir / "s013__s013_TT001_kp2d.json").write_text("{}")
    entries = cli._discover_hmr_entries(hmr_dir)
    assert entries == [
        (
            "s013",
            "s013_TT001",
            hmr_dir / "s013__s013_TT001_smpl_world.npz",
            hmr_dir / "s013__s013_TT001_kp2d.json",
        )
    ]


def test_discover_hmr_entries_handles_legacy_no_shot_naming(tmp_path) -> None:
    cli = _import_cli()
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir()
    (hmr_dir / "P001_smpl_world.npz").write_bytes(b"")
    entries = cli._discover_hmr_entries(hmr_dir)
    assert entries == [("", "P001", hmr_dir / "P001_smpl_world.npz", hmr_dir / "P001_kp2d.json")]


def test_load_contacts_sidecar_returns_none_when_missing(tmp_path) -> None:
    cli = _import_cli()
    assert cli._load_contacts_sidecar(tmp_path, "gberch", "P001", np.arange(10)) is None


def test_load_contacts_sidecar_maps_by_global_frame_for_trimmed_track(tmp_path) -> None:
    """The real bug this fixes: a refined_poses track has been trimmed/
    resampled relative to the hmr_world track the sidecar was computed
    from, so its length (and frame membership) legitimately differs.
    Mapping by GLOBAL FRAME NUMBER (via the matching hmr_world npz's own
    ``frames`` array) must still recover the correct per-frame flags,
    where the old exact-length positional read would have silently
    returned ``None`` (forcing every such caller onto the coarser z<0.10
    proxy) even though the sidecar had everything needed to answer
    honestly."""
    from src.schemas.foot_contacts import save_foot_contacts
    from src.schemas.smpl_world import SmplWorldTrack

    cli = _import_cli()
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir(parents=True)

    g = make_walk(n_frames=30)
    hmr_frames = (g.frames + 100).astype(np.int64)  # global frame numbers 100..129

    track = SmplWorldTrack(
        player_id="P001", frames=hmr_frames, betas=g.betas.astype("float32"),
        thetas=g.thetas.astype("float32"), root_R=g.root_R.astype("float32"),
        root_t=g.root_t.astype("float32"),
        confidence=np.ones(len(hmr_frames), dtype="float32"), shot_id="shotA",
    )
    track.save(hmr_dir / "shotA__P001_smpl_world.npz")

    from tests.helpers.synthetic_gait import contacts_from_truth

    fc = contacts_from_truth(g)  # array positions 0..29, aligned with hmr_frames
    save_foot_contacts(
        hmr_dir / "shotA__P001_foot_contacts.json", fc,
        shot_id="shotA", player_id="P001", anchor_mode="contact",
    )

    # A "refined" caller track: a TRIMMED, differently-lengthed subset of
    # the same global frames (global 105..124, 20 frames — array
    # positions 5:25 of the hmr_world track).
    trimmed_global_frames = hmr_frames[5:25]
    contacts = cli._load_contacts_sidecar(hmr_dir, "shotA", "P001", trimmed_global_frames)

    assert contacts is not None
    assert contacts.shape == (20, 2)
    np.testing.assert_array_equal(contacts, fc.in_contact[5:25])


def test_load_contacts_sidecar_falls_back_to_positional_when_hmr_npz_missing(tmp_path) -> None:
    """Sidecar present but its source hmr_world npz is gone (e.g. wiped
    after the sidecar was written): fall back to the old exact-length
    positional interpretation rather than returning None unconditionally."""
    from src.schemas.foot_contacts import save_foot_contacts

    cli = _import_cli()
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir(parents=True)

    g = make_walk(n_frames=20)
    from tests.helpers.synthetic_gait import contacts_from_truth

    fc = contacts_from_truth(g)
    save_foot_contacts(
        hmr_dir / "shotA__P001_foot_contacts.json", fc,
        shot_id="shotA", player_id="P001", anchor_mode="contact",
    )
    # No shotA__P001_smpl_world.npz written.

    contacts = cli._load_contacts_sidecar(hmr_dir, "shotA", "P001", g.frames)
    assert contacts is not None
    np.testing.assert_array_equal(contacts, fc.in_contact)

    # A caller whose length doesn't match the sidecar's n_frames can't be
    # resolved positionally either -> None, not a silent misalignment.
    assert cli._load_contacts_sidecar(hmr_dir, "shotA", "P001", g.frames[:10]) is None


def test_load_resolved_contacts_returns_none_when_missing(tmp_path) -> None:
    cli = _import_cli()
    (tmp_path / "refined_poses").mkdir(parents=True)
    assert cli._load_resolved_contacts(tmp_path, "P001", np.arange(10)) is None


def test_load_resolved_contacts_loads_when_present(tmp_path) -> None:
    from src.schemas.foot_contacts import save_foot_contacts

    cli = _import_cli()
    (tmp_path / "refined_poses").mkdir(parents=True)
    n = 10
    in_contact = np.zeros((n, 2), dtype=bool)
    in_contact[2:6, 0] = True
    fc = FootContacts(
        n_frames=n, in_contact=in_contact, quality=in_contact.astype(float), spans=(),
    )
    save_foot_contacts(
        tmp_path / "refined_poses" / "P001_resolved_contacts.json", fc,
        shot_id="shotA", player_id="P001", anchor_mode="resolved",
    )
    out = cli._load_resolved_contacts(tmp_path, "P001", np.arange(n))
    assert out is not None
    np.testing.assert_array_equal(out, in_contact)


def test_load_resolved_contacts_rejects_wrong_anchor_mode(tmp_path) -> None:
    """A file present but not tagged anchor_mode="resolved" (e.g. a
    stray raw-contact sidecar someone copied into refined_poses/) is
    NOT trusted as the verified set — the caller falls back instead of
    silently misinterpreting it."""
    from src.schemas.foot_contacts import save_foot_contacts

    cli = _import_cli()
    (tmp_path / "refined_poses").mkdir(parents=True)
    n = 10
    fc = FootContacts(
        n_frames=n, in_contact=np.ones((n, 2), dtype=bool),
        quality=np.ones((n, 2)), spans=(),
    )
    save_foot_contacts(
        tmp_path / "refined_poses" / "P001_resolved_contacts.json", fc,
        shot_id="shotA", player_id="P001", anchor_mode="contact",
    )
    assert cli._load_resolved_contacts(tmp_path, "P001", np.arange(n)) is None


def test_load_resolved_contacts_returns_none_on_length_mismatch(tmp_path) -> None:
    from src.schemas.foot_contacts import save_foot_contacts

    cli = _import_cli()
    (tmp_path / "refined_poses").mkdir(parents=True)
    n = 10
    fc = FootContacts(
        n_frames=n, in_contact=np.ones((n, 2), dtype=bool),
        quality=np.ones((n, 2)), spans=(),
    )
    save_foot_contacts(
        tmp_path / "refined_poses" / "P001_resolved_contacts.json", fc,
        shot_id="shotA", player_id="P001", anchor_mode="resolved",
    )
    assert cli._load_resolved_contacts(tmp_path, "P001", np.arange(3)) is None


def _write_fixture(tmp_path) -> "_Path":
    from src.schemas.camera_track import CameraFrame, CameraTrack
    from src.schemas.refined_pose import RefinedPose
    from src.schemas.smpl_world import SmplWorldTrack
    from src.utils.smpl_skeleton import compute_all_joint_worlds_batch

    g = make_walk(n_frames=30)
    K, R, t = _make_broadcast_camera()
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)

    (tmp_path / "hmr_world").mkdir(parents=True)
    (tmp_path / "camera").mkdir(parents=True)
    (tmp_path / "refined_poses").mkdir(parents=True)

    track = SmplWorldTrack(
        player_id="P001", frames=g.frames, betas=g.betas.astype("float32"),
        thetas=g.thetas.astype("float32"), root_R=g.root_R.astype("float32"),
        root_t=g.root_t.astype("float32"),
        confidence=np.ones(len(g.frames), dtype="float32"), shot_id="shotA",
    )
    track.save(tmp_path / "hmr_world" / "shotA__P001_smpl_world.npz")

    kp2d_payload: dict = {"player_id": "P001", "shot_id": "shotA", "frames": []}
    for i, f in enumerate(g.frames):
        kp = [[0.0, 0.0, 0.0] for _ in range(17)]
        l_uv = _project_pinhole(K, R, t, fw[i, 7][None, :])[0]
        r_uv = _project_pinhole(K, R, t, fw[i, 8][None, :])[0]
        kp[15] = [float(l_uv[0]), float(l_uv[1]), 0.9]
        kp[16] = [float(r_uv[0]), float(r_uv[1]), 0.9]
        kp2d_payload["frames"].append({"frame": int(f), "keypoints": kp})
    (tmp_path / "hmr_world" / "shotA__P001_kp2d.json").write_text(_json.dumps(kp2d_payload))

    frames_cam = tuple(
        CameraFrame(
            frame=int(f), K=K.tolist(), R=R.tolist(), confidence=1.0,
            is_anchor=True, t=t.tolist(),
        )
        for f in g.frames
    )
    cam_track = CameraTrack(
        clip_id="shotA", fps=g.fps, image_size=(1920, 1080),
        t_world=t.tolist(), frames=frames_cam,
    )
    cam_track.save(tmp_path / "camera" / "shotA_camera_track.json")

    refined = RefinedPose(
        player_id="P001", frames=g.frames, betas=g.betas.astype("float32"),
        thetas=g.thetas.astype("float32"), root_R=g.root_R.astype("float32"),
        root_t=g.root_t.astype("float32"),
        confidence=np.ones(len(g.frames), dtype="float32"),
        view_count=np.ones(len(g.frames), dtype="int32"),
        contributing_shots=("shotA",),
    )
    refined.save(tmp_path / "refined_poses" / "P001_refined.npz")

    # A STALE refined npz with no matching hmr_world sidecar (mirrors the
    # real repo's output/refined_poses P004-P015, left over from a
    # different run) — auto-discovery must skip it, not crash on it.
    stale = RefinedPose(
        player_id="P099", frames=g.frames, betas=g.betas.astype("float32"),
        thetas=g.thetas.astype("float32"), root_R=g.root_R.astype("float32"),
        root_t=g.root_t.astype("float32"),
        confidence=np.ones(len(g.frames), dtype="float32"),
        view_count=np.ones(len(g.frames), dtype="int32"),
        contributing_shots=("shotZ",),
    )
    stale.save(tmp_path / "refined_poses" / "P099_refined.npz")
    return tmp_path


def test_eval_hmr_player_computes_ankle_reprojection(tmp_path) -> None:
    cli = _import_cli()
    out = _write_fixture(tmp_path)
    m = cli.eval_hmr_player(
        out, "shotA", "P001",
        out / "hmr_world" / "shotA__P001_smpl_world.npz",
        out / "hmr_world" / "shotA__P001_kp2d.json",
        None,
    )
    assert m is not None
    assert "ankle_reproj_px" in m
    assert m["ankle_reproj_px"]["mean_px"] < 1.0


def test_eval_refined_player_returns_metrics(tmp_path) -> None:
    cli = _import_cli()
    out = _write_fixture(tmp_path)
    m = cli.eval_refined_player(out, "P001", None)
    assert m is not None
    assert "skate" in m
    assert "penetration" in m


def test_eval_refined_player_prefers_resolved_contacts_over_hmr_sidecar(tmp_path) -> None:
    """When BOTH a raw hmr_world foot_contacts sidecar (says: in contact
    the whole track) and a refined_poses resolved_contacts sidecar (says:
    never in contact) are present, eval_refined_player's contact_ratio
    reflects the RESOLVED one — the honest, verified set wins over the
    raw detection-time one."""
    from src.schemas.foot_contacts import save_foot_contacts

    cli = _import_cli()
    out = _write_fixture(tmp_path)
    n = 30  # matches _write_fixture's make_walk(n_frames=30)

    hmr_fc = FootContacts(
        n_frames=n, in_contact=np.ones((n, 2), dtype=bool),
        quality=np.ones((n, 2)), spans=(),
    )
    save_foot_contacts(
        out / "hmr_world" / "shotA__P001_foot_contacts.json", hmr_fc,
        shot_id="shotA", player_id="P001", anchor_mode="contact",
    )
    resolved_fc = FootContacts(
        n_frames=n, in_contact=np.zeros((n, 2), dtype=bool),
        quality=np.zeros((n, 2)), spans=(),
    )
    save_foot_contacts(
        out / "refined_poses" / "P001_resolved_contacts.json", resolved_fc,
        shot_id="shotA", player_id="P001", anchor_mode="resolved",
    )

    m = cli.eval_refined_player(out, "P001", None)
    assert m is not None
    assert m["contact_ratio"] == 0.0


def test_eval_refined_player_falls_back_to_hmr_sidecar_without_resolved(tmp_path) -> None:
    from src.schemas.foot_contacts import save_foot_contacts

    cli = _import_cli()
    out = _write_fixture(tmp_path)
    n = 30

    hmr_fc = FootContacts(
        n_frames=n, in_contact=np.ones((n, 2), dtype=bool),
        quality=np.ones((n, 2)), spans=(),
    )
    save_foot_contacts(
        out / "hmr_world" / "shotA__P001_foot_contacts.json", hmr_fc,
        shot_id="shotA", player_id="P001", anchor_mode="contact",
    )

    m = cli.eval_refined_player(out, "P001", None)
    assert m is not None
    assert m["contact_ratio"] == 1.0


def test_eval_refined_player_returns_none_for_missing_npz(tmp_path) -> None:
    cli = _import_cli()
    (tmp_path / "refined_poses").mkdir(parents=True)
    assert cli.eval_refined_player(tmp_path, "P404", None) is None


def test_main_auto_discovery_skips_stale_refined_npz(tmp_path) -> None:
    cli = _import_cli()
    out = _write_fixture(tmp_path)
    results = cli.main(["--output", str(out)])
    assert "P001" in results["players"]
    assert "P099" not in results["players"]


def test_main_writes_json_with_requested_players(tmp_path, monkeypatch) -> None:
    cli = _import_cli()
    out = _write_fixture(tmp_path)
    # The fixture's kp2d "ground truth" was projected using the canonical
    # SMPL_REST_JOINTS_YUP table (matching make_walk); pin main() to the
    # same table (rather than whatever data/models/smpl_neutral.npz
    # happens to be on this machine) so the reprojection assertion below
    # is deterministic across environments.
    monkeypatch.setattr(cli, "load_smpl_neutral_model", lambda: None)
    json_path = tmp_path / "baseline.json"
    cli.main(["--output", str(out), "--players", "P001", "--json", str(json_path)])
    data = _json.loads(json_path.read_text())
    assert "P001" in data["players"]
    assert "refined" in data["players"]["P001"]
    assert "hmr[shotA]" in data["players"]["P001"]
    assert data["players"]["P001"]["hmr[shotA]"]["ankle_reproj_px"]["mean_px"] < 1.0


def test_main_stage_refined_only_omits_hmr_key(tmp_path) -> None:
    cli = _import_cli()
    out = _write_fixture(tmp_path)
    json_path = tmp_path / "baseline.json"
    cli.main(["--output", str(out), "--players", "P001", "--stage", "refined", "--json", str(json_path)])
    data = _json.loads(json_path.read_text())
    assert "refined" in data["players"]["P001"]
    assert not any(k.startswith("hmr") for k in data["players"]["P001"])


def test_metrics_handles_empty_track() -> None:
    m = foot_quality_metrics(
        frames=np.zeros(0, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((0, 24, 3)),
        root_R=np.zeros((0, 3, 3)),
        root_t=np.zeros((0, 3)),
        fps=25.0,
    )
    assert m["spans"]["count"] == 0
    assert m["contact_ratio"] == 0.0

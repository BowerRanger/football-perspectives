"""Task 3 of the hybrid-device shim campaign: the pure parity-metric
function in ``scripts/bench_gvhmr_inference.py`` used by
``--compare-against`` to gate an ``--extractor-device`` run's raw GVHMR
output arrays against a reference run's arrays.

Pure numpy comparison -- no torch, no GVHMR, no file IO -- so this runs
in unit-test time on any machine (per CLAUDE.md: refined_poses/bench
metric helpers like this one are testable without the GPU box).

Covers the schema documented on ``compute_parity_metrics``: REQUIRED
metrics (kp2d coords/conf, thetas excluding SMPL joint 0, root_R_cam
mean angular delta, anchored-frame fraction, all-finite) gate
``_overall_parity_pass``; ADVISORY metrics (root_t_cam, betas, root_R_cam
max angular delta) are reported only and never gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.bench_gvhmr_inference import (
    _PARITY_ANCHORED_FRACTION_ABS_DELTA,
    _PARITY_KP2D_COORD_MAX_DELTA_PX,
    _PARITY_ROOT_R_MEAN_ANGULAR_DELTA_DEG,
    _PARITY_THETA_EXCL_ROOT_MAX_ABS_RAD,
    _PARITY_THETA_EXCL_ROOT_MEAN_ABS_RAD,
    _overall_parity_pass,
    compute_parity_metrics,
)

_REQUIRED_METRICS = (
    "kp2d_coord_max_delta_px",
    "kp2d_conf_max_delta",
    "thetas_excl_joint0_mean_abs_rad",
    "thetas_excl_joint0_max_abs_rad",
    "root_R_cam_mean_angular_delta_deg",
    "anchored_fraction_abs_delta",
    "all_finite",
)
_ADVISORY_METRICS = (
    "root_t_cam_mean_abs_delta",
    "betas_max_abs_delta",
    "root_R_cam_max_angular_delta_deg",
)


def _rot_z(deg: float) -> np.ndarray:
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _make_output(n_frames: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    thetas = rng.normal(scale=0.05, size=(n_frames, 24, 3)).astype(np.float32)
    betas = rng.normal(scale=0.1, size=(n_frames, 10)).astype(np.float32)
    if n_frames:
        root_R_cam = np.stack([_rot_z(0.0) for _ in range(n_frames)]).astype(np.float32)
    else:
        root_R_cam = np.zeros((0, 3, 3), dtype=np.float32)
    root_t_cam = rng.normal(scale=1.0, size=(n_frames, 3)).astype(np.float32)
    kp2d = np.zeros((n_frames, 17, 3), dtype=np.float32)
    kp2d[..., 0] = rng.uniform(0, 640, size=(n_frames, 17))
    kp2d[..., 1] = rng.uniform(0, 480, size=(n_frames, 17))
    kp2d[..., 2] = 0.9  # confident everywhere, including ankles (15, 16)
    return {
        "thetas": thetas,
        "betas": betas,
        "root_R_cam": root_R_cam,
        "root_t_cam": root_t_cam,
        "kp2d": kp2d,
    }


def _required(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if v["tier"] == "required"}


def _advisory(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if v["tier"] == "advisory"}


class TestMetricSchema:
    def test_all_expected_metric_names_present(self):
        ref = _make_output()
        metrics = compute_parity_metrics(ref, ref)
        assert set(metrics) == set(_REQUIRED_METRICS) | set(_ADVISORY_METRICS)

    def test_advisory_metrics_have_no_threshold_or_pass(self):
        ref = _make_output()
        metrics = compute_parity_metrics(ref, ref)
        for name in _ADVISORY_METRICS:
            assert metrics[name]["tier"] == "advisory"
            assert metrics[name]["threshold"] is None
            assert metrics[name]["pass"] is None

    def test_required_metrics_have_thresholds_except_all_finite(self):
        ref = _make_output()
        metrics = compute_parity_metrics(ref, ref)
        for name in _REQUIRED_METRICS:
            assert metrics[name]["tier"] == "required"
            if name != "all_finite":
                assert metrics[name]["threshold"] is not None


class TestExactEqualReference:
    def test_all_required_metrics_pass(self):
        ref = _make_output()
        metrics = compute_parity_metrics(ref, ref)
        required = _required(metrics)
        assert all(entry["pass"] is True for entry in required.values()), required

    def test_overall_pass_true(self):
        ref = _make_output()
        metrics = compute_parity_metrics(ref, ref)
        assert _overall_parity_pass(metrics) is True

    def test_deltas_are_zero(self):
        ref = _make_output()
        metrics = compute_parity_metrics(ref, ref)
        assert metrics["kp2d_coord_max_delta_px"]["value"] == pytest.approx(0.0)
        assert metrics["kp2d_conf_max_delta"]["value"] == pytest.approx(0.0)
        assert metrics["thetas_excl_joint0_mean_abs_rad"]["value"] == pytest.approx(0.0)
        assert metrics["thetas_excl_joint0_max_abs_rad"]["value"] == pytest.approx(0.0)
        assert metrics["root_R_cam_mean_angular_delta_deg"]["value"] == pytest.approx(0.0)
        assert metrics["anchored_fraction_abs_delta"]["value"] == pytest.approx(0.0)


class TestKp2dDeltaGate:
    def test_1px_coord_delta_fails_kp2d_metric_only(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        candidate["kp2d"][0, 0, 0] += 1.0  # 1px x-coord delta on frame 0, joint 0

        metrics = compute_parity_metrics(candidate, ref)

        # float32 round-trip through kp2d's storage dtype introduces a
        # few ulps of slack at these coordinate magnitudes.
        assert metrics["kp2d_coord_max_delta_px"]["value"] == pytest.approx(1.0, abs=1e-3)
        assert metrics["kp2d_coord_max_delta_px"]["pass"] is False
        assert metrics["kp2d_coord_max_delta_px"]["threshold"] == _PARITY_KP2D_COORD_MAX_DELTA_PX

        # Isolation: nothing else should be perturbed.
        assert metrics["thetas_excl_joint0_max_abs_rad"]["pass"] is True
        assert metrics["root_R_cam_mean_angular_delta_deg"]["pass"] is True
        assert metrics["all_finite"]["pass"] is True

        assert _overall_parity_pass(metrics) is False

    def test_conf_delta_fails_conf_metric(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        candidate["kp2d"][0, 0, 2] += 0.5  # confidence delta, well above 0.02

        metrics = compute_parity_metrics(candidate, ref)

        assert metrics["kp2d_conf_max_delta"]["pass"] is False
        assert metrics["kp2d_coord_max_delta_px"]["pass"] is True  # coords untouched


class TestThetaExclJoint0Gate:
    def test_nonroot_joint_delta_fails_max_not_necessarily_mean(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        candidate["thetas"][0, 5, 0] += 0.1  # joint 5 (non-root), one frame

        metrics = compute_parity_metrics(candidate, ref)

        assert metrics["thetas_excl_joint0_max_abs_rad"]["value"] == pytest.approx(0.1)
        assert metrics["thetas_excl_joint0_max_abs_rad"]["pass"] is False
        assert (
            metrics["thetas_excl_joint0_max_abs_rad"]["threshold"]
            == _PARITY_THETA_EXCL_ROOT_MAX_ABS_RAD
        )
        # Averaged over every non-root element, a single 0.1 outlier stays
        # under the tighter mean threshold -- the mean metric is a
        # separate, distinct gate from the max metric.
        assert metrics["thetas_excl_joint0_mean_abs_rad"]["pass"] is True

        assert _overall_parity_pass(metrics) is False

    def test_joint0_only_delta_is_excluded_and_still_passes(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        candidate["thetas"][:, 0, :] += 5.0  # huge delta, but ONLY on SMPL root joint

        metrics = compute_parity_metrics(candidate, ref)

        assert metrics["thetas_excl_joint0_max_abs_rad"]["value"] == pytest.approx(0.0)
        assert metrics["thetas_excl_joint0_mean_abs_rad"]["value"] == pytest.approx(0.0)
        assert metrics["thetas_excl_joint0_max_abs_rad"]["pass"] is True
        assert metrics["thetas_excl_joint0_mean_abs_rad"]["pass"] is True

        assert _overall_parity_pass(metrics) is True

    def test_mean_threshold_breached_by_widespread_small_delta(self):
        ref = _make_output(n_frames=3)
        candidate = {k: v.copy() for k, v in ref.items()}
        # Every non-root element nudged by 2e-3 rad -- well under the max
        # threshold (1e-2) but above the mean threshold (1e-3).
        candidate["thetas"][:, 1:, :] += 2e-3

        metrics = compute_parity_metrics(candidate, ref)

        assert metrics["thetas_excl_joint0_mean_abs_rad"]["pass"] is False
        assert metrics["thetas_excl_joint0_mean_abs_rad"]["threshold"] == (
            _PARITY_THETA_EXCL_ROOT_MEAN_ABS_RAD
        )
        assert metrics["thetas_excl_joint0_max_abs_rad"]["pass"] is True


class TestRootRAngularDeltaGate:
    def test_1deg_rotation_delta_fails(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        candidate["root_R_cam"] = np.stack(
            [_rot_z(1.0).astype(np.float32) for _ in range(ref["root_R_cam"].shape[0])]
        )

        metrics = compute_parity_metrics(candidate, ref)

        assert metrics["root_R_cam_mean_angular_delta_deg"]["value"] == pytest.approx(
            1.0, abs=1e-4
        )
        assert metrics["root_R_cam_mean_angular_delta_deg"]["pass"] is False
        assert metrics["root_R_cam_mean_angular_delta_deg"]["threshold"] == (
            _PARITY_ROOT_R_MEAN_ANGULAR_DELTA_DEG
        )

    def test_small_rotation_delta_passes(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        candidate["root_R_cam"] = np.stack(
            [_rot_z(0.1).astype(np.float32) for _ in range(ref["root_R_cam"].shape[0])]
        )

        metrics = compute_parity_metrics(candidate, ref)
        assert metrics["root_R_cam_mean_angular_delta_deg"]["pass"] is True


class TestAnchoredFractionGate:
    def test_ankle_confidence_drop_changes_anchored_fraction(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        # Drop both ankle confidences below 0.3 on every frame -- flips
        # anchored_fraction from 1.0 to 0.0, an 0.01+ delta.
        candidate["kp2d"][:, 15, 2] = 0.0
        candidate["kp2d"][:, 16, 2] = 0.0

        metrics = compute_parity_metrics(candidate, ref)

        assert metrics["anchored_fraction_abs_delta"]["value"] == pytest.approx(1.0)
        assert metrics["anchored_fraction_abs_delta"]["pass"] is False
        assert metrics["anchored_fraction_abs_delta"]["threshold"] == (
            _PARITY_ANCHORED_FRACTION_ABS_DELTA
        )

    def test_within_tolerance_passes(self):
        ref = _make_output()
        metrics = compute_parity_metrics(ref, ref)
        assert metrics["anchored_fraction_abs_delta"]["pass"] is True


class TestFiniteGate:
    def test_nan_in_candidate_fails_finite_only(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        candidate["thetas"][0, 1, 0] = float("nan")

        metrics = compute_parity_metrics(candidate, ref)

        assert metrics["all_finite"]["value"] is False
        assert metrics["all_finite"]["pass"] is False
        assert _overall_parity_pass(metrics) is False

    def test_inf_in_reference_fails_finite(self):
        ref = _make_output()
        bad_ref = {k: v.copy() for k, v in ref.items()}
        bad_ref["root_t_cam"][0, 0] = float("inf")

        metrics = compute_parity_metrics(ref, bad_ref)

        assert metrics["all_finite"]["pass"] is False


class TestAdvisoryMetricsNeverGate:
    def test_large_advisory_deltas_do_not_fail_overall_gate(self):
        ref = _make_output()
        candidate = {k: v.copy() for k, v in ref.items()}
        candidate["root_t_cam"] += 1000.0  # advisory-only field, huge delta
        candidate["betas"] += 1000.0  # advisory-only field, huge delta

        metrics = compute_parity_metrics(candidate, ref)

        assert metrics["root_t_cam_mean_abs_delta"]["value"] == pytest.approx(1000.0)
        assert metrics["betas_max_abs_delta"]["value"] == pytest.approx(1000.0)
        assert metrics["root_t_cam_mean_abs_delta"]["pass"] is None
        assert metrics["betas_max_abs_delta"]["pass"] is None

        # Every required metric is still exactly equal -> overall gate holds.
        assert _overall_parity_pass(metrics) is True


class TestOverallParityPassHelper:
    def test_ignores_advisory_entries_entirely(self):
        metrics = {
            "req_ok": {"value": 0.0, "threshold": 1.0, "pass": True, "tier": "required"},
            "adv_bad": {"value": 99.0, "threshold": None, "pass": None, "tier": "advisory"},
        }
        assert _overall_parity_pass(metrics) is True

    def test_any_required_failure_fails_overall(self):
        metrics = {
            "req_ok": {"value": 0.0, "threshold": 1.0, "pass": True, "tier": "required"},
            "req_bad": {"value": 2.0, "threshold": 1.0, "pass": False, "tier": "required"},
        }
        assert _overall_parity_pass(metrics) is False


class TestZeroFrames:
    def test_empty_arrays_do_not_crash_and_pass(self):
        ref = _make_output(n_frames=0)
        metrics = compute_parity_metrics(ref, ref)
        assert _overall_parity_pass(metrics) is True
        assert metrics["kp2d_coord_max_delta_px"]["value"] == 0.0
        assert metrics["all_finite"]["value"] is True

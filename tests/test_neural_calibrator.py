import numpy as np
import pytest

from src.utils.neural_calibrator import (
    _paths_shadowing_pnlcalib,
    convert_pnlcalib_to_ours,
)


class TestPathsShadowingPnlcalib:
    """The shim must prune EVERY src/ entry providing a regular ``utils`` or
    ``model`` package — not just this checkout's — or a run from a git
    worktree leaves the editable install's ``<main>/src`` on sys.path, its
    regular ``utils`` package shadows PnLCalib's namespace ``utils``, and
    every calibrate() silently fails (the origi01 baseline artifact)."""

    def _make_src(self, root, with_utils_init=False, with_model_init=False):
        src = root / "src"
        (src / "utils").mkdir(parents=True)
        (src / "model").mkdir(parents=True)
        if with_utils_init:
            (src / "utils" / "__init__.py").touch()
        if with_model_init:
            (src / "model" / "__init__.py").touch()
        return str(src)

    def test_prunes_any_src_with_regular_utils_package(self, tmp_path):
        main = self._make_src(tmp_path / "main", with_utils_init=True)
        worktree = self._make_src(tmp_path / "wt", with_utils_init=True)
        assert _paths_shadowing_pnlcalib([main, worktree, "/unrelated"]) == [
            main, worktree,
        ]

    def test_prunes_src_with_regular_model_package(self, tmp_path):
        src = self._make_src(tmp_path, with_model_init=True)
        assert _paths_shadowing_pnlcalib([src]) == [src]

    def test_keeps_src_with_namespace_only_packages(self, tmp_path):
        src = self._make_src(tmp_path)  # utils/model exist but no __init__.py
        assert _paths_shadowing_pnlcalib([src]) == []

    def test_never_touches_non_src_entries(self, tmp_path):
        # A site-packages-like dir with a regular utils package must be kept:
        # removing it mid-import would break PnLCalib's own dependencies.
        site = tmp_path / "site-packages"
        (site / "utils").mkdir(parents=True)
        (site / "utils" / "__init__.py").touch()
        assert _paths_shadowing_pnlcalib([str(site), ""]) == []


def test_convert_identity_rotation_centres_to_corner_origin():
    R = np.eye(3)
    C_pnl = np.array([0.0, 0.0, 0.0])
    rvec, tvec, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    assert C_ours == pytest.approx([52.5, 34.0, 0.0])


def test_convert_flips_y_and_z():
    R = np.eye(3)
    C_pnl = np.array([10.0, 5.0, -15.0])
    _, _, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    assert C_ours == pytest.approx([62.5, 29.0, 15.0])


def test_convert_preserves_projection_consistency():
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    C_pnl = np.array([3.0, -2.0, -20.0])
    rvec, tvec, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    import cv2
    R_ours, _ = cv2.Rodrigues(rvec)
    assert tvec == pytest.approx((-R_ours @ C_ours))

"""Tests for the corridor-bounded / speed-clamped moving-camera fix to
``refine_camera_from_lines`` (src/utils/line_camera_refine.py).

Real-clip defect (gberch-2, found by the coordinator's follow-up review
of the 2026-09-09 moving-camera-support feature): the per-frame line
solve's LM bounded the raw OpenCV ``tvec`` to ±300m per component — a
bound on ``t``, not on the camera CENTRE (``-R^T @ t``). For a frame
with only 1-2 detected lines (under-determined: 4 residuals for a 7-DOF
problem), the centre could wander tens to hundreds of metres from the
anchor-interpolated corridor while still reporting a low line RMS (and
therefore high confidence) — 10/181 gberch-2 frames did this, up to
212.6m off, at confidence 0.95-1.00. See docs/superpowers/specs/
2026-09-09-moving-camera-support.md's corridor-drift addendum.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.utils.line_camera_refine as line_camera_refine
from src.utils.anchor_solver import _line_residuals
from src.utils.line_camera_refine import refine_camera_from_lines
from src.utils.line_detector import DetectedLine
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE

CX, CY = 960.0, 540.0
TARGET = np.array([52.5, 25.0, 0.0])
BLANK_FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


def _K(fx: float) -> np.ndarray:
    return np.array([[fx, 0.0, CX], [0.0, fx, CY], [0.0, 0.0, 1.0]])


def _look_at_R(C: np.ndarray, target: np.ndarray = TARGET) -> np.ndarray:
    look = target - C
    look = look / np.linalg.norm(look)
    right = np.cross(look, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(look, right)
    return np.array([right, down, look], dtype=float)


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, world) -> tuple[float, float]:
    cam = R @ np.asarray(world, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _camera_at(C: np.ndarray, fx: float = 1500.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = _look_at_R(C)
    t = -R @ C
    return _K(fx), R, t


def _lines_for(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, names: tuple[str, ...],
) -> list[DetectedLine]:
    out = []
    for name in names:
        seg = LINE_CATALOGUE[name]
        a = _project(K, R, t, seg[0])
        b = _project(K, R, t, seg[1])
        out.append(DetectedLine(
            name=name, image_segment=(a, b), world_segment=seg,
            confidence=1.0, n_samples=60,
        ))
    return out


def _two_lines_for(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> list[DetectedLine]:
    """An under-determined (2-line) observation set consistent with
    (K, R, t) — matches the real failure mode: a mid-span frame where
    the detector only finds 1-2 usable painted lines. 4 residuals for a
    7-DOF (rvec+dC+fx) problem: many (rvec, dC, fx) combinations fit
    exactly, so a solve against this alone can land ANYWHERE the box
    bound allows, not necessarily at the "true" generating camera —
    used only to test that the corridor/speed BOUNDS hold, never to
    test convergence accuracy."""
    return _lines_for(K, R, t, ("near_touchline", "halfway_line"))


def _several_lines_for(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> list[DetectedLine]:
    """A well-determined (4-line) observation set — enough independent
    constraints (8 residuals for 7 unknowns), and all in front of a
    camera near the corridor centre used by these tests (verified: the
    near-side box lines project behind such a camera, which is why
    those aren't used here), that the solve reliably recovers the one
    true generating camera — used for tests that assert convergence
    accuracy rather than just bound compliance."""
    return _lines_for(
        K, R, t,
        ("far_touchline", "halfway_line", "left_18yd_front", "left_goal_left_post"),
    )


def _patch_detector_to(
    monkeypatch, K: np.ndarray, R: np.ndarray, t: np.ndarray, *, several: bool = False,
) -> None:
    """Replace pixel-level detection with a fixed, fully-consistent
    observation set (2 lines by default — under-determined, the real
    failure mode; ``several=True`` for a well-determined 4-line set) —
    isolates the LM/bounds logic from real pixel detection, which this
    test suite doesn't need."""
    lines_fn = _several_lines_for if several else _two_lines_for

    def _fake_detect(frame_bgr, K_boot, R_boot, t_boot, distortion, world_lines, cfg=None):
        return lines_fn(K, R, t)
    monkeypatch.setattr(line_camera_refine, "detect_painted_lines_in_frame", _fake_detect)


# ── Corridor bound ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_corridor_bound_is_respected_even_when_true_camera_is_far_outside_it(monkeypatch):
    """The core fix: an under-determined (2-line) frame whose best-fit
    camera is genuinely far from the anchor-interpolated corridor must
    still come back within max_corridor_deviation_m of it — the bound
    is enforced by the optimiser's own box constraints (a scipy
    guarantee), not a post-hoc best-effort check."""
    corridor_centre = np.array([45.0, 15.0, 15.0])
    C_true = np.array([52.5, 30.0, 15.0])
    assert np.linalg.norm(C_true - corridor_centre) > 10.0  # genuinely far

    K_true, R_true, t_true = _camera_at(C_true)
    _patch_detector_to(monkeypatch, K_true, R_true, t_true)

    K_seed, R_seed, t_seed = _camera_at(corridor_centre)
    result = refine_camera_from_lines(
        BLANK_FRAME, K_seed, R_seed, t_seed, (0.0, 0.0),
        corridor_centre=corridor_centre, max_corridor_deviation_m=3.0,
    )
    assert result.n_detections == 2
    C_result = -result.R.T @ result.t
    dev = float(np.linalg.norm(C_result - corridor_centre))
    assert dev <= 3.0 + 1e-6, f"corridor deviation {dev:.2f}m exceeds the 3.0m bound"
    assert result.corridor_deviation_m <= 3.0 + 1e-6


@pytest.mark.unit
def test_corridor_bound_does_not_disturb_a_well_supported_frame(monkeypatch):
    """When the true camera IS inside the corridor bound, the fit should
    still converge close to the truth (the bound isn't over-tight for
    the common, well-behaved case)."""
    corridor_centre = np.array([45.0, 15.0, 15.0])
    C_true = corridor_centre + np.array([1.0, -0.5, 0.2])  # well within 5m
    K_true, R_true, t_true = _camera_at(C_true)
    _patch_detector_to(monkeypatch, K_true, R_true, t_true, several=True)

    K_seed, R_seed, t_seed = _camera_at(corridor_centre)
    result = refine_camera_from_lines(
        BLANK_FRAME, K_seed, R_seed, t_seed, (0.0, 0.0),
        corridor_centre=corridor_centre, max_corridor_deviation_m=5.0,
    )
    C_result = -result.R.T @ result.t
    err = float(np.linalg.norm(C_result - C_true))
    assert err < 0.5, f"recovered centre {C_result} too far from truth {C_true} (err {err:.2f}m)"
    assert result.line_rms_px < 1.0


# ── Frame-to-frame speed clamp ───────────────────────────────────────────


@pytest.mark.unit
def test_speed_clamp_limits_frame_to_frame_step(monkeypatch):
    corridor_centre = np.array([45.0, 15.0, 15.0])
    C_true = corridor_centre + np.array([2.0, 0.0, 0.0])  # inside corridor bound
    K_true, R_true, t_true = _camera_at(C_true)
    _patch_detector_to(monkeypatch, K_true, R_true, t_true)

    K_seed, R_seed, t_seed = _camera_at(corridor_centre)
    prev_centre = corridor_centre - np.array([0.05, 0.0, 0.0])
    result = refine_camera_from_lines(
        BLANK_FRAME, K_seed, R_seed, t_seed, (0.0, 0.0),
        corridor_centre=corridor_centre, max_corridor_deviation_m=5.0,
        prev_centre=prev_centre, max_step_m=0.1,
    )
    C_result = -result.R.T @ result.t
    step = float(np.linalg.norm(C_result - prev_centre))
    assert step <= 0.1 + 1e-6, f"frame-to-frame step {step:.3f}m exceeds the 0.1m budget"


@pytest.mark.unit
def test_speed_clamp_is_a_noop_when_within_budget(monkeypatch):
    corridor_centre = np.array([45.0, 15.0, 15.0])
    K_true, R_true, t_true = _camera_at(corridor_centre)
    _patch_detector_to(monkeypatch, K_true, R_true, t_true, several=True)

    prev_centre = corridor_centre - np.array([0.01, 0.0, 0.0])
    result = refine_camera_from_lines(
        BLANK_FRAME, K_true, R_true, t_true, (0.0, 0.0),
        corridor_centre=corridor_centre, max_corridor_deviation_m=5.0,
        prev_centre=prev_centre, max_step_m=6.0,  # generous — shouldn't engage
    )
    C_result = -result.R.T @ result.t
    err = float(np.linalg.norm(C_result - corridor_centre))
    assert err < 0.2, "speed clamp should not perturb an already-close fit"


# ── Honest confidence (RMS must reflect the RETURNED pose) ─────────────


@pytest.mark.unit
def test_line_rms_reflects_the_actually_returned_pose_after_speed_clamp(monkeypatch):
    """The reported line_rms_px (which the camera stage turns directly
    into confidence) must be recomputed at whatever pose is actually
    returned — including after a post-hoc speed clamp — never the
    pre-clamp fit's (lower) RMS."""
    corridor_centre = np.array([45.0, 15.0, 15.0])
    C_true = corridor_centre + np.array([2.0, 0.0, 0.0])
    K_true, R_true, t_true = _camera_at(C_true)
    _patch_detector_to(monkeypatch, K_true, R_true, t_true)

    K_seed, R_seed, t_seed = _camera_at(corridor_centre)
    prev_centre = corridor_centre - np.array([0.05, 0.0, 0.0])
    result = refine_camera_from_lines(
        BLANK_FRAME, K_seed, R_seed, t_seed, (0.0, 0.0),
        corridor_centre=corridor_centre, max_corridor_deviation_m=5.0,
        prev_centre=prev_centre, max_step_m=0.1,
    )
    recomputed = float(np.sqrt(
        (_line_residuals(result.detected_lines, result.K, result.R, result.t) ** 2).mean()
    ))
    assert abs(recomputed - result.line_rms_px) < 1e-6, (
        f"reported line_rms_px {result.line_rms_px:.3f} doesn't match the "
        f"returned pose's own residual {recomputed:.3f} — confidence would "
        f"be scored against the wrong (pre-clamp) fit"
    )
    # And the clamp actually moved it away from the (better-fitting) true
    # camera, so the honest RMS should be clearly non-trivial.
    assert result.line_rms_px > 0.5


# ── Back-compat: no corridor supplied ───────────────────────────────────


@pytest.mark.unit
def test_without_corridor_behaves_like_the_legacy_free_solve(monkeypatch):
    """Default params (no corridor/speed args) must still solve normally
    — this is the anchor-frame / explicit-false-legacy code path, which
    must be unaffected by this fix."""
    C_true = np.array([52.5, -30.0, 30.0])
    K_true, R_true, t_true = _camera_at(C_true)
    _patch_detector_to(monkeypatch, K_true, R_true, t_true)

    result = refine_camera_from_lines(
        BLANK_FRAME, K_true, R_true, t_true, (0.0, 0.0),
    )
    assert result.n_detections == 2
    assert result.line_rms_px < 1.0
    assert result.corridor_deviation_m == 0.0

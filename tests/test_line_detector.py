"""Unit tests for the painted-line detector's projection guards."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.line_detector import _project_endpoints, _project_visible_segment


@pytest.mark.unit
def test_project_endpoints_accepts_a_normal_in_view_segment():
    """A short segment with both endpoints comfortably in front of the
    camera projects to sane in-frame coordinates."""
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.zeros(3)
    proj = _project_endpoints(
        K, R, t, (0.0, 0.0), (-5.0, 0.0, 50.0), (5.0, 0.0, 50.0)
    )
    assert proj is not None
    pa, pb = proj
    # ±5 m at depth 50 m → 960 ± 100 px, well within a 1920-wide frame.
    assert abs(pa[0] - 860.0) < 1e-6
    assert abs(pb[0] - 1060.0) < 1e-6


@pytest.mark.unit
def test_project_endpoints_rejects_grazing_incidence_endpoint():
    """A world line whose far endpoint sits at the camera horizon — just in
    front of the image plane (cam_z > the behind-camera threshold, so that
    guard passes) but far off-axis — perspective-divides to millions of
    pixels. Such a segment cannot be meaningfully strip-searched, so
    ``_project_endpoints`` must reject it rather than hand back a
    numerically degenerate line.

    This is the gberch ``near_touchline`` bug: ((0,0,0),(105,0,0)) with the
    far endpoint grazing the horizon for a tight penalty-box shot.
    """
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.zeros(3)
    near = (0.0, 0.0, 50.0)        # cam_z = 50 → projects to image centre
    grazing = (8000.0, 0.0, 4.0)   # cam_z = 4 (> behind-cam threshold), x huge
    # Sanity: the grazing endpoint really does project absurdly far out.
    assert 1000.0 * 8000.0 / 4.0 > 1e6
    assert _project_endpoints(K, R, t, (0.0, 0.0), near, grazing) is None


# ── _project_visible_segment — returns the in-frame run of a world line ─────


@pytest.mark.unit
def test_visible_segment_short_in_view_returns_near_identity():
    """A short world segment fully in view → the in-frame run spans both
    endpoints, so the returned segment is essentially the projected line."""
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.zeros(3)
    seg = _project_visible_segment(
        K, R, t, (0.0, 0.0), (-5.0, 0.0, 50.0), (5.0, 0.0, 50.0), 1920, 1080
    )
    assert seg is not None
    pa, pb = seg
    # ±5 m at depth 50 → 960 ± 100 px.
    assert abs(pa[0] - 860.0) < 1.5
    assert abs(pb[0] - 1060.0) < 1.5


@pytest.mark.unit
def test_visible_segment_recovers_middle_when_endpoints_are_off_frame():
    """The near_touchline failure mode: both world endpoints are outside
    the camera's view — one off-screen-left, one past the camera horizon
    — but a middle sub-segment is plainly in frame. ``_project_visible_segment``
    must return the in-frame run, not give up like a 2-endpoint
    projection would."""
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.zeros(3)
    # Line along world x at depth 50 — but extended out to where the far
    # endpoint sits at grazing depth and the near endpoint sits off-frame.
    near_endpoint    = (-200.0, 0.0,  50.0)   # cam_z=50, projects WAY off-frame-left
    far_endpoint     = (8000.0, 0.0,   4.0)   # grazing horizon, projects to millions
    seg = _project_visible_segment(
        K, R, t, (0.0, 0.0), near_endpoint, far_endpoint, 1920, 1080
    )
    assert seg is not None
    pa, pb = seg
    # The visible portion is whatever crosses the image rect; both
    # endpoints of the returned run must be finite and near the frame.
    for p in (pa, pb):
        assert np.all(np.isfinite(p))
        assert -50 < p[0] < 1920 + 50
        assert -50 < p[1] < 1080 + 50


@pytest.mark.unit
def test_visible_segment_returns_none_for_fully_out_of_view_line():
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.zeros(3)
    # Entire segment behind the camera (negative z).
    seg = _project_visible_segment(
        K, R, t, (0.0, 0.0), (0.0, 0.0, -10.0), (5.0, 0.0, -20.0), 1920, 1080
    )
    assert seg is None

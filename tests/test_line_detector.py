"""Unit tests for the painted-line detector's projection guards."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.line_detector import _project_endpoints


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

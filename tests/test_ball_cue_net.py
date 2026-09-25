"""Tests for ball_cue_net on a synthetic pinhole camera looking at the
near goal: net-region projection (in view / out of view), energy-spike
detection for a synthetic blob placed inside the projected net region,
and player-box exclusion suppressing that same energy."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.ball_cue_net import net_energy_onsets, net_energy_series, net_region_polygon
from src.utils.goal_geometry import GoalGeometry

GEOM = GoalGeometry.from_pitch_config({
    "length_m": 105.0, "width_m": 68.0, "goal_height_m": 2.44,
    "goal_width_m": 7.32, "goal_depth_m": 1.5,
})


def _look_at_rt(C, target, up=(0.0, 0.0, 1.0)):
    """Minimal OpenCV-convention look-at camera: world_cam = R @ world + t,
    R's rows are (right, down, forward) so z_cam > 0 is in front."""
    C = np.asarray(C, float)
    target = np.asarray(target, float)
    forward = target - C
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, float)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    t = -R @ C
    return R, t


def _make_camera(C, target, w=1920, h=1080, f=1200.0):
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    R, t = _look_at_rt(C, target)
    return K, R, t


def test_net_region_polygon_in_view_near_goal():
    C = (-15.0, 34.0, 3.0)
    target = (0.0, 34.0, 1.2)
    K, R, t = _make_camera(C, target)
    out = net_region_polygon(K, R, t, (0.0, 0.0), GEOM, (1920, 1080))
    assert out is not None
    hull, side = out
    assert side == "near"
    assert hull.shape[1] == 2
    cx = hull[:, 0].mean()
    assert 1920 * 0.15 < cx < 1920 * 0.85


def test_net_region_polygon_none_when_goal_far_out_of_view():
    # Camera at pitch centre looking along +y with a narrow lens -- both
    # goals (at x=0 and x=105, far off to either side) are out of frame.
    C = (52.5, -20.0, 8.0)
    target = (52.5, 34.0, 0.0)
    K, R, t = _make_camera(C, target, f=4000.0)
    out = net_region_polygon(K, R, t, (0.0, 0.0), GEOM, (1920, 1080))
    assert out is None


def test_net_energy_series_flags_blob_inside_net():
    C = (-15.0, 34.0, 3.0)
    target = (0.0, 34.0, 1.2)
    K, R, t = _make_camera(C, target)
    cam = (K, R, t, (0.0, 0.0))
    w, h = 1920, 1080
    poly = net_region_polygon(K, R, t, (0.0, 0.0), GEOM, (w, h))
    assert poly is not None
    hull, _side = poly
    cx, cy = int(hull[:, 0].mean()), int(hull[:, 1].mean())

    base = np.full((h, w, 3), 40, dtype=np.uint8)
    cur_spike = base.copy()
    cv2.rectangle(cur_spike, (cx - 10, cy - 10), (cx + 10, cy + 10), (220, 220, 220), -1)
    # A handful of quiet frames establish the adaptive baseline, then one
    # spike frame -- net_energy_onsets needs >= 5 series points to score.
    n_quiet = 6
    spike_idx = n_quiet

    def pairs():
        prev = base
        for i in range(1, n_quiet):
            yield i, prev, base
            prev = base
        yield spike_idx, prev, cur_spike

    cameras = {i: cam for i in range(n_quiet + 1)}
    series = net_energy_series(pairs(), cameras, GEOM, (w, h))
    energies = {s.frame: s.energy for s in series}
    assert energies[spike_idx] > max(
        e for f, e in energies.items() if f != spike_idx) * 5

    onsets = net_energy_onsets(series, cameras, GEOM, k_mad=1.0, min_gap_frames=1)
    assert any(o.frame == spike_idx for o in onsets)
    hit = next(o for o in onsets if o.frame == spike_idx)
    assert hit.kind == "goal_impact"
    assert hit.cue == "net_energy"


def test_net_energy_series_player_box_suppresses_energy():
    C = (-15.0, 34.0, 3.0)
    target = (0.0, 34.0, 1.2)
    K, R, t = _make_camera(C, target)
    cam = (K, R, t, (0.0, 0.0))
    w, h = 1920, 1080
    poly = net_region_polygon(K, R, t, (0.0, 0.0), GEOM, (w, h))
    assert poly is not None
    hull, _side = poly
    cx, cy = int(hull[:, 0].mean()), int(hull[:, 1].mean())

    prev = np.full((h, w, 3), 40, dtype=np.uint8)
    cur = prev.copy()
    box = (cx - 15, cy - 15, cx + 15, cy + 15)
    cv2.rectangle(cur, box[:2], box[2:], (220, 220, 220), -1)

    def pairs():
        yield 1, prev, cur

    cameras = {1: cam}
    boxes = {1: [tuple(float(v) for v in box)]}
    series = net_energy_series(pairs(), cameras, GEOM, (w, h), player_boxes=boxes)
    assert len(series) == 1
    assert series[0].energy < 1.0


def test_net_energy_onsets_empty_series_returns_empty():
    assert net_energy_onsets([], {}, GEOM) == []

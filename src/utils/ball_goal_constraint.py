"""Goal-mouth constraint for the decisive shot (design D6, gap G16).

When the event list contains a goal -- an operator ``goal_impact`` anchor --
the flight span ending in it must cross the goal line inside the mouth
(``|y - 34| < 3.66``, ``z < 2.44``). Two pieces live here, both pure and
camera-geometry only (no IO, no solver imports):

``infer_line_cross_knots``
    Turns an *operator airborne anchor near the goal* into an in-memory
    **line-cross knot** = (anchor pixel ray) intersect (goal-line plane).
    Two cases:

    * the goal_impact itself sits ON the goal line (``post``/``crossbar``/
      explicit ``mouth``): it already is the line-cross, nothing to infer;
    * the goal_impact is a net contact (``back_net``/``side_net``): the
      latest operator ``airborne_*`` anchor within ``window_frames`` before
      it whose ray hits the goal-line plane inside the mouth (+ margin) is
      promoted to a depth-hard knot. Two anchored knots + gravity fully
      determine monocular depth (the well-posedness the gberch finish
      lacked: the frame-394 anchor is only ``airborne_low``, which pins
      depth at z=1 m and sends the ball wide of the far post).

    The anchor file is never touched -- the knot lives in memory and its
    provenance is reported as ``knot_source``.

``goal_check``
    Reads a solved dense track and reports where it crossed the goal line:
    ``{goal_frame, goal_end_x, line_cross:{frame,xyz}|None, status,
    knot_source}`` with status ``ok | misses_mouth | over_crossbar |
    no_line_cross``. The quality report fails loudly on anything but ok.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils.ball_hybrid_physics import BALL_RADIUS_M
from src.utils.ball_hybrid_types import HybridShotCtx, Knot
from src.utils.goal_geometry import GoalGeometry, resolve_goal_impact_world

DEFAULT_CFG: dict[str, Any] = {
    "enabled": True,
    # how far back (frames) from the goal_impact an operator airborne
    # anchor may sit and still be promoted to the line-cross knot
    "window_frames": 24,
    # the ray must hit the line inside the mouth widened by this much (m)
    # laterally / vertically, so a click on the post or bar still counts
    "mouth_margin_m": 0.6,
    "bar_margin_m": 0.4,
}

_AIRBORNE = frozenset({"airborne_low", "airborne_mid", "airborne_high"})
_ON_LINE_ELEMENTS = frozenset({"post", "crossbar", "mouth"})
_GEOMETRY = GoalGeometry.from_pitch_config({})

STATUS_OK = "ok"
STATUS_MISSES_MOUTH = "misses_mouth"
STATUS_OVER_CROSSBAR = "over_crossbar"
STATUS_NO_LINE_CROSS = "no_line_cross"


def constraint_cfg(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(DEFAULT_CFG)
    if cfg:
        out.update(cfg)
    return out


@dataclass(frozen=True)
class GoalEvent:
    """The goal the shot ended in: impact frame, which goal line, and the
    provenance of the line-cross knot (when one was inferred/observed)."""

    frame: int
    goal_end_x: float
    element: str | None
    knot_source: str | None = None


def _attrs(a: Any) -> tuple[int, tuple[float, float] | None, str, str | None]:
    if isinstance(a, Mapping):
        raw = a.get("image_xy")
        xy = (float(raw[0]), float(raw[1])) if raw is not None else None
        return int(a["frame"]), xy, str(a["state"]), a.get("goal_element")
    return int(a.frame), a.image_xy, str(a.state), getattr(a, "goal_element", None)


def _goal_end_for_x(x: float) -> float:
    mid = 0.5 * (_GEOMETRY.goal_line_x_near + _GEOMETRY.goal_line_x_far)
    return _GEOMETRY.goal_line_x_near if x < mid else _GEOMETRY.goal_line_x_far


def _resolve_impact_world(ctx: HybridShotCtx, frame: int, xy, element: str):
    try:
        return resolve_goal_impact_world(
            xy, element, K=ctx.per_frame_K[frame], R=ctx.per_frame_R[frame],
            t=ctx.per_frame_t[frame], distortion=ctx.distortion,
            geometry=_GEOMETRY)
    except ValueError:
        return None


def ray_goal_line_hit(
    ctx: HybridShotCtx, frame: int, uv, goal_x: float,
    *, mouth_margin_m: float, bar_margin_m: float,
) -> np.ndarray | None:
    """Pixel ray ∩ goal-line plane ``x = goal_x``, or ``None`` when the
    ray misses the (margin-widened) mouth or points away from the plane."""
    C, d = ctx.ray(frame, uv)
    if abs(float(d[0])) < 1e-9:
        return None
    s = (goal_x - float(C[0])) / float(d[0])
    if s <= 0.0:
        return None
    p = C + s * d
    y_lo = _GEOMETRY.post_y_left - mouth_margin_m
    y_hi = _GEOMETRY.post_y_right + mouth_margin_m
    if not (y_lo <= p[1] <= y_hi):
        return None
    if not (0.0 <= p[2] <= _GEOMETRY.crossbar_z + bar_margin_m):
        return None
    return p


def find_goal_event(
    ctx: HybridShotCtx, anchors: Sequence[Any],
) -> GoalEvent | None:
    """The shot's goal: the latest operator ``goal_impact`` anchor that
    resolves to a goal line, or ``None``."""
    best: GoalEvent | None = None
    for a in sorted(anchors, key=lambda x: _attrs(x)[0]):
        frame, xy, state, element = _attrs(a)
        if state != "goal_impact" or xy is None or not element:
            continue
        if not ctx.has_frame(frame):
            continue
        world = _resolve_impact_world(ctx, frame, xy, element)
        if world is None:
            continue
        source = "operator_mouth" if element == "mouth" else (
            "goal_impact_on_line" if element in _ON_LINE_ELEMENTS else None)
        best = GoalEvent(frame=frame, goal_end_x=_goal_end_for_x(float(world[0])),
                         element=element, knot_source=source)
    return best


SNAP_REPORT_M = 0.01


def snap_into_mouth(y: float, z: float) -> tuple[float, float]:
    """Clamp a goal-line point so the whole ball is inside the mouth
    (between the posts, under the bar, above the turf)."""
    g = _GEOMETRY
    y = min(max(y, g.post_y_left + BALL_RADIUS_M), g.post_y_right - BALL_RADIUS_M)
    z = min(max(z, BALL_RADIUS_M), g.crossbar_z - BALL_RADIUS_M)
    return y, z


def infer_line_cross_knots(
    ctx: HybridShotCtx,
    anchors: Sequence[Any],
    cfg: Mapping[str, Any] | None = None,
) -> tuple[list[Knot], GoalEvent | None]:
    """In-memory line-cross knot(s) inferred from operator ``anchors``
    (never mutated). Returns ``(extra_knots, goal_event)``; the event's
    ``knot_source`` is ``operator_airborne_ray`` when a knot was inferred,
    the on-line source when none was needed, else ``None``."""
    c = constraint_cfg(cfg)
    event = find_goal_event(ctx, anchors)
    if event is None or not c["enabled"]:
        return [], event
    if event.knot_source is not None:  # impact already on the line
        return [], event

    candidates = []
    for a in anchors:
        frame, xy, state, _el = _attrs(a)
        if state not in _AIRBORNE or xy is None or not ctx.has_frame(frame):
            continue
        if not (event.frame - int(c["window_frames"]) <= frame < event.frame):
            continue
        candidates.append((frame, xy))
    for frame, xy in sorted(candidates, reverse=True):  # latest first
        p = ray_goal_line_hit(ctx, frame, xy, event.goal_end_x,
                              mouth_margin_m=float(c["mouth_margin_m"]),
                              bar_margin_m=float(c["bar_margin_m"]))
        if p is None:
            continue
        # A scored ball crossed inside the mouth. A ray landing within the
        # margin OUTSIDE it is camera-calibration error near the goal
        # (kroupi01: 0.5 m wide of a near post the footage shows it inside),
        # so snap the knot just inside the frame and say so in knot_source.
        y, z = snap_into_mouth(float(p[1]), float(p[2]))
        moved = abs(y - float(p[1])) + abs(z - float(p[2]))
        knot = Knot(frame=frame, xyz=(float(p[0]), y, z),
                    kind="line_cross", depth_hard=True, source="auto", uv=xy)
        source = "operator_airborne_ray_snapped" if moved > SNAP_REPORT_M else "operator_airborne_ray"
        return [knot], GoalEvent(event.frame, event.goal_end_x, event.element, source)
    return [], event


def contain_in_net(
    frames: Mapping[int, Mapping[str, Any]],
    event: GoalEvent | None,
    check: Mapping[str, Any] | None,
    *,
    margin_m: float = 0.05,
    protect_frames: Sequence[int] = (),
    max_window_frames: float = 15.0,
) -> tuple[dict[int, dict[str, Any]], int]:
    """Keep the ball inside the net volume between the line-cross and the
    goal_impact frame.

    The span from the line-cross to the net contact is short and weakly
    observed (the ball decelerates into the netting), so its fit can leave
    the goal box entirely (a roll fallback drifting 2 m behind the back
    netting). For a goal that DID cross inside the mouth, clamp those
    frames to the goal box (``x`` within net depth, ``y`` between the posts,
    ``z`` under the crossbar) and mark them ``flight``. Returns
    ``(new_frames, n_clamped)``; frames outside the window and the
    line-cross / impact frames themselves are untouched.
    """
    out = {f: dict(v) for f, v in frames.items()}
    if event is None or check is None or check.get("status") != STATUS_OK:
        return out, 0
    lc = check.get("line_cross")
    if not lc:
        return out, 0
    g = _GEOMETRY
    near = event.goal_end_x <= 0.5 * (g.goal_line_x_near + g.goal_line_x_far)
    x_in, x_back = ((0.0, -g.net_depth) if near
                    else (g.goal_line_x_far, g.goal_line_x_far + g.net_depth))
    x_lo, x_hi = sorted((x_in, x_back))
    fc, fi = float(lc["frame"]), float(event.frame)
    protect = {int(f) for f in protect_frames}
    if fi - fc > max_window_frames:  # not a net-entry window; leave the fit alone
        return out, 0
    p_cross = np.asarray(lc["xyz"], dtype=float)
    if event.frame not in out:
        return out, 0
    p_imp = np.asarray(out[event.frame]["xyz"], dtype=float)
    n = 0
    for f in sorted(out):
        if not (fc < f < fi) or f in protect:
            continue
        # ease-out (quadratic) from the line-cross point to the net contact:
        # the netting stops the ball, so speed falls to ~0 at the impact
        a = (f - fc) / (fi - fc)
        e = 1.0 - (1.0 - a) ** 2
        x, y, z = (float(v) for v in (p_cross + e * (p_imp - p_cross)))
        cx = min(max(x, x_lo + margin_m), x_hi - margin_m)
        cy = min(max(y, g.post_y_left + margin_m), g.post_y_right - margin_m)
        cz = min(max(z, BALL_RADIUS_M), g.crossbar_z - margin_m)
        if (cx, cy, cz) != (x, y, z):
            n += 1
        out[f]["xyz"] = (cx, cy, cz)
        out[f]["state"] = "flight"
    return out, n


def goal_check(
    frames: Mapping[int, Mapping[str, Any]],
    event: GoalEvent | None,
) -> dict[str, Any] | None:
    """Where the solved dense track crossed the goal line. ``None`` when
    the shot has no goal event; otherwise the diag ``goal_check`` block."""
    if event is None:
        return None
    gx = event.goal_end_x
    near = gx <= 0.5 * (_GEOMETRY.goal_line_x_near + _GEOMETRY.goal_line_x_far)
    out: dict[str, Any] = {
        "goal_frame": int(event.frame), "goal_end_x": float(gx),
        "line_cross": None, "status": STATUS_NO_LINE_CROSS,
        "knot_source": event.knot_source,
    }
    fs = sorted(f for f in frames if f <= event.frame + 1)
    # signed distance beyond the line (positive = past it)
    def beyond(f: int) -> float:
        x = float(frames[f]["xyz"][0])
        return (gx - x) if near else (x - gx)

    cross = None
    for f0, f1 in zip(fs, fs[1:]):
        if f1 != f0 + 1:
            continue
        b0, b1 = beyond(f0), beyond(f1)
        if b0 < 0.0 <= b1:
            a = (0.0 - b0) / (b1 - b0) if b1 != b0 else 0.0
            p0 = np.asarray(frames[f0]["xyz"], dtype=float)
            p1 = np.asarray(frames[f1]["xyz"], dtype=float)
            xyz = p0 + a * (p1 - p0)
            # the FIRST inward crossing is the shot's line-cross; later
            # inward "crossings" are post-line fit wobble in the net
            cross = (f0 + a, xyz)
            break
    if cross is None:
        # a knot sitting exactly on the line (beyond == 0 at the first frame)
        for f in fs:
            if abs(beyond(f)) < 1e-6:
                cross = (float(f), np.asarray(frames[f]["xyz"], dtype=float))
    if cross is None:
        return out
    fc, xyz = cross
    out["line_cross"] = {"frame": round(float(fc), 3),
                         "xyz": [round(float(v), 3) for v in xyz]}
    half = 0.5 * (_GEOMETRY.post_y_right - _GEOMETRY.post_y_left)
    mid = 0.5 * (_GEOMETRY.post_y_right + _GEOMETRY.post_y_left)
    if abs(float(xyz[1]) - mid) >= half:
        out["status"] = STATUS_MISSES_MOUTH
    elif float(xyz[2]) > _GEOMETRY.crossbar_z:
        out["status"] = STATUS_OVER_CROSSBAR
    else:
        out["status"] = STATUS_OK
    return out

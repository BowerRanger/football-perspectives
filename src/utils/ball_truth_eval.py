"""Score a pipeline ball track against Ball Studio truth.

Pure functions over arrays/dicts (no IO); ``scripts/eval_ball_truth.py``
loads the files. Truth = the dense track written next to the truth file
(``<group>_ball_truth_dense.json``): reference-frame ``xyz`` plus the
operator's events and outcome.

Metrics (all on the reference timeline):

* per-frame 3-D error p50/p95/mean over truth frames the pipeline covers,
  plus coverage and the share of ALL truth frames within 0.2 m / 0.5 m
  (uncovered frames count as misses);
* per-view reprojection error (px) of the pipeline point vs the truth point,
  when a camera for that view is supplied;
* event timing: each truth event vs the nearest compatible pipeline event;
* goal line-cross: crossing frame and (y, z) point, truth vs pipeline.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

GOAL_LINES = (0.0, 105.0)
# truth event kind -> pipeline anchor states that count as the same event
EVENT_STATE_MAP: dict[str, tuple[str, ...]] = {
    "touch": ("player_touch", "kick", "header", "volley", "chest", "catch"),
    "keeper_save": ("player_touch", "catch"),
    "bounce": ("bounce",),
    "post": ("goal_impact",),
    "crossbar": ("goal_impact",),
    "net": ("goal_impact",),
    "line_cross": ("goal_impact",),
}


def _pct(a: np.ndarray, q: float) -> float | None:
    return round(float(np.percentile(a, q)), 4) if len(a) else None


def _stats(a: np.ndarray) -> dict:
    return {"n": int(len(a)), "p50": _pct(a, 50), "p95": _pct(a, 95),
            "mean": round(float(a.mean()), 4) if len(a) else None}


def pipeline_by_ref(
    frames: Sequence[int], xyz: Sequence[Sequence[float] | None],
) -> dict[int, np.ndarray]:
    """``{ref_frame: xyz}`` for frames where the pipeline has a position."""
    return {int(f): np.asarray(p, float) for f, p in zip(frames, xyz) if p is not None}


def error_3d(truth_frames: Sequence[int], truth_xyz: np.ndarray,
             pipe: Mapping[int, np.ndarray]) -> dict:
    n_all = len(truth_frames)
    errs, covered = [], []
    for f, p in zip(truth_frames, truth_xyz):
        q = pipe.get(int(f))
        if q is None:
            continue
        covered.append(int(f))
        errs.append(float(np.linalg.norm(q - p)))
    e = np.asarray(errs)
    out = _stats(e)
    out.update({
        "truth_frames": int(n_all),
        "coverage": round(len(e) / n_all, 4) if n_all else None,
        "pct_within_0.2m": round(float((e <= 0.2).sum() / n_all), 4) if n_all else None,
        "pct_within_0.5m": round(float((e <= 0.5).sum() / n_all), 4) if n_all else None,
        "pct_within_0.2m_of_covered": round(float((e <= 0.2).mean()), 4) if len(e) else None,
        "pct_within_0.5m_of_covered": round(float((e <= 0.5).mean()), 4) if len(e) else None,
    })
    return out


def reprojection_error(
    truth_frames: Sequence[int], truth_xyz: np.ndarray,
    pipe: Mapping[int, np.ndarray],
    cameras: Mapping[str, Callable[[int], object]],
    offsets: Mapping[str, int],
) -> dict[str, dict]:
    """Per-view px distance between the projected pipeline and truth points.

    ``cameras[shot](ref_frame)`` returns an object with
    ``project(pts) -> (uv, depth)`` (a ``ball_truth_solver.Cam``) or None.
    """
    out: dict[str, dict] = {}
    for sid, cam_at in cameras.items():
        errs = []
        for f, p in zip(truth_frames, truth_xyz):
            q = pipe.get(int(f))
            cam = cam_at(int(f) + int(offsets.get(sid, 0)))
            if q is None or cam is None:
                continue
            ut, _ = cam.project(p)
            uq, _ = cam.project(q)
            if np.isfinite(ut).all() and np.isfinite(uq).all():
                errs.append(float(np.hypot(*(ut[0] - uq[0]))))
        out[sid] = _stats(np.asarray(errs))
    return out


def event_timing(
    truth_events: Sequence[dict],
    pipeline_events: Sequence[dict],
    *,
    max_window: int = 15,
) -> dict:
    """Match each truth event to the nearest compatible pipeline event.

    ``pipeline_events`` rows: ``{"frame": ref_frame, "state": str}``.
    Unmatched (nothing within ``max_window`` frames) counts as a miss.
    """
    rows = []
    for ev in truth_events:
        states = EVENT_STATE_MAP.get(ev["kind"])
        if states is None:
            continue
        cands = [p for p in pipeline_events if p["state"] in states
                 and abs(p["frame"] - ev["frame"]) <= max_window]
        if not cands:
            rows.append({"frame": ev["frame"], "kind": ev["kind"], "matched": False,
                         "error_frames": None})
            continue
        best = min(cands, key=lambda p: abs(p["frame"] - ev["frame"]))
        rows.append({"frame": ev["frame"], "kind": ev["kind"], "matched": True,
                     "error_frames": int(best["frame"] - ev["frame"])})
    matched = [r["error_frames"] for r in rows if r["matched"]]
    return {
        "events": rows,
        "n_truth_events": len(rows),
        "n_matched": len(matched),
        "mean_abs_error_frames": round(float(np.mean(np.abs(matched))), 3) if matched else None,
    }


def line_cross(frames: Sequence[int], xyz: np.ndarray) -> dict | None:
    """First goal-line crossing of a track (x passes 0 or 105), interpolated
    linearly between the straddling frames. ``None`` if it never crosses."""
    frames = np.asarray(frames, int)
    xyz = np.asarray(xyz, float)
    for line in GOAL_LINES:
        s = np.where(xyz[:, 0] >= line, 1.0, -1.0)
        # crossing from the pitch side: s changes sign; ignore pure touches
        idx = np.nonzero((s[:-1] * s[1:]) < 0)[0]
        # only count crossings that happen near the goal mouth region later
        for i in idx:
            u = (line - xyz[i, 0]) / (xyz[i + 1, 0] - xyz[i, 0])
            p = xyz[i] + u * (xyz[i + 1] - xyz[i])
            f = frames[i] + u * (frames[i + 1] - frames[i])
            return {"line_x": line, "frame": round(float(f), 3),
                    "point": [round(float(v), 4) for v in p]}
    return None


def line_cross_error(truth: dict | None, pipe: dict | None) -> dict | None:
    if truth is None:
        return None
    if pipe is None or pipe["line_x"] != truth["line_x"]:
        return {"truth": truth, "pipeline": pipe, "matched": False,
                "point_error_m": None, "frame_error": None}
    dyz = np.asarray(pipe["point"][1:]) - np.asarray(truth["point"][1:])
    return {"truth": truth, "pipeline": pipe, "matched": True,
            "point_error_m": round(float(np.linalg.norm(dyz)), 4),
            "frame_error": round(pipe["frame"] - truth["frame"], 3)}


def filled_pipeline_track(
    truth_frames: Sequence[int], pipe: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pipeline positions on the truth frames, linearly interpolated across
    gaps (used only for the line-cross comparison)."""
    known = sorted(pipe)
    if len(known) < 2:
        return None
    fr = np.asarray(truth_frames, int)
    lo, hi = known[0], known[-1]
    fr = fr[(fr >= lo) & (fr <= hi)]
    if len(fr) < 2:
        return None
    arr = np.stack([pipe[k] for k in known])
    xyz = np.stack([np.interp(fr, known, arr[:, a]) for a in range(3)], axis=1)
    return fr, xyz


def evaluate(
    truth_dense: dict,
    pipe: Mapping[int, np.ndarray],
    *,
    pipeline_events: Sequence[dict] = (),
    cameras: Mapping[str, Callable[[int], object]] | None = None,
    offsets: Mapping[str, int] | None = None,
) -> dict:
    d = truth_dense["dense"]
    frames = d["frames"]
    xyz = np.asarray(d["xyz"], float).reshape(-1, 3)
    out: dict = {"error_3d_m": error_3d(frames, xyz, pipe)}
    if cameras:
        out["reprojection_px"] = reprojection_error(frames, xyz, pipe, cameras, offsets or {})
    out["event_timing"] = event_timing(truth_dense.get("events", []), pipeline_events)
    if truth_dense.get("outcome") == "goal" and len(frames) > 1:
        t_cross = line_cross(frames, xyz)
        filled = filled_pipeline_track(frames, pipe)
        p_cross = line_cross(*filled) if filled is not None else None
        out["line_cross"] = line_cross_error(t_cross, p_cross)
    return out

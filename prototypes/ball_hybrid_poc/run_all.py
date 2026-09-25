"""CLI: assemble per-clip ``Results`` dicts for the ball hybrid-extraction
PoC (``CONTRACT.md``'s "Results" schema), from whatever combination of A1
(truth/synth), A2 (current-method tracks) and B (hybrid) artifacts exist on
disk at the time it runs.

Deliberately tolerant of missing pieces: A1/A2/B are concurrent ICs on this
same PoC, and their files/modules may not exist yet (or may still be
running in the background for the real-detector cases). Every place this
script depends on one of them records a ``PENDING``/``error: ...`` status
string in the assembled results rather than raising, so a partial run still
produces a valid, viewer-loadable ``results.json``.

Usage:
    .venv311/bin/python prototypes/ball_hybrid_poc/run_all.py \
        --clips gberch,origi01,kroupi01,s013 --scenarios base,mismatch,sparse
    .venv311/bin/python prototypes/ball_hybrid_poc/run_all.py --skip-current
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parents[1]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.schemas.ball_anchor import BallAnchorSet  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.utils import ball_eval as BE  # noqa: E402

from prototypes.ball_hybrid_poc import ctx as poc_ctx  # noqa: E402
from prototypes.ball_hybrid_poc import metrics  # noqa: E402
from prototypes.ball_hybrid_poc import types  # noqa: E402

# --- concurrent ICs' modules: optional at import time -----------------------
try:
    from prototypes.ball_hybrid_poc import truth_builder  # type: ignore
except Exception:  # noqa: BLE001 — A1 may not have landed yet
    truth_builder = None
try:
    from prototypes.ball_hybrid_poc import synth_detector  # type: ignore
except Exception:  # noqa: BLE001
    synth_detector = None
try:
    from prototypes.ball_hybrid_poc import hybrid  # type: ignore
except Exception:  # noqa: BLE001 — B may not have landed yet
    hybrid = None
try:
    from prototypes.ball_hybrid_poc import run_current  # type: ignore
except Exception:  # noqa: BLE001 — A2's module; used only for its
    run_current = None  # BallTrack -> contract Track converter (shipped-track fallback)

DEFAULT_CLIPS = ("gberch", "origi01", "kroupi01", "s013")
DEFAULT_SCENARIOS = ("base", "mismatch", "sparse")
N_FOLDS = 2
_CONTACT_ANCHOR_STATES = frozenset({
    "player_touch", "bounce", "kick", "header", "volley", "chest", "catch",
    "goal_impact",
})
# Real-mode methods beyond "current": (cfg_override, is_events) — an
# "events" method also folds in the current stage's own auto-anchor
# sidecar (kinematic touches / velocity-break bounces / auto goal
# impacts) as extra, softly-gated knots on top of the manual anchors
# (see hybrid._integrate_auto_knots); "hybrid" alone stays manual-only.
_HYBRID_REAL_METHODS: dict[str, tuple[Optional[dict], bool]] = {
    "hybrid": (None, False),
    "hybrid_events": (None, True),
    "hybrid_events_nodrag": ({"cd": 0.0, "fit_cd": False}, True),
}


def _out_root() -> Path:
    return Path(poc_ctx.M) / "output-ball-poc"


def _pctl(vals: Sequence[float], q: float) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


# ---------------------------------------------------------------------------
# A1 truth / synth load-or-build
# ---------------------------------------------------------------------------

def _load_or_build_truth(clip_ctx, scenario: str, clip_dir: Path):
    path = clip_dir / f"truth_{scenario}.json"
    if path.exists():
        try:
            return types.TruthTrack.from_json(types.load_json(path)), "loaded"
        except Exception as exc:  # noqa: BLE001
            return None, f"error loading {path.name}: {exc!r}"
    if truth_builder is None:
        return None, "PENDING (truth_builder.py not present yet)"
    try:
        truth = truth_builder.build_truth(clip_ctx, scenario, seed=0)
    except Exception as exc:  # noqa: BLE001
        return None, f"PENDING (truth_builder.build_truth error: {exc!r})"
    types.save_json(path, truth)
    return truth, "built"


def _load_or_build_synth(clip_ctx, truth, scenario: str, clip_dir: Path):
    path = clip_dir / f"synth_obs_{scenario}.json"
    if path.exists():
        try:
            return types.SynthRun.from_json(types.load_json(path)), "loaded"
        except Exception as exc:  # noqa: BLE001
            return None, f"error loading {path.name}: {exc!r}"
    if synth_detector is None:
        return None, "PENDING (synth_detector.py not present yet)"
    try:
        synth = synth_detector.make_synth_run(clip_ctx, truth, scenario, seed=0)
    except Exception as exc:  # noqa: BLE001
        return None, f"PENDING (synth_detector.make_synth_run error: {exc!r})"
    types.save_json(path, synth)
    return synth, "built"


def _load_track_file(path: Path):
    if not path.exists():
        return None, "PENDING (file not found)"
    try:
        return types.Track.from_json(types.load_json(path)), "loaded"
    except Exception as exc:  # noqa: BLE001
        return None, f"error loading {path.name}: {exc!r}"


def _nodrag_cfg():
    """Best-effort ``cd=0`` (gravity-only ablation) config for B's
    ``run_hybrid``. B's config type isn't fixed by the contract beyond
    "cfg (cd=0 -> gravity-only ablation)", so this tries the plausible
    constructor names before falling back to a plain dict; ``run_hybrid``
    callers catch any resulting TypeError and mark the ablation PENDING
    rather than crash the run."""
    if hybrid is None:
        return None
    # fit_cd must be off too, or the span solver re-fits Cd whenever a span
    # has enough evidence and the ablation silently becomes the drag model.
    return {"cd": 0.0, "fit_cd": False}


def _run_hybrid_track(clip_ctx, observations, anchors, fixes,
                       auto_anchors=(), cfg=None):
    """Returns ``(Track|None, status)``. Never raises."""
    if hybrid is None:
        return None, "PENDING (hybrid.py not present yet)"
    try:
        track = hybrid.run_hybrid(clip_ctx, observations, anchors,
                                   fixes=fixes, auto_anchors=auto_anchors, cfg=cfg)
    except Exception as exc:  # noqa: BLE001
        return None, f"error: {exc!r}"
    return track, "ok"


def _load_auto_anchors(path: Path) -> tuple[tuple, str]:
    """The current stage's auto-event sidecar (same ``BallAnchorSet``
    schema as the manual anchors file) -> its ``.anchors`` tuple, for
    ``hybrid.run_hybrid``'s ``auto_anchors`` param. These files appear as
    A2's runs (``run_current.py``) complete; PENDING until then."""
    if not path.exists():
        return (), "PENDING (auto-anchor sidecar not found)"
    try:
        aset = BallAnchorSet.load(path)
        return tuple(aset.anchors), "loaded"
    except Exception as exc:  # noqa: BLE001
        return (), f"error loading {path.name}: {exc!r}"


def _track_from_shipped_ball_track(clip_ctx) -> tuple[Any, str]:
    """Fallback for the real 'current' full track when A2's own
    ``track_current_real_full.json`` isn't there yet: convert the
    shipped main-repo ``ball/<shot>_ball_track.json`` (the current
    method's actual production output) into the contract ``Track``.
    Reuses A2's own ``_track_to_contract`` converter when importable, so
    this can't silently drift from how A2 does the same conversion."""
    path = clip_ctx.output_dir / "ball" / f"{clip_ctx.shot_id}_ball_track.json"
    if not path.exists():
        return None, "PENDING (no track_current_real_full.json or shipped ball_track.json)"
    try:
        bt = BallTrack.load(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"error loading shipped ball_track.json: {exc!r}"
    if run_current is not None:
        try:
            return run_current._track_to_contract(clip_ctx.clip_id, bt), "shipped track"
        except Exception as exc:  # noqa: BLE001
            return None, f"error converting shipped ball_track.json: {exc!r}"
    # run_current.py not importable: replicate its (trivial) conversion inline.
    track = types.Track(
        clip_id=clip_ctx.clip_id, method="current",
        frames=tuple(types.TrackFrame(frame=f.frame, xyz=f.world_xyz,
                                       mode=f.state, conf=f.confidence)
                     for f in bt.frames))
    return track, "shipped track"


def _load_current_full_with_fallback(clip_ctx, clip_dir: Path):
    track, status = _load_track_file(clip_dir / "track_current_real_full.json")
    if track is not None:
        return track, status
    return _track_from_shipped_ball_track(clip_ctx)


# ---------------------------------------------------------------------------
# synthetic scenarios
# ---------------------------------------------------------------------------

def process_scenario(clip_ctx, clip_dir: Path, scenario: str,
                      *, skip_current: bool) -> dict[str, Any]:
    status: dict[str, str] = {}

    truth, tstatus = _load_or_build_truth(clip_ctx, scenario, clip_dir)
    status["truth"] = tstatus
    if truth is None:
        return {
            "truth": {"scenario": scenario, "pending": True, "reason": tstatus},
            "tracks": {}, "metrics": {}, "metrics_detail": {},
            "per_frame_err": {}, "_status": status,
        }

    synth, sstatus = _load_or_build_synth(clip_ctx, truth, scenario, clip_dir)
    status["synth"] = sstatus

    side_cam = metrics.build_side_camera([f.xyz for f in truth.frames])

    tracks: dict[str, Any] = {}
    flat_metrics: dict[str, Any] = {}
    detail_metrics: dict[str, Any] = {}
    per_frame_err: dict[str, Any] = {}

    def _record(method: str, track):
        tracks[method] = track.to_json()
        flat, detail = metrics.compute_scenario_metrics(
            clip_ctx, track, truth, side_camera=side_cam)
        flat_metrics[method] = flat
        detail_metrics[method] = detail
        per_frame_err[method] = metrics.per_frame_error(track, truth)

    if not skip_current:
        cur_track, cstatus = _load_track_file(
            clip_dir / f"track_current_{scenario}.json")
        status["current"] = cstatus
        if cur_track is not None:
            _record("current", cur_track)
    else:
        status["current"] = "skipped (--skip-current)"

    if synth is not None:
        # No fixes on synthetic scenarios per CONTRACT.
        hyb_track, hstatus = _run_hybrid_track(
            clip_ctx, list(synth.observations), list(synth.anchors), ())
        status["hybrid"] = hstatus
        if hyb_track is not None:
            _record("hybrid", hyb_track)

        nodrag_track, nstatus = _run_hybrid_track(
            clip_ctx, list(synth.observations), list(synth.anchors), (),
            cfg=_nodrag_cfg())
        status["hybrid_nodrag"] = nstatus
        if nodrag_track is not None:
            _record("hybrid_nodrag", nodrag_track)

        # hybrid_events{,_nodrag}: manual anchors + the current stage's
        # OWN auto events for this same scenario (auto_anchors_current_
        # <scenario>.json, written by run_current.run_synthetic).
        auto_anchors, auto_status = _load_auto_anchors(
            clip_dir / f"auto_anchors_current_{scenario}.json")
        status["auto_anchors"] = auto_status
        if auto_anchors:
            ev_track, evstatus = _run_hybrid_track(
                clip_ctx, list(synth.observations), list(synth.anchors), (),
                auto_anchors=auto_anchors)
            status["hybrid_events"] = evstatus
            if ev_track is not None:
                _record("hybrid_events", ev_track)

            ev_nodrag_track, evnstatus = _run_hybrid_track(
                clip_ctx, list(synth.observations), list(synth.anchors), (),
                auto_anchors=auto_anchors, cfg=_nodrag_cfg())
            status["hybrid_events_nodrag"] = evnstatus
            if ev_nodrag_track is not None:
                _record("hybrid_events_nodrag", ev_nodrag_track)
        else:
            status["hybrid_events"] = status["hybrid_events_nodrag"] = (
                f"PENDING (no auto anchors: {auto_status})")
    else:
        status["hybrid"] = status["hybrid_nodrag"] = (
            f"PENDING (no synth run: {sstatus})")
        status["hybrid_events"] = status["hybrid_events_nodrag"] = (
            f"PENDING (no synth run: {sstatus})")

    return {
        "truth": truth.to_json(),
        "tracks": tracks,
        "metrics": flat_metrics,
        "metrics_detail": detail_metrics,
        "per_frame_err": per_frame_err,
        "side_camera": side_cam.to_json(),
        "_status": status,
    }


# ---------------------------------------------------------------------------
# real-footage evaluation
# ---------------------------------------------------------------------------

def _split_fixes_even_odd(fixes: Sequence) -> tuple[tuple, tuple]:
    ordered = tuple(sorted(fixes, key=lambda fx: fx.frame))
    half_a = tuple(fx for i, fx in enumerate(ordered) if i % 2 == 0)
    half_b = tuple(fx for i, fx in enumerate(ordered) if i % 2 == 1)
    return half_a, half_b


def _fix_halves_for_fold(fold: int, half_a: tuple, half_b: tuple):
    """fold 0 knots on half_a, grades on half_b; fold 1 the reverse — so
    running both folds knots+grades every fix exactly once each way."""
    return (half_a, half_b) if fold == 0 else (half_b, half_a)


def _anchor_err(row) -> Optional[float]:
    """Mirrors ``scripts/eval_ball_accuracy.py``'s anchor grading: 3-D
    error where ground truth is known, else the ray-lateral distance as a
    (necessarily optimistic) lower bound so a missing GT never masks a
    method that's actually off."""
    return row.err_3d_m if row.err_3d_m is not None else row.lateral_m


def _get_or_build_split(clip_ctx, fold: int, clip_dir: Path) -> tuple[tuple, tuple]:
    """Returns ``(kept_anchors, held_anchors)`` for ``fold``, using A2's
    ``real_split_foldK.json`` (frame lists) if present, else computing +
    persisting it via ``BE.split_anchors`` so the split is written exactly
    once and every consumer (A2's own current-method run included) agrees
    on the same frames."""
    path = clip_dir / f"real_split_fold{fold}.json"
    all_anchors = clip_ctx.anchors.anchors
    if path.exists():
        data = types.load_json(path)
        kept_frames = set(int(f) for f in data.get("kept_frames", []))
        held_frames = set(int(f) for f in data.get("heldout_frames", []))
        kept = tuple(a for a in all_anchors if a.frame in kept_frames)
        held = tuple(a for a in all_anchors if a.frame in held_frames)
        return kept, held
    kept, held = BE.split_anchors(all_anchors, fold=fold, n_folds=N_FOLDS)
    types.save_json(path, {
        "kept_frames": sorted(a.frame for a in kept),
        "heldout_frames": sorted(a.frame for a in held),
    })
    return kept, held


def _faithfulness_px_error(clip_ctx, track, observations) -> dict[str, Any]:
    """Reprojection error of the method's xyz against the REAL detector
    observation pixel at the same frame (no dense truth exists for real
    footage, so faithfulness is graded against the evidence itself)."""
    if track is None:
        return {"n": 0, "p50": None, "p95": None}
    world = {tf.frame: tf.xyz for tf in track.frames if tf.xyz is not None}
    errs = []
    for obs in observations:
        xyz = world.get(obs.frame)
        if xyz is None or obs.frame not in clip_ctx.per_frame_K:
            continue
        u = clip_ctx.project(obs.frame, xyz)
        errs.append(float(np.hypot(u[0] - obs.uv[0], u[1] - obs.uv[1])))
    return {"n": len(errs), "p50": _pctl(errs, 50), "p95": _pctl(errs, 95)}


def process_real(clip_ctx, clip_dir: Path, *, skip_current: bool) -> dict[str, Any]:
    status: dict[str, Any] = {}
    all_anchors = clip_ctx.anchors.anchors
    cams = {f: (clip_ctx.per_frame_K[f], clip_ctx.per_frame_R[f],
                clip_ctx.per_frame_t[f]) for f in clip_ctx.frames}
    observations = list(clip_ctx.observations)
    ball_radius = metrics.BALL_RADIUS_M
    is_origi01 = clip_ctx.clip_id == "origi01" and len(clip_ctx.fixes) > 0
    half_a, half_b = (_split_fixes_even_odd(clip_ctx.fixes)
                      if is_origi01 else ((), ()))

    event_frames_real = [a.frame for a in all_anchors
                          if a.state in _CONTACT_ANCHOR_STATES]

    def _joint_world_fn(frame, player_id, bone):
        try:
            return clip_ctx.player_context().joint_world(frame, player_id, bone)
        except Exception:  # noqa: BLE001
            return None

    tracks_full: dict[str, Any] = {}
    metrics_flat: dict[str, Any] = {}
    metrics_detail: dict[str, Any] = {}
    fold_detail: dict[str, Any] = {}
    anchors_heldout: list[dict] = []
    fixes_rows: list[dict] = []

    for method in ("current", *_HYBRID_REAL_METHODS):
        held_rows_all: list = []
        fix_err_vals: list[float] = []
        per_fold_info: dict[str, Any] = {}
        hyb_cfg, use_events = _HYBRID_REAL_METHODS.get(method, (None, False))

        # --- full run (all anchors as knots; faithfulness + naturalness) ---
        if method == "current":
            if skip_current:
                full_track, fstatus = None, "skipped (--skip-current)"
            else:
                full_track, fstatus = _load_current_full_with_fallback(clip_ctx, clip_dir)
        else:
            full_fixes = clip_ctx.fixes if is_origi01 else ()
            full_auto: tuple = ()
            if use_events:
                full_auto, auto_status = _load_auto_anchors(
                    clip_dir / "auto_anchors_current_real_full.json")
                status[f"{method}_full_auto"] = auto_status
            if use_events and not full_auto:
                full_track, fstatus = None, f"PENDING (no auto anchors: {auto_status})"
            else:
                full_track, fstatus = _run_hybrid_track(
                    clip_ctx, observations, list(all_anchors), full_fixes,
                    auto_anchors=full_auto, cfg=hyb_cfg)
        status[f"{method}_full"] = fstatus

        if full_track is not None:
            tracks_full[method] = full_track.to_json()
            bpx = _faithfulness_px_error(clip_ctx, full_track, observations)
            nat = metrics.naturalness_summary_real(
                full_track, fps=clip_ctx.fps, event_frames=event_frames_real)
        else:
            bpx = {"n": 0, "p50": None, "p95": None}
            nat = {"n_violations": None, "by_kind": {}}

        # --- folds: held-out anchor 3-D error (+ fix error on origi01) ---
        for fold in range(N_FOLDS):
            kept, held = _get_or_build_split(clip_ctx, fold, clip_dir)
            held_frames = frozenset(a.frame for a in held)
            knot_fixes, graded_fixes = _fix_halves_for_fold(fold, half_a, half_b)

            if method == "current":
                fold_track, tstatus = (
                    (None, "skipped (--skip-current)") if skip_current else
                    _load_track_file(
                        clip_dir / f"track_current_real_fold{fold}.json"))
            else:
                fold_auto: tuple = ()
                if use_events:
                    fold_auto, fold_auto_status = _load_auto_anchors(
                        clip_dir / f"auto_anchors_current_real_fold{fold}.json")
                    status[f"{method}_fold{fold}_auto"] = fold_auto_status
                if use_events and not fold_auto:
                    fold_track, tstatus = None, f"PENDING (no auto anchors: {fold_auto_status})"
                else:
                    fold_track, tstatus = _run_hybrid_track(
                        clip_ctx, observations, list(kept), knot_fixes,
                        auto_anchors=fold_auto, cfg=hyb_cfg)
            status[f"{method}_fold{fold}"] = tstatus
            per_fold_info[f"fold{fold}"] = {"status": tstatus}

            if fold_track is None:
                continue
            world = {tf.frame: tf.xyz for tf in fold_track.frames
                      if tf.xyz is not None}
            rows = BE.eval_rows_at_anchors(
                world, all_anchors, cams, ball_radius=ball_radius,
                distortion=clip_ctx.distortion, joint_world_fn=_joint_world_fn,
                held_out_frames=held_frames,
                evidence_frames=frozenset(o.frame for o in observations))
            fold_held = [r for r in rows if r.held_out]
            held_rows_all.extend(fold_held)
            fold_errs = [_anchor_err(r) for r in fold_held
                         if _anchor_err(r) is not None]
            per_fold_info[f"fold{fold}"]["heldout_p50_m"] = _pctl(fold_errs, 50)
            per_fold_info[f"fold{fold}"]["heldout_n"] = len(fold_held)

            if is_origi01 and graded_fixes:
                fx_rows = BE.eval_rows_at_fixes(
                    world, [(fx.frame, fx.xyz, fx.ray_miss_m)
                            for fx in graded_fixes])
                fold_fix_errs = [r.err_3d_m for r in fx_rows
                                  if r.err_3d_m is not None]
                fix_err_vals.extend(fold_fix_errs)
                per_fold_info[f"fold{fold}"]["fix_err_p50_m"] = _pctl(
                    fold_fix_errs, 50)

        all_held_errs = [_anchor_err(r) for r in held_rows_all
                          if _anchor_err(r) is not None]
        flat = {
            "p50": None, "p95": None, "max": None, "pct_le_20cm": None,
            "contact_gap": None, "ground_float_sink": None,
            "broadcast_px_error": bpx["p50"],
            "side_px_error": None,
            "naturalness_violations": nat["n_violations"],
            "anchor_heldout_err_m": _pctl(all_held_errs, 50),
            "anchor_heldout_err_p95_m": _pctl(all_held_errs, 95),
            "anchor_heldout_n": len(held_rows_all),
            "fix_err_m": _pctl(fix_err_vals, 50) if fix_err_vals else None,
            "fix_err_n": len(fix_err_vals),
            "broadcast_px_error_p95": bpx["p95"],
        }
        metrics_flat[method] = flat
        metrics_detail[method] = {
            "faithfulness_px": bpx, "naturalness": nat,
            "anchor_heldout_all": {"n": len(held_rows_all),
                                    "p50": _pctl(all_held_errs, 50),
                                    "p95": _pctl(all_held_errs, 95)},
            "fix_errors": {"n": len(fix_err_vals),
                            "p50": _pctl(fix_err_vals, 50) if fix_err_vals else None},
        }
        fold_detail[method] = per_fold_info

    # anchors_heldout / fixes context lists (method-agnostic; built once
    # from the anchor/fix objects themselves, not from any method's track).
    for fold in range(N_FOLDS):
        _, held = _get_or_build_split(clip_ctx, fold, clip_dir)
        for a in held:
            gt, _kind = BE.anchor_gt_world(
                a, *cams.get(a.frame, (None, None, None)), clip_ctx.distortion,
                ball_radius=ball_radius,
                joint_world=(_joint_world_fn(a.frame, a.player_id, a.bone)
                              if a.state == "player_touch" else None),
            ) if a.frame in cams else (None, "none")
            anchors_heldout.append({
                "frame": a.frame,
                "xyz_gt": (list(gt) if gt is not None else None),
                "fold": fold,
            })
        knot_fixes, graded_fixes = _fix_halves_for_fold(fold, half_a, half_b)
        for fx in knot_fixes:
            fixes_rows.append({"frame": fx.frame, "xyz": list(fx.xyz),
                                "used_as": "knot", "fold": fold})
        for fx in graded_fixes:
            fixes_rows.append({"frame": fx.frame, "xyz": list(fx.xyz),
                                "used_as": "graded", "fold": fold})

    return {
        "tracks": tracks_full,
        "metrics": metrics_flat,
        "metrics_detail": metrics_detail,
        "anchors_heldout": anchors_heldout,
        "fixes": fixes_rows,
        "_status": status,
        "_fold_detail": fold_detail,
    }


# ---------------------------------------------------------------------------
# per-clip assembly
# ---------------------------------------------------------------------------

def process_clip(clip_id: str, scenario_names: Sequence[str], out_root: Path,
                  *, skip_current: bool) -> dict[str, Any]:
    clip_dir = out_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    try:
        clip_ctx = poc_ctx.load_clip(clip_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "clip_id": clip_id,
            "_status": {"load_clip": f"PENDING ({exc!r})"},
            "scenarios": {}, "real": {"tracks": {}, "metrics": {}},
        }

    scenarios = {
        name: process_scenario(clip_ctx, clip_dir, name,
                                skip_current=skip_current)
        for name in scenario_names
    }
    real = process_real(clip_ctx, clip_dir, skip_current=skip_current)

    results = {
        "clip_id": clip_id,
        "fps": clip_ctx.fps,
        "image_size": list(clip_ctx.image_size),
        "pitch": {"length": 105.0, "width": 68.0},
        "camera": {"centre_xyz_per_frame": clip_ctx.camera_centres()},
        "scenarios": scenarios,
        "real": real,
        "players": {"frames": [],
                     "_status": "skipped (optional; out of PoC-C timebox)"},
    }
    problems = types.validate_results(results)
    if problems:
        results.setdefault("_status", {})["validate_results"] = problems
    types.save_json(clip_dir / "results.json", results)
    return results


# ---------------------------------------------------------------------------
# summary assembly
# ---------------------------------------------------------------------------

def _fmt(v, nd=3) -> str:
    if v is None:
        return "PENDING"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_summary(all_results: dict[str, dict]) -> tuple[dict, str]:
    rows = []
    for clip_id, res in all_results.items():
        for scen_name, sc in res.get("scenarios", {}).items():
            m = sc.get("metrics", {})
            for method, mm in m.items():
                rows.append({
                    "clip_id": clip_id, "scenario": scen_name, "method": method,
                    "p50": mm.get("p50"), "p95": mm.get("p95"),
                    "pct_le_20cm": mm.get("pct_le_20cm"),
                    "contact_gap": mm.get("contact_gap"),
                    "ground_float_sink": mm.get("ground_float_sink"),
                    "side_px_error": mm.get("side_px_error"),
                    "naturalness_violations_minus_truth":
                        mm.get("naturalness_violations_minus_truth"),
                })
            if not m:
                rows.append({"clip_id": clip_id, "scenario": scen_name,
                              "method": "(none)", "p50": None, "p95": None,
                              "pct_le_20cm": None, "contact_gap": None,
                              "ground_float_sink": None, "side_px_error": None,
                              "naturalness_violations_minus_truth": None})
        real_rows = []
        real_m = res.get("real", {}).get("metrics", {})
        for method, mm in real_m.items():
            real_rows.append({
                "clip_id": clip_id, "method": method,
                "anchor_heldout_p50_m": mm.get("anchor_heldout_err_m"),
                "anchor_heldout_p95_m": mm.get("anchor_heldout_err_p95_m"),
                "fix_err_m": mm.get("fix_err_m"),
            })
        res.setdefault("_summary_real_rows", real_rows)

    lines = ["# Ball hybrid-extraction PoC — summary", "",
             "## Synthetic (scenario x method)", "",
             "| clip | scenario | method | p50 (m) | p95 (m) | %<=20cm | "
             "contact gap (m) | float/sink (m) | side px p50 | nat. viol. Δ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['clip_id']} | {r['scenario']} | {r['method']} | "
            f"{_fmt(r['p50'])} | {_fmt(r['p95'])} | "
            f"{_fmt(r['pct_le_20cm'])} | {_fmt(r['contact_gap'])} | "
            f"{_fmt(r['ground_float_sink'])} | {_fmt(r['side_px_error'], 1)} | "
            f"{_fmt(r['naturalness_violations_minus_truth'], 0)} |")

    lines += ["", "## Real footage (held-out anchor error, per method)", "",
              "| clip | method | held-out p50 (m) | held-out p95 (m) | "
              "fix err p50 (m) |", "|---|---|---|---|---|"]
    for clip_id, res in all_results.items():
        for r in res.get("_summary_real_rows", []):
            lines.append(
                f"| {r['clip_id']} | {r['method']} | "
                f"{_fmt(r['anchor_heldout_p50_m'])} | "
                f"{_fmt(r['anchor_heldout_p95_m'])} | {_fmt(r['fix_err_m'])} |")

    summary_dict = {
        "clips": sorted(all_results),
        "synthetic_rows": rows,
        "real_rows": [r for res in all_results.values()
                       for r in res.get("_summary_real_rows", [])],
    }
    for res in all_results.values():
        res.pop("_summary_real_rows", None)
    return summary_dict, "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--skip-current", action="store_true")
    ap.add_argument("--out-dir", default=None,
                     help="Defaults to $M/output-ball-poc.")
    args = ap.parse_args()

    clip_names = [c.strip() for c in args.clips.split(",") if c.strip()]
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    out_root = Path(args.out_dir) if args.out_dir else _out_root()
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    for clip_id in clip_names:
        print(f"=== {clip_id} ===")
        all_results[clip_id] = process_clip(
            clip_id, scenario_names, out_root, skip_current=args.skip_current)

    summary_dict, summary_md = build_summary(all_results)
    types.save_json(out_root / "summary.json", summary_dict)
    (out_root / "summary.md").write_text(summary_md)
    print(summary_md)
    print(f"wrote {out_root / 'summary.json'} and {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()

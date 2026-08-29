#!/usr/bin/env python
"""Bench harness for the GVHMR inference-time campaign (Phase A).

Compares wall-clock + per-phase timing (Task 1's ``GVHMREstimator.timings``)
across three ``estimate_sequence``/``run_on_track`` call modes, for one or
more player tracks pulled from an existing (read-only) pipeline output
directory:

``legacy``
    ``legacy_decode=True``, ``per_frame_R=None``. Pre-Task-2/3 behaviour:
    ViTPose and the HMR2 feature extractor each independently decode +
    preprocess the temp video (``get_batch()`` runs twice), and GVHMR's
    internal SimpleVO estimates per-frame camera rotation.
``shared_decode``
    Task 2 only: a single shared ``get_batch()`` decode feeds both
    extractors. Still uses SimpleVO for camera rotation (``per_frame_R``
    not supplied).
``calibrated_r``
    Task 2 + Task 3: single shared decode AND the shot's calibrated
    ``camera_track`` R (via ``build_track_camera_R``) replaces GVHMR's
    internal SimpleVO estimate entirely.

All three modes pass the SAME calibrated per-frame K (from the shot's
camera_track) in every mode — K conditioning isn't the axis under test
here, and the pipeline never runs GVHMR decoupled from calibrated K (see
CLAUDE.md's "hmr_world camera-K coupling" note).

One ``GVHMREstimator`` is constructed and reused for every (player, mode)
combination in a run — the GVHMR + ViTPose-Huge + HMR2-ViT + SMPLX load
only happens once (lazily, on the first ``estimate_sequence`` call).

Usage
-----
    .venv311/bin/python scripts/bench_gvhmr_inference.py \\
        --output output/ --shot gberch --players P012,P014 \\
        --modes legacy,shared_decode,calibrated_r \\
        --scratch /path/to/scratch/bench_phase_a

``--output`` is read-only; every write goes under ``--scratch``
(required) — ``<scratch>/<player>/<mode>/arrays.npz`` (raw GVHMR output
arrays) and ``<scratch>/bench_summary.json`` (timing + comparison
summary).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.schemas.tracks import TracksResult  # noqa: E402
from src.stages.hmr_world import build_track_camera_R  # noqa: E402
from src.utils.gvhmr_estimator import (  # noqa: E402
    _TIMING_KEYS,
    GVHMREstimator,
    run_on_track,
)

_MODES = ("legacy", "shared_decode", "calibrated_r")

_DEFAULT_CHECKPOINT = (
    "third_party/gvhmr/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
)

# Mirrors hmr_world._ANKLE_CONF_MIN — used here only as a comparison
# proxy for "was this frame foot-anchor-eligible", not to reproduce the
# full foot-anchor pipeline.
_ANKLE_CONF_MIN = 0.3
_COCO_LEFT_ANKLE = 15
_COCO_RIGHT_ANKLE = 16


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", type=Path, default=Path("output/"),
        help="Read-only pipeline output dir (default: output/)",
    )
    p.add_argument("--shot", required=True, help="Shot id, e.g. gberch")
    p.add_argument(
        "--players", required=True,
        help="Comma-separated player_id list, e.g. P012,P014",
    )
    p.add_argument(
        "--modes", default=",".join(_MODES),
        help=f"Comma-separated subset of {_MODES} (default: all three)",
    )
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p.add_argument(
        "--extractor-device", default="cpu", dest="extractor_device",
        help=(
            "Device for the ViTPose/HMR2 feature extractors only "
            "(default: cpu, matching every non-bench caller). "
            "'auto'|'mps'|'cpu'|'cuda' -- passed straight to "
            "GVHMREstimator(extractor_device=...). Deliberately NOT "
            "gated on PYTORCH_ENABLE_MPS_FALLBACK: an unsupported op on "
            "the extractor device should surface as a RuntimeError so "
            "the estimator's built-in fallback engages and gets "
            "counted (resolved_extractor_device / "
            "extractor_fallback_count in bench_summary.json) -- the "
            "env-var guard below stays scoped to --device mps only."
        ),
    )
    p.add_argument(
        "--max-seq-len", type=int, default=120, dest="max_seq_len",
        help="Max frames per estimate_sequence() chunk (default: 120)",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, dest="max_frames",
        help="Optional per-track truncation, e.g. for smoke runs",
    )
    p.add_argument(
        "--checkpoint", default=_DEFAULT_CHECKPOINT,
        help="GVHMR checkpoint path (relative to repo root or absolute)",
    )
    p.add_argument(
        "--scratch", type=Path, required=True,
        help="Required: every write (npz arrays + bench_summary.json) goes here",
    )
    p.add_argument(
        "--compare-against", type=Path, default=None, dest="compare_against",
        help=(
            "Optional reference scratch dir from a prior bench run (e.g. "
            "one recorded with --extractor-device cpu). For each player "
            "also present in this run's 'calibrated_r' mode output, "
            "loads <dir>/<player>/calibrated_r/arrays.npz as the "
            "reference and emits a per-player parity table (value / "
            "threshold / pass per metric) into bench_summary.json under "
            "'parity'. Read-only."
        ),
    )
    args = p.parse_args(argv)

    if args.device.strip().lower() == "mps" and os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) != "1":
        p.error(
            "--device mps requires PYTORCH_ENABLE_MPS_FALLBACK=1 in the "
            "environment (some GVHMR/vendored ops fall back to CPU under "
            "MPS) -- refusing to run without it"
        )
    return args


def _load_shot_camera(output_dir: Path, shot: str) -> CameraTrack:
    path = output_dir / "camera" / f"{shot}_camera_track.json"
    if not path.exists():
        raise FileNotFoundError(f"camera track not found at {path}")
    return CameraTrack.load(path)


def _load_track_frames(
    output_dir: Path, shot: str, player_id: str
) -> list[tuple[int, tuple[int, int, int, int]]]:
    path = output_dir / "tracks" / f"{shot}_tracks.json"
    if not path.exists():
        raise FileNotFoundError(f"tracks file not found at {path}")
    tr = TracksResult.load(path)
    for t in tr.tracks:
        if t.player_id == player_id:
            frames = sorted(t.frames, key=lambda f: f.frame)
            return [
                (int(f.frame), tuple(int(x) for x in f.bbox)) for f in frames
            ]
    available = sorted(
        {t.player_id for t in tr.tracks if t.player_id}
    )
    raise ValueError(
        f"player_id {player_id!r} not found in {path}; available: {available}"
    )


def _build_dense_K(
    track_frames: list[tuple[int, tuple[int, int, int, int]]],
    per_frame_K: dict[int, np.ndarray],
) -> np.ndarray | None:
    """Mirrors hmr_world.process_player's gvhmr_K construction exactly —
    median-K fallback for any frame missing from the camera track."""
    if not per_frame_K:
        return None
    K_values = np.stack(list(per_frame_K.values()))
    K_median = np.median(K_values, axis=0)
    return np.stack(
        [per_frame_K.get(int(fi), K_median) for fi, _ in track_frames]
    ).astype(np.float32)


def _warm_page_cache(video_path: Path) -> float:
    """Read the shot's mp4 bytes once so the OS page cache is warm
    before any mode is timed — keeps first-mode-run disk-IO variance out
    of the cross-mode comparison."""
    t0 = time.perf_counter()
    with open(video_path, "rb") as fh:
        while fh.read(4 * 1024 * 1024):
            pass
    return time.perf_counter() - t0


def _anchored_fraction(kp2d: np.ndarray, conf_min: float = _ANKLE_CONF_MIN) -> float:
    if kp2d.shape[0] == 0:
        return 0.0
    left = kp2d[:, _COCO_LEFT_ANKLE, 2]
    right = kp2d[:, _COCO_RIGHT_ANKLE, 2]
    min_conf = np.minimum(left, right)
    return float(np.mean(min_conf >= conf_min))


def _mode_kwargs(
    mode: str,
    track_frames: list[tuple[int, tuple[int, int, int, int]]],
    per_frame_R_dict: dict[int, np.ndarray],
) -> dict:
    if mode == "legacy":
        return {"per_frame_R": None, "legacy_decode": True}
    if mode == "shared_decode":
        return {"per_frame_R": None, "legacy_decode": False}
    if mode == "calibrated_r":
        gvhmr_R = build_track_camera_R(track_frames, per_frame_R_dict)
        return {"per_frame_R": gvhmr_R, "legacy_decode": False}
    raise ValueError(f"unknown mode {mode!r}, must be one of {_MODES}")


def _run_mode(
    mode: str,
    *,
    estimator: GVHMREstimator,
    track_frames: list[tuple[int, tuple[int, int, int, int]]],
    video_path: Path,
    checkpoint: Path,
    device: str,
    max_seq_len: int,
    per_frame_K: np.ndarray | None,
    per_frame_R_dict: dict[int, np.ndarray],
) -> dict:
    kwargs = _mode_kwargs(mode, track_frames, per_frame_R_dict)

    n_chunks_before = len(estimator.timings)
    t0 = time.perf_counter()
    out = run_on_track(
        track_frames=track_frames,
        video_path=video_path,
        checkpoint=checkpoint,
        device=device,
        batch_size=16,
        max_sequence_length=max_seq_len,
        estimator=estimator,
        per_frame_K=per_frame_K,
        **kwargs,
    )
    wall_s = time.perf_counter() - t0
    chunk_records = estimator.timings[n_chunks_before:]

    breakdown_s = {k: 0.0 for k in _TIMING_KEYS}
    for rec in chunk_records:
        for k in _TIMING_KEYS:
            breakdown_s[k] += rec[k]

    return {
        "wall_s": wall_s,
        "n_frames": len(track_frames),
        "n_chunks": len(chunk_records),
        "breakdown_s": breakdown_s,
        "out": out,
    }


def _compare_legacy_vs_shared(legacy_out: dict, shared_out: dict) -> dict:
    detail = {}
    gate_ok = True
    for key in ("thetas", "betas", "root_R_cam", "root_t_cam", "kp2d"):
        a, b = legacy_out[key], shared_out[key]
        close = bool(np.allclose(a, b, atol=1e-6))
        exact = bool(np.array_equal(a, b))
        detail[key] = {"allclose_atol_1e-6": close, "array_equal": exact}
        gate_ok = gate_ok and close
    return {"gate_allclose_atol_1e-6_PASS": gate_ok, "detail": detail}


def _root_R_angular_delta_deg(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Per-frame relative rotation angle (degrees) between two (N,3,3)
    rotation-matrix arrays, via trace(R_a^T @ R_b) = 1 + 2*cos(angle)."""
    R_rel = np.einsum("nij,njk->nik", np.transpose(R_a, (0, 2, 1)), R_b)
    trace = np.trace(R_rel, axis1=1, axis2=2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _compare_shared_vs_calibrated(shared_out: dict, calibrated_out: dict) -> dict:
    kp2d_equal = bool(np.array_equal(shared_out["kp2d"], calibrated_out["kp2d"]))
    all_finite = bool(
        np.all(np.isfinite(calibrated_out["thetas"]))
        and np.all(np.isfinite(calibrated_out["root_R_cam"]))
        and np.all(np.isfinite(calibrated_out["root_t_cam"]))
    )
    dtheta = calibrated_out["thetas"].astype(np.float64) - shared_out["thetas"].astype(np.float64)
    max_abs_dtheta_rad = float(np.max(np.abs(dtheta))) if dtheta.size else 0.0
    mean_abs_dtheta_rad = float(np.mean(np.abs(dtheta))) if dtheta.size else 0.0

    angles_deg = _root_R_angular_delta_deg(
        shared_out["root_R_cam"].astype(np.float64),
        calibrated_out["root_R_cam"].astype(np.float64),
    )
    mean_root_R_angular_delta_deg = float(np.mean(angles_deg)) if angles_deg.size else 0.0

    anchored_shared = _anchored_fraction(shared_out["kp2d"])
    anchored_calibrated = _anchored_fraction(calibrated_out["kp2d"])

    return {
        "kp2d_array_equal_REQUIRED": kp2d_equal,
        "all_finite": all_finite,
        "max_abs_dtheta_rad": max_abs_dtheta_rad,
        "mean_abs_dtheta_rad": mean_abs_dtheta_rad,
        "mean_root_R_angular_delta_deg": mean_root_R_angular_delta_deg,
        "anchored_fraction_shared_decode": anchored_shared,
        "anchored_fraction_calibrated_r": anchored_calibrated,
        "anchored_fraction_identical_REQUIRED": anchored_shared == anchored_calibrated,
    }


# ---------------------------------------------------------------------------
# --compare-against parity metrics (Task 3 of the hybrid-device shim
# campaign). Pure function: raw GVHMR output arrays in, {metric: {value,
# threshold, pass, tier}} out -- unit-testable without a GPU, a checkpoint,
# or file IO (see tests/test_bench_gvhmr_parity_metrics.py).
#
# REQUIRED metrics gate the run (their "pass" bools are ANDed by
# _overall_parity_pass); ADVISORY metrics are reported for visibility only
# -- "threshold"/"pass" are left None and never affect the gate.
#
# thetas comparisons EXCLUDE joint 0 (SMPL root): thetas[0] is ignored by
# the FK path (root_R_cam alone carries root world orientation -- see
# CLAUDE.md's "SMPL FK root orientation" note), so a delta confined to
# joint 0 is not a real pose regression.
# ---------------------------------------------------------------------------

_PARITY_KP2D_COORD_MAX_DELTA_PX = 0.5
_PARITY_KP2D_CONF_MAX_DELTA = 0.02
_PARITY_THETA_EXCL_ROOT_MEAN_ABS_RAD = 1e-3
_PARITY_THETA_EXCL_ROOT_MAX_ABS_RAD = 1e-2
_PARITY_ROOT_R_MEAN_ANGULAR_DELTA_DEG = 0.5
_PARITY_ANCHORED_FRACTION_ABS_DELTA = 0.01


def _parity_entry(
    value: float | bool, threshold: float | None, passed: bool | None, tier: str
) -> dict[str, object]:
    return {"value": value, "threshold": threshold, "pass": passed, "tier": tier}


def compute_parity_metrics(
    candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    """Compare one GVHMR raw-output dict (``candidate``, e.g. an
    ``--extractor-device mps`` run) against a ``reference`` dict (e.g. a
    prior ``--extractor-device cpu`` baseline) with the same schema as
    ``run_on_track``'s return value: ``thetas`` (N,24,3), ``betas``
    (N,10), ``root_R_cam`` (N,3,3), ``root_t_cam`` (N,3), ``kp2d``
    (N,17,3).

    Pure function -- no file IO, no torch. Returns one entry per metric
    keyed by name; each entry has ``value``/``threshold``/``pass``/
    ``tier`` (``"required"`` or ``"advisory"``). Use
    ``_overall_parity_pass`` to reduce to a single gate bool.
    """
    cand_thetas = np.asarray(candidate["thetas"], dtype=np.float64)
    ref_thetas = np.asarray(reference["thetas"], dtype=np.float64)
    cand_betas = np.asarray(candidate["betas"], dtype=np.float64)
    ref_betas = np.asarray(reference["betas"], dtype=np.float64)
    cand_root_R = np.asarray(candidate["root_R_cam"], dtype=np.float64)
    ref_root_R = np.asarray(reference["root_R_cam"], dtype=np.float64)
    cand_root_t = np.asarray(candidate["root_t_cam"], dtype=np.float64)
    ref_root_t = np.asarray(reference["root_t_cam"], dtype=np.float64)
    cand_kp2d = np.asarray(candidate["kp2d"], dtype=np.float64)
    ref_kp2d = np.asarray(reference["kp2d"], dtype=np.float64)

    metrics: dict[str, dict[str, object]] = {}

    # ---- REQUIRED ----
    kp2d_coord_delta = np.abs(cand_kp2d[..., :2] - ref_kp2d[..., :2])
    kp2d_coord_max = float(np.max(kp2d_coord_delta)) if kp2d_coord_delta.size else 0.0
    metrics["kp2d_coord_max_delta_px"] = _parity_entry(
        kp2d_coord_max,
        _PARITY_KP2D_COORD_MAX_DELTA_PX,
        bool(kp2d_coord_max < _PARITY_KP2D_COORD_MAX_DELTA_PX),
        "required",
    )

    kp2d_conf_delta = np.abs(cand_kp2d[..., 2] - ref_kp2d[..., 2])
    kp2d_conf_max = float(np.max(kp2d_conf_delta)) if kp2d_conf_delta.size else 0.0
    metrics["kp2d_conf_max_delta"] = _parity_entry(
        kp2d_conf_max,
        _PARITY_KP2D_CONF_MAX_DELTA,
        bool(kp2d_conf_max < _PARITY_KP2D_CONF_MAX_DELTA),
        "required",
    )

    theta_delta_excl_root = np.abs(cand_thetas[:, 1:, :] - ref_thetas[:, 1:, :])
    theta_mean_abs_rad = (
        float(np.mean(theta_delta_excl_root)) if theta_delta_excl_root.size else 0.0
    )
    theta_max_abs_rad = (
        float(np.max(theta_delta_excl_root)) if theta_delta_excl_root.size else 0.0
    )
    metrics["thetas_excl_joint0_mean_abs_rad"] = _parity_entry(
        theta_mean_abs_rad,
        _PARITY_THETA_EXCL_ROOT_MEAN_ABS_RAD,
        bool(theta_mean_abs_rad < _PARITY_THETA_EXCL_ROOT_MEAN_ABS_RAD),
        "required",
    )
    metrics["thetas_excl_joint0_max_abs_rad"] = _parity_entry(
        theta_max_abs_rad,
        _PARITY_THETA_EXCL_ROOT_MAX_ABS_RAD,
        bool(theta_max_abs_rad < _PARITY_THETA_EXCL_ROOT_MAX_ABS_RAD),
        "required",
    )

    root_R_angles_deg = _root_R_angular_delta_deg(ref_root_R, cand_root_R)
    root_R_mean_angular_delta_deg = (
        float(np.mean(root_R_angles_deg)) if root_R_angles_deg.size else 0.0
    )
    root_R_max_angular_delta_deg = (
        float(np.max(root_R_angles_deg)) if root_R_angles_deg.size else 0.0
    )
    metrics["root_R_cam_mean_angular_delta_deg"] = _parity_entry(
        root_R_mean_angular_delta_deg,
        _PARITY_ROOT_R_MEAN_ANGULAR_DELTA_DEG,
        bool(root_R_mean_angular_delta_deg < _PARITY_ROOT_R_MEAN_ANGULAR_DELTA_DEG),
        "required",
    )

    anchored_cand = _anchored_fraction(cand_kp2d)
    anchored_ref = _anchored_fraction(ref_kp2d)
    anchored_abs_delta = abs(anchored_cand - anchored_ref)
    metrics["anchored_fraction_abs_delta"] = _parity_entry(
        anchored_abs_delta,
        _PARITY_ANCHORED_FRACTION_ABS_DELTA,
        bool(anchored_abs_delta <= _PARITY_ANCHORED_FRACTION_ABS_DELTA),
        "required",
    )

    all_finite = bool(
        np.all(np.isfinite(cand_thetas)) and np.all(np.isfinite(ref_thetas))
        and np.all(np.isfinite(cand_betas)) and np.all(np.isfinite(ref_betas))
        and np.all(np.isfinite(cand_root_R)) and np.all(np.isfinite(ref_root_R))
        and np.all(np.isfinite(cand_root_t)) and np.all(np.isfinite(ref_root_t))
        and np.all(np.isfinite(cand_kp2d)) and np.all(np.isfinite(ref_kp2d))
    )
    metrics["all_finite"] = _parity_entry(all_finite, None, all_finite, "required")

    # ---- ADVISORY (report only -- never gates) ----
    root_t_delta = np.abs(cand_root_t - ref_root_t)
    root_t_mean_abs_delta = float(np.mean(root_t_delta)) if root_t_delta.size else 0.0
    metrics["root_t_cam_mean_abs_delta"] = _parity_entry(
        root_t_mean_abs_delta, None, None, "advisory"
    )

    betas_delta = np.abs(cand_betas - ref_betas)
    betas_max_abs_delta = float(np.max(betas_delta)) if betas_delta.size else 0.0
    metrics["betas_max_abs_delta"] = _parity_entry(betas_max_abs_delta, None, None, "advisory")

    metrics["root_R_cam_max_angular_delta_deg"] = _parity_entry(
        root_R_max_angular_delta_deg, None, None, "advisory"
    )

    return metrics


def _overall_parity_pass(metrics: dict[str, dict[str, object]]) -> bool:
    """AND of every REQUIRED metric's ``pass`` bool. ADVISORY metrics
    never participate -- a bad advisory number is reported, not gated."""
    return all(
        bool(entry["pass"]) for entry in metrics.values() if entry.get("tier") == "required"
    )


class _WarningLineCapture(logging.Handler):
    """Collects formatted WARNING+ log records emitted anywhere during a
    ``with`` block (installed on the root logger), so bench_summary.json
    can report a scan-count of fallback/warning lines alongside the
    estimator's own ``extractor_fallback_count`` counter -- the counter
    tracks fallback EVENTS the estimator itself triggered; this counts
    every warning-level line actually logged (fallback lines included,
    since ``_run_extractor_phase`` logs its fallback via
    ``logger.warning``), which also surfaces any other warning noise
    (e.g. from vendored GVHMR/ViTPose code) that wouldn't bump the
    estimator's own counter.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        self.lines.append(self.format(record))


def _gvhmr_model_param_device(estimator: GVHMREstimator) -> str | None:
    """Best-effort read of the loaded GVHMR transformer's parameter
    device -- crash-safety evidence for the hybrid-device shim (the
    transformer's RoPE implementation SIGABRTs on MPS, so this is
    expected to read "cpu" whenever --device is cpu regardless of
    --extractor-device). Returns None if the model never loaded (e.g.
    every mode failed before the first estimate_sequence() call)."""
    model = getattr(estimator, "_model", None)
    if model is None:
        return None
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in _MODES:
            raise SystemExit(f"unknown mode {m!r}, must be one of {_MODES}")
    players = [p.strip() for p in args.players.split(",") if p.strip()]
    if not players:
        raise SystemExit("--players must name at least one player_id")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = _REPO_ROOT / checkpoint

    video_path = args.output / "shots" / f"{args.shot}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"shot clip not found at {video_path}")

    cam = _load_shot_camera(args.output, args.shot)
    per_frame_K = {f.frame: np.array(f.K, dtype=float) for f in cam.frames}
    per_frame_R = {f.frame: np.array(f.R, dtype=float) for f in cam.frames}

    warm_s = _warm_page_cache(video_path)
    print(f"[bench] warmed page cache for {video_path.name} in {warm_s:.3f}s", flush=True)

    # One estimator for the whole run — the GVHMR + ViTPose-Huge +
    # HMR2-ViT + SMPLX load happens once, lazily, on the first
    # estimate_sequence() call below.
    estimator = GVHMREstimator(
        checkpoint=str(checkpoint), device=args.device, extractor_device=args.extractor_device,
    )

    results: dict[str, dict[str, dict]] = {}
    raw_outputs: dict[tuple[str, str], dict] = {}

    warning_capture = _WarningLineCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(warning_capture)
    try:
        run_start = time.time()
        for player_id in players:
            track_frames = _load_track_frames(args.output, args.shot, player_id)
            if args.max_frames is not None:
                track_frames = track_frames[: args.max_frames]
            dense_K = _build_dense_K(track_frames, per_frame_K)

            results[player_id] = {}
            for mode in modes:
                print(
                    f"[bench] running {player_id} / {mode} "
                    f"({len(track_frames)} frames)...",
                    flush=True,
                )
                t0 = time.time()
                r = _run_mode(
                    mode,
                    estimator=estimator,
                    track_frames=track_frames,
                    video_path=video_path,
                    checkpoint=checkpoint,
                    device=args.device,
                    max_seq_len=args.max_seq_len,
                    per_frame_K=dense_K,
                    per_frame_R_dict=per_frame_R,
                )
                print(
                    f"[bench]   {player_id} / {mode}: {r['wall_s']:.2f}s wall "
                    f"({time.time() - t0:.2f}s incl. bench overhead)",
                    flush=True,
                )

                raw_outputs[(player_id, mode)] = r["out"]
                results[player_id][mode] = {
                    "wall_s": r["wall_s"],
                    "n_frames": r["n_frames"],
                    "n_chunks": r["n_chunks"],
                    "breakdown_s": r["breakdown_s"],
                }

                out_dir = args.scratch / player_id / mode
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez(out_dir / "arrays.npz", **r["out"])

        print(f"[bench] all runs done in {time.time() - run_start:.1f}s total", flush=True)
    finally:
        root_logger.removeHandler(warning_capture)

    model_param_device = _gvhmr_model_param_device(estimator)

    # ---- In-harness comparisons ----
    comparisons: dict[str, dict] = {}
    for player_id in players:
        player_comp: dict = {}
        legacy_out = raw_outputs.get((player_id, "legacy"))
        shared_out = raw_outputs.get((player_id, "shared_decode"))
        calibrated_out = raw_outputs.get((player_id, "calibrated_r"))

        if legacy_out is not None and shared_out is not None:
            player_comp["legacy_vs_shared_decode"] = _compare_legacy_vs_shared(
                legacy_out, shared_out
            )
        if shared_out is not None and calibrated_out is not None:
            player_comp["shared_decode_vs_calibrated_r"] = _compare_shared_vs_calibrated(
                shared_out, calibrated_out
            )
        comparisons[player_id] = player_comp

    # ---- Speedups ----
    speedups: dict[str, dict] = {}
    for player_id in players:
        pr = results[player_id]
        entry: dict[str, float | None] = {}
        if "legacy" in pr and "shared_decode" in pr and pr["shared_decode"]["wall_s"] > 0:
            entry["legacy_to_shared_decode"] = pr["legacy"]["wall_s"] / pr["shared_decode"]["wall_s"]
        if "legacy" in pr and "calibrated_r" in pr and pr["calibrated_r"]["wall_s"] > 0:
            entry["legacy_to_calibrated_r"] = pr["legacy"]["wall_s"] / pr["calibrated_r"]["wall_s"]
        speedups[player_id] = entry

    # ---- --compare-against parity (Task 3) ----
    parity: dict[str, dict] = {}
    if args.compare_against is not None:
        for player_id in players:
            candidate_out = raw_outputs.get((player_id, "calibrated_r"))
            if candidate_out is None:
                parity[player_id] = {
                    "error": (
                        "this run has no 'calibrated_r' mode output for "
                        f"{player_id!r} to compare (modes run: {modes})"
                    )
                }
                continue
            ref_path = args.compare_against / player_id / "calibrated_r" / "arrays.npz"
            if not ref_path.exists():
                parity[player_id] = {"error": f"reference not found at {ref_path}"}
                continue
            with np.load(ref_path) as npz:
                reference_out = {k: npz[k] for k in npz.files}
            metrics = compute_parity_metrics(candidate_out, reference_out)
            parity[player_id] = {
                "reference_path": str(ref_path),
                "metrics": metrics,
                "overall_required_PASS": _overall_parity_pass(metrics),
            }

    summary = {
        "args": {
            "output": str(args.output),
            "shot": args.shot,
            "players": players,
            "modes": modes,
            "device": args.device,
            "extractor_device": args.extractor_device,
            "max_seq_len": args.max_seq_len,
            "max_frames": args.max_frames,
            "checkpoint": str(checkpoint),
            "compare_against": str(args.compare_against) if args.compare_against else None,
        },
        "page_cache_warm_s": warm_s,
        "results": results,
        "comparisons": comparisons,
        "speedups": speedups,
        "parity": parity,
        "estimator": {
            "device": args.device,
            "extractor_device_requested": args.extractor_device,
            "resolved_extractor_device": estimator.resolved_extractor_device,
            "extractor_fallback_count": estimator.extractor_fallback_count,
            "fallback_warning_log_line_count": len(warning_capture.lines),
            "fallback_warning_log_lines": warning_capture.lines[:50],
            # Crash-safety evidence for the hybrid-device shim: the main
            # GVHMR transformer's parameter device (expected "cpu" given
            # its RoPE implementation SIGABRTs on MPS -- see
            # _gvhmr_model_param_device's docstring). None if the model
            # never loaded (e.g. every mode errored before the first
            # estimate_sequence() call).
            "model_param_device": model_param_device,
        },
    }
    summary_path = args.scratch / "bench_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[bench] wrote {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

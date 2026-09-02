"""Foot-contact locomotion quality metrics — the eval harness [A] of
docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md.

Quantifies penetration, flight preservation, stance-foot skate, and
(optionally) image fidelity so every later change (contact-aware
anchoring in hmr_world, the foot-lock finale in refined_poses) can be
measured before/after on the same yardstick. numpy/scipy only — no
torch, so this runs on the Mac dev box without the GPU box.

Skate is measured only from WITHIN a stance span/run (consecutive
frame pairs both inside the same contiguous stance run), never across
a stance/swing boundary — a foot's velocity is not smooth there (it is
frozen during stance and moves during swing), so a naive whole-track
central difference would report a spurious spike at every span edge
even for a perfectly-planted foot.
"""

from __future__ import annotations

import numpy as np

from src.utils.smpl_skeleton import SMPL_JOINT_NAMES, compute_all_joint_worlds_batch

_SOLE_CLEARANCE_M = 0.025
_FLIGHT_THRESHOLD_M = 0.05
_LOW_FOOT_THRESHOLD_M = 0.10
_ANKLE_CONF_MIN = 0.5

_FOOT_IDX = (SMPL_JOINT_NAMES.index("l_foot"), SMPL_JOINT_NAMES.index("r_foot"))
_ANKLE_IDX = (SMPL_JOINT_NAMES.index("l_ankle"), SMPL_JOINT_NAMES.index("r_ankle"))
# COCO-17 ankle keypoint indices (left, right).
_COCO_ANKLE_IDX = (15, 16)
_SIDE_NAMES = ("L", "R")


def _runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs in a 1-D boolean array, as [start, end) pairs."""
    n = int(mask.shape[0])
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _within_run_speeds(pos_xy: np.ndarray, runs: list[tuple[int, int]], fps: float) -> np.ndarray:
    """Per-consecutive-frame-pair XY speeds, computed only between frames
    that are both inside the same run (see module docstring)."""
    samples: list[float] = []
    for a, b in runs:
        if b - a < 2:
            continue
        seg = pos_xy[a:b]
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1) * float(fps)
        samples.extend(d.tolist())
    return np.array(samples, dtype=float)


def _within_run_path_lengths(pos_xy: np.ndarray, runs: list[tuple[int, int]]) -> list[float]:
    lengths: list[float] = []
    for a, b in runs:
        if b - a < 2:
            lengths.append(0.0)
            continue
        seg = pos_xy[a:b]
        lengths.append(float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))))
    return lengths


def _stat_block(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean_mps": 0.0, "p50_mps": 0.0, "p95_mps": 0.0}
    return {
        "mean_mps": float(np.mean(values)),
        "p50_mps": float(np.percentile(values, 50)),
        "p95_mps": float(np.percentile(values, 95)),
    }


def _project_pinhole(K: np.ndarray, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Pinhole-only world->image projection (no distortion). v1 assumes
    the caller has already undistorted kp2d, per the design doc."""
    cam = pts @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float)
    uv = cam[:, :2] / cam[:, 2:3]
    return uv @ np.asarray(K, dtype=float)[:2, :2].T + np.asarray(K, dtype=float)[:2, 2]


def foot_quality_metrics(
    *,
    frames: np.ndarray,
    betas: np.ndarray,
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    fps: float,
    contacts: np.ndarray | None = None,
    kp2d: np.ndarray | None = None,
    cameras: dict | None = None,
    rest_joints: np.ndarray | None = None,
    sole_clearance_m: float = _SOLE_CLEARANCE_M,
) -> dict:
    """Compute foot-contact locomotion quality metrics for one track.

    Args:
        frames: (F,) frame indices (used only to look up ``cameras``).
        betas/thetas/root_R/root_t: SMPL track arrays, same conventions
            as the rest of the pipeline (thetas[:, 0] ignored).
        fps: clip frame rate, drives the skate (m/s) conversion.
        contacts: optional (F, 2) bool [L, R] — when given, skate/spans
            are measured within these contact spans; when ``None``,
            "foot z < 0.10 m" is used as the stance proxy instead.
        kp2d: optional (F, 17, 3) COCO-17 keypoints (u, v, conf).
        cameras: optional ``{frame: (K, R, t)}`` — when both ``kp2d``
            and ``cameras`` are given, ``ankle_reproj_px`` is computed;
            projection is pinhole-only (v1), so callers on a distorted
            lens must undistort kp2d themselves first.
        rest_joints: optional (24, 3) beta-adjusted rest joint override
            (see ``src.utils.smpl_skeleton.beta_adjusted_rest_joints``).
        sole_clearance_m: sole-proxy offset below the foot joint used
            for the penetration metric (mesh sole sits below the joint).

    Returns a dict with keys ``penetration``, ``lower_foot_z``, ``skate``,
    ``spans``, ``flight``, ``contact_ratio``, and (only when kp2d+cameras
    are both given) ``ankle_reproj_px``.
    """
    n = int(np.asarray(frames).shape[0])
    if n == 0:
        return {
            "penetration": {
                "pct_frames_sole_below_0": 0.0,
                "max_depth_cm": 0.0,
                "mean_depth_cm": 0.0,
            },
            "lower_foot_z": {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0},
            "skate": {
                side: {"mean_mps": 0.0, "p50_mps": 0.0, "p95_mps": 0.0}
                for side in _SIDE_NAMES
            },
            "spans": {"count": 0, "mean_m": 0.0, "max_m": 0.0},
            "flight": {"pct_frames_both_up": 0.0},
            "contact_ratio": 0.0,
        }

    fw = compute_all_joint_worlds_batch(thetas, root_R, root_t, rest_joints)
    feet_pos = fw[:, _FOOT_IDX, :]          # (F, 2, 3)
    feet_z = feet_pos[:, :, 2]              # (F, 2)
    lower_foot_z = feet_z.min(axis=1)       # (F,)

    # --- penetration ------------------------------------------------
    sole_z = lower_foot_z - float(sole_clearance_m)
    below = sole_z < 0.0
    depths_cm = np.clip(-sole_z, 0.0, None) * 100.0
    penetration = {
        "pct_frames_sole_below_0": float(100.0 * below.mean()),
        "max_depth_cm": float(depths_cm.max()),
        "mean_depth_cm": float(depths_cm.mean()),
    }

    # --- lower_foot_z -------------------------------------------------
    lower_foot_z_stats = {
        "mean": float(np.mean(lower_foot_z)),
        "p05": float(np.percentile(lower_foot_z, 5)),
        "p50": float(np.percentile(lower_foot_z, 50)),
        "p95": float(np.percentile(lower_foot_z, 95)),
    }

    # --- per-side stance masks -> runs --------------------------------
    if contacts is not None:
        contacts_arr = np.asarray(contacts, dtype=bool)
        side_masks = [contacts_arr[:, 0], contacts_arr[:, 1]]
        contact_ratio = float(contacts_arr.any(axis=1).mean())
        both_up = ~contacts_arr.any(axis=1)
    else:
        side_masks = [feet_z[:, 0] < _LOW_FOOT_THRESHOLD_M, feet_z[:, 1] < _LOW_FOOT_THRESHOLD_M]
        any_low = side_masks[0] | side_masks[1]
        contact_ratio = float(any_low.mean())
        both_up = (feet_z[:, 0] > _FLIGHT_THRESHOLD_M) & (feet_z[:, 1] > _FLIGHT_THRESHOLD_M)

    skate: dict = {}
    all_runs: list[tuple[int, int]] = []
    all_lengths: list[float] = []
    for side, name in enumerate(_SIDE_NAMES):
        runs = _runs_from_mask(side_masks[side])
        all_runs.extend(runs)
        all_lengths.extend(_within_run_path_lengths(feet_pos[:, side, :2], runs))
        speeds = _within_run_speeds(feet_pos[:, side, :2], runs, fps)
        skate[name] = _stat_block(speeds)

    spans = {
        "count": len(all_runs),
        "mean_m": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "max_m": float(np.max(all_lengths)) if all_lengths else 0.0,
    }

    flight = {"pct_frames_both_up": float(100.0 * both_up.mean())}

    out: dict = {
        "penetration": penetration,
        "lower_foot_z": lower_foot_z_stats,
        "skate": skate,
        "spans": spans,
        "flight": flight,
        "contact_ratio": contact_ratio,
    }

    # --- optional ankle reprojection error -----------------------------
    if kp2d is not None and cameras is not None:
        kp2d_arr = np.asarray(kp2d, dtype=float)
        frames_arr = np.asarray(frames)
        errs: list[float] = []
        for i, f in enumerate(frames_arr):
            cam = cameras.get(int(f))
            if cam is None:
                continue
            K, R, t = cam
            for side, ankle_idx, coco_idx in zip(
                (0, 1), _ANKLE_IDX, _COCO_ANKLE_IDX
            ):
                conf = float(kp2d_arr[i, coco_idx, 2])
                if conf < _ANKLE_CONF_MIN:
                    continue
                world_pt = fw[i, ankle_idx][None, :]
                proj = _project_pinhole(K, R, t, world_pt)[0]
                obs = kp2d_arr[i, coco_idx, :2]
                errs.append(float(np.linalg.norm(proj - obs)))
        errs_arr = np.array(errs, dtype=float)
        out["ankle_reproj_px"] = (
            {
                "mean_px": float(np.mean(errs_arr)),
                "p95_px": float(np.percentile(errs_arr, 95)),
            }
            if errs_arr.size
            else {"mean_px": 0.0, "p95_px": 0.0}
        )

    return out

"""Per-player post-HMR cleanup stage.

For each ``(shot, player_id)`` track emitted by ``hmr_world`` this stage:

  1. Rejects root-rotation outliers (90-180° "flip" frames produced
     by monocular HMR's near-frontal-pose ambiguity).
  2. Trims leading/trailing un-anchored frames so the viewer doesn't
     render a "ghost" interpolating from (0, 0) on frames before the
     player was first detected.

For each ``player_id`` with one contributing shot, the cleaned track is
mapped onto the shared reference timeline (via ``sync_map`` offset) and
saved as ``output/refined_poses/{player_id}_refined.npz``.

For ``player_id``s appearing in multiple shots, a simple per-frame
highest-confidence pick assembles the fused track on the reference
timeline. Real cross-shot fusion math (chordal mean over rotations,
MAD outlier rejection on translations) was removed in this rewrite —
it's expected to return as a separate concern when multi-camera work
resumes.

The schema (``src.schemas.refined_pose.RefinedPose``) and file naming
(``{player_id}_refined.npz`` on the reference timeline) are
unchanged, so ``src/stages/export.py`` and the dashboard endpoints
continue to consume refined_poses output without modification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.refined_pose import (
    RefinedPose,
    RefinedPoseDiagnostics,
)
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import SyncMap
from src.utils.pose_fusion import so3_chordal_mean, so3_geodesic_distance
from src.utils.smpl_skeleton import (
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    SMPL_REST_JOINTS_YUP,
    axis_angle_to_matrix,
)
from src.utils.temporal_smoothing import savgol_axis, slerp_window

logger = logging.getLogger(__name__)

# Mirror of the hmr_world constant — matches the ankle-confidence cutoff
# below which a track frame never received a fresh foot-anchor ray-cast.
# Frames whose confidence is below this are candidates for the
# leading/trailing trim.
_FRESH_ANCHOR_CONF = 0.3

# SMPL canonical ankle offset from the pelvis (in y-up canonical space),
# used as the pivot for the lean-toward-vertical correction so the
# player's foot stays on the pitch when the body is rotated. Mirrors
# ``_ANKLE_IN_ROOT`` in ``hmr_world.py``: left/right ankle midpoint
# with the lateral component zeroed.
_ANKLE_IN_ROOT = 0.5 * (
    SMPL_REST_JOINTS_YUP[7] + SMPL_REST_JOINTS_YUP[8]
).astype(float)
_ANKLE_IN_ROOT[0] = 0.0

# SMPL joint indices for the foot tips (one below each ankle). Used by
# the ground-snap pass to find the lower foot per frame.
_L_FOOT_IDX = SMPL_JOINT_NAMES.index("l_foot")
_R_FOOT_IDX = SMPL_JOINT_NAMES.index("r_foot")


def _load_smpl_neutral_model() -> dict | None:
    """Load the SMPL neutral shape data so we can beta-adjust the rest
    joint table per player. Returns ``None`` when the file is absent
    (e.g. CI without ``data/models/smpl_neutral.npz``) — callers must
    fall back to the constant ``SMPL_REST_JOINTS_YUP`` in that case.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "data" / "models" / "smpl_neutral.npz"
    )
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return None
    out: dict = {"joint_positions": np.asarray(z["joint_positions"])}
    if "joint_shapedirs" in z.files:
        out["joint_shapedirs"] = np.asarray(z["joint_shapedirs"])
    return out


def _beta_adjusted_rest_joints(
    betas: np.ndarray | None, smpl_model: dict | None,
) -> np.ndarray:
    """Build a (24, 3) pelvis-relative rest joint table for one player.

    Without ``smpl_model`` (file missing), returns the constant
    ``SMPL_REST_JOINTS_YUP`` so callers still get something usable.

    With ``smpl_model`` and ``betas``, applies the per-shape
    ``joint_shapedirs`` delta on top of the neutral joint positions,
    then shifts the whole table so the pelvis joint sits at the
    origin. This matches the canonical convention used by the FK
    routines and yields the player's actual leg length, fixing the
    ~8-10 cm gap between mean-betas canonical feet and beta-shaped
    mesh feet that left players floating above the pitch.
    """
    if smpl_model is None:
        return np.asarray(SMPL_REST_JOINTS_YUP, dtype=float)
    jp = np.asarray(smpl_model["joint_positions"], dtype=float).copy()
    jsd = smpl_model.get("joint_shapedirs")
    if jsd is not None and betas is not None:
        betas = np.asarray(betas, dtype=float).reshape(-1)
        K = min(jsd.shape[2], len(betas))
        if K > 0:
            jp = jp + jsd[:, :, :K] @ betas[:K]
    # Shift so pelvis is at origin (matches src table convention).
    return jp - jp[0]


def _reject_root_R_outliers(
    root_R: np.ndarray,
    *,
    max_rotation_deg: float = 45.0,
    anchor_gap: int = 5,
    anchor_window: int = 3,
    max_passes: int = 3,
) -> np.ndarray:
    """Replace per-frame root rotation outliers with neighbour SO(3) mean.

    HMR networks produce sporadic large rotation errors on goalkeepers
    and near-frontal poses — sometimes a clean 180° front/back flip,
    sometimes a 90-110° "quarter / half-flip" that snaps the body's
    facing direction by an implausible amount in a single frame and
    snaps back a few frames later.

    Detection compares each frame *i*'s rotation to two **anchor
    windows** placed ``anchor_gap`` frames before and after *i* —
    deliberately wider than any expected error-run length so the
    anchors fall safely outside the bad region (e.g. for a 4-frame
    flip run, anchors at i±5 sit in the clean before/after stretches).

    A frame is replaced when:
      - Its geodesic distance to the anchor SO(3) mean exceeds
        ``max_rotation_deg``.
      - The anchors themselves are temporally consistent (median
        spread from their mean is below ``max_rotation_deg``), so we
        know we're looking at outliers inside a stable run rather
        than a genuine fast turn.

    A genuine fast turn (e.g. a goalkeeper spinning to face a shot)
    is preserved because the anchors span a portion of the turn,
    their spread from their own mean is large, the consistency check
    fails, and no replacement is made.

    Iterates up to ``max_passes`` so an outlier replaced on pass 1
    can stabilise the anchors for the next outlier on pass 2.
    """
    n = root_R.shape[0]
    if n < 2 * anchor_gap + anchor_window + 1:
        return root_R
    out = root_R.copy()
    thresh = float(np.deg2rad(max_rotation_deg))

    for _pass in range(max_passes):
        snapshot = out.copy()
        fixed_any = False
        for i in range(n):
            left_lo = max(0, i - anchor_gap - anchor_window + 1)
            left_hi = max(0, i - anchor_gap + 1)
            right_lo = min(n, i + anchor_gap)
            right_hi = min(n, i + anchor_gap + anchor_window)
            left = list(range(left_lo, left_hi))
            right = list(range(right_lo, right_hi))
            # Require anchors on BOTH sides so we can bracket *i* with
            # temporal context. Without this, the first few frames of a
            # genuine fast turn (where the left-anchor window doesn't
            # exist yet) would get flagged as outliers because the right
            # anchors are already deep into the turn and consistent with
            # each other — the consistency check would pass and the
            # frame would be wrongly replaced.
            if not left or not right:
                continue
            anchors = left + right
            if len(anchors) < 3:
                continue
            anchor_Rs = np.stack([snapshot[j] for j in anchors])
            weights = np.ones(len(anchors), dtype=float)
            R_mean = so3_chordal_mean(anchor_Rs, weights)
            spreads = [
                float(so3_geodesic_distance(snapshot[j], R_mean))
                for j in anchors
            ]
            if float(np.median(spreads)) >= thresh:
                # Anchors disagree among themselves (genuine fast turn,
                # or the window straddles two distinct stable regimes):
                # don't risk replacing this frame.
                continue
            if float(so3_geodesic_distance(snapshot[i], R_mean)) > thresh:
                out[i] = R_mean
                fixed_any = True
        if not fixed_any:
            break
    return out


def _reduce_root_lean(
    root_R: np.ndarray,
    root_t: np.ndarray,
    *,
    correction_factor: float = 0.7,
    max_lean_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Tilt each frame's body toward pitch +z, pivoting around the ankle.

    The monocular HMR's lean-away-from-camera bias survives even after
    feeding the calibrated per-frame K into GVHMR — bodies still tilt
    a few degrees off vertical. This pass measures the angle between
    the body's up axis (SMPL canonical +y, lifted through ``root_R``
    into pitch coords) and pitch +z, and rotates by
    ``correction_factor`` of that angle around the horizontal axis
    perpendicular to the tilt plane.

    The pivot is the SMPL canonical ankle position rather than the
    pelvis, so the foot stays on the pitch and only the upper body
    moves. ``root_t`` is updated in lockstep with ``root_R`` to keep
    the rotation consistent with that pivot.

    Leans larger than ``max_lean_deg`` are left untouched on the
    assumption they're deliberate (a goalkeeper diving, a player on
    the ground after a tackle). Returns ``(root_R, root_t)`` with the
    same shape and dtype as the inputs.
    """
    n = root_R.shape[0]
    if n == 0:
        return root_R.copy(), root_t.copy()
    out_R = np.asarray(root_R, dtype=float).copy()
    out_t = np.asarray(root_t, dtype=float).copy()
    z_up = np.array([0.0, 0.0, 1.0])
    max_lean_rad = float(np.deg2rad(max_lean_deg))

    for i in range(n):
        R = out_R[i]
        t = out_t[i]
        body_up = R[:, 1]
        bu_norm = float(np.linalg.norm(body_up))
        if bu_norm < 1e-9:
            continue
        bu = body_up / bu_norm
        cos_a = float(np.clip(float(np.dot(bu, z_up)), -1.0, 1.0))
        if cos_a >= 1.0 - 1e-9:
            continue  # already upright
        angle = float(np.arccos(cos_a))
        if angle > max_lean_rad:
            continue
        axis = np.cross(bu, z_up)
        axis_n = float(np.linalg.norm(axis))
        if axis_n < 1e-9:
            # bu == -z_up (body upside-down) — undefined axis, leave it.
            continue
        axis = axis / axis_n
        correction_angle = angle * float(correction_factor)
        ca = float(np.cos(correction_angle))
        sa = float(np.sin(correction_angle))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        correction = np.eye(3) + sa * K + (1.0 - ca) * (K @ K)
        # Pivot around the ankle so the foot stays in place. The ankle
        # offset is in SMPL canonical y-up; ``R @ _ANKLE_IN_ROOT`` lifts
        # it into pitch-world.
        ankle_world = t + R @ _ANKLE_IN_ROOT
        out_t[i] = ankle_world + correction @ (t - ankle_world)
        out_R[i] = correction @ R
    return out_R, out_t


def _foot_world_zs(
    theta: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    *,
    rest_joints: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return ``(l_foot_z, r_foot_z)`` in pitch coords using the
    viewer's FK convention: ``thetas[0]`` is ignored (``root_R``
    already carries canonical-y-up → pitch z-up), per-joint
    ``thetas[1..]`` are applied as local rotations down the SMPL
    hierarchy.

    ``rest_joints`` overrides the canonical rest pose joint table
    (defaults to mean-betas ``SMPL_REST_JOINTS_YUP``). Supply the
    player's beta-adjusted rest joints so the FK matches the actual
    mesh leg length — otherwise the snap pins a canonical foot that
    sits 8-10 cm below the mesh's beta-shaped foot for typical
    players, leaving feet visibly floating in the viewer.

    Only the joints on the leg chain are visited (pelvis → hip →
    knee → ankle → foot), so this is cheap to run per frame even on
    long tracks.
    """
    rest = (
        SMPL_REST_JOINTS_YUP if rest_joints is None
        else np.asarray(rest_joints)
    )
    # Cache for joints we'll need: 0 (pelvis), 1/2 (hips), 4/5 (knees),
    # 7/8 (ankles), 10/11 (feet). Walk the chain twice (l and r side).
    rot_cache: dict[int, np.ndarray] = {0: np.asarray(root_R, dtype=float)}
    pos_cache: dict[int, np.ndarray] = {0: np.asarray(root_t, dtype=float)}

    def _walk(target: int) -> np.ndarray:
        chain: list[int] = []
        cur = target
        while cur not in pos_cache:
            chain.append(cur)
            cur = SMPL_PARENTS[cur]
        for j in reversed(chain):
            par = SMPL_PARENTS[j]
            Rl = axis_angle_to_matrix(theta[j])
            rot_cache[j] = rot_cache[par] @ Rl
            offset = rest[j] - rest[par]
            pos_cache[j] = pos_cache[par] + rot_cache[par] @ offset
        return pos_cache[target]

    l = _walk(_L_FOOT_IDX)
    r = _walk(_R_FOOT_IDX)
    return float(l[2]), float(r[2])


def _ground_snap(
    root_R: np.ndarray,
    root_t: np.ndarray,
    thetas: np.ndarray,
    *,
    target_foot_z: float = 0.02,
    max_snap_distance: float = 0.30,
    rest_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Shift each frame's ``root_t.z`` so the lower foot joint sits at
    ``target_foot_z`` above the pitch.

    The upstream foot anchor in ``hmr_world`` places the body so its
    **canonical-rest-pose** mid-ankle midpoint lands at z = 0.05. But
    canonical rest pose has both legs straight down; when a player is
    running with one leg raised, the canonical midpoint sits roughly
    halfway between the planted ankle and the raised one, so the
    planted foot ends up floating ~10 cm above the pitch.

    This pass measures the lower foot joint per frame (via FK over the
    leg chain) and slides ``root_t.z`` so that foot lands at the small
    positive offset ``target_foot_z`` (a few cm above z = 0 because
    the SMPL foot *joint* sits above the sole of the mesh).

    Frames where the lower foot is more than ``max_snap_distance``
    above the pitch are assumed airborne (jump, header) and skipped —
    yanking those down would mangle the motion. ``root_R`` and
    ``thetas`` are unchanged; only translation moves.
    """
    n = root_R.shape[0]
    if n == 0 or max_snap_distance <= 0.0:
        return root_t.copy()
    out_t = np.asarray(root_t, dtype=float).copy()
    for i in range(n):
        l_z, r_z = _foot_world_zs(
            thetas[i], root_R[i], out_t[i], rest_joints=rest_joints,
        )
        lowest = min(l_z, r_z)
        if lowest > max_snap_distance:
            continue  # airborne — leave alone
        delta = float(target_foot_z) - lowest
        out_t[i, 2] += delta
    return out_t


def _smooth_track(
    track: SmplWorldTrack,
    *,
    root_R_slerp_window: int = 7,
    root_t_savgol_window: int = 7,
    root_t_savgol_order: int = 2,
    thetas_savgol_window: int = 9,
    thetas_savgol_order: int = 2,
) -> SmplWorldTrack:
    """Run temporal smoothers on ``root_R`` / ``root_t`` / ``thetas``.

    SLERP on ``root_R`` because rotation interpolation in quaternion
    space avoids gimbal-lock edge cases; Savgol on ``root_t`` and
    ``thetas`` because they're scalar signals with predictable
    polynomial structure within the smoothing window.

    The smoothers are intended to run AFTER the trim in
    ``_clean_single_track``, so the input frames are all anchored and
    the edge taps never reach zero-padded un-anchored frames. Each
    window can be set to ``<= 1`` to disable that specific smoother
    (e.g. for tests or to leave a particular signal raw).
    """
    n = int(len(track.frames))
    if n == 0:
        return track

    root_R = np.asarray(track.root_R)
    root_t = np.asarray(track.root_t)
    thetas = np.asarray(track.thetas)

    if root_R_slerp_window > 1 and n >= 3:
        root_R = slerp_window(root_R, window=root_R_slerp_window)
    if root_t_savgol_window > 1 and n >= root_t_savgol_window:
        root_t = savgol_axis(
            root_t,
            window=root_t_savgol_window,
            order=root_t_savgol_order,
            axis=0,
        )
    if thetas_savgol_window > 1 and n >= thetas_savgol_window:
        thetas = savgol_axis(
            thetas,
            window=thetas_savgol_window,
            order=thetas_savgol_order,
            axis=0,
        )

    return SmplWorldTrack(
        player_id=track.player_id,
        frames=track.frames,
        betas=np.asarray(track.betas, dtype=np.float32),
        thetas=thetas.astype(np.float32),
        root_R=root_R.astype(np.float32),
        root_t=root_t.astype(np.float32),
        confidence=np.asarray(track.confidence, dtype=np.float32),
        shot_id=track.shot_id,
    )


def _apply_jitter_correction(
    tracks: list[SmplWorldTrack],
    *,
    fps: float,
    enabled: bool = True,
    min_players: int = 4,
    baseline_window_s: float = 1.0,
    baseline_floor_m: float = 0.05,
    tempo_ratio: float = 2.5,
    max_spread_ratio: float = 0.30,
) -> tuple[list[SmplWorldTrack], dict]:
    """Subtract a per-frame cumulative XY offset from every player's
    ``root_t`` whenever the players' frame-to-frame deltas agree on a
    common pitch-plane shift much larger than the recent scene tempo.

    Assumes all input ``tracks`` share a single shot timeline. The
    consensus is computed by walking the union of frame indices and
    asking, at each frame, what fraction of visible confident players
    moved in the same XY direction with the same magnitude. If at
    least ``min_players`` players have a fresh anchor at frame ``f``
    and ``f-1``, agree (spread ≤ ``max_spread_ratio × |median Δ|``),
    and the median shift is ≥ ``tempo_ratio`` times the recent
    baseline (median |Δ| over a ``baseline_window_s`` window, floored
    at ``baseline_floor_m``), the median Δ is added to the cumulative
    offset for ``f`` and every later frame. Snap-back jitter cancels
    naturally — the reverse-direction Δ subtracts the offset back out.

    Returns the input list with new ``SmplWorldTrack`` instances (the
    schema is frozen so we never mutate in place) plus a stats dict
    suitable for ``refined_poses_summary.json`` and
    ``quality_report.json``.

    XY only — ``z`` is owned by the per-player ground-snap pass and
    is not touched here.
    """
    stats: dict = {
        "corrected_frames": 0,
        "total_frames_evaluated": 0,
        "max_offset_m": 0.0,
        "mean_offset_m": 0.0,
    }
    if not enabled or len(tracks) < min_players:
        return list(tracks), stats

    # Build a frame → {player_idx: (xy, confidence)} index across the
    # union of frames from every track.
    frame_data: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    for pi, tr in enumerate(tracks):
        frames = np.asarray(tr.frames, dtype=np.int64)
        rt_xy = np.asarray(tr.root_t, dtype=np.float64)[:, :2]
        confs = np.asarray(tr.confidence, dtype=np.float64)
        for i, f in enumerate(frames):
            frame_data.setdefault(int(f), {})[pi] = (rt_xy[i], float(confs[i]))
    if not frame_data:
        return list(tracks), stats

    shot_frames = sorted(frame_data.keys())

    # Per-frame consensus median Δ across confident players who have
    # a valid prev-frame sample. Frames with no Δ are simply absent.
    deltas: dict[int, np.ndarray] = {}
    spreads: dict[int, float] = {}
    visible: dict[int, int] = {}
    for f in shot_frames:
        prev = frame_data.get(f - 1)
        if prev is None:
            continue
        cur = frame_data[f]
        ds: list[np.ndarray] = []
        for pi, (xy, c) in cur.items():
            prev_sample = prev.get(pi)
            if prev_sample is None:
                continue
            xy_p, c_p = prev_sample
            if c < _FRESH_ANCHOR_CONF or c_p < _FRESH_ANCHOR_CONF:
                continue
            ds.append(xy - xy_p)
        if not ds:
            continue
        D = np.stack(ds)
        deltas[f] = np.median(D, axis=0)
        # Pythagorean combination of per-axis std — large when players
        # disagree on direction, zero when they march in lockstep.
        spreads[f] = float(np.linalg.norm(np.std(D, axis=0)))
        visible[f] = D.shape[0]

    if not deltas:
        return list(tracks), stats

    half_w = max(1, int(round(baseline_window_s * fps / 2.0)))
    mag_signal = {f: float(np.linalg.norm(d)) for f, d in deltas.items()}

    # Decide per frame which Δ contribute to the cumulative offset.
    triggers: dict[int, np.ndarray] = {}
    for f, d in deltas.items():
        if visible[f] < min_players:
            continue
        # Baseline excludes the frame under test so a single huge
        # spike doesn't inflate its own threshold to the point of
        # missing the snap-back at f+1.
        window_mags = [
            mag_signal[g] for g in mag_signal
            if abs(g - f) <= half_w and g != f
        ]
        baseline = max(
            float(np.median(window_mags)) if window_mags else 0.0,
            baseline_floor_m,
        )
        mag = float(np.linalg.norm(d))
        stats["total_frames_evaluated"] += 1
        if mag < tempo_ratio * baseline:
            continue
        if spreads[f] > max_spread_ratio * mag:
            continue
        triggers[f] = d

    if not triggers:
        return list(tracks), stats

    # Build the cumulative offset signal across every shot frame, then
    # apply it to each player's root_t in place of a copy.
    cum = np.zeros(2, dtype=np.float64)
    offset_at: dict[int, np.ndarray] = {}
    for f in shot_frames:
        if f in triggers:
            cum = cum + triggers[f]
        offset_at[f] = cum.copy()

    new_tracks: list[SmplWorldTrack] = []
    applied_mags: list[float] = []
    for tr in tracks:
        rt = np.asarray(tr.root_t, dtype=np.float32).copy()
        for i, f in enumerate(tr.frames):
            off = offset_at.get(int(f))
            if off is None:
                continue
            rt[i, 0] -= float(off[0])
            rt[i, 1] -= float(off[1])
        new_tracks.append(SmplWorldTrack(
            player_id=tr.player_id,
            frames=tr.frames,
            betas=tr.betas,
            thetas=tr.thetas,
            root_R=tr.root_R,
            root_t=rt,
            confidence=tr.confidence,
            shot_id=tr.shot_id,
        ))

    nonzero = [float(np.linalg.norm(v)) for v in offset_at.values() if np.any(v)]
    stats["corrected_frames"] = len(triggers)
    stats["max_offset_m"] = float(max(nonzero) if nonzero else 0.0)
    stats["mean_offset_m"] = float(np.mean(nonzero) if nonzero else 0.0)
    return new_tracks, stats


def _apply_residual_consensus_correction(
    tracks: list[SmplWorldTrack],
    *,
    fps: float,
    enabled: bool = True,
    savgol_window_s: float = 1.2,
    savgol_poly: int = 2,
    min_players: int = 3,
    min_offset_m: float = 0.02,
    max_offset_m: float = 2.0,
    iterations: int = 2,
) -> tuple[list[SmplWorldTrack], dict]:
    """Subtract a per-frame XY common-mode offset from every player's
    ``root_t``. The offset at frame ``f`` is the cross-player median of
    each visible player's residual against their own heavy per-player
    Savgol-smoothed baseline.

    This catches the camera-tracking artifact the Δ-based
    ``_apply_jitter_correction`` cannot see: a sub-second wobble that
    rides on every player's track. Frame-to-frame deltas are dominated
    by individual player motion (a sprinter at 7 m/s contributes
    ~0.3 m/frame Δ at 25 fps), so the Δ-based detector's consensus
    gate fails on real running scenes. The residual signal sidesteps
    this — each player's heavy Savgol baseline absorbs their own
    trajectory, leaving the common-mode wobble in the residual where
    a per-frame cross-player median isolates it cleanly.

    Assumes all input ``tracks`` share a single shot timeline (callers
    are expected to group by ``shot_id`` upstream — jitter is a per-
    camera-frame artifact and must be measured on each shot's native
    timeline, before sync_map remapping).

    XY only — ``z`` is owned by the per-player ground-snap pass.

    Parameters
    ----------
    savgol_window_s:
        Per-player Savgol window in seconds. Anything shorter than
        ~0.5 s lets short wobbles leak into the baseline and reduces
        the recoverable signal. ~1.2 s leaves running velocity
        unaffected (it's well below the Nyquist of the polynomial)
        while putting all of the wobble band into the residual.
    min_players:
        Minimum confident players contributing to a frame's residual
        before its median is trusted as the consensus.
    min_offset_m:
        Frames whose median residual magnitude falls below this
        floor are not corrected — keeps the pass from chasing the
        per-player HMR-noise floor when there's no actual wobble.
    max_offset_m:
        Safety clamp; offsets above this are treated as suspect and
        skipped (prevents a single bad track from yanking everyone).
    iterations:
        Number of correction passes. A single Savgol baseline leaves
        ~30-40 % of a band-limited wobble in the lowpass — iterating
        cuts that residual geometrically (a second pass strips ~36 %
        of the remainder, a third another ~36 %). Two passes are
        usually enough to push residual error below the per-player
        HMR noise floor.
    """
    stats: dict = {
        "corrected_frames": 0,
        "total_frames_evaluated": 0,
        "max_offset_m": 0.0,
        "mean_offset_m": 0.0,
        "iterations_run": 0,
    }
    if not enabled or len(tracks) < min_players or iterations < 1:
        return list(tracks), stats

    window = max(3, int(round(savgol_window_s * fps)))
    if window % 2 == 0:
        window += 1
    poly = int(savgol_poly)

    raw_rt = [np.asarray(tr.root_t, dtype=np.float64).copy() for tr in tracks]
    confs = [np.asarray(tr.confidence, dtype=np.float64) for tr in tracks]
    frames_lst = [np.asarray(tr.frames, dtype=np.int64) for tr in tracks]
    cumulative: dict[int, np.ndarray] = {}

    for _it in range(int(iterations)):
        # Apply the cumulative offset built up so far before re-fitting
        # the per-player baseline — so the next residual reflects only
        # the wobble that survived previous passes.
        by_frame: dict[int, list[np.ndarray]] = {}
        for rt_raw, conf, frames in zip(raw_rt, confs, frames_lst):
            n = int(len(frames))
            if n < poly + 2:
                continue
            rt = rt_raw[:, :2].copy()
            for i, f in enumerate(frames):
                off = cumulative.get(int(f))
                if off is not None:
                    rt[i] -= off
            base = savgol_axis(rt, window=window, order=poly, axis=0)
            resid = rt - base
            for i, f in enumerate(frames):
                if conf[i] < _FRESH_ANCHOR_CONF:
                    continue
                by_frame.setdefault(int(f), []).append(resid[i])

        if not by_frame:
            break

        any_applied = False
        for f, contribs in by_frame.items():
            if _it == 0:
                stats["total_frames_evaluated"] += 1
            if len(contribs) < min_players:
                continue
            med = np.median(np.stack(contribs), axis=0)
            mag = float(np.linalg.norm(med))
            if mag < min_offset_m or mag > max_offset_m:
                continue
            prev = cumulative.get(f)
            cumulative[f] = med if prev is None else prev + med
            any_applied = True

        stats["iterations_run"] = _it + 1
        if not any_applied:
            break

    if not cumulative:
        return list(tracks), stats

    new_tracks: list[SmplWorldTrack] = []
    for tr, frames in zip(tracks, frames_lst):
        rt = np.asarray(tr.root_t, dtype=np.float32).copy()
        for i, f in enumerate(frames):
            off = cumulative.get(int(f))
            if off is None:
                continue
            rt[i, 0] -= float(off[0])
            rt[i, 1] -= float(off[1])
        new_tracks.append(SmplWorldTrack(
            player_id=tr.player_id,
            frames=tr.frames,
            betas=tr.betas,
            thetas=tr.thetas,
            root_R=tr.root_R,
            root_t=rt,
            confidence=tr.confidence,
            shot_id=tr.shot_id,
        ))

    applied_mags = [float(np.linalg.norm(v)) for v in cumulative.values()]
    stats["corrected_frames"] = len(cumulative)
    stats["max_offset_m"] = float(max(applied_mags))
    stats["mean_offset_m"] = float(np.mean(applied_mags))
    return new_tracks, stats


def _velocity_limit_xy(xy: np.ndarray, max_step: float) -> np.ndarray:
    """Forward/backward displacement-limited filter on a 2-D path.

    Caps each consecutive frame-to-frame displacement at ``max_step``
    metres by running a one-sided limiter forward and backward and
    averaging the two. Averaging keeps the result symmetric (no lag)
    while preserving the ``|Δ| ≤ max_step`` bound — so a genuine sprint
    below the cap passes through untouched and only super-physical jumps
    (HMR depth wobble, tracking teleports) are reined in.
    """
    arr = np.asarray(xy, dtype=float)
    n = arr.shape[0]
    if n < 2 or max_step <= 0.0:
        return arr.copy()

    def _limit(seq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        for i in range(1, len(out)):
            step = out[i] - out[i - 1]
            d = float(np.linalg.norm(step))
            if d > max_step:
                out[i] = out[i - 1] + step * (max_step / d)
        return out

    fwd = _limit(arr)
    bwd = _limit(arr[::-1])[::-1]
    return 0.5 * (fwd + bwd)


def _hampel_outlier_mask(
    xy: np.ndarray, *, window: int, k: float, abs_floor: float,
) -> np.ndarray:
    """Boolean mask (``True`` = outlier) over a 2-D path via a Hampel test.

    For each frame, compares its distance to the local median (over a
    centred ``window`` of frames) against ``max(k · 1.4826 · MAD,
    abs_floor)``. The ``abs_floor`` (metres) keeps a perfectly still run
    — where the MAD collapses to ~0 — from flagging sub-floor noise as
    outliers, while ``k · 1.4826 · MAD`` adapts the threshold to the
    local motion so a genuine fast move is not mistaken for a spike.
    """
    arr = np.asarray(xy, dtype=float)
    n = arr.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    half = max(1, int(window) // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = arr[lo:hi]
        med = np.median(win, axis=0)
        dists = np.linalg.norm(win - med, axis=1)
        mad = float(np.median(dists))
        thresh = max(k * 1.4826 * mad, abs_floor)
        if float(np.linalg.norm(arr[i] - med)) > thresh:
            mask[i] = True
    return mask


def _slerp_fill(
    src_frames: np.ndarray, src_R: np.ndarray, dense_frames: np.ndarray,
) -> np.ndarray:
    """SLERP a sparse stack of rotations onto a dense frame grid.

    ``dense_frames`` is assumed to span ``[src_frames[0],
    src_frames[-1]]`` (the resampler only fills *inside* a run), so no
    extrapolation occurs; values are clamped defensively regardless.
    """
    from scipy.spatial.transform import Rotation, Slerp

    sf = np.asarray(src_frames, dtype=float)
    df = np.asarray(dense_frames, dtype=float)
    src_R = np.asarray(src_R, dtype=float)
    if len(sf) == 1:
        return np.tile(src_R[0], (len(df), 1, 1))
    rots = Rotation.from_matrix(src_R)
    slerp = Slerp(sf, rots)
    return slerp(np.clip(df, sf[0], sf[-1])).as_matrix()


def _clean_player_translation(
    track: SmplWorldTrack,
    *,
    fps: float,
    enabled: bool = True,
    max_gap_fill_frames: int = 15,
    hampel_window_s: float = 0.4,
    hampel_k: float = 3.0,
    hampel_floor_m: float = 0.6,
    v_max_m_s: float = 12.0,
) -> tuple[SmplWorldTrack, dict]:
    """Per-player robust translation cleanup, run before the cross-player
    consensus and the final smoother.

    Three things, all in physical units so the same config generalises
    across clips and frame rates:

      1. **Gap fill.** The anchored span is resampled onto a uniform
         per-frame grid; missing frames are interpolated (linear for
         ``root_t`` / ``thetas`` / ``confidence``, SLERP for ``root_R``).
         Gaps wider than ``max_gap_fill_frames`` are *not* fabricated —
         the track is split into runs at those points so a multi-second
         hole isn't turned into a straight-line glide. (Artifact #3.)
      2. **Outlier rejection.** A Hampel test on pitch XY flags
         single-frame teleports / large excursions among the originally
         present frames; flagged positions are re-interpolated from
         trusted neighbours. (Artifact #2, spikes.)
      3. **Velocity limit.** A physical speed cap (``v_max_m_s``) bounds
         any residual high-frequency depth wobble without flattening a
         genuine sprint. (Artifact #2, wobble.)

    XY only — ``z`` is owned by the per-player ground-snap pass and is
    carried (interpolated) but not clamped here. Returns a new dense
    ``SmplWorldTrack`` plus a stats dict.
    """
    stats = {"filled_frames": 0, "rejected_frames": 0, "clamped_frames": 0}
    frames = np.asarray(track.frames, dtype=np.int64)
    n = int(len(frames))
    if not enabled or n < 2:
        return track, stats

    # Sort + dedupe frames (first occurrence wins) so the interp is
    # monotone and a duplicated frame index can't break np.interp.
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    root_t = np.asarray(track.root_t, dtype=float)[order]
    root_R = np.asarray(track.root_R, dtype=float)[order]
    thetas = np.asarray(track.thetas, dtype=float)[order]
    conf = np.asarray(track.confidence, dtype=float)[order]
    _, first_idx = np.unique(frames, return_index=True)
    first_idx = np.sort(first_idx)
    frames = frames[first_idx]
    root_t, root_R = root_t[first_idx], root_R[first_idx]
    thetas, conf = thetas[first_idx], conf[first_idx]
    n = int(len(frames))
    if n < 2:
        return track, stats

    # Split into runs at gaps wider than we are willing to fabricate.
    diffs = np.diff(frames)
    split_after = np.where(diffs > max_gap_fill_frames)[0]
    bounds: list[tuple[int, int]] = []
    start = 0
    for s in split_after:
        bounds.append((start, int(s) + 1))
        start = int(s) + 1
    bounds.append((start, n))

    hwin = max(3, int(round(hampel_window_s * fps)))
    max_step = float(v_max_m_s) / float(fps) if fps > 0 else float("inf")

    out_f, out_t, out_R, out_th, out_c = [], [], [], [], []
    for a, b in bounds:
        rf = frames[a:b]
        f0, f1 = int(rf[0]), int(rf[-1])
        dense_f = np.arange(f0, f1 + 1)
        present = np.zeros(len(dense_f), dtype=bool)
        present[rf - f0] = True
        stats["filled_frames"] += int((~present).sum())

        # Densify translation (xyz) + confidence by linear interpolation.
        dense_t = np.empty((len(dense_f), 3))
        for ax in range(3):
            dense_t[:, ax] = np.interp(dense_f, rf, root_t[a:b, ax])
        dense_c = np.interp(dense_f, rf, conf[a:b])

        # Densify thetas (linear per element) and rotation (SLERP).
        rth = thetas[a:b].reshape(b - a, -1)
        dth = np.empty((len(dense_f), rth.shape[1]))
        for j in range(rth.shape[1]):
            dth[:, j] = np.interp(dense_f, rf, rth[:, j])
        dense_th = dth.reshape((len(dense_f),) + thetas.shape[1:])
        dense_R = _slerp_fill(rf, root_R[a:b], dense_f)

        # Reject position outliers among the originally-present frames,
        # then re-interpolate the rejected positions from trusted ones.
        xy = dense_t[:, :2]
        outliers = _hampel_outlier_mask(
            xy, window=hwin, k=hampel_k, abs_floor=hampel_floor_m,
        ) & present
        stats["rejected_frames"] += int(outliers.sum())
        trusted = present & ~outliers
        if outliers.any() and int(trusted.sum()) >= 2:
            ti = np.where(trusted)[0]
            grid = np.arange(len(dense_f))
            for ax in range(2):
                xy[:, ax] = np.interp(grid, ti, xy[ti, ax])

        # Physical velocity limit on the de-spiked path.
        before = xy.copy()
        xy = _velocity_limit_xy(xy, max_step)
        stats["clamped_frames"] += int(
            (np.linalg.norm(xy - before, axis=1) > 1e-6).sum()
        )
        dense_t[:, :2] = xy

        out_f.append(dense_f)
        out_t.append(dense_t)
        out_R.append(dense_R)
        out_th.append(dense_th)
        out_c.append(dense_c)

    new = SmplWorldTrack(
        player_id=track.player_id,
        frames=np.concatenate(out_f).astype(np.int64),
        betas=np.asarray(track.betas, dtype=np.float32),
        thetas=np.concatenate(out_th).astype(np.float32),
        root_R=np.concatenate(out_R).astype(np.float32),
        root_t=np.concatenate(out_t).astype(np.float32),
        confidence=np.concatenate(out_c).astype(np.float32),
        shot_id=track.shot_id,
    )
    return new, stats


def _clean_refined_translation(
    refined: RefinedPose, **cleanup_kwargs,
) -> tuple[RefinedPose, dict]:
    """Re-apply the per-player translation cleanup to an assembled
    multi-shot track.

    The multi-shot assembly picks the highest-confidence shot per
    reference frame, which teleports a player between two disagreeing
    camera solves at every switch frame. Running the same physical
    cleanup (Hampel reject + velocity limit) on the merged track bounds
    those switches without re-introducing the per-shot work. ``view_count``
    is carried onto the (possibly densified) frame set — gap-filled
    frames inherit the nearest neighbour's count.
    """
    tr = SmplWorldTrack(
        player_id=refined.player_id,
        frames=np.asarray(refined.frames),
        betas=np.asarray(refined.betas),
        thetas=np.asarray(refined.thetas),
        root_R=np.asarray(refined.root_R),
        root_t=np.asarray(refined.root_t),
        confidence=np.asarray(refined.confidence),
        shot_id="",
    )
    cleaned, stats = _clean_player_translation(tr, **cleanup_kwargs)
    vc_map = {
        int(f): int(v)
        for f, v in zip(np.asarray(refined.frames), np.asarray(refined.view_count))
    }
    new_vc = np.array(
        [vc_map.get(int(f), 1) for f in cleaned.frames], dtype=np.int32,
    )
    out = RefinedPose(
        player_id=refined.player_id,
        frames=cleaned.frames,
        betas=cleaned.betas,
        thetas=cleaned.thetas,
        root_R=cleaned.root_R,
        root_t=cleaned.root_t,
        confidence=cleaned.confidence,
        view_count=new_vc,
        contributing_shots=refined.contributing_shots,
    )
    return out, stats


def _load_clip_fps(output_dir: Path) -> float:
    """Return the clip FPS from any available camera_track JSON, or
    25.0 when none has been written yet (e.g. unit-test fixtures)."""
    candidates: list[Path] = [output_dir / "camera" / "camera_track.json"]
    candidates.extend(sorted((output_dir / "camera").glob("*_camera_track.json")))
    for p in candidates:
        if not p.exists():
            continue
        try:
            from src.schemas.camera_track import CameraTrack
            return float(CameraTrack.load(p).fps)
        except Exception:
            continue
    return 25.0


def _clean_single_track(
    track: SmplWorldTrack,
    *,
    lean_correction_factor: float = 0.7,
    lean_max_correction_deg: float = 30.0,
    ground_snap_target_z: float = 0.02,
    ground_snap_max_distance: float = 0.30,
    smpl_model: dict | None = None,
    smoothing: dict | None = None,
    cleanup: dict | None = None,
) -> tuple[SmplWorldTrack, dict]:
    """Apply outlier rejection + lean reduction + leading/trailing trim.

    Returns a new ``SmplWorldTrack`` carrying the same ``shot_id`` /
    ``player_id`` / ``betas`` but with the per-frame arrays sliced to
    the span where ``confidence >= _FRESH_ANCHOR_CONF`` (i.e. a fresh
    foot-anchor ray-cast actually succeeded for at least one frame).
    If no frame was freshly anchored, an empty track is returned so
    downstream code can still locate the player_id by filename.

    Lean reduction is applied **before** the trim so all output frames
    carry the corrected orientation. ``lean_correction_factor=0`` (or
    ``lean_max_correction_deg=0``) effectively disables it.
    """
    root_R = np.asarray(track.root_R, dtype=float)
    root_R_fixed = _reject_root_R_outliers(root_R)

    root_t = np.asarray(track.root_t, dtype=float)
    root_R_fixed, root_t = _reduce_root_lean(
        root_R_fixed,
        root_t,
        correction_factor=lean_correction_factor,
        max_lean_deg=lean_max_correction_deg,
    )

    thetas = np.asarray(track.thetas)
    # Beta-adjusted rest joints so FK uses the player's actual leg
    # length, not mean-betas canonical (which is 8-10 cm too long for
    # typical players and leaves feet floating above the pitch).
    rest_joints = _beta_adjusted_rest_joints(track.betas, smpl_model)
    # Ground-snap: the hmr_world foot anchor uses the canonical
    # mid-ankle (which assumes both legs straight). For running /
    # walking players that midpoint sits halfway between the planted
    # and raised feet, so the planted foot floats off the pitch. Slide
    # root_t.z per frame so the lower foot lands at ground level.
    root_t = _ground_snap(
        root_R_fixed,
        root_t,
        thetas,
        target_foot_z=ground_snap_target_z,
        max_snap_distance=ground_snap_max_distance,
        rest_joints=rest_joints,
    )

    frames = np.asarray(track.frames, dtype=np.int64)
    confidence = np.asarray(track.confidence)

    empty_cleanup_stats = {
        "filled_frames": 0, "rejected_frames": 0, "clamped_frames": 0,
    }
    anchored = confidence >= _FRESH_ANCHOR_CONF
    if not anchored.any():
        return (
            SmplWorldTrack(
                player_id=track.player_id,
                frames=np.zeros(0, dtype=np.int64),
                betas=np.asarray(track.betas, dtype=np.float32),
                thetas=np.zeros((0, 24, 3), dtype=np.float32),
                root_R=np.zeros((0, 3, 3), dtype=np.float32),
                root_t=np.zeros((0, 3), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                shot_id=track.shot_id,
            ),
            empty_cleanup_stats,
        )

    idx = np.where(anchored)[0]
    i_first = int(idx[0])
    i_last = int(idx[-1])
    sl = slice(i_first, i_last + 1)

    trimmed = SmplWorldTrack(
        player_id=track.player_id,
        frames=frames[sl],
        betas=np.asarray(track.betas, dtype=np.float32),
        thetas=thetas[sl].astype(np.float32),
        root_R=root_R_fixed[sl].astype(np.float32),
        root_t=root_t[sl].astype(np.float32),
        confidence=confidence[sl].astype(np.float32),
        shot_id=track.shot_id,
    )

    # Per-player robust translation cleanup AFTER trim (so it runs on the
    # anchored span) and BEFORE the cross-player consensus + final
    # smoother (so per-player teleports don't corrupt the consensus
    # median and the Savgol isn't fitting through outliers). Gap-fill
    # densifies the track onto a uniform grid; outlier rejection +
    # velocity limit bound implausible per-frame motion.
    cleanup_stats = dict(empty_cleanup_stats)
    if cleanup and cleanup.get("enabled", False):
        trimmed, cleanup_stats = _clean_player_translation(trimmed, **cleanup)

    # Temporal smoothing AFTER trim so the smoother's edge taps stay
    # inside the anchored span. ``hmr_world`` already applies a mild
    # Savgol on root_t (window 5); this pass is additional and targets
    # residual jitter that survives the foot-anchor / SLERP smoothers
    # upstream. Disable individual smoothers by setting their window
    # to ``<= 1`` in the ``refined_poses`` config block.
    if smoothing:
        trimmed = _smooth_track(trimmed, **smoothing)
    return trimmed, cleanup_stats


class RefinedPosesStage(BaseStage):
    name = "refined_poses"

    def is_complete(self) -> bool:
        hmr_dir = self.output_dir / "hmr_world"
        out_dir = self.output_dir / "refined_poses"
        if not hmr_dir.exists():
            return True
        player_ids = _discover_player_ids(hmr_dir)
        if not player_ids:
            return True
        return all(
            (out_dir / f"{pid}_refined.npz").exists() for pid in player_ids
        )

    def run(self) -> None:
        hmr_dir = self.output_dir / "hmr_world"
        out_dir = self.output_dir / "refined_poses"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not hmr_dir.exists():
            logger.info(
                "[refined_poses] hmr_world output missing — nothing to refine"
            )
            return

        cfg = (self.config.get("refined_poses") or {})
        lean_factor = float(cfg.get("lean_correction_factor", 0.7))
        lean_max_deg = float(cfg.get("lean_max_correction_deg", 30.0))
        ground_target = float(cfg.get("ground_snap_target_z", 0.02))
        ground_max_dist = float(cfg.get("ground_snap_max_distance", 0.30))
        smoothing = {
            "root_R_slerp_window": int(cfg.get("smooth_root_R_window", 7)),
            "root_t_savgol_window": int(cfg.get("smooth_root_t_window", 7)),
            "root_t_savgol_order": int(cfg.get("smooth_root_t_order", 2)),
            "thetas_savgol_window": int(cfg.get("smooth_thetas_window", 9)),
            "thetas_savgol_order": int(cfg.get("smooth_thetas_order", 2)),
        }
        jitter_cfg = (cfg.get("jitter") or {})
        jitter_kwargs = {
            "enabled": bool(jitter_cfg.get("enabled", True)),
            "min_players": int(jitter_cfg.get("min_players", 4)),
            "baseline_window_s": float(jitter_cfg.get("baseline_window_s", 1.0)),
            "baseline_floor_m": float(jitter_cfg.get("baseline_floor_m", 0.05)),
            "tempo_ratio": float(jitter_cfg.get("tempo_ratio", 2.5)),
            "max_spread_ratio": float(jitter_cfg.get("max_spread_ratio", 0.30)),
        }
        residual_kwargs = {
            "enabled": bool(jitter_cfg.get("residual_pass_enabled", True)),
            "savgol_window_s": float(jitter_cfg.get("residual_savgol_window_s", 1.2)),
            "savgol_poly": int(jitter_cfg.get("residual_savgol_poly", 2)),
            "min_players": int(jitter_cfg.get("residual_min_players", 3)),
            "min_offset_m": float(jitter_cfg.get("residual_min_offset_m", 0.02)),
            "max_offset_m": float(jitter_cfg.get("residual_max_offset_m", 2.0)),
            "iterations": int(jitter_cfg.get("residual_iterations", 2)),
        }

        # fps drives the physical-unit cleanup + jitter passes; load it
        # once up front (before the per-player clean loop, which now uses
        # it for the velocity limit / Hampel window).
        fps = _load_clip_fps(self.output_dir)

        # Per-player robust translation cleanup (gap-fill + outlier
        # rejection + velocity limit). Disabled when the config block is
        # absent so minimal-config callers keep the legacy behaviour;
        # config/default.yaml turns it on for production.
        cleanup_cfg = (cfg.get("cleanup") or {})
        cleanup_kwargs = {
            "enabled": bool(cleanup_cfg.get("enabled", False)),
            "fps": fps,
            "max_gap_fill_frames": int(cleanup_cfg.get("max_gap_fill_frames", 15)),
            "hampel_window_s": float(cleanup_cfg.get("hampel_window_s", 0.4)),
            "hampel_k": float(cleanup_cfg.get("hampel_k", 3.0)),
            "hampel_floor_m": float(cleanup_cfg.get("hampel_floor_m", 0.6)),
            "v_max_m_s": float(cleanup_cfg.get("v_max_m_s", 12.0)),
        }

        sync_map = self._load_sync_map()
        per_player = self._gather_contributions(hmr_dir)
        # Loaded once per stage run so each per-player call can apply
        # the right shape correction without re-reading the npz.
        smpl_model = _load_smpl_neutral_model()
        if smpl_model is None:
            logger.warning(
                "[refined_poses] data/models/smpl_neutral.npz missing — "
                "ground-snap will use mean-betas canonical rest joints "
                "and may leave 5-10 cm gaps for players whose shape "
                "differs from mean"
            )

        # 1. Per-(shot, player) CLEAN (outlier reject + lean + ground
        # snap + trim + per-player translation cleanup). No final
        # smoothing yet — the cross-player jitter pass needs to operate
        # on the cleaned signal so any residual error gets mopped up by
        # the smoother afterwards.
        cleanup_summary = {
            "enabled": cleanup_kwargs["enabled"],
            "filled_frames": 0,
            "rejected_frames": 0,
            "clamped_frames": 0,
            "assembly_rejected_frames": 0,
            "assembly_clamped_frames": 0,
        }
        cleaned_by_shot: dict[str, list[tuple[str, SmplWorldTrack]]] = {}
        for pid, contribs in per_player.items():
            for shot_id, track in contribs:
                cleaned, c_stats = _clean_single_track(
                    track,
                    lean_correction_factor=lean_factor,
                    lean_max_correction_deg=lean_max_deg,
                    ground_snap_target_z=ground_target,
                    ground_snap_max_distance=ground_max_dist,
                    smpl_model=smpl_model,
                    smoothing=None,
                    cleanup=cleanup_kwargs,
                )
                cleanup_summary["filled_frames"] += int(c_stats["filled_frames"])
                cleanup_summary["rejected_frames"] += int(c_stats["rejected_frames"])
                cleanup_summary["clamped_frames"] += int(c_stats["clamped_frames"])
                cleaned_by_shot.setdefault(shot_id, []).append((pid, cleaned))

        # 2. Per-shot cross-player JITTER pass. Jitter is a camera
        # frame artifact so consensus must be computed on each shot's
        # native timeline before any sync_map remapping.
        jitter_summary = {
            "enabled": jitter_kwargs["enabled"],
            "corrected_frames": 0,
            "total_frames_evaluated": 0,
            "max_offset_m": 0.0,
            "mean_offset_m": 0.0,
            "shots": [],
        }
        for shot_id, items in cleaned_by_shot.items():
            tracks_only = [tr for _, tr in items]
            corrected, shot_stats = _apply_jitter_correction(
                tracks_only, fps=fps, **jitter_kwargs,
            )
            cleaned_by_shot[shot_id] = [
                (pid, ct) for (pid, _), ct in zip(items, corrected)
            ]
            shot_stats["shot_id"] = shot_id
            jitter_summary["shots"].append(shot_stats)
            jitter_summary["corrected_frames"] += int(shot_stats["corrected_frames"])
            jitter_summary["total_frames_evaluated"] += int(
                shot_stats["total_frames_evaluated"]
            )
            jitter_summary["max_offset_m"] = float(max(
                jitter_summary["max_offset_m"], shot_stats["max_offset_m"],
            ))
        # Aggregate mean across shots: weighted by per-shot corrected_frames.
        weighted_sum = 0.0
        weight = 0
        for s in jitter_summary["shots"]:
            n = int(s["corrected_frames"])
            if n > 0:
                weighted_sum += float(s["mean_offset_m"]) * n
                weight += n
        if weight > 0:
            jitter_summary["mean_offset_m"] = weighted_sum / weight

        # 2b. Per-shot RESIDUAL-CONSENSUS pass. The Δ-based pass above
        # catches single-frame spikes but is blind to sustained wobble
        # because per-player Δ noise dominates the cross-player std
        # gate on real running scenes. This residual pass takes the
        # cross-player median residual against a heavy per-player
        # Savgol baseline as the common-mode camera bias and subtracts
        # it. Two iterations strip ~85% of a band-limited wobble.
        residual_summary = {
            "enabled": residual_kwargs["enabled"],
            "corrected_frames": 0,
            "total_frames_evaluated": 0,
            "max_offset_m": 0.0,
            "mean_offset_m": 0.0,
            "shots": [],
        }
        for shot_id, items in cleaned_by_shot.items():
            tracks_only = [tr for _, tr in items]
            corrected, shot_stats = _apply_residual_consensus_correction(
                tracks_only, fps=fps, **residual_kwargs,
            )
            cleaned_by_shot[shot_id] = [
                (pid, ct) for (pid, _), ct in zip(items, corrected)
            ]
            shot_stats["shot_id"] = shot_id
            residual_summary["shots"].append(shot_stats)
            residual_summary["corrected_frames"] += int(shot_stats["corrected_frames"])
            residual_summary["total_frames_evaluated"] += int(
                shot_stats["total_frames_evaluated"]
            )
            residual_summary["max_offset_m"] = float(max(
                residual_summary["max_offset_m"], shot_stats["max_offset_m"],
            ))
        weighted_sum_r = 0.0
        weight_r = 0
        for s in residual_summary["shots"]:
            n = int(s["corrected_frames"])
            if n > 0:
                weighted_sum_r += float(s["mean_offset_m"]) * n
                weight_r += n
        if weight_r > 0:
            residual_summary["mean_offset_m"] = weighted_sum_r / weight_r

        # 3. Per-(shot, player) SMOOTH on the jitter-corrected tracks.
        # 4. Regroup back into per_player so the assemble step sees the
        # same shape as the original contribs but with cleaned +
        # jitter-corrected + smoothed data.
        prepared_per_player: dict[str, list[tuple[str, SmplWorldTrack]]] = {}
        for shot_id, items in cleaned_by_shot.items():
            for pid, tr in items:
                if int(len(tr.frames)) > 0:
                    tr = _smooth_track(tr, **smoothing)
                prepared_per_player.setdefault(pid, []).append((shot_id, tr))

        summary: dict = {
            "players_refined": 0,
            "single_shot_players": 0,
            "multi_shot_players": 0,
            "total_frames": 0,
            "cleanup": cleanup_summary,
            "jitter": jitter_summary,
            "residual_consensus": residual_summary,
        }

        # 5. Assemble each player on the reference timeline and save.
        for pid in sorted(prepared_per_player.keys()):
            contribs = prepared_per_player[pid]
            refined, diag = _assemble_player(pid, contribs, sync_map)
            distinct_shots = {sid for sid, _ in contribs}
            # Multi-shot assembly merges by per-frame highest-confidence
            # pick, which teleports a player between disagreeing camera
            # solves at switch frames. Bound that with the same physical
            # cleanup. Single-shot tracks were already cleaned per-shot.
            if (
                cleanup_kwargs["enabled"]
                and len(distinct_shots) > 1
                and int(len(refined.frames)) > 1
            ):
                refined, a_stats = _clean_refined_translation(
                    refined, **cleanup_kwargs,
                )
                cleanup_summary["assembly_rejected_frames"] += int(
                    a_stats["rejected_frames"]
                )
                cleanup_summary["assembly_clamped_frames"] += int(
                    a_stats["clamped_frames"]
                )
            refined.save(out_dir / f"{pid}_refined.npz")
            diag.save(out_dir / f"{pid}_diagnostics.json")
            summary["players_refined"] += 1
            if len(distinct_shots) <= 1:
                summary["single_shot_players"] += 1
            else:
                summary["multi_shot_players"] += 1
            summary["total_frames"] += int(len(refined.frames))

        (out_dir / "refined_poses_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        logger.info(
            "[refined_poses] %d player(s) refined, %d frame(s) total, "
            "cleanup[%d filled / %d rejected / %d clamped], "
            "%d Δ-jitter / %d residual-consensus frame(s) corrected",
            summary["players_refined"], summary["total_frames"],
            cleanup_summary["filled_frames"],
            cleanup_summary["rejected_frames"],
            cleanup_summary["clamped_frames"],
            jitter_summary["corrected_frames"],
            residual_summary["corrected_frames"],
        )

    # ------------------------------------------------------------------

    def _load_sync_map(self) -> SyncMap:
        sync_path = self.output_dir / "shots" / "sync_map.json"
        if sync_path.exists():
            return SyncMap.load(sync_path)
        logger.warning(
            "[refined_poses] sync_map.json missing; treating offsets as 0"
        )
        return SyncMap(reference_shot="", alignments=[])

    def _gather_contributions(
        self, hmr_dir: Path,
    ) -> dict[str, list[tuple[str, SmplWorldTrack]]]:
        out: dict[str, list[tuple[str, SmplWorldTrack]]] = {}
        for npz in sorted(hmr_dir.glob("*_smpl_world.npz")):
            try:
                track = SmplWorldTrack.load(npz)
            except Exception as exc:
                logger.warning(
                    "[refined_poses] skipping %s — load failed (%s)",
                    npz.name, exc,
                )
                continue
            out.setdefault(track.player_id, []).append((track.shot_id, track))
        return out


# ----------------------------------------------------------------------
# Helpers (module-level so tests can import them directly).


def _discover_player_ids(hmr_dir: Path) -> set[str]:
    """Pull player_ids out of ``hmr_world`` filenames (``{shot}__{pid}``).

    Legacy single-shot filenames with no ``__`` separator are also
    accepted so older outputs still register here.
    """
    out: set[str] = set()
    for p in hmr_dir.glob("*_smpl_world.npz"):
        stem = p.name.removesuffix("_smpl_world.npz")
        if "__" in stem:
            out.add(stem.split("__", 1)[1])
        else:
            out.add(stem)
    return out


def _assemble_player(
    player_id: str,
    contribs: list[tuple[str, SmplWorldTrack]],
    sync_map: SyncMap,
) -> tuple[RefinedPose, RefinedPoseDiagnostics]:
    """Assemble already-cleaned/jitter-corrected/smoothed per-shot
    tracks onto the shared reference timeline."""
    cleaned: list[tuple[str, SmplWorldTrack, int]] = []
    for shot_id, track in contribs:
        offset = sync_map.offset_for(shot_id) if shot_id else 0
        cleaned.append((shot_id, track, offset))

    # Single-shot fast path: pass cleaned data through with sync offset.
    if len(cleaned) == 1:
        shot_id, tr, offset = cleaned[0]
        ref_frames = np.asarray(tr.frames, dtype=np.int64) - offset
        n = int(len(ref_frames))
        contributing = (shot_id,) if shot_id else ()
        return (
            RefinedPose(
                player_id=player_id,
                frames=ref_frames,
                betas=np.asarray(tr.betas, dtype=np.float32),
                thetas=np.asarray(tr.thetas, dtype=np.float32),
                root_R=np.asarray(tr.root_R, dtype=np.float32),
                root_t=np.asarray(tr.root_t, dtype=np.float32),
                confidence=np.asarray(tr.confidence, dtype=np.float32),
                view_count=np.ones(n, dtype=np.int32),
                contributing_shots=contributing,
            ),
            RefinedPoseDiagnostics(
                player_id=player_id,
                contributing_shots=contributing,
                frames=tuple(),
                summary={"total_frames": n, "shots": list(contributing)},
            ),
        )

    # Multi-shot: per-frame highest-confidence pick on the reference
    # timeline. Placeholder for the cross-shot fusion redesign — keeps
    # the schema populated without trying to combine rotations.
    by_frame: dict[int, dict] = {}
    contributing_shots: list[str] = []
    betas_stack: list[np.ndarray] = []
    for shot_id, tr, offset in cleaned:
        ref_frames = np.asarray(tr.frames, dtype=np.int64) - offset
        for i, f in enumerate(ref_frames):
            conf = float(tr.confidence[i])
            key = int(f)
            if key not in by_frame or conf > by_frame[key]["conf"]:
                by_frame[key] = {
                    "conf": conf,
                    "theta": np.asarray(tr.thetas[i]),
                    "R": np.asarray(tr.root_R[i]),
                    "t": np.asarray(tr.root_t[i]),
                }
        if shot_id:
            contributing_shots.append(shot_id)
        betas_stack.append(np.asarray(tr.betas, dtype=np.float32))

    if not by_frame:
        empty_n = 0
        contributing = tuple(sorted(set(contributing_shots)))
        return (
            RefinedPose(
                player_id=player_id,
                frames=np.zeros(0, dtype=np.int64),
                betas=(
                    np.mean(np.stack(betas_stack), axis=0)
                    if betas_stack else np.zeros(10, dtype=np.float32)
                ),
                thetas=np.zeros((0, 24, 3), dtype=np.float32),
                root_R=np.zeros((0, 3, 3), dtype=np.float32),
                root_t=np.zeros((0, 3), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                view_count=np.zeros(0, dtype=np.int32),
                contributing_shots=contributing,
            ),
            RefinedPoseDiagnostics(
                player_id=player_id,
                contributing_shots=contributing,
                frames=tuple(),
                summary={"total_frames": empty_n, "shots": list(contributing)},
            ),
        )

    sorted_keys = sorted(by_frame.keys())
    frames = np.array(sorted_keys, dtype=np.int64)
    n = int(len(frames))
    thetas = np.stack([by_frame[k]["theta"] for k in sorted_keys]).astype(
        np.float32
    )
    root_R = np.stack([by_frame[k]["R"] for k in sorted_keys]).astype(
        np.float32
    )
    root_t = np.stack([by_frame[k]["t"] for k in sorted_keys]).astype(
        np.float32
    )
    confidence = np.array(
        [by_frame[k]["conf"] for k in sorted_keys], dtype=np.float32
    )
    betas = (
        np.mean(np.stack(betas_stack), axis=0).astype(np.float32)
        if betas_stack else np.zeros(10, dtype=np.float32)
    )
    contributing = tuple(sorted(set(contributing_shots)))
    return (
        RefinedPose(
            player_id=player_id,
            frames=frames,
            betas=betas,
            thetas=thetas,
            root_R=root_R,
            root_t=root_t,
            confidence=confidence,
            view_count=np.ones(n, dtype=np.int32),
            contributing_shots=contributing,
        ),
        RefinedPoseDiagnostics(
            player_id=player_id,
            contributing_shots=contributing,
            frames=tuple(),
            summary={"total_frames": n, "shots": list(contributing)},
        ),
    )

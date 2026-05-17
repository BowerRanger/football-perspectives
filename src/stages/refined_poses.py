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


def _clean_single_track(
    track: SmplWorldTrack,
    *,
    lean_correction_factor: float = 0.7,
    lean_max_correction_deg: float = 30.0,
    ground_snap_target_z: float = 0.02,
    ground_snap_max_distance: float = 0.30,
    smpl_model: dict | None = None,
    smoothing: dict | None = None,
) -> SmplWorldTrack:
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

    anchored = confidence >= _FRESH_ANCHOR_CONF
    if not anchored.any():
        return SmplWorldTrack(
            player_id=track.player_id,
            frames=np.zeros(0, dtype=np.int64),
            betas=np.asarray(track.betas, dtype=np.float32),
            thetas=np.zeros((0, 24, 3), dtype=np.float32),
            root_R=np.zeros((0, 3, 3), dtype=np.float32),
            root_t=np.zeros((0, 3), dtype=np.float32),
            confidence=np.zeros(0, dtype=np.float32),
            shot_id=track.shot_id,
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

    # Temporal smoothing AFTER trim so the smoother's edge taps stay
    # inside the anchored span. ``hmr_world`` already applies a mild
    # Savgol on root_t (window 5); this pass is additional and targets
    # residual jitter that survives the foot-anchor / SLERP smoothers
    # upstream. Disable individual smoothers by setting their window
    # to ``<= 1`` in the ``refined_poses`` config block.
    if smoothing:
        trimmed = _smooth_track(trimmed, **smoothing)
    return trimmed


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

        summary: dict = {
            "players_refined": 0,
            "single_shot_players": 0,
            "multi_shot_players": 0,
            "total_frames": 0,
        }

        for pid, contribs in sorted(per_player.items()):
            refined, diag = _refine_player(
                pid, contribs, sync_map,
                lean_correction_factor=lean_factor,
                lean_max_correction_deg=lean_max_deg,
                ground_snap_target_z=ground_target,
                ground_snap_max_distance=ground_max_dist,
                smpl_model=smpl_model,
                smoothing=smoothing,
            )
            refined.save(out_dir / f"{pid}_refined.npz")
            diag.save(out_dir / f"{pid}_diagnostics.json")
            summary["players_refined"] += 1
            distinct_shots = {sid for sid, _ in contribs}
            if len(distinct_shots) <= 1:
                summary["single_shot_players"] += 1
            else:
                summary["multi_shot_players"] += 1
            summary["total_frames"] += int(len(refined.frames))

        (out_dir / "refined_poses_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        logger.info(
            "[refined_poses] %d player(s) refined, %d frame(s) total",
            summary["players_refined"], summary["total_frames"],
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


def _refine_player(
    player_id: str,
    contribs: list[tuple[str, SmplWorldTrack]],
    sync_map: SyncMap,
    *,
    lean_correction_factor: float = 0.7,
    lean_max_correction_deg: float = 30.0,
    ground_snap_target_z: float = 0.02,
    ground_snap_max_distance: float = 0.30,
    smpl_model: dict | None = None,
    smoothing: dict | None = None,
) -> tuple[RefinedPose, RefinedPoseDiagnostics]:
    """Clean each shot's track, then assemble onto the reference timeline."""
    cleaned: list[tuple[str, SmplWorldTrack, int]] = []
    for shot_id, track in contribs:
        cleaned_track = _clean_single_track(
            track,
            lean_correction_factor=lean_correction_factor,
            lean_max_correction_deg=lean_max_correction_deg,
            ground_snap_target_z=ground_snap_target_z,
            ground_snap_max_distance=ground_snap_max_distance,
            smpl_model=smpl_model,
            smoothing=smoothing,
        )
        offset = sync_map.offset_for(shot_id) if shot_id else 0
        cleaned.append((shot_id, cleaned_track, offset))

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

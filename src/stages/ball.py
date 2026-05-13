"""Ball stage: per-frame detection, IMM smoothing, ground projection,
and 3D trajectory reconstruction.

The stage owns the entire ball pipeline.  It reads only the clip video
files (via the shots manifest) and the camera track from earlier
stages; it does **not** read any ball detections from
``output/tracks/``.

Run flow per shot:

1. **Detect** — iterate the clip frames and ask the configured
   :class:`BallDetector` for ``(u, v, confidence)`` per frame.
2. **Smooth** — feed the per-frame detections through
   :class:`BallTracker`, a 2-mode IMM Kalman filter. Output includes a
   per-frame mode posterior ``p_flight`` and bounded gap-fill.
3. **Reconstruct 3D position** — ground-project each smoothed pixel to
   the world frame at ``z = ball_radius_m`` via
   :func:`src.utils.foot_anchor.ankle_ray_to_pitch`.
4. **Flight fit** — run-length encode frames where ``p_flight >= 0.5``;
   for each run run a parabola fit and (when the segment is long
   enough) a Magnus refinement. Accept Magnus only if it improves the
   pixel residual by ``ball.spin.min_residual_improvement`` and
   ``|ω|`` is within ``ball.spin.max_omega_rad_s``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.ball_track import BallFrame, BallTrack, FlightSegment
from src.schemas.camera_track import CameraTrack
from src.schemas.shots import ShotsManifest
from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.utils.ball_anchor_heights import (
    AIRBORNE_STATES,
    EVENT_STATES,
    GROUND_LEVEL_STATES,
    HARD_KNOT_STATES,
    airborne_bucket_range,
    state_to_height,
)
from src.utils.ball_detector import BallDetector, YOLOBallDetector
from src.utils.ball_tracker import BallTracker, TrackerStep
from src.utils.ball_plausibility import (
    GroundPromotionCfg,
    GroundedRun,
    PitchDims,
    PlausibilityCfg,
    find_implausible_grounded_runs,
    is_plausible_trajectory,
)
from src.utils.bundle_adjust import (
    _integrate_magnus_positions,
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)
from src.utils.ball_appearance_bridge import (
    AppearanceBridge,
    AppearanceBridgeCfg,
)
from src.utils.ball_kick_anchor import KickAnchorCfg, find_kick_anchor
from src.utils.foot_anchor import ankle_ray_to_pitch
from src.utils.goal_geometry import GoalGeometry, resolve_goal_impact_world
from src.utils.ball_spin_presets import (
    SPIN_ENABLED_STATES,
    omega_seed_from_preset,
)


logger = logging.getLogger(__name__)


def _build_detector(cfg: dict) -> BallDetector:
    """Construct a BallDetector from the ``ball.detector`` config key."""
    backend = str(cfg.get("detector", "yolo")).strip().lower()
    if backend == "wasb":
        from src.utils.ball_detector import WASBBallDetector  # lazy import
        wasb_cfg = cfg.get("wasb", {})
        return WASBBallDetector(
            checkpoint=wasb_cfg.get("checkpoint"),
            confidence=float(wasb_cfg.get("confidence", 0.3)),
            input_size=tuple(wasb_cfg.get("input_size", (512, 288))),
        )
    if backend == "yolo":
        return YOLOBallDetector(
            model_name=cfg.get("yolo_model", "yolov8n.pt"),
            confidence=float(cfg.get("confidence_threshold", 0.3)),
        )
    raise ValueError(f"Unknown ball.detector backend: {backend!r}")


def _demote_run_to_missing(
    per_frame_world: dict[int, tuple[np.ndarray, float]],
    a: int,
    b: int,
) -> None:
    """Drop world positions for frames [a, b] so they emit state='missing'."""
    for fi in range(a, b + 1):
        per_frame_world.pop(fi, None)


def _load_foot_uvs_for_shot(
    output_dir: Path, shot_id: str
) -> dict[int, list[tuple[float, float]]]:
    """Aggregate ankle pixel positions across all players for a shot.

    Reads ``output/hmr_world/<shot>__<player>_kp2d.json`` files (COCO-17
    keypoints; indices 15 = left_ankle, 16 = right_ankle). Returns a dict
    keyed by frame index with a list of ankle pixel positions, ignoring
    any with confidence below 0.3.
    """
    hmr_dir = output_dir / "hmr_world"
    if not hmr_dir.exists():
        return {}
    if shot_id:
        pattern = f"{shot_id}__*_kp2d.json"
    else:
        pattern = "*_kp2d.json"
    feet_by_frame: dict[int, list[tuple[float, float]]] = {}
    for path in hmr_dir.glob(pattern):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for entry in payload.get("frames", []):
            fi = int(entry.get("frame", -1))
            if fi < 0:
                continue
            kps = entry.get("keypoints", [])
            for idx in (15, 16):
                if idx >= len(kps):
                    continue
                kp = kps[idx]
                if len(kp) < 3 or kp[2] < 0.3:
                    continue
                feet_by_frame.setdefault(fi, []).append((float(kp[0]), float(kp[1])))
    return feet_by_frame


def _load_ball_anchors(
    output_dir: Path, shot_id: str
) -> dict[int, BallAnchor]:
    """Load per-frame ball anchors keyed by frame index.

    Returns an empty dict when no anchor file exists.
    """
    if shot_id:
        path = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
    else:
        path = output_dir / "ball" / "ball_anchors.json"
    if not path.exists():
        return {}
    try:
        aset = BallAnchorSet.load(path)
    except Exception as exc:
        logger.warning("ball stage: failed to load anchors at %s: %s", path, exc)
        return {}
    return {a.frame: a for a in aset.anchors}


class _MagnusRefinement:
    """Result of attempting a Magnus refinement on a flight segment.

    The caller uses ``effective_p0`` / ``effective_v0`` / ``effective_resid``
    for per-frame evaluation and reporting, ``omega_world`` to decide
    whether to integrate via Magnus vs. plain parabola, and the
    ``spin_axis`` / ``spin_omega`` / ``spin_confidence`` triple to
    populate ``FlightSegment.parabola``. When ``omega_world is None``
    the refinement was rejected and the inputs (parabola fit) win.
    """

    __slots__ = (
        "effective_p0", "effective_v0", "effective_resid",
        "omega_world", "spin_axis", "spin_omega", "spin_confidence",
    )

    def __init__(
        self,
        effective_p0: np.ndarray,
        effective_v0: np.ndarray,
        effective_resid: float,
        omega_world: np.ndarray | None,
        spin_axis: list[float] | None,
        spin_omega: float | None,
        spin_confidence: float | None,
    ) -> None:
        self.effective_p0 = effective_p0
        self.effective_v0 = effective_v0
        self.effective_resid = effective_resid
        self.omega_world = omega_world
        self.spin_axis = spin_axis
        self.spin_omega = spin_omega
        self.spin_confidence = spin_confidence


def _refine_with_magnus(
    *,
    obs: list[tuple[int, tuple[float, float]]],
    Ks_seg: list[np.ndarray],
    Rs_seg: list[np.ndarray],
    ts_seg: list[np.ndarray],
    fps: float,
    drag: float,
    plaus_cfg: PlausibilityCfg,
    pitch_dims: PitchDims,
    p0: np.ndarray,
    v0: np.ndarray,
    parab_resid: float,
    anchor_world: np.ndarray | None,
    duration_s: float,
    spin_enabled: bool,
    spin_min_seconds: float,
    spin_max_omega: float,
    spin_min_improve: float,
    spin_min_improve_hinted: float,
    omega_seed: np.ndarray,
    hint_provided: bool,
    segment_label: str,
    knot_frames: dict[int, np.ndarray] | None = None,
    knot_max_violation_m: float = 2.0,
) -> _MagnusRefinement:
    """Attempt a Magnus refinement of a parabola fit on a flight segment.

    When ``hint_provided`` is True (the segment's start anchor carries an
    explicit spin preset), the accept threshold relaxes from
    ``spin_min_improve`` to ``spin_min_improve_hinted`` — the user has
    asserted that this flight has spin, so we lower the bar for ω ≠ 0.
    """
    fallback = _MagnusRefinement(
        effective_p0=p0,
        effective_v0=v0,
        effective_resid=parab_resid,
        omega_world=None,
        spin_axis=None,
        spin_omega=None,
        spin_confidence=None,
    )
    if not spin_enabled or duration_s < spin_min_seconds:
        return fallback
    # When the user explicitly requests knuckle (no spin), skip the LM
    # entirely — Magnus would only ever drift away from omega=0.
    if hint_provided and np.linalg.norm(omega_seed) < 1e-9:
        # Knuckle case (preset == 'knuckle'): zero seed + hint. Trust the
        # user and stick with the parabola fit.
        return fallback
    # Two modes:
    #   - hint_provided: lock the spin axis to the preset direction
    #     (omega_seed is a non-zero vector from omega_seed_from_preset),
    #     letting the LM optimise only the spin magnitude alongside v0.
    #     The user told us the curl direction; we trust that.
    #   - no hint: bound each omega component to spin_max_omega / √3 so
    #     the recovered |omega| can't exceed spin_max_omega even at the
    #     cube corner. Without this the LM ran off to |omega| ≈ 700+
    #     rad/s on real-world data and got silently rejected.
    seed_norm = float(np.linalg.norm(omega_seed))
    try:
        if hint_provided and seed_norm > 1e-9:
            mp0, mv0, momega, magnus_resid = fit_magnus_trajectory(
                obs,
                Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                fps=fps, drag_k_over_m=drag,
                p0_seed=p0, v0_seed=v0,
                omega_seed=omega_seed,
                p0_fixed=anchor_world,
                omega_axis_fixed=omega_seed / seed_norm,
                omega_mag_bound=spin_max_omega,
                # Keep v0 inside the same physical envelope as the
                # plausibility check (horizontal_speed_max + ~50% margin
                # for vertical component) so the LM can't fit by
                # inventing 80 m/s velocities.
                v0_abs_bound=max(
                    plaus_cfg.horizontal_speed_max_m_s * 1.5,
                    40.0,
                ),
            )
        else:
            mp0, mv0, momega, magnus_resid = fit_magnus_trajectory(
                obs,
                Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                fps=fps, drag_k_over_m=drag,
                p0_seed=p0, v0_seed=v0,
                omega_seed=omega_seed,
                p0_fixed=anchor_world,
                omega_abs_bound=spin_max_omega / np.sqrt(3.0),
            )
    except Exception as exc:
        logger.debug("magnus fit failed on %s: %s", segment_label, exc)
        return fallback
    omega_mag = float(np.linalg.norm(momega))
    improvement = (
        (parab_resid - magnus_resid) / parab_resid
        if parab_resid > 0 else 0.0
    )
    # When a hint is provided the LM is already box-bounded on v0 and on
    # the omega scalar (with the axis locked to the preset direction),
    # so any fit it produces is physically inside the same envelope the
    # plausibility check enforces. Skipping the strict plausibility gate
    # here lets a locked-axis fit through when the LM saturated at the
    # v0 bound — common on hard real data where no single Magnus arc
    # exactly fits the user's pixel obs. Without a hint the LM is freer
    # so plausibility remains as the safety net.
    if hint_provided:
        magnus_plausible = True
    else:
        magnus_plausible = is_plausible_trajectory(
            mp0, mv0, omega=momega,
            duration_s=duration_s, fps=fps,
            cfg=plaus_cfg, pitch=pitch_dims,
        )
    accept_threshold = spin_min_improve_hinted if hint_provided else spin_min_improve
    # Validate against any anchor-derived knot constraints. Magnus
    # itself optimises only against pixel residuals (no knot support),
    # so on a Phase 2 span with kick + goal_impact knots the LM happily
    # produces a v0 that satisfies the airborne pixels at the cost of
    # missing the goal — apex z dives below ground, x flies off-pitch.
    # If Magnus violates any knot by more than ``knot_max_violation_m``,
    # reject and keep the parabola fit (which DOES respect the knots).
    knot_violation_ok = True
    if knot_frames:
        g_vec_local = np.array([0.0, 0.0, -9.81])
        for rel_idx, target_world in knot_frames.items():
            dt_k = rel_idx / fps
            positions = _integrate_magnus_positions(
                mp0, mv0, momega,
                g_vec_local,
                drag,
                np.array([0.0, dt_k]),
            )
            pos_at_knot = positions[-1]
            err = float(np.linalg.norm(
                np.asarray(pos_at_knot) - np.asarray(target_world)
            ))
            if err > knot_max_violation_m:
                logger.info(
                    "magnus refinement on %s violates knot at rel=%d by "
                    "%.2f m > %.2f m — rejecting, keeping parabola",
                    segment_label, rel_idx, err, knot_max_violation_m,
                )
                knot_violation_ok = False
                break
    if not (
        knot_violation_ok
        and omega_mag > 0
        and omega_mag <= spin_max_omega
        and improvement >= accept_threshold
        and magnus_plausible
    ):
        return fallback
    duration_factor = min(1.0, duration_s / 1.0)
    spin_confidence = float(min(1.0, (improvement / 0.5) * duration_factor))
    return _MagnusRefinement(
        effective_p0=mp0,
        effective_v0=mv0,
        effective_resid=magnus_resid,
        omega_world=momega,
        spin_axis=list((momega / omega_mag).astype(float)),
        spin_omega=omega_mag,
        spin_confidence=spin_confidence,
    )


def _spin_seed_for_segment(
    anchor_by_frame: dict[int, BallAnchor],
    a: int,
    b: int,
    *,
    v0: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    """Find a player_touch anchor with a ``spin`` preset inside the
    [a, b] flight segment and translate it to an angular-velocity seed.

    Spin is carried on ``player_touch`` anchors whose ``touch_type`` is
    ``"shot"`` or ``"volley"`` — the schema validates that pairing.
    Returns ``(omega_seed, hint_provided)``. ``hint_provided`` is True
    when an explicit non-``"none"`` preset was found — that flag drives
    the relaxed Magnus-acceptance threshold downstream.
    """
    for fi in range(a, b + 1):
        anc = anchor_by_frame.get(fi)
        if anc is None or anc.state not in SPIN_ENABLED_STATES:
            continue
        if not anc.spin or anc.spin == "none":
            continue
        return omega_seed_from_preset(anc.spin, v0), True
    return np.zeros(3, dtype=float), False


def _resolve_anchor_world(
    *,
    anc: BallAnchor,
    fi: int,
    ground_touch_frames: set[int],
    bone_lookup: "_BoneWorldLookup",
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ball_radius: float,
    goal_geometry: GoalGeometry,
) -> np.ndarray | None:
    """Single source of truth for resolving a hard-knot anchor to its
    world position. Called from every site that needs anchor world
    coordinates: the initial pin pass, the IMM-segment knot setup, the
    Phase 2 span knot setup, and the final end-of-run override.

    Rules:
      • ``goal_impact`` → intersect clicked-pixel ray with the goal
        element geometry (post / crossbar / back_net / side_net).
        Fallback (rare: ray parallel to surface) is ankle_ray_to_pitch
        at z = state_to_height("goal_impact") = 2.44 m.
      • ``player_touch`` ground-touch → clicked-pixel ray-cast at
        z = ball_radius. SMPL bone XY drifts 0.5–2 m due to monocular
        HMR depth ambiguity, so using it rubber-bands every dribble.
      • ``player_touch`` airborne → SMPL bone XYZ at the named body
        part. Fallback (bone lookup unavailable: missing player track,
        out-of-range frame, bad bone name) is ankle_ray_to_pitch at
        the player_touch default height of 1.0 m — NOT ball_radius,
        which would teleport the airborne touch to ground level.
      • All other hard-knot states → ankle_ray_to_pitch at the state's
        canonical height.
    """
    if anc.image_xy is None:
        return None
    K = per_frame_K.get(fi)
    R = per_frame_R.get(fi)
    t = per_frame_t.get(fi)
    if K is None or R is None or t is None:
        return None
    uv = (float(anc.image_xy[0]), float(anc.image_xy[1]))

    if anc.state == "goal_impact" and anc.goal_element is not None:
        try:
            return np.asarray(
                resolve_goal_impact_world(
                    uv, anc.goal_element,
                    K=K, R=R, t=t,
                    distortion=distortion, geometry=goal_geometry,
                ),
                dtype=float,
            )
        except Exception as exc:
            logger.debug(
                "ball goal_impact resolver failed at frame %d (%s): %s",
                fi, anc.goal_element, exc,
            )
            # Fall through to ankle_ray_to_pitch fallback below.

    if anc.state == "player_touch" and fi not in ground_touch_frames:
        bone_world = bone_lookup.bone_world(anc)
        if bone_world is not None:
            return np.asarray(bone_world, dtype=float)
        # Fall through to fallback ray-cast at z=1.0 below.

    # Ray-cast fallback path. Plane height depends on state semantics.
    if anc.state == "player_touch" and fi in ground_touch_frames:
        plane_z = ball_radius
    else:
        try:
            plane_z = state_to_height(anc.state)
        except ValueError:
            plane_z = ball_radius
    try:
        return np.asarray(
            ankle_ray_to_pitch(
                uv, K=K, R=R, t=t,
                plane_z=plane_z, distortion=distortion,
            ),
            dtype=float,
        )
    except Exception as exc:
        logger.debug("ball anchor projection failed at frame %d: %s", fi, exc)
        return None


def _apply_hard_knot_anchor_overrides(
    *,
    per_frame_world: dict[int, tuple[np.ndarray, float]],
    anchor_by_frame: dict[int, BallAnchor],
    ground_touch_frames: set[int],
    bone_lookup: "_BoneWorldLookup",
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ball_radius: float,
    goal_geometry: GoalGeometry,
) -> None:
    """Pin per-frame world positions for HARD_KNOT_STATES anchors.

    Idempotent. Every trajectory-writing pass (IMM parabola fit,
    promotion refit, Phase 2 fit, ground-level interp) is allowed to
    overwrite arbitrary frames; this helper then pulls anchored
    hard-knot frames back to the user's clicked pixel + state-height
    ray-cast (or SMPL bone for ``player_touch``, or goal-element
    geometry for ``goal_impact``). Anchors are the user's ground truth,
    so they win.
    """
    for fi, anc in anchor_by_frame.items():
        if anc.state not in HARD_KNOT_STATES:
            continue
        world = _resolve_anchor_world(
            anc=anc, fi=fi,
            ground_touch_frames=ground_touch_frames,
            bone_lookup=bone_lookup,
            per_frame_K=per_frame_K,
            per_frame_R=per_frame_R,
            per_frame_t=per_frame_t,
            distortion=distortion,
            ball_radius=ball_radius,
            goal_geometry=goal_geometry,
        )
        if world is None:
            continue
        per_frame_world[fi] = (world, 1.0)


class _BoneWorldLookup:
    """Resolve ``player_touch`` anchors to bone world positions via SMPL FK.

    Caches per-player SmplWorldTrack loads to avoid repeated NPZ reads.
    Returns ``None`` when the player track, the requested frame, or the
    bone name is unavailable — caller falls back to the pixel-only
    behaviour for that anchor.
    """

    def __init__(self, output_dir: Path, shot_id: str) -> None:
        from src.utils.ball_anchor_heights import BONE_TO_SMPL_INDEX

        self._output_dir = output_dir
        self._shot_id = shot_id
        self._bone_map = BONE_TO_SMPL_INDEX
        self._tracks: dict[str, object] = {}

    def _load_track(self, player_id: str) -> object | None:
        if player_id in self._tracks:
            return self._tracks[player_id]
        from src.schemas.smpl_world import SmplWorldTrack
        candidates = []
        if self._shot_id:
            candidates.append(
                self._output_dir / "hmr_world" / f"{self._shot_id}__{player_id}_smpl_world.npz"
            )
        candidates.append(
            self._output_dir / "hmr_world" / f"{player_id}_smpl_world.npz"
        )
        for path in candidates:
            if path.exists():
                try:
                    track = SmplWorldTrack.load(path)
                except Exception as exc:
                    logger.warning(
                        "ball stage: failed to load SMPL track %s: %s", path, exc
                    )
                    self._tracks[player_id] = None  # type: ignore[assignment]
                    return None
                self._tracks[player_id] = track
                return track
        logger.warning(
            "ball stage: no SmplWorldTrack found for player %r (shot=%r)",
            player_id, self._shot_id,
        )
        self._tracks[player_id] = None  # type: ignore[assignment]
        return None

    def bone_world(self, anchor: BallAnchor) -> np.ndarray | None:
        from src.utils.smpl_skeleton import compute_joint_world

        if anchor.state != "player_touch":
            return None
        if not anchor.player_id or not anchor.bone:
            return None
        joint_idx = self._bone_map.get(anchor.bone)
        if joint_idx is None:
            return None
        track = self._load_track(anchor.player_id)
        if track is None:
            return None
        frames = np.asarray(track.frames)  # type: ignore[attr-defined]
        match = np.where(frames == anchor.frame)[0]
        if len(match) == 0:
            logger.debug(
                "ball stage: SMPL track for %s has no frame %d",
                anchor.player_id, anchor.frame,
            )
            return None
        i = int(match[0])
        return compute_joint_world(
            track.thetas[i],         # type: ignore[attr-defined]
            track.root_R[i],         # type: ignore[attr-defined]
            track.root_t[i],         # type: ignore[attr-defined]
            joint_idx,
        )


class BallStage(BaseStage):
    name = "ball"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        ball_detector: BallDetector | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.ball_detector = ball_detector

    def is_complete(self) -> bool:
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return (self.output_dir / "ball" / "ball_track.json").exists()
        manifest = ShotsManifest.load(manifest_path)
        return all(
            (self.output_dir / "ball" / f"{shot.id}_ball_track.json").exists()
            for shot in manifest.shots
        )

    def run(self) -> None:
        cfg = self.config.get("ball", {})
        detector = self.ball_detector if self.ball_detector is not None else _build_detector(cfg)

        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            # Legacy single-shot path. Use the unprefixed file names.
            cam_path = self.output_dir / "camera" / "camera_track.json"
            ball_out = self.output_dir / "ball" / "ball_track.json"
            if not cam_path.exists():
                raise FileNotFoundError(
                    f"ball stage requires manifest at {manifest_path}; run prepare_shots first"
                )
            clip_path = self._guess_legacy_clip()
            self._run_shot("", clip_path, cam_path, ball_out, cfg, detector)
            return

        manifest = ShotsManifest.load(manifest_path)
        shot_filter = getattr(self, "shot_filter", None)
        for shot in manifest.shots:
            if shot_filter is not None and shot.id != shot_filter:
                continue
            cam_path = self.output_dir / "camera" / f"{shot.id}_camera_track.json"
            ball_out = self.output_dir / "ball" / f"{shot.id}_ball_track.json"
            if not cam_path.exists():
                logger.warning(
                    "ball stage skipping shot %s — no camera_track at %s",
                    shot.id, cam_path,
                )
                continue
            clip_path = self.output_dir / shot.clip_file
            if not clip_path.exists():
                logger.warning(
                    "ball stage skipping shot %s — clip missing at %s",
                    shot.id, clip_path,
                )
                continue
            self._run_shot(shot.id, clip_path, cam_path, ball_out, cfg, detector)

    def _guess_legacy_clip(self) -> Path:
        """Find a clip file under shots/ for the legacy no-manifest path."""
        shots_dir = self.output_dir / "shots"
        candidates = sorted(shots_dir.glob("*.mp4")) if shots_dir.exists() else []
        if not candidates:
            raise FileNotFoundError(
                f"ball stage: no clip files found under {shots_dir}"
            )
        return candidates[0]

    def _run_shot(
        self,
        shot_id: str,
        clip_path: Path,
        camera_path: Path,
        ball_out_path: Path,
        cfg: dict,
        detector: BallDetector,
    ) -> None:
        camera = CameraTrack.load(camera_path)
        per_frame_K = {f.frame: np.array(f.K) for f in camera.frames}
        per_frame_R = {f.frame: np.array(f.R) for f in camera.frames}
        t_world_fallback = np.array(camera.t_world)
        per_frame_t = {
            f.frame: (np.array(f.t) if f.t is not None else t_world_fallback)
            for f in camera.frames
        }
        distortion = camera.distortion
        n_frames = max(per_frame_K) + 1 if per_frame_K else 0

        ball_radius = float(cfg.get("ball_radius_m", 0.11))
        tracker_cfg = cfg.get("tracker", {})
        spin_cfg = cfg.get("spin", {})
        max_residual = float(cfg.get("flight_max_residual_px", 5.0))
        plaus_cfg = PlausibilityCfg(
            z_max_m=float(cfg.get("plausibility", {}).get("z_max_m", 50.0)),
            horizontal_speed_max_m_s=float(cfg.get("plausibility", {}).get("horizontal_speed_max_m_s", 40.0)),
            pitch_margin_m=float(cfg.get("plausibility", {}).get("pitch_margin_m", 5.0)),
        )
        pitch_cfg = self.config.get("pitch", {})
        pitch_dims = PitchDims(
            length_m=float(pitch_cfg.get("length_m", 105.0)),
            width_m=float(pitch_cfg.get("width_m", 68.0)),
        )
        goal_geometry = GoalGeometry.from_pitch_config(pitch_cfg)

        tracker = BallTracker(
            process_noise_grounded_px=float(tracker_cfg.get("process_noise_grounded_px", 4.0)),
            process_noise_flight_px=float(tracker_cfg.get("process_noise_flight_px", 12.0)),
            measurement_noise_px=float(tracker_cfg.get("measurement_noise_px", 2.0)),
            gating_sigma=float(tracker_cfg.get("gating_sigma", 4.0)),
            max_gap_frames=int(cfg.get("max_gap_frames", 6)),
            initial_p_flight=float(tracker_cfg.get("initial_p_flight", 0.1)),
        )

        feet_pixel_by_frame = _load_foot_uvs_for_shot(self.output_dir, shot_id)
        anchor_by_frame = _load_ball_anchors(self.output_dir, shot_id)
        bone_lookup = _BoneWorldLookup(self.output_dir, shot_id)
        if anchor_by_frame:
            logger.info(
                "ball stage: loaded %d anchors for shot %s",
                len(anchor_by_frame), shot_id or "(legacy)",
            )

        # Classify each player_touch by the surrounding anchors. The
        # ball is at ground level (z = ball radius) UNLESS the touch
        # sits between two airborne-implying anchors — only then is
        # the ball mid-flight at the contact (e.g. a volley between
        # two airborne anchors, a header between airborne_mid and
        # airborne_high). A ground-to-air transition (grounded → pt
        # → airborne) keeps the ball at ground level on the touch
        # frame: the kick launches it on the next frame, not at the
        # touch frame itself. This is robust to HMR foot Z drift
        # (~0.1–1.5 m depending on pose).
        # For a pt to be "mid-flight" (ball at bone height at the
        # contact), the ball must be airborne BOTH approaching and
        # leaving the touch.
        #   - approaching airborne: previous anchor is airborne_*,
        #     header/volley/chest/catch, off_screen_flight, bounce
        #     (ball was descending into the bounce), or kick (kick
        #     launches the ball — anything after kick until the next
        #     ground state is in flight).
        #   - leaving airborne: next anchor is airborne_*,
        #     header/volley/chest/catch, off_screen_flight, or bounce
        #     (ball was airborne until it hit the ground at bounce).
        #     NOT kick — kick is "ball on the ground here, then
        #     launches", which means the ball was on the ground at the
        #     pt frame.
        _PREV_AIRBORNE_STATES = frozenset({
            "airborne_low", "airborne_mid", "airborne_high",
            "header", "volley", "chest", "catch", "off_screen_flight",
            "bounce", "kick",
        })
        _NEXT_AIRBORNE_STATES = frozenset({
            "airborne_low", "airborne_mid", "airborne_high",
            "header", "volley", "chest", "catch", "off_screen_flight",
            "bounce",
        })
        sorted_anchor_frames = sorted(anchor_by_frame.keys())

        def _neighbor_implies_flight(
            idx: int, step: int, airborne_set: frozenset[str]
        ) -> bool:
            """Walk through any adjacent player_touch chain; return True
            if the first non-pt anchor in that direction is in the
            given airborne-implying set."""
            j = idx + step
            while 0 <= j < len(sorted_anchor_frames):
                anc_j = anchor_by_frame[sorted_anchor_frames[j]]
                if anc_j.state != "player_touch":
                    return anc_j.state in airborne_set
                j += step
            return False

        ground_touch_frames: set[int] = set()
        for idx in range(len(sorted_anchor_frames)):
            fi = sorted_anchor_frames[idx]
            anc = anchor_by_frame[fi]
            if anc.state != "player_touch":
                continue
            prev_flight = _neighbor_implies_flight(idx, -1, _PREV_AIRBORNE_STATES)
            next_flight = _neighbor_implies_flight(idx, +1, _NEXT_AIRBORNE_STATES)
            if not (prev_flight and next_flight):
                ground_touch_frames.add(fi)
        if ground_touch_frames:
            logger.info(
                "ball stage: %d player_touch anchor(s) classified as ground-level",
                len(ground_touch_frames),
            )
        if ground_touch_frames:
            logger.info(
                "ball stage: %d player_touch anchor(s) classified as ground-level",
                len(ground_touch_frames),
            )

        forced_flight: set[int] = {
            fi for fi, a in anchor_by_frame.items()
            if a.state in AIRBORNE_STATES and fi not in ground_touch_frames
        }
        # Raw anchor pixels keyed by frame for exact world-position override
        # after the tracker loop. Off_screen_flight anchors have no pixel and
        # are excluded here.
        anchor_pixels: dict[int, tuple[float, float]] = {
            fi: (float(a.image_xy[0]), float(a.image_xy[1]))
            for fi, a in anchor_by_frame.items()
            if a.image_xy is not None
        }
        kick_cfg = KickAnchorCfg(
            enabled=bool(cfg.get("kick_anchor", {}).get("enabled", True))
                    and bool(feet_pixel_by_frame),
            max_pixel_distance_px=float(cfg.get("kick_anchor", {}).get("max_pixel_distance_px", 30.0)),
            lookahead_frames=int(cfg.get("kick_anchor", {}).get("lookahead_frames", 4)),
            min_pixel_acceleration_px_per_frame=float(cfg.get("kick_anchor", {}).get("min_pixel_acceleration_px_per_frame", 6.0)),
            foot_anchor_z_m=float(cfg.get("kick_anchor", {}).get("foot_anchor_z_m", 0.11)),
        )
        if not feet_pixel_by_frame and cfg.get("kick_anchor", {}).get("enabled", True):
            logger.warning(
                "ball stage: kick_anchor enabled but no kp2d sidecars found under %s",
                self.output_dir / "hmr_world",
            )

        bridge_cfg = AppearanceBridgeCfg(
            enabled=bool(cfg.get("appearance_bridge", {}).get("enabled", True)),
            max_gap_frames=int(cfg.get("appearance_bridge", {}).get("max_gap_frames", 8)),
            template_size_px=int(cfg.get("appearance_bridge", {}).get("template_size_px", 32)),
            search_radius_px=int(cfg.get("appearance_bridge", {}).get("search_radius_px", 64)),
            min_ncc=float(cfg.get("appearance_bridge", {}).get("min_ncc", 0.6)),
            template_max_age_frames=int(cfg.get("appearance_bridge", {}).get("template_max_age_frames", 30)),
            template_update_confidence=float(cfg.get("appearance_bridge", {}).get("template_update_confidence", 0.5)),
        )
        bridge = AppearanceBridge(bridge_cfg)
        consecutive_misses = 0

        steps: list[TrackerStep] = []
        raw_confidences: dict[int, float] = {}
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                anchor = anchor_by_frame.get(frame_idx)
                if anchor is not None:
                    if anchor.state == "off_screen_flight":
                        # No pixel; let the IMM predict, but record the
                        # forced flight marker for the flight-run pass below.
                        uv: tuple[float, float] | None = None
                    else:
                        uv = (float(anchor.image_xy[0]), float(anchor.image_xy[1]))
                        raw_confidences[frame_idx] = 1.0
                        bridge.update_template(
                            frame=frame_idx, frame_image=frame,
                            uv=uv, confidence=1.0,
                        )
                    consecutive_misses = 0
                else:
                    det = detector.detect(frame)
                    if det is None:
                        consecutive_misses += 1
                        bridge_result = bridge.try_bridge(
                            frame=frame_idx,
                            frame_image=frame,
                            predicted_uv=(
                                (float(steps[-1].uv[0]), float(steps[-1].uv[1]))
                                if steps and steps[-1].uv is not None else None
                            ),
                            consecutive_misses=consecutive_misses,
                        )
                        if bridge_result is None:
                            uv = None
                        else:
                            uv, bridged_conf = bridge_result
                            raw_confidences[frame_idx] = bridged_conf
                    else:
                        consecutive_misses = 0
                        uv = (float(det[0]), float(det[1]))
                        raw_confidences[frame_idx] = float(det[2])
                        bridge.update_template(
                            frame=frame_idx,
                            frame_image=frame,
                            uv=uv,
                            confidence=float(det[2]),
                        )
                step = tracker.update(frame_idx, uv)
                steps.append(step)
                frame_idx += 1
        finally:
            cap.release()

        if frame_idx == 0:
            logger.warning("ball stage: clip %s contained no frames", clip_path)
            return

        n_frames = max(n_frames, frame_idx)

        # 3D ground projection of every smoothed step.
        # World positions far outside the pitch are dropped: when the
        # IMM-smoothed UV approaches the camera horizon the ray-to-plane
        # intersection blows up to hundreds (or thousands) of metres,
        # producing visible teleports in the 3D viewer. An honest
        # state="missing" is better than a wrong world position.
        offpitch_clamp_m = max(
            50.0, 2.0 * max(pitch_dims.length_m, pitch_dims.width_m)
        )
        per_frame_world: dict[int, tuple[np.ndarray, float]] = {}
        for step in steps:
            if step.uv is None:
                continue
            fi = step.frame
            if fi not in per_frame_K:
                continue
            try:
                world = ankle_ray_to_pitch(
                    step.uv,
                    K=per_frame_K[fi],
                    R=per_frame_R[fi],
                    t=per_frame_t[fi],
                    plane_z=ball_radius,
                    distortion=distortion,
                )
            except Exception as exc:
                logger.debug("ball ground projection failed at frame %d: %s", fi, exc)
                continue
            if (
                not np.all(np.isfinite(world))
                or abs(float(world[0])) > offpitch_clamp_m
                or abs(float(world[1])) > offpitch_clamp_m
            ):
                logger.debug(
                    "ball: dropping ground projection at frame %d — world "
                    "(%.1f, %.1f) far off-pitch (near-horizon ray-cast blow-up)",
                    fi, float(world[0]), float(world[1]),
                )
                continue
            base_conf = raw_confidences.get(fi, 0.5)
            # Gap-filled frames have no direct detection — discount.
            conf = base_conf * (0.3 if step.is_gap_fill else 1.0)
            per_frame_world[fi] = (world, conf)

        # Pin world positions for HARD_KNOT_STATES anchors to the exact
        # pixel + state-height ray-cast. This runs three times: here
        # (before any trajectory fitting), again after IMM + promotion
        # (so the ground-level interp endpoints come from anchor truth
        # not parabola eval), and finally at the end (so Phase 2 fits
        # never get the last word over the user's anchor).
        _apply_hard_knot_anchor_overrides(
            per_frame_world=per_frame_world,
            anchor_by_frame=anchor_by_frame,
            ground_touch_frames=ground_touch_frames,
            bone_lookup=bone_lookup,
            per_frame_K=per_frame_K,
            per_frame_R=per_frame_R,
            per_frame_t=per_frame_t,
            distortion=distortion,
            ball_radius=ball_radius,
            goal_geometry=goal_geometry,
        )

        # Flight segmentation by IMM mode posterior.
        min_flight = int(tracker_cfg.get("min_flight_frames", 6))
        max_flight = int(tracker_cfg.get("max_flight_frames", 90))
        flight_runs = self._flight_runs(steps, min_flight, max_flight)

        # Layer 5 — event-splitting: kick/catch/bounce anchors split flight runs.
        # Semantics: a cut at the run start (cut == a_run) keeps the event
        # frame as the start of the remaining segment (e.g. kick starts a new
        # arc here); a cut strictly inside the run excludes the event frame
        # from both sub-runs (e.g. bounce frame is ground contact, not in air).
        if anchor_by_frame:
            event_frames = sorted(
                fi for fi, a_ev in anchor_by_frame.items()
                if a_ev.state in EVENT_STATES
            )
            if event_frames:
                split_runs: list[tuple[int, int]] = []
                for (a_run, b_run) in flight_runs:
                    cuts = [fi for fi in event_frames if a_run <= fi <= b_run]
                    if not cuts:
                        split_runs.append((a_run, b_run))
                        continue
                    prev = a_run
                    for cut in cuts:
                        if cut - 1 >= prev:
                            split_runs.append((prev, cut - 1))
                        # When the cut is at the very start of the run there is
                        # no pre-segment; keep the event frame as the new start
                        # so that kick anchors remain in their flight segment.
                        prev = cut if cut == a_run else cut + 1
                    if prev <= b_run:
                        split_runs.append((prev, b_run))
                flight_runs = split_runs

        flight_segments: list[FlightSegment] = []
        flight_membership: dict[int, int] = {}
        spin_enabled = bool(spin_cfg.get("enabled", True))
        spin_min_seconds = float(spin_cfg.get("min_flight_seconds", 0.5))
        spin_min_improve = float(spin_cfg.get("min_residual_improvement", 0.2))
        spin_min_improve_hinted = float(
            spin_cfg.get("min_residual_improvement_with_hint", 0.05)
        )
        spin_max_omega = float(spin_cfg.get("max_omega_rad_s", 200.0))
        drag = float(spin_cfg.get("drag_k_over_m", 0.005))
        g = -9.81
        g_vec = np.array([0.0, 0.0, g])

        for sid, (a, b) in enumerate(flight_runs):
            obs_pairs = [
                (fi, steps[fi].uv)
                for fi in range(a, b + 1)
                if steps[fi].uv is not None and fi in per_frame_K
            ]
            if len(obs_pairs) < min_flight:
                continue
            obs = [(o[0], (float(o[1][0]), float(o[1][1]))) for o in obs_pairs]
            Ks_seg = [per_frame_K[o[0]] for o in obs]
            Rs_seg = [per_frame_R[o[0]] for o in obs]
            ts_seg = [per_frame_t[o[0]] for o in obs]

            ball_uvs_seg = {fi: uv for fi, uv in obs}
            anchor_world: np.ndarray | None = None
            if kick_cfg.enabled:
                # Pick the nearest foot per frame in the segment seed region.
                seed_feet: dict[int, tuple[float, float]] = {}
                for fi in range(a, min(a + kick_cfg.lookahead_frames + 1, b + 1)):
                    feet = feet_pixel_by_frame.get(fi, [])
                    if not feet or fi not in ball_uvs_seg:
                        continue
                    bu, bv = ball_uvs_seg[fi]
                    nearest = min(feet, key=lambda f: (f[0] - bu) ** 2 + (f[1] - bv) ** 2)
                    seed_feet[fi] = nearest
                if a in per_frame_K:
                    anchor_world = find_kick_anchor(
                        segment_start_frame=a,
                        ball_uvs=ball_uvs_seg,
                        foot_uvs_by_frame=seed_feet,
                        K=per_frame_K[a],
                        R=per_frame_R[a],
                        t=per_frame_t[a],
                        cfg=kick_cfg,
                        distortion=distortion,
                    )

            # Layer 5 — hard knots from anchored frames within this
            # segment. Uses the same resolver as the end-of-run override
            # so the parabola fit sees exactly the world position the
            # final track will emit at anchored frames.
            knot_frames_arg: dict[int, np.ndarray] = {}
            for fi in range(a, b + 1):
                anc = anchor_by_frame.get(fi)
                if anc is None or anc.state not in HARD_KNOT_STATES:
                    continue
                world_at_anchor = _resolve_anchor_world(
                    anc=anc, fi=fi,
                    ground_touch_frames=ground_touch_frames,
                    bone_lookup=bone_lookup,
                    per_frame_K=per_frame_K,
                    per_frame_R=per_frame_R,
                    per_frame_t=per_frame_t,
                    distortion=distortion,
                    ball_radius=ball_radius,
                    goal_geometry=goal_geometry,
                )
                if world_at_anchor is None:
                    continue
                knot_frames_arg[fi - a] = world_at_anchor

            # If the seed frame is a hard knot AND Layer 3 didn't set
            # anchor_world, promote frame 0 to p0_fixed.
            if 0 in knot_frames_arg and anchor_world is None:
                anchor_world = knot_frames_arg.pop(0)

            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                    p0_fixed=anchor_world,
                    knot_frames=knot_frames_arg or None,
                )
            except Exception as exc:
                logger.debug("parabola fit failed on segment %d: %s", sid, exc)
                continue
            if parab_resid > max_residual:
                continue
            segment_duration_s = (b - a) / camera.fps
            if not is_plausible_trajectory(
                p0, v0, omega=None,
                duration_s=segment_duration_s, fps=camera.fps,
                cfg=plaus_cfg, pitch=pitch_dims,
            ):
                logger.info(
                    "ball seg %d (%d-%d): parabola failed plausibility, dropping",
                    sid, a, b,
                )
                continue

            duration_s = (b - a) / camera.fps
            # Look up an explicit spin preset on the segment's start
            # anchor (kick/volley) — the user is telling us the strike
            # imparted spin. omega_seed defaults to zeros otherwise.
            omega_seed, hint_provided = _spin_seed_for_segment(
                anchor_by_frame, a, b, v0=v0,
            )
            refinement = _refine_with_magnus(
                obs=obs, Ks_seg=Ks_seg, Rs_seg=Rs_seg, ts_seg=ts_seg,
                fps=camera.fps, drag=drag,
                plaus_cfg=plaus_cfg, pitch_dims=pitch_dims,
                p0=p0, v0=v0, parab_resid=parab_resid,
                anchor_world=anchor_world, duration_s=duration_s,
                spin_enabled=spin_enabled,
                spin_min_seconds=spin_min_seconds,
                spin_max_omega=spin_max_omega,
                spin_min_improve=spin_min_improve,
                spin_min_improve_hinted=spin_min_improve_hinted,
                omega_seed=omega_seed,
                hint_provided=hint_provided,
                segment_label=f"segment {sid}",
                knot_frames=knot_frames_arg,
            )
            effective_p0 = refinement.effective_p0
            effective_v0 = refinement.effective_v0
            effective_resid = refinement.effective_resid
            omega_world = refinement.omega_world
            spin_axis = refinement.spin_axis
            spin_omega = refinement.spin_omega
            spin_confidence = refinement.spin_confidence

            # Replace per-frame world_xyz inside the flight with the fitted
            # trajectory evaluation. Preserves original BallStage behaviour
            # and gives clean curves through the gltf export.
            for fi in range(a, b + 1):
                if fi not in per_frame_K:
                    continue
                flight_membership[fi] = sid
                dt = (fi - a) / camera.fps
                if omega_world is not None:
                    positions = _integrate_magnus_positions(
                        effective_p0,
                        effective_v0,
                        omega_world,
                        g_vec,
                        drag,
                        np.array([0.0, dt]),
                    )
                    pos = positions[-1]
                else:
                    pos = effective_p0 + effective_v0 * dt + 0.5 * g_vec * dt ** 2
                prev_conf = per_frame_world.get(fi, (None, 0.5))[1]
                per_frame_world[fi] = (pos, prev_conf)

            flight_segments.append(
                FlightSegment(
                    id=sid,
                    frame_range=(a, b),
                    parabola={
                        "p0": [float(x) for x in effective_p0],
                        "v0": [float(x) for x in effective_v0],
                        "g": g,
                        "spin_axis_world": spin_axis,
                        "spin_omega_rad_s": spin_omega,
                        "spin_confidence": spin_confidence,
                    },
                    fit_residual_px=effective_resid,
                )
            )

        promote_cfg = GroundPromotionCfg(
            enabled=bool(cfg.get("flight_promotion", {}).get("enabled", True)),
            min_run_frames=int(cfg.get("flight_promotion", {}).get("min_run_frames", 6)),
            off_pitch_margin_m=float(cfg.get("flight_promotion", {}).get("off_pitch_margin_m", 5.0)),
            max_ground_speed_m_s=float(cfg.get("flight_promotion", {}).get("max_ground_speed_m_s", 35.0)),
        )

        # Provisional state map matching what would be emitted below.
        provisional_state: dict[int, str] = {}
        for fi in range(n_frames):
            if fi in per_frame_world:
                provisional_state[fi] = "flight" if fi in flight_membership else "grounded"
            else:
                provisional_state[fi] = "missing"

        runs_to_promote = find_implausible_grounded_runs(
            per_frame_xyz=per_frame_world,
            per_frame_state=provisional_state,
            fps=camera.fps,
            cfg=promote_cfg,
            pitch=pitch_dims,
        )

        # Frames where the user has explicitly anchored a non-flight
        # intent: grounded, kick, catch, bounce, or a ground-touch
        # player_touch. The promotion stage must not lift these into a
        # parabola — doing so would override the user's ground truth.
        non_flight_anchored: set[int] = {
            fi for fi, a in anchor_by_frame.items()
            if a.state in ("grounded", "kick", "catch", "bounce")
            or (a.state == "player_touch" and fi in ground_touch_frames)
        }

        next_segment_id = (max(flight_membership.values()) + 1) if flight_membership else 0
        min_flight_frames_for_refit = int(tracker_cfg.get("min_flight_frames", 6))
        for run in runs_to_promote:
            if any(run.start <= fi <= run.end for fi in non_flight_anchored):
                logger.info(
                    "ball: promotion skipped run %d-%d — overlaps user "
                    "non-flight anchor",
                    run.start, run.end,
                )
                continue
            obs_pairs = [
                (fi, steps[fi].uv) for fi in range(run.start, run.end + 1)
                if 0 <= fi < len(steps) and steps[fi].uv is not None and fi in per_frame_K
            ]
            if len(obs_pairs) < min_flight_frames_for_refit:
                continue
            obs = [(o[0], (float(o[1][0]), float(o[1][1]))) for o in obs_pairs]
            Ks_seg = [per_frame_K[o[0]] for o in obs]
            Rs_seg = [per_frame_R[o[0]] for o in obs]
            ts_seg = [per_frame_t[o[0]] for o in obs]
            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                )
            except Exception as exc:
                # Refit failure means the data isn't actually a clean
                # flight arc — the original ground projection (noisy but
                # bounded) is a better fallback than nothing.
                logger.debug("promotion refit failed at run %d-%d: %s — leaving as grounded",
                             run.start, run.end, exc)
                continue
            if parab_resid > max_residual:
                logger.info(
                    "ball: promotion refit for run %d-%d residual %.1f px > "
                    "%.1f px cap, leaving as grounded",
                    run.start, run.end, parab_resid, max_residual,
                )
                continue
            seg_duration = (run.end - run.start) / camera.fps
            if not is_plausible_trajectory(
                p0, v0, omega=None,
                duration_s=seg_duration, fps=camera.fps,
                cfg=plaus_cfg, pitch=pitch_dims,
            ):
                logger.info(
                    "ball: promotion refit for run %d-%d failed plausibility; "
                    "leaving as grounded",
                    run.start, run.end,
                )
                continue

            sid_new = next_segment_id
            next_segment_id += 1
            for fi in range(run.start, run.end + 1):
                if fi not in per_frame_K:
                    continue
                dt = (fi - run.start) / camera.fps
                pos = p0 + v0 * dt + 0.5 * g_vec * dt ** 2
                prev_conf = per_frame_world.get(fi, (None, 0.5))[1]
                per_frame_world[fi] = (pos, prev_conf)
                flight_membership[fi] = sid_new
            flight_segments.append(
                FlightSegment(
                    id=sid_new,
                    frame_range=(run.start, run.end),
                    parabola={
                        "p0": [float(x) for x in p0],
                        "v0": [float(x) for x in v0],
                        "g": g,
                        "spin_axis_world": None,
                        "spin_omega_rad_s": None,
                        "spin_confidence": None,
                    },
                    fit_residual_px=parab_resid,
                )
            )

        # Pull anchored hard-knot frames back to their exact state-
        # height ray-cast. IMM segments and promotion refits may have
        # written parabola values over the user's anchors; the
        # ground-level interp pass below reads pa/pb from
        # per_frame_world, so this MUST happen before interp or the
        # interp endpoints carry the parabola error into every
        # in-between frame.
        _apply_hard_knot_anchor_overrides(
            per_frame_world=per_frame_world,
            anchor_by_frame=anchor_by_frame,
            ground_touch_frames=ground_touch_frames,
            bone_lookup=bone_lookup,
            per_frame_K=per_frame_K,
            per_frame_R=per_frame_R,
            per_frame_t=per_frame_t,
            distortion=distortion,
            ball_radius=ball_radius,
            goal_geometry=goal_geometry,
        )

        # Layer 5: forced-flight frames from airborne_* / off_screen_flight
        # anchors. We do NOT create FlightSegment entries here — the user
        # marked the frame airborne but we have no parabola data to fit
        # (single-frame runs are not real flights). Instead the BallFrame
        # assembly below uses the `forced_flight` set directly to classify
        # those frames as state="flight". Avoids polluting the segments
        # table with zero-parabola placeholders.

        # Layer 5: linear interpolation between consecutive ground-level
        # anchors. grounded / kick / bounce are all physically at
        # z = 0.11 m, so XY interpolates smoothly between them along
        # the pitch surface. catch is not in this set (z = 1.5 m); any
        # airborne / header / volley / chest anchor between two
        # ground-level anchors blocks the interp (the ball was airborne
        # in between).
        if anchor_by_frame:
            # ground-level pool: grounded/kick/bounce anchors PLUS any
            # player_touch whose bone Z is below the ground threshold
            # (small dribble / short ground-pass touches).
            ground_level_frames = sorted(
                fi for fi, a in anchor_by_frame.items()
                if (
                    (a.state in GROUND_LEVEL_STATES and a.image_xy is not None)
                    or fi in ground_touch_frames
                )
            )
            for i in range(len(ground_level_frames) - 1):
                fa = ground_level_frames[i]
                fb = ground_level_frames[i + 1]
                if fb - fa <= 1:
                    continue
                # Skip if any anchor that is NOT ground-level lies
                # strictly between fa and fb. Other ground-level anchors
                # — and ground-touch player_touches — are fine.
                if any(
                    fa < fi < fb
                    and anchor_by_frame[fi].state not in GROUND_LEVEL_STATES
                    and fi not in ground_touch_frames
                    for fi in anchor_by_frame.keys()
                ):
                    continue
                wa = per_frame_world.get(fa)
                wb = per_frame_world.get(fb)
                if wa is None or wb is None:
                    continue
                pa, _ = wa
                pb, _ = wb
                span = fb - fa
                # Smooth ground-level interpolation: fit a single
                # quadratic curve from anchor A through the kept WASB
                # observations to anchor B.
                #
                #   pos(t) = line(t) + D · t · (1−t)
                #
                # where line(t) = (1−t)·A + t·B and D is a 2-vector
                # "bulge" magnitude fit by least squares to the
                # WASB-vs-line residuals. The quadratic
                # • passes through both anchors exactly (no
                #   discontinuity at the touch),
                # • follows the actual rolling trajectory (no
                #   straight-line lag the user used to see), and
                # • has no frame-to-frame jitter (the curve is
                #   analytical, not a sample-by-sample copy of WASB).
                #
                # WASB observations are filtered for sanity: must be
                # on the pitch and within a generous distance of the
                # anchor-to-anchor line. Bogus detections (WASB
                # locking onto a player's foot or line marking, or a
                # ground-projection of an airborne ball) are dropped
                # before the LSQ fit.
                _wasb_offline_tolerance_m = max(2.0, 0.4 * float(np.linalg.norm(pb[:2] - pa[:2])))
                ts: list[float] = []
                residuals_xy: list[np.ndarray] = []
                for fi in range(fa + 1, fb):
                    existing = per_frame_world.get(fi)
                    if existing is None:
                        continue
                    t_frac = (fi - fa) / span
                    line_pos = pa[:2] * (1.0 - t_frac) + pb[:2] * t_frac
                    pos_existing = np.asarray(existing[0][:2])
                    on_pitch = (
                        -plaus_cfg.pitch_margin_m
                        <= pos_existing[0]
                        <= pitch_dims.length_m + plaus_cfg.pitch_margin_m
                        and -plaus_cfg.pitch_margin_m
                        <= pos_existing[1]
                        <= pitch_dims.width_m + plaus_cfg.pitch_margin_m
                    )
                    offline = float(np.linalg.norm(pos_existing - line_pos))
                    if not on_pitch or offline > _wasb_offline_tolerance_m:
                        continue
                    ts.append(t_frac)
                    residuals_xy.append(pos_existing - line_pos)
                if ts:
                    ts_arr = np.asarray(ts)
                    res_arr = np.asarray(residuals_xy)
                    weights = ts_arr * (1.0 - ts_arr)
                    denom = float(np.sum(weights * weights))
                    if denom > 1e-6:
                        bulge_xy = (res_arr * weights[:, None]).sum(axis=0) / denom
                    else:
                        bulge_xy = np.zeros(2)
                else:
                    bulge_xy = np.zeros(2)
                # Cap the bulge so the interp curve cannot overshoot
                # past either anchor and cannot swing more than half the
                # anchor-to-anchor distance perpendicular to the line.
                # Without this cap the LSQ readily produces |D| ≫ |AB|
                # when a handful of WASB observations land on the same
                # side of the line, and the ball "rubber-bands" past
                # the next anchor and back. The constraint |D_par| ≤
                # |AB| is exactly the no-overshoot bound on the
                # quadratic pos(t) = (1−t)A + tB + D·t·(1−t); the
                # symmetric |D_perp| ≤ |AB| keeps sideways swings
                # bounded by the same scale.
                ab_vec = np.asarray(pb[:2] - pa[:2], dtype=float)
                ab_len = float(np.linalg.norm(ab_vec))
                if ab_len > 1e-6:
                    ab_unit = ab_vec / ab_len
                    d_par = float(np.dot(bulge_xy, ab_unit))
                    d_perp_vec = bulge_xy - d_par * ab_unit
                    d_par = max(-ab_len, min(ab_len, d_par))
                    d_perp_mag = float(np.linalg.norm(d_perp_vec))
                    if d_perp_mag > ab_len:
                        d_perp_vec = d_perp_vec * (ab_len / d_perp_mag)
                    bulge_xy = d_par * ab_unit + d_perp_vec
                for fi in range(fa + 1, fb):
                    flight_membership.pop(fi, None)
                    t_frac = (fi - fa) / span
                    line_pos_xy = pa[:2] * (1.0 - t_frac) + pb[:2] * t_frac
                    bulge = bulge_xy * (t_frac * (1.0 - t_frac))
                    pos = np.array([
                        line_pos_xy[0] + bulge[0],
                        line_pos_xy[1] + bulge[1],
                        ball_radius,
                    ])
                    per_frame_world[fi] = (pos, 0.9)

        # Layer 5 Phase 2: parabola fit through maximal non-grounded
        # anchor spans. For each contiguous run of non-grounded anchors
        # (no grounded anchor between them), gather all anchor pixels
        # as observations and all anchor-state heights as soft knots,
        # then fit a parabola. If the fit is plausible, fill the span's
        # unanchored frames with the parabola's per-frame evaluation
        # and add a real FlightSegment. The linear-interp pass below
        # serves as the fallback when the parabola fit fails or the
        # span has too few pixels.
        parabola_handled_spans: list[tuple[int, int]] = []
        if anchor_by_frame:
            ordered_for_spans = sorted(anchor_by_frame.items(), key=lambda kv: kv[0])
            _NON_GROUNDED = AIRBORNE_STATES | EVENT_STATES
            spans: list[list[tuple[int, BallAnchor]]] = []
            # Span boundary rules:
            #   grounded  → close current span (state change).
            #   kick      → close current AND start a new span (kick is
            #               the start of a fresh flight). kick is in the
            #               new span only.
            #   bounce    → close current including the bounce (flight
            #   catch       ends here; subsequent flight needs a fresh
            #               kick). Event is in the old span only.
            #   header    → close current including the header AND start
            #               a new span starting with the header (head
            #               contact ends one parabola and starts the
            #               next). Header is in BOTH adjacent spans.
            #   airborne_*, off_screen_flight → continues current.
            current_span: list[tuple[int, BallAnchor]] = []
            for fi, anc in ordered_for_spans:
                if anc.state == "grounded":
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = []
                elif anc.state == "kick":
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = [(fi, anc)]
                elif anc.state == "player_touch" and fi in ground_touch_frames:
                    # Ground-touch player_touch (dribble / short ground
                    # pass): the ball comes down to the player's foot
                    # but might or might not launch back up — depends
                    # on the NEXT anchor. Behaves like `bounce`: close
                    # the current span AND start a new span with the
                    # touch as the bookend. If the next anchor is also
                    # ground-level, the new span ends up with only
                    # ground-level anchors and is dropped by the
                    # has_flight_evidence filter below — leaving the
                    # gap to be filled by the ground-level interp pass
                    # above. If the next anchor is airborne, the new
                    # span carries the touch + airborne anchors and is
                    # parabola-fit.
                    current_span.append((fi, anc))
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = [(fi, anc)]
                elif anc.state in (
                    "header", "volley", "chest", "bounce",
                    "player_touch", "goal_impact",
                ):
                    # Contact events that are both an end and a start of
                    # a flight: header/volley/chest/player_touch/
                    # goal_impact are mid-flight contacts (body part or
                    # goal frame); bounce is a ground contact that may
                    # launch the ball back up to a subsequent airborne
                    # event (e.g. bounce → volley apex). The event frame
                    # is in BOTH adjacent spans so Phase 2 fits the
                    # incoming parabola to END at the contact point and
                    # the outgoing parabola to START from it — without
                    # this, the surrounding trajectory is fit ignoring
                    # the contact and visibly teleports to/from the
                    # pinned impact frame.
                    current_span.append((fi, anc))
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = [(fi, anc)]
                elif anc.state == "catch":
                    current_span.append((fi, anc))
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = []
                elif anc.state in _NON_GROUNDED:
                    current_span.append((fi, anc))
                else:
                    if len(current_span) >= 2:
                        spans.append(current_span)
                    current_span = []
            if len(current_span) >= 2:
                spans.append(current_span)

            for span in spans:
                fa_span = span[0][0]
                fb_span = span[-1][0]
                # Defensive filter: drop spans where every anchor is
                # ground-level (grounded/bounce, or ground-touch
                # player_touch). These represent ground passes /
                # dribbles, not flights, and are handled by the
                # ground-level interp pass above. `kick` counts as
                # flight evidence because it marks the start of a
                # flight, even though the ball is at z=0.11 at the
                # kick frame.
                has_flight_evidence = any(
                    anc.state == "kick"
                    or (
                        anc.state in AIRBORNE_STATES
                        and not (
                            anc.state == "player_touch"
                            and fi in ground_touch_frames
                        )
                    )
                    for fi, anc in span
                )
                if not has_flight_evidence:
                    continue
                # Build obs from every anchor pixel in the span. Only
                # HARD_KNOT_STATES contribute to knot_frames — airborne
                # bucket heights (1/6/15 m) are too coarse to pin Z,
                # and using them as soft constraints would pull the fit
                # toward the wrong height when the user picks the wrong
                # bucket. The pixel observations alone constrain XY
                # along each camera ray; gravity + hard knots constrain
                # Z.
                obs_p2: list[tuple[int, tuple[float, float]]] = []
                Ks_p2: list[np.ndarray] = []
                Rs_p2: list[np.ndarray] = []
                ts_p2: list[np.ndarray] = []
                knots: dict[int, np.ndarray] = {}
                z_ranges: dict[int, tuple[float, float]] = {}
                p0_pin: np.ndarray | None = None
                for fi, anc in span:
                    if anc.image_xy is None or fi not in per_frame_K:
                        continue
                    obs_p2.append((fi, (float(anc.image_xy[0]), float(anc.image_xy[1]))))
                    Ks_p2.append(per_frame_K[fi])
                    Rs_p2.append(per_frame_R[fi])
                    ts_p2.append(per_frame_t[fi])
                    rel = fi - fa_span
                    if anc.state in HARD_KNOT_STATES:
                        # Use the same resolver as the end-of-run
                        # override so the parabola fit sees the exact
                        # world position the final track will emit.
                        # Previously this site used bone_world for
                        # ground-touch player_touch (wrong: bone XY
                        # drifts), causing p0_pin to be 1–2 m off and
                        # the LM to produce nonsense v0 for the shot.
                        world_at_anchor = _resolve_anchor_world(
                            anc=anc, fi=fi,
                            ground_touch_frames=ground_touch_frames,
                            bone_lookup=bone_lookup,
                            per_frame_K=per_frame_K,
                            per_frame_R=per_frame_R,
                            per_frame_t=per_frame_t,
                            distortion=distortion,
                            ball_radius=ball_radius,
                            goal_geometry=goal_geometry,
                        )
                        if world_at_anchor is None:
                            continue
                        knots[rel] = world_at_anchor
                    else:
                        # airborne_low/mid/high → one-sided Z hinge that
                        # forces z into the bucket range but lets the
                        # pixel obs determine the exact value inside it.
                        bucket = airborne_bucket_range(anc.state)
                        if bucket is not None:
                            z_ranges[rel] = bucket
                # Need at least 2 anchors. With p0 pinned to a hard-knot
                # or airborne-start ray-cast, the fit only optimises v0
                # (3 unknowns). One additional anchor (pixel obs or
                # knot) gives 2-3 residuals — enough to determine v0
                # exactly for a kick→bounce or kick→airborne pair.
                if len(obs_p2) < 2:
                    continue
                # Pin p0 to a known world position when possible:
                #   - Hard-knot start (kick/header/etc.): use its exact
                #     state-height ray-cast (already in knots[0]).
                #   - Airborne start (no kick anchor was placed): use
                #     the bucket-midpoint ray-cast. The LM otherwise has
                #     free reign over p0 along the camera ray and drifts
                #     several metres from where the user clicked.
                first_anc = span[0][1]
                if 0 in knots and first_anc.state in HARD_KNOT_STATES:
                    p0_pin = knots.pop(0)
                elif (
                    0 not in knots
                    and first_anc.state in AIRBORNE_STATES
                    and first_anc.image_xy is not None
                    and fa_span in per_frame_K
                ):
                    try:
                        p0_pin = ankle_ray_to_pitch(
                            first_anc.image_xy,
                            K=per_frame_K[fa_span], R=per_frame_R[fa_span], t=per_frame_t[fa_span],
                            plane_z=state_to_height(first_anc.state),
                            distortion=distortion,
                        )
                        p0_pin = np.asarray(p0_pin, dtype=float)
                        # Drop the z-range hinge at rel=0 since position
                        # is now pinned directly.
                        z_ranges.pop(0, None)
                    except (ValueError, Exception):
                        p0_pin = None
                try:
                    p2_p0, p2_v0, p2_resid = fit_parabola_to_image_observations(
                        obs_p2, Ks=Ks_p2, Rs=Rs_p2, t_world=ts_p2,
                        fps=camera.fps, distortion=distortion,
                        p0_fixed=p0_pin, knot_frames=knots or None,
                        z_range_frames=z_ranges or None,
                    )
                except Exception as exc:
                    logger.debug(
                        "Phase 2 parabola fit failed for span %d-%d: %s",
                        fa_span, fb_span, exc,
                    )
                    continue
                duration_s = (fb_span - fa_span) / camera.fps
                if duration_s <= 0:
                    continue
                # Phase 2 uses a LOOSER plausibility than Layer 1: the
                # user explicitly anchored every frame in this span, so
                # we trust their intent even when bucket choices imply a
                # slightly off-pitch or very high arc. Layer 1's strict
                # bounds (pitch margin, apex 50 m, speed 40 m/s + 5)
                # would reject perfectly usable fits when the user's
                # bucket misclassifies the ball's height by a few metres.
                # We still reject NaN / runaway parameters as obvious
                # nonsense.
                if (
                    not np.all(np.isfinite(p2_p0))
                    or not np.all(np.isfinite(p2_v0))
                    or float(np.linalg.norm(p2_v0)) > 100.0
                    or float(np.max(np.abs(p2_p0))) > 1000.0
                ):
                    logger.info(
                        "Phase 2 fit for span %d-%d returned non-finite or "
                        "runaway parameters (p0=%s v0=%s) — falling back",
                        fa_span, fb_span,
                        np.asarray(p2_p0).round(1).tolist(),
                        np.asarray(p2_v0).round(1).tolist(),
                    )
                    continue
                if not is_plausible_trajectory(
                    p2_p0, p2_v0, omega=None,
                    duration_s=duration_s, fps=camera.fps,
                    cfg=plaus_cfg, pitch=pitch_dims,
                ):
                    logger.info(
                        "Phase 2 fit for span %d-%d failed Layer 1 plausibility but "
                        "passed sanity bounds — accepting (user-anchored trajectories "
                        "are trusted over the strict envelope)",
                        fa_span, fb_span,
                    )
                # Success: drop any pre-existing flight segments inside
                # this span (the parabola we just fit is the authoritative
                # answer) and emit a single new segment.
                surviving: list[FlightSegment] = []
                for seg in flight_segments:
                    seg_a, seg_b = seg.frame_range
                    if seg_a >= fa_span and seg_b <= fb_span:
                        for fi in range(seg_a, seg_b + 1):
                            if flight_membership.get(fi) == seg.id:
                                flight_membership.pop(fi, None)
                        continue
                    surviving.append(seg)
                flight_segments[:] = surviving

                sid_new = (max(flight_membership.values()) + 1) if flight_membership else 0
                # Avoid colliding with any segment IDs still in the list.
                existing_ids = {s.id for s in flight_segments}
                while sid_new in existing_ids:
                    sid_new += 1
                g_vec = np.array([0.0, 0.0, -9.81])
                # Attempt Magnus refinement on the Phase 2 parabola when
                # the span's start anchor (kick / volley) carries a spin
                # preset. Without a hint Magnus still runs but at the
                # strict acceptance threshold — equivalent to the IMM
                # path's behaviour, and a no-op for most spans.
                omega_seed_p2, hint_provided_p2 = _spin_seed_for_segment(
                    anchor_by_frame, fa_span, fb_span, v0=p2_v0,
                )
                refinement = _refine_with_magnus(
                    obs=obs_p2, Ks_seg=Ks_p2, Rs_seg=Rs_p2, ts_seg=ts_p2,
                    fps=camera.fps, drag=drag,
                    plaus_cfg=plaus_cfg, pitch_dims=pitch_dims,
                    p0=(p0_pin if p0_pin is not None else p2_p0),
                    v0=p2_v0, parab_resid=p2_resid,
                    anchor_world=p0_pin,
                    duration_s=duration_s,
                    spin_enabled=spin_enabled,
                    spin_min_seconds=spin_min_seconds,
                    spin_max_omega=spin_max_omega,
                    spin_min_improve=spin_min_improve,
                    spin_min_improve_hinted=spin_min_improve_hinted,
                    omega_seed=omega_seed_p2,
                    hint_provided=hint_provided_p2,
                    segment_label=f"phase-2 span {fa_span}-{fb_span}",
                    knot_frames=knots,
                )
                # Inside a successful Phase 2 span the parabola (or
                # Magnus-refined trajectory) is the authoritative answer
                # for every frame — including the anchored ones, whose
                # pixels constrained the fit. The per-anchor ray-cast at
                # bucket heights would otherwise land 10s of metres off
                # the truth for coarse buckets.
                for fi in range(fa_span, fb_span + 1):
                    forced_flight.add(fi)
                    flight_membership[fi] = sid_new
                    dt_k = (fi - fa_span) / camera.fps
                    if refinement.omega_world is not None:
                        positions = _integrate_magnus_positions(
                            refinement.effective_p0,
                            refinement.effective_v0,
                            refinement.omega_world,
                            g_vec, drag,
                            np.array([0.0, dt_k]),
                        )
                        pos = positions[-1]
                    else:
                        pos = (
                            refinement.effective_p0
                            + refinement.effective_v0 * dt_k
                            + 0.5 * (dt_k ** 2) * g_vec
                        )
                    per_frame_world[fi] = (pos, 0.92)
                flight_segments.append(FlightSegment(
                    id=sid_new,
                    frame_range=(fa_span, fb_span),
                    parabola={
                        "p0": [float(x) for x in refinement.effective_p0],
                        "v0": [float(x) for x in refinement.effective_v0],
                        "g": -9.81,
                        "spin_axis_world": refinement.spin_axis,
                        "spin_omega_rad_s": refinement.spin_omega,
                        "spin_confidence": refinement.spin_confidence,
                    },
                    fit_residual_px=refinement.effective_resid,
                ))
                parabola_handled_spans.append((fa_span, fb_span))

        # Layer 5 Phase 1: forced-flight extension for non-grounded spans
        # that Phase 2 did not cover. We mark frames between consecutive
        # non-grounded anchors as state="flight" but DO NOT emit a
        # world_xyz for unanchored or airborne-anchored frames — the
        # bucket-midpoint ray-cast was producing 50+ metre depth swings
        # between adjacent airborne_low/mid/high anchors and showing up
        # as a top-down zigzag. Honest gaps are better than wrong dots.
        # Hard-knot anchored frames (kick/catch/bounce/grounded — set
        # earlier via the exact-world override) keep their world_xyz.
        def _in_handled_span(fi: int) -> bool:
            for sa, sb in parabola_handled_spans:
                if sa <= fi <= sb:
                    return True
            return False

        if anchor_by_frame:
            ordered = sorted(anchor_by_frame.items(), key=lambda kv: kv[0])
            _NON_GROUNDED = AIRBORNE_STATES | EVENT_STATES
            for (fa, anc_a), (fb, anc_b) in zip(ordered, ordered[1:]):
                if anc_a.state == "grounded" or anc_b.state == "grounded":
                    continue
                if anc_a.state not in _NON_GROUNDED or anc_b.state not in _NON_GROUNDED:
                    continue
                # If BOTH endpoints are ground-level (e.g. bounce→kick,
                # kick→bounce), the ground-level linear-interp pass
                # already filled the world XY at z=0.11. Don't clobber
                # it — that pair represents the ball rolling, not flying.
                if (
                    anc_a.state in GROUND_LEVEL_STATES
                    and anc_b.state in GROUND_LEVEL_STATES
                ):
                    continue
                if fb - fa <= 1:
                    continue
                if _in_handled_span(fa) and _in_handled_span(fb):
                    continue
                # Mark every in-between frame as flight (state-only).
                # Also clear any world_xyz the IMM/ground-projection
                # already wrote — those are bucket-midpoint ray-casts
                # at bogus heights for airborne frames.
                for fi in range(fa + 1, fb):
                    forced_flight.add(fi)
                    flight_membership.pop(fi, None)
                    per_frame_world.pop(fi, None)
                # Also drop world_xyz at airborne-anchored endpoints —
                # the bucket-midpoint ray-cast there is just as wrong as
                # the in-between frames. Hard-knot endpoints keep their
                # exact-height ray-cast.
                for fi_endpoint, anc_endpoint in ((fa, anc_a), (fb, anc_b)):
                    if anc_endpoint.state not in HARD_KNOT_STATES:
                        per_frame_world.pop(fi_endpoint, None)

        # Final hard-knot pin: Phase 2 and Phase 1 may have written
        # parabola eval values over body-part contact anchors (header /
        # volley / chest / catch) inside a span. Anchors are the user's
        # ground truth, so they take the last word.
        _apply_hard_knot_anchor_overrides(
            per_frame_world=per_frame_world,
            anchor_by_frame=anchor_by_frame,
            ground_touch_frames=ground_touch_frames,
            bone_lookup=bone_lookup,
            per_frame_K=per_frame_K,
            per_frame_R=per_frame_R,
            per_frame_t=per_frame_t,
            distortion=distortion,
            ball_radius=ball_radius,
            goal_geometry=goal_geometry,
        )

        per_frame_out: list[BallFrame] = []
        for fi in range(n_frames):
            in_flight = fi in flight_membership or fi in forced_flight
            if fi in per_frame_world:
                world, conf = per_frame_world[fi]
                state = "flight" if in_flight else "grounded"
                per_frame_out.append(
                    BallFrame(
                        frame=fi,
                        world_xyz=tuple(float(x) for x in world),
                        state=state,
                        confidence=float(conf),
                        flight_segment_id=flight_membership.get(fi),
                    )
                )
            else:
                if in_flight:
                    per_frame_out.append(
                        BallFrame(
                            frame=fi,
                            world_xyz=None,
                            state="flight",
                            confidence=0.0,
                            flight_segment_id=flight_membership.get(fi),
                        )
                    )
                else:
                    per_frame_out.append(
                        BallFrame(
                            frame=fi,
                            world_xyz=None,
                            state="missing",
                            confidence=0.0,
                        )
                    )

        track = BallTrack(
            clip_id=camera.clip_id,
            fps=camera.fps,
            frames=tuple(per_frame_out),
            flight_segments=tuple(flight_segments),
        )
        track.save(ball_out_path)

    @staticmethod
    def _flight_runs(
        steps: list[TrackerStep], min_flight: int, max_flight: int
    ) -> list[tuple[int, int]]:
        """Run-length encode frames with ``p_flight >= 0.5``.

        Returns ``(start_frame, end_frame)`` pairs, both inclusive.
        Runs shorter than ``min_flight`` or longer than ``max_flight``
        are dropped (long runs are likely tracker confusion, not real
        flights).
        """
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for step in steps:
            in_flight = step.p_flight >= 0.5 and step.uv is not None
            if in_flight and start is None:
                start = step.frame
            elif not in_flight and start is not None:
                end = step.frame - 1
                if min_flight <= (end - start + 1) <= max_flight:
                    runs.append((start, end))
                start = None
        if start is not None and steps:
            end = steps[-1].frame
            if min_flight <= (end - start + 1) <= max_flight:
                runs.append((start, end))
        return runs

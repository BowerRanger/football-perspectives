"""Camera-tracking stage: anchors + propagation + smoothing → camera_track.json."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.anchor import Anchor, AnchorSet
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.anchor_solver import (
    AnchorSolveError,
    _estimate_lens_from_best_anchor,
    _estimate_lens_jointly,
    refine_with_shared_translation,
    reprojection_residual_for_anchor,
    solve_anchors_jointly,
)
from src.utils.bidirectional_smoother import smooth_between_anchors
from src.utils.camera_confidence import FrameSignals, confidence_from_signals
from src.utils.feature_propagator import propagate_one_frame
from src.utils.static_line_solver import StaticCameraSolution

logger = logging.getLogger(__name__)


def _angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    cos_t = (np.trace(R1.T @ R2) - 1) / 2
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(np.degrees(np.arccos(cos_t)))


def _circle_in_view_fraction(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, image_size: tuple[int, int],
) -> float:
    """Fraction of the catalogue centre circle that projects inside the image
    under (K, R, t). Gate for circle-aided solves: when the true circle is
    mostly out of view, the circle detector strip-searches over unrelated
    painted ridges (box lines, the D-arc) and hallucinates a lock that yanks
    the solve (origi02's behind-goal start)."""
    from src.utils.circle_detector import (
        CENTRE_CIRCLE_CENTRE,
        CENTRE_CIRCLE_RADIUS,
    )
    ang = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
    world = np.stack([
        CENTRE_CIRCLE_CENTRE[0] + CENTRE_CIRCLE_RADIUS * np.cos(ang),
        CENTRE_CIRCLE_CENTRE[1] + CENTRE_CIRCLE_RADIUS * np.sin(ang),
        np.zeros_like(ang),
    ], axis=1)
    cam = world @ np.asarray(R).T + np.asarray(t)
    in_front = cam[:, 2] > 0.1
    if not in_front.any():
        return 0.0
    pix = cam[in_front] @ np.asarray(K).T
    uv = pix[:, :2] / pix[:, 2:3]
    w, h = image_size
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w)
              & (uv[:, 1] >= 0) & (uv[:, 1] < h))
    return float(inside.sum()) / len(ang)


def _generate_auto_anchors(shot_id, clip_path, cfg):
    """Run the PnLCalib auto-anchor pipeline for one shot. Returns an
    AnchorSet or None. Heavy imports are local so the camera stage has no
    hard torch dependency unless auto-anchors are actually used."""
    from src.utils.auto_anchor import generate
    from src.utils.neural_calibrator import PnLCalibrator

    aa = cfg.get("auto_anchors", {})
    model_cfg = aa.get("model", {})
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    calibrator = PnLCalibrator(
        device=model_cfg.get("device", "auto"),
        kp_threshold=float(model_cfg.get("kp_threshold", 0.3434)),
        line_threshold=float(model_cfg.get("line_threshold", 0.7867)),
    )

    def _frames_reader(indices, image_size):
        cap = cv2.VideoCapture(str(clip_path))
        out = {}
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    out[idx] = frame
        finally:
            cap.release()
        return out

    return generate(
        calibrator=calibrator, clip_id=shot_id, n_frames=n_frames,
        image_size=(w, h), cfg=aa, frames_reader=_frames_reader,
    )


class CameraStage(BaseStage):
    name = "camera"

    def is_complete(self) -> bool:
        from src.schemas.shots import ShotsManifest
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            # Legacy: no manifest, but a single camera_track may exist.
            return (self.output_dir / "camera" / "camera_track.json").exists()
        manifest = ShotsManifest.load(manifest_path)
        return all(
            (self.output_dir / "camera" / f"{shot.id}_camera_track.json").exists()
            for shot in manifest.shots
        )

    def run(self) -> None:
        from src.schemas.shots import ShotsManifest
        cfg = self.config.get("camera", {})
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"camera stage requires a shots manifest at {manifest_path}; "
                "run prepare_shots first"
            )
        manifest = ShotsManifest.load(manifest_path)
        shot_filter = getattr(self, "shot_filter", None)
        any_processed = False
        for shot in manifest.shots:
            if shot_filter is not None and shot.id != shot_filter:
                continue
            anchors_path = (
                self.output_dir / "camera" / f"{shot.id}_anchors.json"
            )
            if not anchors_path.exists() and not cfg.get("auto_anchors", {}).get("enabled", False):
                logger.warning(
                    "camera stage skipping shot %s — no anchors at %s. Open "
                    "the anchor editor and place keyframes before re-running.",
                    shot.id, anchors_path,
                )
                continue
            clip_path = self.output_dir / shot.clip_file
            self._run_shot(shot.id, anchors_path, clip_path, cfg)
            any_processed = True
        if not any_processed and shot_filter is None:
            logger.warning(
                "camera stage produced no output — no shot in the manifest "
                "had matching anchors. Place keyframes via the anchor editor."
            )

    def _ensure_anchors(self, shot_id, anchors_path, clip_path, cfg):
        """Auto-generate anchors when enabled and appropriate. On any failure,
        leave the file as-is so existing manual path/warnings apply."""
        aa = cfg.get("auto_anchors", {})
        if not aa.get("enabled", False):
            return
        mode = aa.get("mode", "replace_when_empty")
        if anchors_path.exists() and mode == "replace_when_empty":
            return
        try:
            generated = _generate_auto_anchors(shot_id, clip_path, cfg)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "auto_anchors: generation failed for shot %s (%s); "
                "falling back to manual anchors", shot_id, exc,
            )
            return
        if generated is None or not generated.anchors:
            logger.warning(
                "auto_anchors: no usable anchors for shot %s; "
                "falling back to manual anchors", shot_id,
            )
            return
        if mode == "augment" and anchors_path.exists():
            existing = AnchorSet.load(anchors_path)
            seen = {e.frame for e in existing.anchors}
            merged = existing.anchors + tuple(
                a for a in generated.anchors if a.frame not in seen
            )
            generated = AnchorSet(
                clip_id=generated.clip_id, image_size=generated.image_size,
                anchors=merged,
            )
        elif mode == "force" and anchors_path.exists():
            logger.warning(
                "auto_anchors mode=force: overwriting existing anchors at %s",
                anchors_path,
            )
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        generated.save(anchors_path)
        logger.info(
            "auto_anchors: wrote %d generated anchors for shot %s to %s",
            len(generated.anchors), shot_id, anchors_path,
        )

    def _run_shot(
        self,
        shot_id: str,
        anchors_path: Path,
        clip_path: Path,
        cfg: dict,
    ) -> None:
        """Single-shot camera solve. The body is the original run() logic
        with file paths parameterised on shot_id."""
        self._ensure_anchors(shot_id, anchors_path, clip_path, cfg)
        if not anchors_path.exists():
            logger.warning(
                "camera stage: no anchors for shot %s (auto-generation "
                "produced none and no manual anchors exist); skipping shot.",
                shot_id,
            )
            return
        anchors = AnchorSet.load(anchors_path)

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open clip: {clip_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        anchor_max_residual = float(cfg.get("anchor_max_reprojection_px", 4.0))
        subsequent_min_landmarks = int(cfg.get("subsequent_anchor_min_landmarks", 4))
        subsequent_min_lines = 2

        # Step 1: filter anchors that don't have enough constraints to
        # contribute to the joint solve, then call the joint solver.
        qualifying: list[Anchor] = []
        for a in anchors.anchors:
            if (
                len(a.landmarks) >= subsequent_min_landmarks
                or len(a.lines) >= subsequent_min_lines
            ):
                qualifying.append(a)
            else:
                logger.warning(
                    "anchor at frame %d has only %d landmarks and %d lines "
                    "(need ≥%d points or ≥%d lines); skipping",
                    a.frame, len(a.landmarks), len(a.lines),
                    subsequent_min_landmarks, subsequent_min_lines,
                )
        if not qualifying:
            raise AnchorSolveError(
                "no anchor has enough landmarks or line correspondences to "
                "contribute to the camera solve; place more anchors in the "
                "web editor"
            )

        lens_prior = None
        if bool(cfg.get("lens_from_anchor", True)):
            # Joint estimator first — fits across every rich anchor with
            # shared (cx, cy, k1, k2) and per-anchor (rvec, tvec, fx).
            # Far better-determined than the single-anchor estimator on
            # real broadcast clips (≥2 rich anchors required). The single-
            # anchor fallback only fires when the joint LM rejects or
            # there's fewer than 2 rich anchors.
            lens_prior = _estimate_lens_jointly(
                tuple(qualifying), image_size=anchors.image_size,
            )
            if lens_prior is None:
                lens_prior = _estimate_lens_from_best_anchor(
                    tuple(qualifying), image_size=anchors.image_size,
                )
            if lens_prior is not None:
                logger.info(
                    "lens-from-anchor: prior recovered for shot %s — "
                    "cx=%.1f, cy=%.1f, k1=%+.4f, k2=%+.4f",
                    shot_id, *lens_prior,
                )

        try:
            sol = solve_anchors_jointly(
                tuple(qualifying),
                image_size=anchors.image_size,
                lens_prior=lens_prior,
            )
        except AnchorSolveError as exc:
            raise RuntimeError(f"camera stage failed: {exc}") from exc

        # Per-anchor (K, R, t) — each anchor has its own translation by
        # default. For static stadium-mounted cameras (the documented
        # design intent in CLAUDE.md), translation should be locked: the
        # camera body doesn't move, only its pan/tilt/zoom. With t free
        # the LM finds K/R/t combinations that reproject anchors well
        # but place the camera in physically inconsistent positions
        # (jumps of tens of metres between adjacent anchors), which
        # downstream foot-anchor ray-casting interprets as players
        # moving across the pitch.
        static_camera = bool(cfg.get("static_camera", True))
        if static_camera:
            # Joint LM over all anchors with t shared. Replaces the joint
            # solution wholesale so t_world / principal_point reflect the
            # refined values.
            sol = refine_with_shared_translation(tuple(qualifying), sol)
            logger.info(
                "static_camera=true: joint shared-t refine produced "
                "t=%s across %d anchors",
                np.round(sol.t_world, 3).tolist(), len(sol.per_anchor_KRt),
            )
            # TRIMMED re-relock: C is near-unidentifiable along the viewing
            # axis from any single-azimuth anchor subset (solo box-anchor
            # centres slide ~4.5 m along one line), so the shared-C relock is
            # the C estimator — but one or two wrong-basin anchors (origi01's
            # f0/f108: implied focal 2.4x off every other anchor) drag it by
            # metres. Drop anchors whose post-relock residual is grossly
            # inconsistent and re-relock with the survivors; the downstream
            # C-profile/bundle stay within the trust radius of the result.
            for _trim in range(2):
                res_by_frame: dict[int, float] = {}
                for a in qualifying:
                    got = sol.per_anchor_KRt.get(a.frame)
                    if got is None or not a.landmarks:
                        continue
                    K_a, R_a, t_a = got
                    res_by_frame[a.frame] = reprojection_residual_for_anchor(
                        a, K_a, R_a, t_a, tuple(sol.distortion[:2]))
                if len(res_by_frame) < 4:
                    break
                med_res = float(np.median(list(res_by_frame.values())))
                thr_res = max(12.0, 3.0 * med_res)
                keep = [a for a in qualifying
                        if res_by_frame.get(a.frame, 0.0) <= thr_res]
                if len(keep) == len(qualifying) or len(keep) < 3:
                    break
                dropped = sorted(
                    a.frame for a in qualifying if a not in keep)
                trial = solve_anchors_jointly(
                    tuple(keep), image_size=anchors.image_size,
                    lens_prior=lens_prior)
                trial = refine_with_shared_translation(tuple(keep), trial)
                # Accept the trim ONLY if the surviving anchors actually fit
                # better without the dropped ones — a re-solve on the reduced
                # set can land in a worse basin entirely (kroupi: dropping
                # its one bad anchor collapsed the relock to a centre INSIDE
                # the pitch and took the whole clip down with it).
                kept_before = float(np.median([
                    res_by_frame[a.frame] for a in keep
                    if a.frame in res_by_frame]))
                kept_after_vals = []
                for a in keep:
                    got = trial.per_anchor_KRt.get(a.frame)
                    if got is None or not a.landmarks:
                        continue
                    K_a, R_a, t_a = got
                    kept_after_vals.append(reprojection_residual_for_anchor(
                        a, K_a, R_a, t_a, tuple(trial.distortion[:2])))
                kept_after = (float(np.median(kept_after_vals))
                              if kept_after_vals else float("inf"))
                if kept_after >= 0.9 * kept_before:
                    logger.info(
                        "static_camera=true: trimmed relock REJECTED "
                        "(kept-anchor fit %.1f -> %.1f px)",
                        kept_before, kept_after)
                    break
                qualifying = keep
                sol = trial
                logger.info(
                    "static_camera=true: trimmed relock dropped anchor(s) %s "
                    "(residual > %.1f px); kept-anchor fit %.1f -> %.1f px; "
                    "C now %s over %d anchors",
                    dropped, thr_res, kept_before, kept_after,
                    np.round(-np.asarray(
                        sol.per_anchor_KRt[qualifying[0].frame][1]).T
                        @ sol.per_anchor_KRt[qualifying[0].frame][2],
                        2).tolist(),
                    len(qualifying))
        t_world_median = sol.t_world
        principal_point = sol.principal_point
        anchor_solutions: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = (
            sol.per_anchor_KRt
        )
        anchor_confidence_override: dict[int, float] = {}
        for af, residual in sol.per_anchor_residual_px.items():
            if residual > anchor_max_residual:
                logger.warning(
                    "anchor at frame %d has reprojection residual %.2f px > "
                    "%.2f px threshold; flagging as low-confidence",
                    af, residual, anchor_max_residual,
                )
                anchor_confidence_override[af] = 0.5

        # Step 2: per-frame propagate (K, R) forward/backward between
        # consecutive anchor pairs. Per-frame t is linearly interpolated
        # between the two anchor t values (smooth camera motion under
        # steadicam/handheld assumption).
        per_frame_K: list[np.ndarray | None] = [None] * n_frames
        per_frame_R: list[np.ndarray | None] = [None] * n_frames
        per_frame_t: list[np.ndarray | None] = [None] * n_frames
        per_frame_conf: list[float] = [0.0] * n_frames
        is_anchor: list[bool] = [False] * n_frames

        for af, (K, R, t) in anchor_solutions.items():
            per_frame_K[af] = K
            per_frame_R[af] = R
            per_frame_t[af] = t
            per_frame_conf[af] = anchor_confidence_override.get(af, 1.0)
            is_anchor[af] = True

        anchor_frames = sorted(anchor_solutions.keys())
        # Inter-anchor frames: LERP K and t, SLERP R between adjacent
        # anchors. The legacy feature-propagator (ORB homography frame-to-
        # frame) was tuned for the broadcast-fixed-body assumption; with a
        # moving camera (per-anchor t) it tends to drift visibly between
        # anchors. Direct interpolation of (K, R, t) is smoother and more
        # predictable: it trusts the anchor solves and produces a steady
        # camera-motion model between them.
        from scipy.spatial.transform import Rotation, Slerp
        # When the camera body is static, every per-frame t must satisfy
        # -R^T @ t == C_locked. LERP'ing t between two anchors with
        # different R does NOT honour that constraint for the SLERP'd R,
        # so the camera body wanders between anchors. Rebuild t = -R @ C
        # instead. C_locked is None for moving-camera clips; fall back
        # to LERP in that case.
        C_locked = (
            np.asarray(sol.camera_centre) if sol.camera_centre is not None else None
        )
        for a, b in zip(anchor_frames, anchor_frames[1:]):
            K_a, R_a, t_a = anchor_solutions[a]
            K_b, R_b, t_b = anchor_solutions[b]
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R_a, R_b]))
            for offset in range(1, b - a):
                idx = a + offset
                # Don't reuse the name `w` — the outer scope holds image
                # width and we'd clobber image_size on save (D27).
                lerp_w = offset / (b - a)
                per_frame_K[idx] = (1.0 - lerp_w) * K_a + lerp_w * K_b
                R_inter = slerp([lerp_w]).as_matrix()[0]
                per_frame_R[idx] = R_inter
                if C_locked is not None:
                    per_frame_t[idx] = -R_inter @ C_locked
                else:
                    per_frame_t[idx] = (1.0 - lerp_w) * t_a + lerp_w * t_b
                # Lower confidence than the anchors but still high since
                # interpolation is well-behaved between trusted anchors.
                per_frame_conf[idx] = 0.7

        # Step 2.5 (optional): line-extraction refinement. When
        # camera.line_extraction is enabled, every per-frame camera from
        # the propagation above is treated as a bootstrap and re-fitted
        # against painted pitch lines detected directly in the frame.
        # This is the experimental sub-pixel path — see
        # docs/superpowers/notes/2026-05-14-camera-1px-experiment.md.
        # The bootstrap from Step 2 is what makes per-frame detection
        # tractable: the detector only searches a strip around the
        # projected line, so it needs a roughly-right camera to start.
        detected_lines_by_frame: dict[int, list] = {}
        static_line_sol: StaticCameraSolution | None = None
        if bool(cfg.get("line_extraction", False)):
            if static_camera:
                static_line_sol = self._refine_with_static_line_solve(
                    cap, shot_id, anchors, cfg,
                    per_frame_K, per_frame_R, per_frame_t, per_frame_conf,
                    is_anchor, tuple(sol.distortion),
                    detected_lines_by_frame,
                )
            else:
                self._refine_with_line_extraction(
                    cap, shot_id, anchors, cfg,
                    per_frame_K, per_frame_R, per_frame_t, per_frame_conf,
                    is_anchor, tuple(sol.distortion),
                    detected_lines_by_frame,
                )

        cap.release()

        # Frames outside the [first_anchor, last_anchor] span are not currently
        # covered by the bidirectional propagator — warn the user so they can
        # add anchors instead of silently losing them.
        dropped_before = anchor_frames[0]
        dropped_after = max(0, n_frames - 1 - anchor_frames[-1])
        if dropped_before > 0 or dropped_after > 0:
            logger.warning(
                "camera stage dropped %d frames before first anchor (frame %d) "
                "and %d frames after last anchor (frame %d); add anchors to "
                "cover them",
                dropped_before, anchor_frames[0],
                dropped_after, anchor_frames[-1],
            )

        # Step 3: assemble output.
        frames_out: list[CameraFrame] = []
        for i in range(n_frames):
            K = per_frame_K[i]
            R = per_frame_R[i]
            t = per_frame_t[i]
            if K is None or R is None or t is None:
                continue  # frames outside any anchor span are skipped in v1
            frames_out.append(
                CameraFrame(
                    frame=i,
                    K=K.tolist(),
                    R=R.tolist(),
                    confidence=per_frame_conf[i],
                    is_anchor=is_anchor[i],
                    t=list(t),
                )
            )

        # When the static-camera line solve ran, its refined lens + locked
        # centre supersede the anchor solve's: the per-frame (K, R, t) in
        # the track were produced by that solve, so the track's stated
        # principal point / distortion / camera centre must match it.
        # ``distortion`` is truncated to (k1, k2) for the CameraTrack schema
        # — under brown_conrady the tangential / k3 terms are not persisted.
        if static_line_sol is not None:
            camera_centre_out: tuple[float, float, float] | None = tuple(
                float(x) for x in static_line_sol.camera_centre
            )
            principal_point_out = static_line_sol.principal_point
            distortion_out = tuple(
                float(x) for x in static_line_sol.distortion[:2]
            )
        else:
            camera_centre_out = (
                tuple(float(x) for x in sol.camera_centre)
                if sol.camera_centre is not None
                else None
            )
            principal_point_out = principal_point
            distortion_out = tuple(float(x) for x in sol.distortion)

        track = CameraTrack(
            clip_id=anchors.clip_id,
            fps=float(fps),
            image_size=(w, h),
            t_world=list(t_world_median),
            frames=tuple(frames_out),
            principal_point=(
                float(principal_point_out[0]), float(principal_point_out[1])
            ),
            camera_centre=camera_centre_out,
            distortion=distortion_out,
        )
        track.save(self.output_dir / "camera" / f"{shot_id}_camera_track.json")

        # Persist detected lines as a debug side-output when line
        # extraction ran. Lets the dashboard / anchor editor overlay the
        # detected painted lines and compare against the projected
        # catalogue lines.
        if detected_lines_by_frame:
            import json
            debug_path = (
                self.output_dir / "camera" / f"{shot_id}_detected_lines.json"
            )
            debug_path.write_text(json.dumps({
                "shot_id": shot_id,
                "image_size": [w, h],
                "fps": float(fps),
                "frames": {
                    str(k): {
                        "lines": v,
                        "K": per_frame_K[k].tolist(),
                        "R": per_frame_R[k].tolist(),
                        "t": list(per_frame_t[k]),
                    }
                    for k, v in sorted(detected_lines_by_frame.items())
                    if per_frame_K[k] is not None
                },
            }))
            logger.info(
                "line_extraction: wrote %d frames of detected lines to %s",
                len(detected_lines_by_frame), debug_path,
            )

    def _refine_with_line_extraction(
        self,
        cap: cv2.VideoCapture,
        shot_id: str,
        anchors: AnchorSet,
        cfg: dict,
        per_frame_K: list,
        per_frame_R: list,
        per_frame_t: list,
        per_frame_conf: list,
        is_anchor: list,
        distortion: tuple[float, float],
        detected_lines_by_frame: dict[int, list],
    ) -> None:
        """In-place per-frame camera refinement against detected painted
        lines. Replaces ``per_frame_{K,R,t}`` entries with line-fitted
        values where detection succeeds, and records the detected lines
        in ``detected_lines_by_frame`` for the debug JSON.

        Frames where line detection fails (occlusion, too few lines)
        keep their propagated camera untouched.
        """
        from src.utils.line_detector import DetectorConfig
        from src.utils.line_camera_refine import refine_camera_from_lines

        det_cfg = DetectorConfig(
            search_strip_px=int(cfg.get("line_extraction_strip_px", 25)),
            min_gradient=float(cfg.get("line_extraction_min_gradient", 10.0)),
        )
        max_iters = int(cfg.get("line_extraction_max_iters", 4))
        # Anchor-frame landmark clicks become low-weight point hints so
        # the line solve doesn't slide into a geometrically wrong basin.
        anchor_landmarks: dict[int, list] = {
            a.frame: list(a.landmarks)
            for a in anchors.anchors if a.landmarks
        }

        n_frames = len(per_frame_K)
        n_refined = 0
        n_failed = 0
        # Can't use list.count(None) — the list holds numpy arrays and
        # `array == None` is element-wise, raising on the truth test.
        n_covered = sum(1 for k in per_frame_K if k is not None)
        rms_values: list[float] = []
        for idx in range(n_frames):
            if per_frame_K[idx] is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            result = refine_camera_from_lines(
                frame,
                per_frame_K[idx], per_frame_R[idx], per_frame_t[idx],
                distortion,
                point_hint_landmarks=anchor_landmarks.get(idx),
                detector_cfg=det_cfg,
                max_iters=max_iters,
            )
            if result.n_detections == 0:
                n_failed += 1
                continue
            per_frame_K[idx] = result.K
            per_frame_R[idx] = result.R
            per_frame_t[idx] = result.t
            # Line-refined frames are high-confidence where the fit is
            # tight; degrade smoothly with line RMS so the dashboard
            # confidence timeline still surfaces poorly-fit spans.
            per_frame_conf[idx] = max(
                0.3, min(1.0, 1.0 - result.line_rms_px / 6.0)
            )
            rms_values.append(result.line_rms_px)
            detected_lines_by_frame[idx] = [
                {
                    "name": ln.name,
                    "image_segment": [list(ln.image_segment[0]),
                                      list(ln.image_segment[1])],
                    "world_segment": [list(ln.world_segment[0]),
                                      list(ln.world_segment[1])],
                }
                for ln in result.detected_lines
            ]
            n_refined += 1

        if rms_values:
            arr = np.array(rms_values)
            logger.info(
                "line_extraction: refined %d/%d frames (%d had too few "
                "detected lines, kept propagated camera). Line RMS: "
                "mean=%.3f px, median=%.3f px, max=%.3f px, frac<1px=%.2f",
                n_refined, n_covered, n_failed,
                float(arr.mean()), float(np.median(arr)), float(arr.max()),
                float((arr < 1.0).mean()),
            )
        else:
            logger.warning(
                "line_extraction: no frame produced usable line detections; "
                "camera track unchanged from the propagated solution",
            )

    def _refine_with_static_line_solve(
        self,
        cap: cv2.VideoCapture,
        shot_id: str,
        anchors: AnchorSet,
        cfg: dict,
        per_frame_K: list,
        per_frame_R: list,
        per_frame_t: list,
        per_frame_conf: list,
        is_anchor: list,
        distortion: tuple[float, float],
        detected_lines_by_frame: dict[int, list],
    ) -> StaticCameraSolution | None:
        """Static-camera line solve: detect painted lines on every
        propagated frame, profile the camera centre, bundle-adjust one
        shared centre, then iteratively re-detect under the coherent
        cameras. Writes per-frame ``(K, R, t)`` back in place and returns
        the :class:`StaticCameraSolution` (whose ``camera_centre``,
        ``principal_point`` and ``distortion`` the caller writes into the
        track), or ``None`` if it bailed and left the propagated cameras
        untouched.
        """
        from src.utils.anchor_solver import _is_rich
        from src.utils.line_detector import DetectorConfig
        from src.utils.line_camera_refine import (
            detect_lines_for_frames,
            drop_underdetermined_frames,
        )
        from src.utils.static_c_profile import make_c_grid, profile_camera_centre
        from src.utils.static_line_solver import solve_static_camera_from_lines

        det_cfg = DetectorConfig(
            search_strip_px=int(cfg.get("line_extraction_strip_px", 25)),
            min_gradient=float(cfg.get("line_extraction_min_gradient", 10.0)),
        )
        lens_model = str(cfg.get("line_extraction_lens_model", "pinhole_k1k2"))
        n_rounds = int(cfg.get("line_extraction_static_rounds", 1))
        # A per-frame solve recovers 4 DOF (rvec + fx); frames with too few
        # detected lines are under-determined and yield non-physical cameras
        # (extreme focal, rotation flips). Exclude them so they keep the smooth
        # interpolated camera instead. See the static-line glitch investigation.
        min_lines = int(cfg.get("line_extraction_min_lines_per_frame", 4))
        # Per-detection acceptance gates (tunable). Lowering min_n_samples lets
        # SHORT lines through (e.g. the far touchline, which projects short in
        # the image) — adding a perpendicular constraint that improves solve
        # conditioning where near-parallel near-side lines dominate.
        det_min_n_samples = int(cfg.get("line_extraction_det_min_n_samples", 40))
        det_min_confidence = float(cfg.get("line_extraction_det_min_confidence", 0.5))
        point_hint_weight = float(
            cfg.get("line_extraction_point_hint_weight", 0.05)
        )
        dist2 = (float(distortion[0]), float(distortion[1]))

        covered = [
            i for i in range(len(per_frame_K)) if per_frame_K[i] is not None
        ]
        frames_bgr: dict[int, np.ndarray] = {}
        for i in covered:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                frames_bgr[i] = frame

        def _cameras_from_arrays() -> dict[int, dict]:
            return {
                i: {"K": per_frame_K[i], "R": per_frame_R[i], "t": per_frame_t[i]}
                for i in frames_bgr
            }

        # Step 0 — detect lines under the propagated bootstrap. Keep the RAW
        # (pre-filter) detections so midfield frames with too few straight
        # lines can be rescued by the centre circle below.
        prop_cams = _cameras_from_arrays()
        raw_lines = detect_lines_for_frames(
            frames_bgr, prop_cams, dist2, det_cfg,
            min_confidence=det_min_confidence, min_n_samples=det_min_n_samples,
            min_lines=1,
        )
        seed_cams = prop_cams
        well_lined = drop_underdetermined_frames(raw_lines, min_lines)

        # Clip-adaptive fallback: poor anchor-bootstrap coverage -> re-bootstrap
        # detection from PnLCalib's accurate per-frame camera. gberch (~100%
        # coverage) never triggers this, so its path is unchanged.
        coverage = len(well_lined) / max(1, len(covered))
        min_cov = float(
            cfg.get("line_extraction_pnlcalib_bootstrap_min_coverage", 0.5)
        )
        if (
            bool(cfg.get("line_extraction_pnlcalib_bootstrap", True))
            and coverage < min_cov
        ):
            pnl_cams = self._pnlcalib_bootstrap_cameras(frames_bgr, cfg)
            if pnl_cams:
                # Detect under ZERO distortion, not the anchor-solve dist2
                # (often saturated at its bounds — non-physical). The saturated
                # value throws midfield line projections off the search strip,
                # so those frames detect <2 lines and never become circle-rescue
                # candidates. PnLCalib's camera + ~zero distortion projects the
                # catalogue accurately; the solve re-estimates real distortion.
                raw_pnl = detect_lines_for_frames(
                    frames_bgr, pnl_cams, (0.0, 0.0), det_cfg,
                    min_confidence=det_min_confidence,
                    min_n_samples=det_min_n_samples, min_lines=1,
                )
                well_pnl = drop_underdetermined_frames(raw_pnl, min_lines)
                if len(well_pnl) > len(well_lined):
                    logger.info(
                        "static line solve: anchor-bootstrap coverage %.0f%% "
                        "< %.0f%%; switched to PnLCalib per-frame bootstrap "
                        "(%d -> %d line-detected frames)",
                        100 * coverage, 100 * min_cov,
                        len(well_lined), len(well_pnl),
                    )
                    raw_lines, well_lined, seed_cams = raw_pnl, well_pnl, pnl_cams

        if len(well_lined) < 2:
            logger.warning(
                "static line solve: only %d well-lined frame(s); keeping the "
                "propagated cameras unchanged", len(well_lined),
            )
            return None

        # Q1 — centre-circle rescue: frames the straight-line filter dropped
        # (too few lines, e.g. midfield where the box is out of view) but where
        # the centre circle is visible get the circle as a weighted point
        # constraint, so they can still be solved. Only DROPPED frames are
        # rescued, so well-lined clips (gberch) are untouched.
        per_frame_circle: dict[int, list] = {}
        rescued_lines: dict[int, list] = {}
        if bool(cfg.get("line_extraction_detect_circle", True)):
            from src.schemas.anchor import LandmarkObservation
            from src.utils.circle_detector import detect_circle
            circ_min_lines = int(cfg.get("line_extraction_circle_min_lines", 2))
            n_circ = int(cfg.get("line_extraction_circle_points", 20))
            for fid in frames_bgr:
                if fid in well_lined or len(raw_lines.get(fid, [])) < circ_min_lines:
                    continue
                cam = seed_cams.get(fid)
                if cam is None:
                    continue
                # Use zero distortion for the circle projection, NOT the
                # anchor-solve dist2 (often saturated at its bounds — a
                # non-physical LM artifact). The circle spans the image, so a
                # bogus k1/k2 throws its projection off the search strip; the
                # solve re-estimates real distortion from zero anyway.
                det = detect_circle(
                    frames_bgr[fid], np.asarray(cam["K"]), np.asarray(cam["R"]),
                    np.asarray(cam["t"]), (0.0, 0.0), det_cfg,
                )
                if det is None:
                    continue
                k = min(n_circ, len(det.image_points))
                idx = np.linspace(0, len(det.image_points) - 1, k).astype(int)
                per_frame_circle[fid] = [
                    LandmarkObservation(
                        name=det.name, image_xy=det.image_points[j],
                        world_xyz=det.world_points[j],
                    )
                    for j in idx
                ]
                rescued_lines[fid] = raw_lines[fid]
            if rescued_lines:
                logger.info(
                    "static line solve: centre circle rescued %d midfield "
                    "frame(s)", len(rescued_lines),
                )

        per_frame_lines = {**well_lined, **rescued_lines}

        # Per-frame (rvec, fx) bootstrap seeds from the chosen camera source.
        bootstrap: dict[int, tuple[np.ndarray, float]] = {}
        for fid in per_frame_lines:
            cam = seed_cams[fid]
            rv, _ = cv2.Rodrigues(np.asarray(cam["R"]))
            bootstrap[fid] = (rv.reshape(3), float(np.asarray(cam["K"])[0, 0]))

        # Seed C from the chosen cameras' centres (rich-anchor frames preferred
        # when using the propagated bootstrap; all detected frames otherwise).
        rich = {a.frame for a in anchors.anchors if _is_rich(a)}

        def _centre(f: int) -> np.ndarray:
            R = np.asarray(seed_cams[f]["R"]); t = np.asarray(seed_cams[f]["t"])
            return -R.T @ t

        seed_cs = [
            _centre(f) for f in well_lined if f in rich
        ] or [
            _centre(f) for f in well_lined
        ]
        c_center = np.median(np.stack(seed_cs), axis=0)
        # The anchor-stage shared-t relock C (read from the anchor frames'
        # per-frame cameras, which carry it verbatim) is the consensus over
        # ALL anchor azimuths. It is NOT used as the search seed — a broken
        # auto-anchor relock poisons everything downstream (kroupi's relock
        # lands INSIDE the pitch while its seed-cam median is fine) — it
        # enters only as the HELD candidate in the C arbitration below,
        # where the anchor-fit comparison can reject it.
        _anchor_centres = [
            -np.asarray(per_frame_R[a.frame]).T @ np.asarray(
                per_frame_t[a.frame])
            for a in anchors.anchors
            if a.frame < len(per_frame_R) and per_frame_R[a.frame] is not None
        ]
        c_anchor_consensus = (
            np.median(np.stack(_anchor_centres), axis=0)
            if len(_anchor_centres) >= 2 else c_center)
        logger.info(
            "static line solve: C seed=%s (%d seed-cam centre(s)); anchor "
            "consensus=%s (%d anchor centre(s))",
            np.round(c_center, 2).tolist(), len(seed_cs),
            np.round(c_anchor_consensus, 2).tolist(), len(_anchor_centres))
        cx0 = float(per_frame_K[covered[0]][0, 2])
        cy0 = float(per_frame_K[covered[0]][1, 2])
        # Seed the lens with zero distortion: the anchor-solve distortion
        # on real clips is often a saturated, non-physical value (the LM
        # absorbing click noise). The C-profile holds the lens fixed and
        # the static-C bundle adjustment refines distortion from here.
        lens_seed = (cx0, cy0, 0.0, 0.0)

        # Step 1 — C-profile: coarse grid then a fine grid around its argmin.
        # profile_camera_centre subsamples frames for the grid sweep, so the
        # cost scales with the grid size, not the (often hundreds of) frames.
        # C-profile uses only the WELL-LINED frames so the shared centre is
        # found from well-constrained geometry; the circle-rescued midfield
        # frames join the final bundle solve (where the circle constrains them).
        #
        # C TRUST RADIUS: detected lines are strip-searched around the current
        # cameras' projections and partially self-confirm whatever C they were
        # found under — left free, origi01's profile+bundle walked C ~3 m onto
        # its own detections, and per-frame fits at the anchor-consensus C
        # showed EVERY span (start/midfield/box) fitting the hand clicks at
        # 3-11 px simultaneously where the drifted C failed the midfield by
        # metres. The anchor consensus (c_center) is the only C evidence not
        # subject to that bias; the profile/bundle refine WITHIN its trust
        # radius rather than wander.
        c_trust = float(cfg.get("line_extraction_c_trust_m", 1.5))
        coarse = profile_camera_centre(
            well_lined, anchors.image_size,
            c_grid=make_c_grid(c_center, extent_m=7.5, n_steps=5),
            lens_seed=lens_seed, per_frame_bootstrap=bootstrap,
        )
        fine = profile_camera_centre(
            well_lined, anchors.image_size,
            c_grid=make_c_grid(coarse.argmin_c, extent_m=2.0, n_steps=5),
            lens_seed=lens_seed, per_frame_bootstrap=coarse.per_frame_seeds,
        )
        logger.info(
            "static line solve: C-profile argmin=%s mean line RMS=%.3f px",
            np.round(fine.argmin_c, 3).tolist(),
            float(np.min(fine.mean_rms)),
        )

        # Steps 2 + 3 — bundle adjustment + iterative re-detection.
        anchor_landmarks = {
            a.frame: list(a.landmarks) for a in anchors.anchors if a.landmarks
        }
        c_seed = fine.argmin_c
        c_bound_bundle = 5.0
        # Refined seeds for well-lined frames; bootstrap seeds for the
        # circle-rescued frames (which the C-profile didn't see).
        seeds = {**bootstrap, **fine.per_frame_seeds}
        circle_weight = float(cfg.get("line_extraction_circle_weight", 0.3))
        # The centre-circle lens refinement is a POST-propagation step (the
        # circle lives on midfield frames covered only after propagation, and it
        # only helps where the box-line lens is wrong) — see below.
        # Re-detecting under the static-C cameras (a per-frame compromise)
        # is not guaranteed to improve every round, so keep the best round
        # by mean line RMS rather than blindly taking the last.
        best_sol: StaticCameraSolution | None = None
        best_mean = float("inf")
        best_lines = per_frame_lines
        n_rounds = max(1, n_rounds)
        for round_idx in range(n_rounds):
            sol = solve_static_camera_from_lines(
                per_frame_lines, anchors.image_size,
                c_seed=c_seed, lens_seed=lens_seed,
                per_frame_seeds=seeds, point_hints=anchor_landmarks,
                circle_points=per_frame_circle,
                lens_model=lens_model, point_hint_weight=point_hint_weight,
                circle_weight=circle_weight,
                c_bound_m=c_bound_bundle,
            )
            round_rms = np.array(
                [v for v in sol.per_frame_line_rms.values() if np.isfinite(v)]
            )
            round_mean = float(round_rms.mean()) if round_rms.size else float("inf")
            logger.info(
                "static line solve: round %d/%d — line RMS mean=%.3f "
                "median=%.3f max=%.3f frac<1px=%.2f (%d frames, %d lines)",
                round_idx + 1, n_rounds, round_mean,
                float(np.median(round_rms)) if round_rms.size else float("nan"),
                float(round_rms.max()) if round_rms.size else float("nan"),
                float((round_rms < 1.0).mean()) if round_rms.size else 0.0,
                len(sol.per_frame_KRt),
                sum(len(v) for v in per_frame_lines.values()),
            )
            if round_mean < best_mean:
                best_sol, best_mean, best_lines = sol, round_mean, per_frame_lines
            if round_idx < n_rounds - 1:
                cams = {
                    fid: {"K": K, "R": R, "t": t}
                    for fid, (K, R, t) in sol.per_frame_KRt.items()
                }
                redet = detect_lines_for_frames(
                    frames_bgr, cams, tuple(sol.distortion[:2]), det_cfg,
                    min_confidence=det_min_confidence,
                    min_n_samples=det_min_n_samples,
                )
                redet = drop_underdetermined_frames(redet, min_lines)
                if len(redet) >= 2:
                    per_frame_lines = redet
                c_seed = sol.camera_centre
                seeds = {
                    fid: (cv2.Rodrigues(R)[0].reshape(3), float(K[0, 0]))
                    for fid, (K, R, _t) in sol.per_frame_KRt.items()
                }

        assert best_sol is not None
        sol = best_sol
        per_frame_lines = best_lines

        # C ARBITRATION on the only evidence that is NOT self-confirming.
        # Detections are strip-searched around the cameras' own projections,
        # so an azimuth-poor line set lets the free bundle WALK C metres down
        # a shallow valley while its line-RMS keeps improving (origi01: ~3 m,
        # every span's clicks paid). Conversely a clip whose anchor consensus
        # is the noisy estimate must keep the free result (gberch: clamping
        # cost 2.1 -> 5.4 px line-RMS). Arbiter: solve BOTH (free bundle vs
        # bundle held to the anchor-consensus C) and keep whichever fits the
        # ANCHOR KEYPOINTS better — clicks/PnLCalib points were placed
        # independently of any camera.
        def _anchor_fit(s: StaticCameraSolution) -> float:
            res = []
            for a in anchors.anchors:
                if not a.landmarks:
                    continue
                got = s.per_frame_KRt.get(a.frame)
                if got is None:
                    continue
                K_a, R_a, t_a = got
                res.append(reprojection_residual_for_anchor(
                    a, K_a, R_a, t_a, tuple(s.distortion[:2])))
            return float(np.median(res)) if res else float("inf")

        _gap = float(np.linalg.norm(
            np.asarray(sol.camera_centre) - c_anchor_consensus))
        if _gap > 0.75:
            sol_held = solve_static_camera_from_lines(
                per_frame_lines, anchors.image_size,
                c_seed=c_anchor_consensus, lens_seed=lens_seed,
                per_frame_seeds=seeds, point_hints=anchor_landmarks,
                circle_points=per_frame_circle,
                lens_model=lens_model,
                point_hint_weight=point_hint_weight,
                circle_weight=circle_weight,
                c_bound_m=max(0.25, c_trust / 3.0),
            )
            # Third candidate: held C AND a modest lens. With C pinned, an
            # azimuth-poor line set compensates through (k1, fx) instead
            # (origi01: k1 walked to 0.395 with fx ~12% high; the wide-field
            # anchors paid ~300 px). The anchor fit arbitrates all three.
            sol_held_k = solve_static_camera_from_lines(
                per_frame_lines, anchors.image_size,
                c_seed=c_anchor_consensus, lens_seed=lens_seed,
                per_frame_seeds=seeds, point_hints=anchor_landmarks,
                circle_points=per_frame_circle,
                lens_model=lens_model,
                point_hint_weight=point_hint_weight,
                circle_weight=circle_weight,
                c_bound_m=max(0.25, c_trust / 3.0),
                dist_bound=float(cfg.get(
                    "line_extraction_modest_dist_bound", 0.12)),
            )
            cands = [
                ("free", sol, _anchor_fit(sol)),
                ("held", sol_held, _anchor_fit(sol_held)),
                ("held+modest-k", sol_held_k, _anchor_fit(sol_held_k)),
            ]
            logger.info(
                "static line solve: C arbitration — %s",
                " | ".join(
                    f"{name} C={np.round(s.camera_centre, 2).tolist()} "
                    f"k1={s.distortion[0]:+.3f} fit={fit:.1f}px"
                    for name, s, fit in cands))
            name, best, fit = min(cands, key=lambda c: c[2])
            if best is not sol:
                logger.info(
                    "static line solve: anchor evidence prefers %s "
                    "(%.1f px) — keeping it", name, fit)
                sol = best
        C = sol.camera_centre

        def _anchor_click_checkpoint(stage: str) -> None:
            """Per-stage anchor-click fit — locates which post-bundle pass
            degrades the arbitrated solution (the bundle fits the good
            anchors at ~8 px; the final track reads 11-22 px at the same
            anchors)."""
            if not bool(cfg.get("line_extraction_debug_anchor_fit", False)):
                return
            from src.utils.camera_projection import project_world_to_image
            dist_dbg = tuple(float(x) for x in sol.distortion[:2])
            per = []
            for a in anchors.anchors:
                if not a.landmarks or a.frame >= len(per_frame_K):
                    continue
                if per_frame_K[a.frame] is None:
                    continue
                rs = []
                for lm in a.landmarks:
                    p = project_world_to_image(
                        per_frame_K[a.frame], per_frame_R[a.frame],
                        per_frame_t[a.frame], dist_dbg,
                        np.array([lm.world_xyz], dtype=float))[0]
                    rs.append(float(np.linalg.norm(
                        p - np.asarray(lm.image_xy))))
                per.append((a.frame, float(np.median(rs))))
            if per:
                med = float(np.median([v for _, v in per]))
                worst = sorted(per, key=lambda x: -x[1])[:3]
                logger.info(
                    "anchor-fit checkpoint [%s]: med %.1f px | worst %s",
                    stage, med,
                    ", ".join(f"f{f}={v:.0f}" for f, v in worst))

        # Write the solved cameras back in place.
        for fid, (K, R, t) in sol.per_frame_KRt.items():
            per_frame_K[fid] = K
            per_frame_R[fid] = R
            per_frame_t[fid] = t
            rms = sol.per_frame_line_rms.get(fid, float("nan"))
            if np.isfinite(rms):
                per_frame_conf[fid] = max(0.3, min(1.0, 1.0 - rms / 6.0))
            detected_lines_by_frame[fid] = [
                {
                    "name": ln.name,
                    "image_segment": [list(ln.image_segment[0]),
                                      list(ln.image_segment[1])],
                    "world_segment": [list(ln.world_segment[0]),
                                      list(ln.world_segment[1])],
                }
                for ln in per_frame_lines.get(fid, [])
            ]

        _anchor_click_checkpoint("post-bundle+arbitration")

        # Frames the bundle skipped (no/too-few straight lines) carry rotations
        # solved under the PRE-bundle geometry (anchor-stage C / principal
        # point). Keeping that R while re-deriving t against the bundle's C
        # silently shifts their projections by metres (origi01's midfield sat
        # ~330 px off the user's clicks). Demote them to UNCOVERED so the
        # propagation / cold-start passes below re-solve them against their own
        # detected lines (+ centre circle) at the final locked geometry; frames
        # nothing can re-solve are SLERP-filled between solved neighbours at
        # the end — interpolation between consistent frames, not stale poses.
        # gberch (every frame bundle-solved) demotes nothing.
        anchor_resolved_frames: set[int] = set()
        if bool(cfg.get("line_extraction_resolve_underlined", True)):
            from src.utils.static_c_profile import (
                _solve_frame_at_fixed_c as _dm_solve,
            )
            from src.utils.static_line_solver import _dist5 as _dm_d5
            dm_d5 = _dm_d5(sol.distortion)
            dm_cx, dm_cy = sol.principal_point
            dm_gate = float(cfg.get("line_extraction_anchor_resolve_max_rms", 30.0))
            demoted = [
                i for i in covered
                if i not in sol.per_frame_KRt and per_frame_R[i] is not None
            ]
            n_anchor_resolved = 0
            for i in demoted:
                # Anchor frames carry real constraints (their landmark points
                # — PnLCalib or manual): re-solve (rvec, fx) against them at
                # the LOCKED geometry instead of discarding. These become
                # solved islands inside otherwise feature-poor spans, which
                # the propagation pass then fills between (origi01's midfield
                # circle span is unreachable from the box end without them).
                lms = anchor_landmarks.get(i)
                if lms and len(lms) >= 4:
                    rv, _ = cv2.Rodrigues(np.asarray(per_frame_R[i]))
                    fx0 = float(per_frame_K[i][0, 0])
                    rvec, fx, rms = _dm_solve(
                        [], dm_cx, dm_cy, dm_d5, C, rv.reshape(3), fx0,
                        circle_obs=lms, circle_weight=1.0)
                    if np.isfinite(rms) and rms <= dm_gate:
                        R2, _ = cv2.Rodrigues(rvec)
                        per_frame_K[i] = np.array(
                            [[fx, 0.0, dm_cx], [0.0, fx, dm_cy],
                             [0.0, 0.0, 1.0]])
                        per_frame_R[i] = R2
                        per_frame_t[i] = -R2 @ C
                        per_frame_conf[i] = 0.5
                        n_anchor_resolved += 1
                        anchor_resolved_frames.add(i)
                        continue
                per_frame_K[i] = None
                per_frame_R[i] = None
                per_frame_t[i] = None
                per_frame_conf[i] = 0.0
            if demoted:
                logger.info(
                    "static line solve: demoted %d pre-bundle interp frame(s) "
                    "for re-solve at the locked geometry (%d anchor frame(s) "
                    "point-re-solved in place)",
                    len(demoted), n_anchor_resolved,
                )
        else:
            # One-C consistency: frames the solve skipped still share C.
            for i in covered:
                if i not in sol.per_frame_KRt and per_frame_R[i] is not None:
                    per_frame_t[i] = -per_frame_R[i] @ C

        # Q2 — coverage extension. Solve frames BEYOND the anchor span where
        # PnLCalib gives a plausible camera and lines are detectable (e.g. the
        # clip tail where the box is in view but no anchor cold-started there).
        # Each frame's (rvec, fx) is solved with the locked C, so the camera
        # body stays fixed. Only ADDS frames -> full-coverage clips (gberch)
        # get nothing to extend.
        if bool(cfg.get("line_extraction_extend_coverage", True)):
            from src.utils.static_c_profile import _solve_frame_at_fixed_c
            from src.utils.static_line_solver import _dist5
            # Only frames BEYOND the solved span: interior frames (incl. the
            # demoted pre-bundle ones) are the propagation pass's job and
            # don't warrant a per-frame PnLCalib inference here.
            _solved_now = [
                i for i in range(len(per_frame_K)) if per_frame_K[i] is not None
            ]
            _lo = min(_solved_now) if _solved_now else 0
            _hi = max(_solved_now) if _solved_now else -1
            uncovered = [
                i for i in range(len(per_frame_K))
                if per_frame_K[i] is None and not (_lo < i < _hi)
            ]
            ext_bgr: dict[int, np.ndarray] = {}
            for i in uncovered:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, fr = cap.read()
                if ok:
                    ext_bgr[i] = fr
            ext_cams = (
                self._pnlcalib_bootstrap_cameras(ext_bgr, cfg) if ext_bgr else {}
            )
            if ext_cams:
                # Detect under the solve's re-estimated (sane) distortion, not
                # the saturated anchor dist2.
                ext_dist = tuple(float(x) for x in sol.distortion[:2])
                ext_lines = detect_lines_for_frames(
                    ext_bgr, ext_cams, ext_dist, det_cfg,
                    min_confidence=det_min_confidence,
                    min_n_samples=det_min_n_samples, min_lines=1,
                )
                ext_lines = drop_underdetermined_frames(ext_lines, min_lines)
                cx_s, cy_s = sol.principal_point
                dist5 = _dist5(sol.distortion)
                max_ext_rms = float(cfg.get("line_extraction_extend_max_rms", 4.0))
                n_ext = 0
                for fid, lines in ext_lines.items():
                    cam = ext_cams[fid]
                    rv_seed, _ = cv2.Rodrigues(np.asarray(cam["R"]))
                    fx_seed = float(np.asarray(cam["K"])[0, 0])
                    rvec, fx, rms = _solve_frame_at_fixed_c(
                        lines, cx_s, cy_s, dist5, C,
                        rv_seed.reshape(3), fx_seed,
                    )
                    if not np.isfinite(rms) or rms > max_ext_rms:
                        continue
                    R_e, _ = cv2.Rodrigues(rvec)
                    per_frame_K[fid] = np.array(
                        [[fx, 0.0, cx_s], [0.0, fx, cy_s], [0.0, 0.0, 1.0]]
                    )
                    per_frame_R[fid] = R_e
                    per_frame_t[fid] = -R_e @ C
                    per_frame_conf[fid] = max(0.3, min(1.0, 1.0 - rms / 6.0))
                    detected_lines_by_frame[fid] = [
                        {
                            "name": ln.name,
                            "image_segment": [list(ln.image_segment[0]),
                                              list(ln.image_segment[1])],
                            "world_segment": [list(ln.world_segment[0]),
                                              list(ln.world_segment[1])],
                        }
                        for ln in lines
                    ]
                    n_ext += 1
                if n_ext:
                    logger.info(
                        "static line solve: coverage-extended %d frame(s) "
                        "beyond the anchor span", n_ext,
                    )

        # Propagating coverage extension: fill remaining uncovered frames where
        # the pitch is still visible by seeding each from its nearest COVERED
        # neighbour. The camera is static, so the locked C applies everywhere;
        # consecutive frames differ by ~one pan step, so the neighbour is a good
        # detection seed. Sweep down then up, repeat until stable. Each frame is
        # line-solved against its OWN detected lines, so seeds don't drift. This
        # reaches spans PnLCalib never cold-started (e.g. origi02 0-246). gberch
        # is already full-coverage -> nothing to propagate.
        if bool(cfg.get("line_extraction_propagate_coverage", True)):
            from scipy.spatial.transform import Rotation, Slerp

            from src.schemas.anchor import LandmarkObservation
            from src.utils.circle_detector import detect_circle
            from src.utils.static_c_profile import _solve_frame_at_fixed_c
            from src.utils.static_line_solver import _dist5
            cx_p, cy_p = sol.principal_point
            dist5_p = _dist5(sol.distortion)
            prop_dist = tuple(float(x) for x in sol.distortion[:2])
            max_prop_rms = float(cfg.get("line_extraction_extend_max_rms", 4.0))
            prop_circle = bool(cfg.get("line_extraction_propagate_circle", True))
            # Circle-aided sparse frames sit on the wide/midfield end where a
            # still-imperfect global lens inflates the residual (the same
            # lens-limited acceptance the cold-start uses); the post-
            # propagation circle-lens refinement then corrects the lens.
            circle_max_rms = float(
                cfg.get("line_extraction_propagate_circle_max_rms", 12.0))
            # A CORRECT circle refines the (adjacent-frame) velocity seed by a
            # fraction of a degree; a FALSE lock (wrong ridge under a wrong
            # pre-refinement lens — origi02's wide start) yanks it. Bound the
            # pull so false locks can't poison the propagation boundary.
            circle_max_dev = float(
                cfg.get("line_extraction_propagate_circle_max_dev_deg", 3.0))
            # Propagation can accept fewer lines than the cold main solve: it
            # has a strong per-frame seed (the covered neighbour), so a frame
            # with 3 well-conditioned box lines solves cleanly, and the rms gate
            # + post-hoc rotation-outlier rejection guard against any
            # under-determined drift.
            prop_min_lines = int(cfg.get("line_extraction_propagate_min_lines", 3))
            n_total = len(per_frame_K)
            _cache: dict[int, np.ndarray | None] = {}

            def _read(i: int):
                if i not in _cache:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, fr = cap.read()
                    _cache[i] = fr if ok else None
                return _cache[i]

            def _seed(nb1: int, nb2: int, steps: int) -> tuple[np.ndarray, float]:
                """Velocity-extrapolate ``steps`` pan-steps beyond nb1 using the
                nb2->nb1 delta (tracks the pan so detection still finds lines)."""
                R1 = per_frame_R[nb1]
                fx1 = float(per_frame_K[nb1][0, 0])
                if (0 <= nb2 < n_total and per_frame_R[nb2] is not None):
                    D = Rotation.from_matrix(R1 @ np.asarray(per_frame_R[nb2]).T)
                    Dk = Rotation.from_rotvec(D.as_rotvec() * steps)
                    seed_R = (Dk * Rotation.from_matrix(R1)).as_matrix()
                    dfx = fx1 - float(per_frame_K[nb2][0, 0])
                    seed_fx = float(np.clip(fx1 + dfx * steps, 0.7 * fx1, 1.3 * fx1))
                    return seed_R, seed_fx
                return R1, fx1

            def _solve_at(
                i: int, seed_R: np.ndarray, seed_fx: float,
                use_circle: bool = True, fx0: float | None = None,
            ) -> bool:
                img = _read(i)
                if img is None:
                    return False
                seed_t = -seed_R @ C
                seed_K = np.array(
                    [[seed_fx, 0.0, cx_p], [0.0, seed_fx, cy_p], [0.0, 0.0, 1.0]])
                det = detect_lines_for_frames(
                    {i: img}, {i: {"K": seed_K, "R": seed_R, "t": seed_t}},
                    prop_dist, det_cfg, min_confidence=det_min_confidence,
                    min_n_samples=det_min_n_samples, min_lines=1,
                )
                lines = det.get(i, [])
                # Centre circle: a strong, well-spread constraint where straight
                # lines are sparse — lets propagation cross featureless spans and
                # better-pins under-constrained frames. Detected under zero
                # distortion (the catalogue-projection lesson). Only consulted
                # when the frame is line-SPARSE (< the under-determined
                # threshold): well-lined frames solve from lines alone, so the
                # circle can never trade away their straight-line fit (the
                # regression that originally got propagate_circle disabled).
                circ_obs = None
                circ_det = None
                if (use_circle and prop_circle and len(lines) < min_lines
                        and _circle_in_view_fraction(
                            seed_K, seed_R, seed_t,
                            anchors.image_size) >= 0.3):
                    # The circle gets a wider search strip than lines: it has
                    # no adjacent-parallel-feature to mis-lock onto, and at
                    # long focal a fraction of a degree of seed error already
                    # displaces its projection by the line strip width.
                    circ_det = detect_circle(
                        img, seed_K, seed_R, seed_t, (0.0, 0.0),
                        DetectorConfig(
                            search_strip_px=max(
                                50, det_cfg.search_strip_px),
                            min_gradient=det_cfg.min_gradient))
                    if circ_det is not None:
                        k = min(20, len(circ_det.image_points))
                        idx = np.linspace(
                            0, len(circ_det.image_points) - 1, k).astype(int)
                        circ_obs = [
                            LandmarkObservation(
                                name=circ_det.name,
                                image_xy=circ_det.image_points[j],
                                world_xyz=circ_det.world_points[j])
                            for j in idx
                        ]
                if len(lines) < prop_min_lines and not circ_obs:
                    logger.debug(
                        "propagation: f%d rejected — %d line(s), no circle",
                        i, len(lines))
                    return False
                # Sparse line sets without angular spread (a near-PARALLEL
                # pair — far touchline + an 18yd edge in origi01's f296-342
                # dead zone) fit at ~0 rms while fx slides along the line
                # direction. They may still solve (origi02's short marches
                # rely on them and are correct) but they cannot MEASURE fx,
                # so they get a much tighter fx envelope below.
                parallel_only = False
                if len(lines) < min_lines:
                    angs = []
                    for ln in lines:
                        d = (np.asarray(ln.image_segment[1], float)
                             - np.asarray(ln.image_segment[0], float))
                        angs.append(np.arctan2(d[1], d[0]) % np.pi)
                    spread = 0.0
                    for ai in range(len(angs)):
                        for bi in range(ai + 1, len(angs)):
                            d = abs(angs[ai] - angs[bi])
                            spread = max(spread, min(d, np.pi - d))
                    parallel_only = np.degrees(spread) < 20.0
                rv_seed, _ = cv2.Rodrigues(seed_R)
                # Sparse frames: tight-band the focal around the (reliable)
                # seed so the few lines can't fit a wrong rotation by drifting
                # fx. Parallel-only sets WITHOUT a circle cannot measure fx at
                # all — clamp to ~the seed (1 %/step): slow real zoom still
                # tracks across a march (origi02's span drifts 0.07 %/frame)
                # while a drift-into-rotation failure (kroupi's tail,
                # 1.7 %/frame) cannot follow its fx and fails the rms gate
                # instead. A detected circle DOES pin fx (its projected size
                # is proportional to it), so circle-aided frames keep the
                # normal band — clamping them starves the circle commits the
                # lens refinement depends on.
                _fxr = (0.01 if (parallel_only and not circ_obs)
                        else (0.05 if len(lines) < 4 else None))
                rvec, fx, rms = _solve_frame_at_fixed_c(
                    lines, cx_p, cy_p, dist5_p, C, rv_seed.reshape(3), seed_fx,
                    fx_rel=_fxr, circle_obs=circ_obs)
                if ((not np.isfinite(rms) or rms > max_prop_rms)
                        and circ_obs and len(lines) >= prop_min_lines):
                    # A bad/partial circle must NOT block a solvable frame:
                    # retry lines-only (else propagation halts at the barrier —
                    # this was collapsing origi02 coverage). The retry is
                    # lines-only by construction, so a parallel-only set gets
                    # the seed-clamped fx band here too.
                    rvec, fx, rms = _solve_frame_at_fixed_c(
                        lines, cx_p, cy_p, dist5_p, C, rv_seed.reshape(3),
                        seed_fx,
                        fx_rel=(0.01 if parallel_only
                                else (0.05 if len(lines) < 4 else None)))
                    circ_det = None  # circle rejected -> don't draw it
                    circ_obs = None
                rms_gate = circle_max_rms if circ_det is not None else max_prop_rms
                if not np.isfinite(rms) or rms > rms_gate:
                    logger.debug(
                        "propagation: f%d rejected — rms %.2f > %.2f "
                        "(%d line(s), circle=%s)", i, rms, rms_gate,
                        len(lines), circ_obs is not None)
                    return False
                if circ_det is not None:
                    R_chk, _ = cv2.Rodrigues(rvec)
                    dev = _angle_between(np.asarray(seed_R), R_chk)
                    if dev > circle_max_dev:
                        logger.debug(
                            "propagation: f%d rejected — circle-aided solve "
                            "pulled %.1f deg from the seed (> %.1f)",
                            i, dev, circle_max_dev)
                        return False
                # Origin-anchored fx bound for SPREAD-sparse solves: per-step
                # bands (±5%) cannot stop a long march of exactly-determined
                # 2-3-line solves from random-walking fx (origi01's f296-342
                # dead zone walked 4800 -> 2893 over ~47 frames and the
                # overlay wobbled). Every march inherits the fx of the SOLID
                # frame it started from; no solve may leave a generous zoom
                # envelope of it. Parallel-only frames are exempt — their fx
                # is seed-clamped above, and an origin envelope would reject
                # the slow REAL zoom their long marches legitimately carry
                # (origi02's 70-251 span).
                if (fx0 is not None and len(lines) < 4
                        and not (parallel_only and circ_obs is None)
                        and not (0.75 * fx0 <= fx <= 1.3 * fx0)):
                    logger.debug(
                        "propagation: f%d rejected — fx %.0f left the march "
                        "origin envelope [%.0f, %.0f]", i, fx,
                        0.75 * fx0, 1.3 * fx0)
                    return False
                R_e, _ = cv2.Rodrigues(rvec)
                per_frame_K[i] = np.array(
                    [[fx, 0.0, cx_p], [0.0, fx, cy_p], [0.0, 0.0, 1.0]])
                per_frame_R[i] = R_e
                per_frame_t[i] = -R_e @ C
                per_frame_conf[i] = max(0.3, min(1.0, 1.0 - rms / 6.0))
                entries = [
                    {
                        "name": ln.name,
                        "image_segment": [list(ln.image_segment[0]),
                                          list(ln.image_segment[1])],
                        "world_segment": [list(ln.world_segment[0]),
                                          list(ln.world_segment[1])],
                    }
                    for ln in lines
                ]
                if circ_det is not None:
                    # write the circle as a polyline so the viewer's detected-
                    # lines overlay renders it (consecutive detected points).
                    ip = circ_det.image_points
                    wp = circ_det.world_points
                    for a in range(len(ip) - 1):
                        entries.append({
                            "name": "centre_circle",
                            "image_segment": [list(ip[a]), list(ip[a + 1])],
                            "world_segment": [list(wp[a]), list(wp[a + 1])],
                        })
                detected_lines_by_frame[i] = entries
                return True

            def _bridge(lo: int, hi: int) -> None:
                """SLERP/LERP-fill the (feature-poor) frames strictly between two
                solved brackets lo and hi — interpolated, lower confidence."""
                slerp = Slerp([float(lo), float(hi)], Rotation.from_matrix(
                    [per_frame_R[lo], per_frame_R[hi]]))
                fxlo = float(per_frame_K[lo][0, 0])
                fxhi = float(per_frame_K[hi][0, 0])
                for m in range(lo + 1, hi):
                    if per_frame_K[m] is not None:
                        continue
                    Rm = slerp([float(m)]).as_matrix()[0]
                    w = (m - lo) / (hi - lo)
                    fxm = (1 - w) * fxlo + w * fxhi
                    per_frame_K[m] = np.array(
                        [[fxm, 0.0, cx_p], [0.0, fxm, cy_p], [0.0, 0.0, 1.0]])
                    per_frame_R[m] = Rm
                    per_frame_t[m] = -Rm @ C
                    per_frame_conf[m] = 0.35

            max_bridge = int(cfg.get("line_extraction_propagate_max_bridge", 8))

            def _run_propagation(use_circle: bool) -> int:
                n_prop = 0
                # fx of the SOLID frame each march started from — frames
                # covered before this call are their own origin; propagated
                # frames inherit. Bounds the cumulative fx walk of long
                # marches of exactly-determined sparse solves.
                fx_origin: dict[int, float] = {}
                progress = True
                while progress:
                    progress = False
                    for direction in (-1, +1):
                        order = (range(n_total) if direction > 0
                                 else range(n_total - 1, -1, -1))
                        for f in order:
                            if per_frame_K[f] is None:
                                continue
                            i = f + direction
                            if not (0 <= i < n_total) or per_frame_K[i] is not None:
                                continue
                            fx0 = fx_origin.get(
                                f, float(per_frame_K[f][0, 0]))
                            nb2 = f - direction
                            # 1) direct solve one step out
                            sR, sfx = _seed(f, nb2, 1)
                            if _solve_at(i, sR, sfx, use_circle, fx0):
                                fx_origin[i] = fx0
                                n_prop += 1; progress = True
                                continue
                            # 1b) circle pass: the velocity extrapolation can
                            # be poisoned where two solve regimes meet (an
                            # anchor island next to a circle-solved frame);
                            # retry with the plain neighbour camera.
                            if use_circle and _solve_at(
                                    i, np.asarray(per_frame_R[f]),
                                    float(per_frame_K[f][0, 0]), True, fx0):
                                fx_origin[i] = fx0
                                n_prop += 1; progress = True
                                continue
                            # 2) barrier: look outward up to max_bridge for a
                            #    frame that re-acquires; if found, bridge the
                            #    gap between.
                            reacq = None
                            for k in range(2, max_bridge + 2):
                                j = f + direction * k
                                if not (0 <= j < n_total) or per_frame_K[j] is not None:
                                    continue
                                sRk, sfxk = _seed(f, nb2, k)
                                if _solve_at(j, sRk, sfxk, use_circle, fx0):
                                    fx_origin[j] = fx0
                                    reacq = j
                                    break
                            if reacq is not None:
                                lo, hi = (reacq, f) if reacq < f else (f, reacq)
                                _bridge(lo, hi)
                                n_prop += 1; progress = True
                return n_prop

            # PASS 1 — line-only. Circle-aided solves are lens-limited
            # (accepted at a relaxed rms while the global lens is still
            # imperfect); letting them march past the clean line boundary
            # accumulates drift and walks fx, poisoning the boundary frame the
            # cold-start uses as its reference (origi02's start). Lines first,
            # cold-start from the clean boundary, circle pass after.
            n_prop = _run_propagation(use_circle=False)
            if n_prop:
                logger.info(
                    "static line solve: propagated coverage to %d additional "
                    "frame(s) (incl. bridged feature-poor barriers)", n_prop,
                )
            circle_propagation = _run_propagation if prop_circle else None
        else:
            circle_propagation = None

        # Cold-start bootstrap for the uncovered START of the clip (runs AFTER
        # propagation, so the reference orientation is the midfield boundary
        # frame, not the box-end where the clip is anchored). Clips pan from a
        # wide/midfield start to the box; the start is reached only by a long,
        # degrading propagation, and a disconnected start (where the backward
        # velocity-seed diverges and PnLCalib couldn't cold-start) never gets
        # covered. With C + lens already solved, a start frame needs only
        # orientation: recover it by an ORIENTATION SWEEP around the covered
        # boundary's orientation (sweep pan/tilt, detect+solve, iteratively
        # refine), keep the consistent solves as seeds, then cascade-fill the
        # small gaps between them. The post-propagation lens refinement then uses
        # the now-covered WIDE start (its lines + circle) to better-calibrate the
        # global lens.
        if bool(cfg.get("line_extraction_cold_start", True)):
            from scipy.spatial.transform import Rotation as _CSRot

            from src.schemas.anchor import LandmarkObservation
            from src.utils.circle_detector import detect_circle as _cs_detect_circle
            from src.utils.static_c_profile import _solve_frame_at_fixed_c as _cs_solve_frame
            from src.utils.static_line_solver import _dist5 as _cs_dist5
            from src.utils.static_line_solver import (
                _line_residuals_distorted as _cs_lres,
            )
            _csc = [i for i in range(len(per_frame_K)) if per_frame_K[i] is not None]
            start_end = min(_csc) if _csc else 0
            if start_end > 0:
                cs_cx, cs_cy = sol.principal_point
                cs_d5 = _cs_dist5(sol.distortion)
                cs_dist = tuple(float(x) for x in sol.distortion[:2])
                _csf = sorted(_csc)
                _ax = [
                    _CSRot.from_matrix(
                        per_frame_R[b] @ np.asarray(per_frame_R[a]).T).as_rotvec()
                    for a, b in zip(_csf, _csf[1:]) if b - a == 1
                ]
                _ax = [v for v in _ax if np.linalg.norm(v) > 1e-4]
                if _ax:
                    pan_ax = np.mean([v / np.linalg.norm(v) for v in _ax], axis=0)
                    pan_ax /= np.linalg.norm(pan_ax)
                    R_ref = np.asarray(per_frame_R[start_end])
                    fx_ref = float(per_frame_K[start_end][0, 0])
                    tilt_ax = np.cross(pan_ax, R_ref[2])
                    tilt_ax /= np.linalg.norm(tilt_ax)
                    cs_max_rms = float(cfg.get("line_extraction_cold_start_max_rms", 12.0))
                    # Circle-aided start frames gate on the MEDIAN residual,
                    # which under a still-wrong wide-field lens is dominated by
                    # global lens error across the circle's full arc (origi01:
                    # 13-37 px median at the start under the box-solved lens).
                    # Accept lens-limited locks — the post-propagation circle-
                    # lens refinement + post-lens re-solve then correct them.
                    cs_circle_max_rms = float(cfg.get(
                        "line_extraction_cold_start_circle_max_rms", 40.0))
                    cs_min_lines = int(cfg.get("line_extraction_cold_start_min_lines", 3))

                    def _cs_solve_at(img, fid, R, fx, strip,
                                     allow_circle=False, max_pull=3.0):
                        sK = np.array([[fx, 0, cs_cx], [0, fx, cs_cy], [0, 0, 1.0]])
                        det = detect_lines_for_frames(
                            {fid: img}, {fid: {"K": sK, "R": R, "t": -R @ C}},
                            cs_dist, DetectorConfig(
                                search_strip_px=strip, min_gradient=10.0),
                            min_confidence=det_min_confidence,
                            min_n_samples=det_min_n_samples, min_lines=1).get(fid, [])
                        # Line-sparse start frames (e.g. a parallel 18yd pair,
                        # or halfway line only) are under-determined from
                        # straight lines — the centre circle disambiguates
                        # them, exactly as in propagation. The circle gets a
                        # much wider strip than lines: the sweep grid is
                        # 4-5 deg coarse, which at broadcast focal displaces
                        # the projected circle by >100 px; the iterative
                        # refine converges true locks while the rms + pull +
                        # boundary-deviation gates reject false ones.
                        circ_obs = None
                        circ_det = None
                        if len(det) < cs_min_lines:
                            if (allow_circle
                                    and _circle_in_view_fraction(
                                        sK, R, -R @ C,
                                        anchors.image_size) >= 0.3):
                                circ_det = _cs_detect_circle(
                                    img, sK, R, -R @ C, (0.0, 0.0),
                                    DetectorConfig(
                                        search_strip_px=max(100, 2 * strip),
                                        min_gradient=10.0))
                            if circ_det is None:
                                return None
                            k = min(20, len(circ_det.image_points))
                            idx = np.linspace(
                                0, len(circ_det.image_points) - 1, k).astype(int)
                            circ_obs = [
                                LandmarkObservation(
                                    name=circ_det.name,
                                    image_xy=circ_det.image_points[j],
                                    world_xyz=circ_det.world_points[j])
                                for j in idx
                            ]
                        rv, _ = cv2.Rodrigues(R)
                        rvec, fx2, rms = _cs_solve_frame(
                            det, cs_cx, cs_cy, cs_d5, C, rv.reshape(3), fx,
                            fx_rel=0.05 if len(det) < 4 else None,
                            circle_obs=circ_obs)
                        if not np.isfinite(rms):
                            return None
                        Re, _ = cv2.Rodrigues(rvec)
                        if circ_obs is not None and _angle_between(
                                np.asarray(R), Re) > max_pull:
                            # false circle lock yanking away from the seed
                            return None
                        if circ_obs is not None:
                            # The wide circle strip admits fat-tail outliers
                            # (players, the D-arc) that inflate the RAW rms of
                            # a correct, Huber-solved lock; gate circle-aided
                            # frames on the MEDIAN absolute residual instead.
                            from src.utils.anchor_solver import (
                                _point_residuals_distorted as _cs_pres,
                            )
                            K2 = np.array([[fx2, 0, cs_cx], [0, fx2, cs_cy],
                                           [0, 0, 1.0]])
                            pr = np.asarray(_cs_pres(
                                circ_obs, K2, rvec, -Re @ C,
                                (float(cs_dist[0]), float(cs_dist[1]))))
                            pn = np.linalg.norm(pr.reshape(-1, 2), axis=1)
                            lr = _cs_lres(det, K2, rvec, -Re @ C, cs_d5)
                            allr = np.concatenate(
                                [pn, np.abs(np.asarray(lr).ravel())]
                                if len(det) else [pn])
                            rms = float(np.median(allr))
                        return Re, fx2, rms, det, circ_det

                    def _cs_commit(f, Re, fx, det, circ_det=None):
                        per_frame_K[f] = np.array(
                            [[fx, 0, cs_cx], [0, fx, cs_cy], [0, 0, 1.0]])
                        per_frame_R[f] = Re
                        per_frame_t[f] = -Re @ C
                        per_frame_conf[f] = 0.4
                        entries = [
                            {"name": ln.name,
                             "image_segment": [list(ln.image_segment[0]),
                                               list(ln.image_segment[1])],
                             "world_segment": [list(ln.world_segment[0]),
                                               list(ln.world_segment[1])]}
                            for ln in det
                        ]
                        if circ_det is not None:
                            ip = circ_det.image_points
                            wp = circ_det.world_points
                            for a in range(len(ip) - 1):
                                entries.append({
                                    "name": "centre_circle",
                                    "image_segment": [list(ip[a]),
                                                      list(ip[a + 1])],
                                    "world_segment": [list(wp[a]),
                                                      list(wp[a + 1])],
                                })
                        detected_lines_by_frame[f] = entries

                    _cs_imgs: dict[int, np.ndarray | None] = {}

                    def _cs_read(f):
                        if f not in _cs_imgs:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                            ok, im = cap.read()
                            _cs_imgs[f] = im if ok else None
                        return _cs_imgs[f]

                    def _cold_start_one(fid):
                        img = _cs_read(fid)
                        if img is None:
                            return None
                        best = None
                        for dp in np.arange(-24, 25, 4):
                            for dt in np.arange(-10, 11, 5):
                                Rc = (_CSRot.from_rotvec(pan_ax * np.radians(dp))
                                      * _CSRot.from_rotvec(tilt_ax * np.radians(dt))
                                      ).as_matrix() @ R_ref
                                # Circle allowed: with fx_ref taken from the
                                # covered boundary each candidate solves
                                # independently (no per-step fx walk — the
                                # degeneracy that sinks circle-only
                                # PROPAGATION marches), the ranking prefers
                                # line-rich candidates, and clips whose start
                                # detects >=cs_min_lines lines (origi02) never
                                # consult the circle at all.
                                r = _cs_solve_at(img, fid, Rc, fx_ref, 40,
                                                 allow_circle=True,
                                                 max_pull=6.0)
                                if r is None:
                                    continue
                                if best is None or (len(r[3]), -r[2]) > (
                                        len(best[3]), -best[2]):
                                    best = r
                        if best is None:
                            logger.debug(
                                "cold-start f%d: no sweep candidate", fid)
                            return None
                        Re, fx, rms, det, circ = best
                        for _ in range(4):  # iterative refine: re-detect under the solve
                            r = _cs_solve_at(img, fid, Re, fx, 30,
                                             allow_circle=True, max_pull=6.0)
                            if r is None:
                                break
                            Re, fx, rms, det, circ = r
                        gate = (cs_circle_max_rms if circ is not None
                                else cs_max_rms)
                        if rms > gate:
                            logger.debug(
                                "cold-start f%d: rejected — refined rms %.1f "
                                "> %.1f (%d lines, circle=%s)", fid, rms,
                                gate, len(det), circ is not None)
                            return None
                        return Re, fx, det, circ

                    # Cold-start a handful of evenly-spaced start frames.
                    cs_seeds: dict[int, tuple] = {}
                    cs_step = max(1, start_end // 8)
                    cs_tried = list(range(0, start_end, cs_step))
                    for fid in cs_tried:
                        out = _cold_start_one(fid)
                        if out is not None:
                            cs_seeds[fid] = out
                            logger.debug(
                                "cold-start sweep: f%d seeded (%d lines, "
                                "circle=%s)", fid, len(out[2]),
                                out[3] is not None)
                    logger.info(
                        "static line solve: cold-start sweep tried %d start "
                        "frame(s) -> %d seed(s)%s", len(cs_tried),
                        len(cs_seeds),
                        "" if len(cs_seeds) >= 2 else " (<2 -> start skipped)")

                    def _csgeo(a, b):
                        c = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1) / 2
                        return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))

                    n_cs = 0
                    if len(cs_seeds) >= 2:
                        # Keep cold-starts within a bounded pan of the covered
                        # boundary — rejects gross false orientation locks.
                        cs_max_dev = float(
                            cfg.get("line_extraction_cold_start_max_dev_deg", 25.0))
                        keep = [f for f in sorted(cs_seeds)
                                if _csgeo(cs_seeds[f][0], R_ref) < cs_max_dev]
                        for f in keep:
                            Re, fx, det, circ = cs_seeds[f]
                            _cs_commit(f, Re, fx, det, circ)
                            n_cs += 1
                        # Cascade-fill the gaps between the cold-started seeds and
                        # the covered boundary: each uncovered start frame is
                        # re-solved from a covered neighbour (a strong seed now),
                        # bridging the thin barriers the boundary cascade can't
                        # cross from outside.
                        if n_cs:
                            fillprog = True
                            while fillprog:
                                fillprog = False
                                for f in range(start_end):
                                    if per_frame_K[f] is not None:
                                        continue
                                    nb = None
                                    if f - 1 >= 0 and per_frame_K[f - 1] is not None:
                                        nb = f - 1
                                    elif (f + 1 < len(per_frame_K)
                                          and per_frame_K[f + 1] is not None):
                                        nb = f + 1
                                    if nb is None:
                                        continue
                                    img = _cs_read(f)
                                    if img is None:
                                        continue
                                    r = _cs_solve_at(
                                        img, f, np.asarray(per_frame_R[nb]),
                                        float(per_frame_K[nb][0, 0]), 30,
                                        allow_circle=True)
                                    if r is not None and r[2] <= (
                                            cs_circle_max_rms
                                            if r[4] is not None
                                            else cs_max_rms):
                                        _cs_commit(f, r[0], r[1], r[3], r[4])
                                        fillprog = True
                    if n_cs:
                        n_filled = sum(1 for i in range(start_end)
                                       if per_frame_K[i] is not None)
                        logger.info(
                            "static line solve: cold-started %d seed(s) + "
                            "cascade-filled the start to %d/%d covered frame(s)",
                            n_cs, n_filled, start_end)

        _anchor_click_checkpoint("post-coldstart")

        # PASS 2 — circle-aided propagation, AFTER the cold-start so its
        # lens-limited solves can only fill spans neither lines nor the
        # orientation sweep could reach (origi01's circle-only midfield and
        # start, marching from the point-re-solved anchor islands). Clips the
        # cold-start already completed (origi02) leave nothing to fill.
        if circle_propagation is not None:
            n_prop2 = circle_propagation(True)
            if n_prop2:
                logger.info(
                    "static line solve: circle-aided propagation covered %d "
                    "additional frame(s)", n_prop2,
                )

        # Advertising-hoarding base line — a STATIC SCENE LINE parallel to
        # the far touchline (one offset per clip; h fixed at 0: with a static
        # C the (d, h) family is projectively equivalent). The boards are the
        # highest-contrast feature in exactly the far field where pitch lines
        # are starved, so the calibrated edge constrains tilt + lens wherever
        # it is visible. Runs AFTER propagation/cold-start: calibration needs
        # solved cameras on frames that actually SEE the boards (calibrating
        # on box-end bundle frames poisoned origi02's whole start), and it is
        # applied as a gated RE-SOLVE of covered frames + stored detections
        # for the lens refinement — never inside the coverage marches, where
        # a mis-calibrated plane would propagate its own error.
        board_model = None
        if bool(cfg.get("line_extraction_board_line", True)):
            from src.schemas.anchor import (
                LandmarkObservation,
                LineObservation,
            )
            from src.utils.hoarding_detector import (
                calibrate_board_line,
                detect_board_line,
            )
            _board_dist = tuple(float(x) for x in sol.distortion[:2])

            def _far_touchline_vis(f: int) -> float:
                """Fraction of the far touchline projecting in-image — the
                d-independent proxy for 'this frame sees the board zone'."""
                xs_v = np.linspace(0.0, 105.0, 36)
                world = np.stack(
                    [xs_v, np.full_like(xs_v, 68.0), np.zeros_like(xs_v)],
                    axis=1)
                cam_p = world @ np.asarray(per_frame_R[f]).T + per_frame_t[f]
                ok_z = cam_p[:, 2] > 1.0
                if not ok_z.any():
                    return 0.0
                from src.utils.camera_projection import (
                    project_world_to_image,
                )
                pr = project_world_to_image(
                    per_frame_K[f], per_frame_R[f], per_frame_t[f],
                    _board_dist, world)
                w_i, h_i = anchors.image_size
                inside = (ok_z & np.isfinite(pr).all(axis=1)
                          & (pr[:, 0] >= 0) & (pr[:, 0] < w_i)
                          & (pr[:, 1] >= 0) & (pr[:, 1] < h_i))
                return float(inside.sum()) / len(xs_v)

            def _board_cal_quality(f: int) -> float:
                """How much to trust frame f's camera for calibrating d. The
                d-estimate error tracks the frame's FAR-FIELD accuracy:
                circle-constrained frames are the best available reference
                (the ring spans midfield depth; held-out misfit ~3-4 px),
                line-rich solves next, point-re-solved anchor islands next —
                lens-limited cold-start fills (which dominated the old
                visibility-only selection and blew the d-spread past the
                trust gate on the origi clips) score lowest.
                """
                entries = detected_lines_by_frame.get(f) or []
                n_straight = sum(
                    1 for ln in entries if "circle" not in ln["name"])
                has_circle = any("circle" in ln["name"] for ln in entries)
                return ((2.0 if has_circle else 0.0)
                        + min(n_straight, 4)
                        + (1.0 if f in anchor_resolved_frames else 0.0))

            _covered_b = [i for i in range(len(per_frame_K))
                          if per_frame_K[i] is not None]
            _cands_b = [
                f for f in _covered_b[::max(1, len(_covered_b) // 60)]
                if _far_touchline_vis(f) >= 0.3 and _board_cal_quality(f) >= 2.0
            ]
            _cands_b.sort(
                key=lambda f: (_board_cal_quality(f), _far_touchline_vis(f)),
                reverse=True)
            cal_fids = sorted(_cands_b[:10])
            cal_frames: dict[int, np.ndarray] = {}
            for f in cal_fids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, im = cap.read()
                if ok:
                    cal_frames[f] = im
            if len(cal_frames) >= 3:
                cal_cams = {
                    f: {"K": per_frame_K[f], "R": per_frame_R[f],
                        "t": per_frame_t[f]}
                    for f in cal_frames
                }
                board_model = calibrate_board_line(
                    cal_frames, cal_cams, _board_dist, det_cfg)
            # d-spread <= 0.5 m: the board only ever acts where its per-frame
            # calibration is demonstrably consistent (kroupi: 0.20 m, clear
            # win). Clips whose cameras still wobble in the far field (origi:
            # 0.7 m+) abstain — applying an uncertain plane there regressed
            # origi01's start. As tracks improve, more clips qualify.
            if board_model is not None and (
                    board_model.frames < 3 or board_model.residual > 0.5
                    or board_model.contrast < 20.0):
                logger.info(
                    "board line: rejected calibration (frames=%d, d-spread "
                    "%.2f m, contrast %.0f)", board_model.frames,
                    board_model.residual, board_model.contrast)
                board_model = None
            if board_model is not None:
                logger.info(
                    "board line: calibrated d=%.2f m (h=0 family) over %d "
                    "frame(s), d-spread %.2f m, contrast %.0f",
                    board_model.d, board_model.frames, board_model.residual,
                    board_model.contrast)

            if board_model is not None:
                from src.utils.static_c_profile import (
                    _solve_frame_at_fixed_c as _bd_solve,
                )
                from src.utils.static_line_solver import _dist5 as _bd_d5
                bd_d5 = _bd_d5(sol.distortion)
                bd_cx, bd_cy = sol.principal_point
                n_bd_entries = 0
                n_bd_resolved = 0
                for f in _covered_b:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ok, im = cap.read()
                    if not ok:
                        continue
                    det_b = detect_board_line(
                        im, per_frame_K[f], per_frame_R[f], per_frame_t[f],
                        _board_dist, board_model.d, board_model.h, det_cfg)
                    if det_b is None:
                        continue
                    bob = LineObservation(
                        name="board_line",
                        image_segment=det_b.image_segment,
                        world_segment=det_b.world_segment)
                    entries = detected_lines_by_frame.get(f) or []
                    pitch_lns = [
                        LineObservation(
                            name=ln["name"],
                            image_segment=(tuple(ln["image_segment"][0]),
                                           tuple(ln["image_segment"][1])),
                            world_segment=(tuple(ln["world_segment"][0]),
                                           tuple(ln["world_segment"][1])))
                        for ln in entries if "circle" not in ln["name"]
                        and ln["name"] != "board_line"
                    ]
                    circ_obs_b = [
                        LandmarkObservation(
                            name="centre_circle",
                            image_xy=tuple(ln["image_segment"][0]),
                            world_xyz=tuple(ln["world_segment"][0]))
                        for ln in entries if "circle" in ln["name"]
                    ] or None
                    # Gated re-solve: pitch features + board together. The
                    # board may only ADJUST a frame, never replace it — a
                    # solve that pulls far from the current camera loses
                    # (mis-calibration protection).
                    rv_b, _ = cv2.Rodrigues(np.asarray(per_frame_R[f]))
                    fx_b = float(per_frame_K[f][0, 0])
                    all_lns = pitch_lns + [bob]
                    rvec_b, fx2_b, rms_b = _bd_solve(
                        all_lns, bd_cx, bd_cy, bd_d5, C, rv_b.reshape(3),
                        fx_b, fx_rel=0.05, circle_obs=circ_obs_b)
                    R2b, _ = cv2.Rodrigues(rvec_b)
                    if (np.isfinite(rms_b) and rms_b <= 12.0
                            and _angle_between(
                                np.asarray(per_frame_R[f]), R2b) <= 2.0):
                        per_frame_K[f] = np.array(
                            [[fx2_b, 0.0, bd_cx], [0.0, fx2_b, bd_cy],
                             [0.0, 0.0, 1.0]])
                        per_frame_R[f] = R2b
                        per_frame_t[f] = -R2b @ C
                        n_bd_resolved += 1
                        entries.append({
                            "name": "board_line",
                            "image_segment": [list(bob.image_segment[0]),
                                              list(bob.image_segment[1])],
                            "world_segment": [list(bob.world_segment[0]),
                                              list(bob.world_segment[1])],
                        })
                        detected_lines_by_frame[f] = entries
                        n_bd_entries += 1
                if n_bd_entries:
                    logger.info(
                        "board line: re-solved %d covered frame(s) with the "
                        "far-field constraint", n_bd_resolved)

        # Final interior gap-fill: any frame still uncovered BETWEEN two solved
        # frames (a demoted pre-bundle frame nothing could re-solve, or a
        # featureless barrier longer than propagate_max_bridge) gets SLERP/LERP
        # interpolation between its solved brackets — consistent with the
        # locked geometry, unlike the stale pre-bundle pose it replaced. Head
        # and tail gaps are left to cold-start / coverage extension. No
        # interior gaps -> no-op (gberch). Re-run after the lens refinement,
        # which may invalidate frames it could not re-solve.
        def _fill_interior_gaps() -> int:
            from scipy.spatial.transform import Rotation as _GRot
            from scipy.spatial.transform import Slerp as _GSlerp
            solved = [i for i in range(len(per_frame_K))
                      if per_frame_K[i] is not None]
            n_filled = 0
            for a, b in zip(solved, solved[1:]):
                if b - a <= 1:
                    continue
                slerp = _GSlerp([float(a), float(b)], _GRot.from_matrix(
                    [per_frame_R[a], per_frame_R[b]]))
                fxa = float(per_frame_K[a][0, 0])
                fxb = float(per_frame_K[b][0, 0])
                cxa, cya = float(per_frame_K[a][0, 2]), float(per_frame_K[a][1, 2])
                for m in range(a + 1, b):
                    Rm = slerp([float(m)]).as_matrix()[0]
                    w = (m - a) / (b - a)
                    fxm = (1 - w) * fxa + w * fxb
                    per_frame_K[m] = np.array(
                        [[fxm, 0.0, cxa], [0.0, fxm, cya], [0.0, 0.0, 1.0]])
                    per_frame_R[m] = Rm
                    per_frame_t[m] = -Rm @ C
                    per_frame_conf[m] = 0.3
                    n_filled += 1
            return n_filled

        n_gap_filled = _fill_interior_gaps()
        if n_gap_filled:
            logger.info(
                "static line solve: SLERP-filled %d interior gap frame(s) "
                "between solved brackets", n_gap_filled,
            )

        # Centre-circle global LENS refinement (post-propagation). The wide ring
        # exposes distortion / principal-point error that central box lines can't
        # constrain — on zoomed-out / midfield shots the box-line lens
        # under-estimates distortion, so the circle + far lines project several
        # px off. The circle lives on midfield frames covered only after
        # propagation, so this runs here. GBERCH-SAFE: only refines when the
        # circle is *significantly* mis-fit under the current lens — gberch's
        # lens is already right (the circle fits) so it is skipped untouched.
        # Iterative: each refinement improves the cameras, which lets the
        # banded ellipse search find MORE wide-field rings next round (the
        # cause of origi02's k "breathing" 0.08-0.24 across single-round
        # runs). Loop until the distortion converges or a round refines
        # nothing.
        _lens_rounds = (
            int(cfg.get("line_extraction_circle_lens_rounds", 3))
            if bool(cfg.get("line_extraction_circle_lens", True)) else 0)
        _lens_prev_circ: float | None = None
        for _lens_round in range(_lens_rounds):
            import dataclasses

            from src.schemas.anchor import LandmarkObservation, LineObservation
            from src.utils.ellipse_detector import detect_circle_ellipse
            from src.utils.static_line_solver import _ellipse_residuals_distorted
            cur_dist = tuple(float(x) for x in sol.distortion[:2])
            cx_l, cy_l = sol.principal_point
            band = float(cfg.get("line_extraction_circle_ellipse_band", 50.0))
            covered_now = [
                i for i in range(len(per_frame_K)) if per_frame_K[i] is not None
            ]
            ell: dict[int, tuple] = {}
            for fid in covered_now[::max(1, len(covered_now) // 120)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, img = cap.read()
                if not ok:
                    continue
                ed = detect_circle_ellipse(
                    img, per_frame_K[fid], per_frame_R[fid], per_frame_t[fid],
                    cur_dist, det_cfg, band_px=band)
                if ed is not None:
                    ell[fid] = ed.ellipse
            min_frames = int(cfg.get("line_extraction_circle_lens_min_frames", 4))
            _mm = []
            for fid, e in ell.items():
                rv, _ = cv2.Rodrigues(per_frame_R[fid])
                r = _ellipse_residuals_distorted(
                    e, per_frame_K[fid], rv.reshape(3), per_frame_t[fid], cur_dist)
                nz = r[r != 0]
                if nz.size:
                    _mm.append(float(np.median(np.abs(nz))))
            # STORED circle points (the sub-pixel ridge detections committed by
            # the cold-start / propagation solves) carry the wide-field lens
            # signal on exactly the frames whose freshly-detected ellipses the
            # band search misses (their cameras are lens-limited, so the
            # projected ring sits outside the band). They serve as BOTH a
            # second trigger signal and a refinement input — without them the
            # refinement starves on the >=2-straight-line gate and the wide
            # start stays lens-limited (origi01 30-160 px; origi02's k
            # breathing 0.08-0.24 run-to-run).
            from src.utils.anchor_solver import (
                _point_residuals_distorted as _cl_pres,
            )
            circ_pts: dict[int, list] = {}
            _cm = []
            for fid in covered_now:
                pts = [
                    LandmarkObservation(
                        name="centre_circle",
                        image_xy=tuple(ln["image_segment"][0]),
                        world_xyz=tuple(ln["world_segment"][0]))
                    for ln in detected_lines_by_frame.get(fid, [])
                    if "circle" in ln["name"]
                ]
                if len(pts) >= 8:
                    circ_pts[fid] = pts
                    rv, _ = cv2.Rodrigues(per_frame_R[fid])
                    pr = np.asarray(_cl_pres(
                        pts, per_frame_K[fid], rv.reshape(3),
                        per_frame_t[fid], cur_dist))
                    pn = np.linalg.norm(pr.reshape(-1, 2), axis=1)
                    _cm.append(float(np.median(pn)))
            med_circ = float(np.median(_cm)) if _cm else 0.0
            logger.info(
                "static line solve: centre-circle ellipse on %d frame(s) "
                "(median mis-fit %.1f px) + stored circle points on %d "
                "frame(s) (median mis-fit %.1f px); refine if > %.1f",
                len(ell), float(np.median(_mm)) if _mm else 0.0,
                len(circ_pts), med_circ,
                float(cfg.get("line_extraction_circle_lens_min_misfit", 5.0)))
            thr = float(cfg.get("line_extraction_circle_lens_min_misfit", 5.0))
            # The handful of banded ellipses can disagree with the broad
            # stored-point evidence; iterating on the ellipses alone then
            # WORSENS the wide field round over round (origi01: stored misfit
            # 9.6 -> 10.9 -> 12.0 while k chased 12 midfield rings). Stop
            # when the broad evidence degrades.
            if (_lens_prev_circ is not None and _cm
                    and med_circ > 1.1 * _lens_prev_circ):
                logger.info(
                    "static line solve: lens iteration stopped — stored "
                    "circle misfit worsened (%.1f -> %.1f px)",
                    _lens_prev_circ, med_circ)
                break
            _lens_prev_circ = med_circ if _cm else None
            ell_trigger = len(ell) >= min_frames and _mm and float(
                np.median(_mm)) > thr
            circ_trigger = len(circ_pts) >= min_frames and med_circ > thr
            if (len(ell) >= min_frames and _mm) or circ_trigger:
                med_mis = float(np.median(_mm)) if _mm else med_circ
                if not (ell_trigger or circ_trigger):
                    logger.info(
                        "static line solve: centre circle well-fit (misfit "
                        "%.1f px <= %.1f) on %d frame(s) -> lens refinement "
                        "skipped", med_mis, thr, len(ell))
                else:
                    pfl: dict[int, list] = {}
                    pfs: dict[int, tuple] = {}
                    for fid in covered_now:
                        lns = [
                            LineObservation(
                                name=ln["name"],
                                image_segment=(tuple(ln["image_segment"][0]),
                                               tuple(ln["image_segment"][1])),
                                world_segment=(tuple(ln["world_segment"][0]),
                                               tuple(ln["world_segment"][1])))
                            for ln in detected_lines_by_frame.get(fid, [])
                            if "circle" not in ln["name"]
                        ]
                        # A frame joins the lens solve if its straight lines
                        # alone determine it (>=2), or its stored circle
                        # points do (the solver consumes them as weighted
                        # point residuals; lines may even be empty).
                        if len(lns) >= 2 or (
                                fid in circ_pts
                                and (lns or len(circ_pts[fid]) >= 12)):
                            pfl[fid] = lns
                            rv, _ = cv2.Rodrigues(per_frame_R[fid])
                            pfs[fid] = (rv.reshape(3), float(per_frame_K[fid][0, 0]))
                    ell2 = {fid: e for fid, e in ell.items() if fid in pfl}
                    circ2 = {fid: pts for fid, pts in circ_pts.items()
                             if fid in pfl}
                    if pfl and (ell2 or circ2):
                        # Residual-budget balance: hundreds of circle frames x
                        # 20 points can outvote the straight lines wholesale
                        # (origi01: ~10k circle obs vs ~1.7k line obs bent k1
                        # to ~0 — the ring fit at 0.9 px while the start
                        # poses slid along the circle's degenerate direction
                        # and the user's clicks got WORSE). Cap the circle's
                        # total influence at half the lines'.
                        n_line_res = sum(len(v) for v in pfl.values()) * 2
                        n_circ_res = sum(len(v) for v in circ2.values()) * 2
                        circ_w = min(0.3, 0.5 * n_line_res
                                     / max(1, n_circ_res))
                        # Anchor keypoints (PnLCalib/manual) are the only
                        # evidence NOT strip-searched around the current
                        # cameras' projections, i.e. the only evidence that
                        # cannot self-confirm a mis-identified (C, lens):
                        # origi01's stored detections prefer the wrong C by
                        # construction (med 2.98 px there vs 5.16 at the C
                        # that fits the user's clicks). Feed them to the
                        # refinement so C/lens settle on unbiased points
                        # spanning both pitch ends.
                        hints_l = {
                            fid: list(anchor_landmarks[fid])
                            for fid in pfl if anchor_landmarks.get(fid)
                        }
                        sol_l = solve_static_camera_from_lines(
                            pfl, anchors.image_size, c_seed=C,
                            lens_seed=(cx_l, cy_l, cur_dist[0], cur_dist[1]),
                            per_frame_seeds=pfs, per_frame_ellipses=ell2,
                            circle_points=circ2, circle_weight=circ_w,
                            point_hints=hints_l,
                            point_hint_weight=float(cfg.get(
                                "line_extraction_lens_point_hint_weight",
                                0.5)),
                            lens_model=lens_model,
                            ellipse_weight=float(
                                cfg.get("line_extraction_circle_lens_weight", 1.0)),
                            c_bound_m=float(cfg.get(
                                "line_extraction_c_trust_m", 1.5)) / 2.0,
                        )
                        C = np.asarray(sol_l.camera_centre, dtype=np.float64)
                        cxn, cyn = sol_l.principal_point
                        for fid, (K2, R2, t2) in sol_l.per_frame_KRt.items():
                            per_frame_K[fid] = K2
                            per_frame_R[fid] = R2
                            per_frame_t[fid] = t2
                        # Frames the lens refinement didn't solve carry
                        # rotations from the OLD geometry — patching pp/C
                        # under them shifts their projections (the origi01
                        # midfield bug in miniature). Re-solve each against
                        # its own stored detections (lines + circle points)
                        # under the NEW lens; frames with nothing to re-solve
                        # against (or that fail the gate) are invalidated and
                        # SLERP-filled between refined neighbours below.
                        from src.utils.static_c_profile import (
                            _solve_frame_at_fixed_c as _lr_solve,
                        )
                        from src.utils.static_line_solver import _dist5 as _lr_d5
                        lr_d5 = _lr_d5(sol_l.distortion)
                        lr_gate = float(cfg.get(
                            "line_extraction_propagate_circle_max_rms", 12.0))
                        n_lr_resolved = 0
                        for fid in covered_now:
                            if fid in sol_l.per_frame_KRt:
                                continue
                            entries = detected_lines_by_frame.get(fid, [])
                            lns = [
                                LineObservation(
                                    name=ln["name"],
                                    image_segment=(
                                        tuple(ln["image_segment"][0]),
                                        tuple(ln["image_segment"][1])),
                                    world_segment=(
                                        tuple(ln["world_segment"][0]),
                                        tuple(ln["world_segment"][1])))
                                for ln in entries
                                if "circle" not in ln["name"]
                            ]
                            circ_obs = [
                                LandmarkObservation(
                                    name="centre_circle",
                                    image_xy=tuple(ln["image_segment"][0]),
                                    world_xyz=tuple(ln["world_segment"][0]))
                                for ln in entries
                                if "circle" in ln["name"]
                            ] or None
                            pt_weight = 0.3
                            if not lns and not circ_obs and (
                                    fid in anchor_resolved_frames):
                                # Point-re-solved anchor frames: re-fit against
                                # their landmark points under the new lens.
                                circ_obs = anchor_landmarks.get(fid)
                                pt_weight = 1.0
                            solved = False
                            if lns or circ_obs:
                                rv, _ = cv2.Rodrigues(per_frame_R[fid])
                                fx0 = float(per_frame_K[fid][0, 0])
                                rvec, fx, rms = _lr_solve(
                                    lns, cxn, cyn, lr_d5, C, rv.reshape(3),
                                    fx0, fx_rel=0.05 if len(lns) < 4 else None,
                                    circle_obs=circ_obs,
                                    circle_weight=pt_weight)
                                if np.isfinite(rms) and rms <= lr_gate:
                                    R2, _ = cv2.Rodrigues(rvec)
                                    per_frame_K[fid] = np.array(
                                        [[fx, 0.0, cxn], [0.0, fx, cyn],
                                         [0.0, 0.0, 1.0]])
                                    per_frame_R[fid] = R2
                                    per_frame_t[fid] = -R2 @ C
                                    solved = True
                                    n_lr_resolved += 1
                            if not solved:
                                per_frame_K[fid] = None
                                per_frame_R[fid] = None
                                per_frame_t[fid] = None
                                per_frame_conf[fid] = 0.0
                        n_refilled = _fill_interior_gaps()
                        sol = dataclasses.replace(
                            sol, camera_centre=C,
                            principal_point=sol_l.principal_point,
                            distortion=sol_l.distortion)
                        logger.info(
                            "static line solve: centre-circle lens refinement on "
                            "%d ellipse frame(s) (misfit %.1f px): distortion %s "
                            "-> %s; re-solved %d unrefined frame(s), SLERP-"
                            "refilled %d", len(ell2), med_mis,
                            np.round(cur_dist, 3).tolist(),
                            np.round(sol_l.distortion[:2], 3).tolist(),
                            n_lr_resolved, n_refilled)
            # Converged (or nothing refined this round): |dk1| spans both the
            # "skip paths left sol untouched" case (delta exactly 0) and true
            # convergence.
            if abs(float(sol.distortion[0]) - float(cur_dist[0])) < 0.005:
                break

        _anchor_click_checkpoint("post-pass2+board")

        # GLOBAL POLISH — Gauss-Seidel sweeps where every covered frame
        # re-solves (rvec, fx) at the locked C/lens against its OWN stored
        # constraints (straight lines + board + circle points) PLUS soft
        # continuity priors toward its neighbours. Data and smoothness are
        # optimised JOINTLY, replacing the per-frame-greedy -> mass-outlier-
        # rejection -> interpolation dance that left whole spans as
        # constraint-free interp with collapsed fx (the audited origi01
        # start/gap failure: 188-212 frames rejected, fx swinging +-32%).
        # Sparse frames bend toward continuity unless their evidence
        # disagrees; well-lined frames barely feel the prior. Gated to clips
        # that actually have sparse spans — fully line-solved clips (gberch)
        # skip untouched.
        gp_ran = False
        gp_touched: set[int] = set()
        if bool(cfg.get("line_extraction_global_polish", True)):
            from scipy.spatial.transform import Rotation as _GPRot
            from scipy.spatial.transform import Slerp as _GPSlerp

            from src.schemas.anchor import (
                LandmarkObservation as _GPLm,
            )
            from src.schemas.anchor import (
                LineObservation as _GPLn,
            )
            from src.utils.static_c_profile import (
                _solve_frame_at_fixed_c as _gp_solve,
            )
            from src.utils.static_line_solver import _dist5 as _gp_d5
            gp_covered = [i for i in range(len(per_frame_K))
                          if per_frame_K[i] is not None]
            gp_lines: dict[int, list] = {}
            gp_circ: dict[int, list] = {}
            n_sparse = 0
            for f in gp_covered:
                entries = detected_lines_by_frame.get(f) or []
                lns = [
                    _GPLn(name=ln["name"],
                          image_segment=(tuple(ln["image_segment"][0]),
                                         tuple(ln["image_segment"][1])),
                          world_segment=(tuple(ln["world_segment"][0]),
                                         tuple(ln["world_segment"][1])))
                    for ln in entries if "circle" not in ln["name"]
                ]
                pts = [
                    _GPLm(name="centre_circle",
                          image_xy=tuple(ln["image_segment"][0]),
                          world_xyz=tuple(ln["world_segment"][0]))
                    for ln in entries if "circle" in ln["name"]
                ]
                if lns:
                    gp_lines[f] = lns
                if len(pts) >= 6:
                    gp_circ[f] = pts
                if len(lns) < int(cfg.get(
                        "line_extraction_min_lines_per_frame", 4)):
                    n_sparse += 1
            gp_frac_sparse = n_sparse / max(1, len(gp_covered))
            if gp_frac_sparse >= float(cfg.get(
                    "line_extraction_global_polish_min_sparse", 0.05)):
                gp_d5v = _gp_d5(sol.distortion)
                gp_cx, gp_cy = sol.principal_point
                w_pose = float(cfg.get(
                    "line_extraction_global_polish_pose_weight", 100.0))
                w_fx = float(cfg.get(
                    "line_extraction_global_polish_fx_weight", 0.05))
                max_sweeps = int(cfg.get(
                    "line_extraction_global_polish_sweeps", 5))
                n_polished = 0
                for sweep in range(max_sweeps):
                    order = (gp_covered if sweep % 2 == 0
                             else gp_covered[::-1])
                    max_delta = 0.0
                    for f in order:
                        nbs = [g for g in (f - 1, f + 1)
                               if 0 <= g < len(per_frame_K)
                               and per_frame_R[g] is not None]
                        if not nbs:
                            continue
                        if len(nbs) == 2:
                            prior_R = _GPSlerp(
                                [0.0, 1.0], _GPRot.from_matrix(
                                    [per_frame_R[nbs[0]],
                                     per_frame_R[nbs[1]]]))([0.5]).as_matrix()[0]
                            prior_fx = 0.5 * (
                                float(per_frame_K[nbs[0]][0, 0])
                                + float(per_frame_K[nbs[1]][0, 0]))
                        else:
                            prior_R = np.asarray(per_frame_R[nbs[0]])
                            prior_fx = float(per_frame_K[nbs[0]][0, 0])
                        lns = gp_lines.get(f, [])
                        pts = gp_circ.get(f)
                        n_str = sum(
                            1 for ln in lns if ln.name != "board_line")
                        if n_str >= 3:
                            # Frames with a determined line solve were never
                            # the failure mode (the audit's passing frames are
                            # exactly these) — polishing them traded verified
                            # accuracy for continuity on origi02's start.
                            # (>=3: a 2-line frame can still be a degenerate
                            # parallel pair.)
                            continue
                        if f in anchor_resolved_frames:
                            # Demotion ISLANDS were point-solved against
                            # their landmarks at the locked geometry — the
                            # best estimate a sparse anchor frame can have;
                            # any blend moves f134-class frames 24 -> ~90 px
                            # off their clicks. Ordinarily-covered anchor
                            # frames (origi02's) keep the blend below.
                            continue
                        anch_obs = anchor_landmarks.get(f) or None
                        if not lns and not pts and not anch_obs:
                            # constraint-free: pure chain relaxation
                            R_new = prior_R
                            fx_new = prior_fx
                        else:
                            # sparse frames bend toward continuity
                            wp = w_pose
                            wf = w_fx
                            rv_pr, _ = cv2.Rodrigues(prior_R)
                            rv_cur, _ = cv2.Rodrigues(
                                np.asarray(per_frame_R[f]))
                            rvec_n, fx_n, rms_n = _gp_solve(
                                lns, gp_cx, gp_cy, gp_d5v, C,
                                rv_cur.reshape(3),
                                float(per_frame_K[f][0, 0]),
                                circle_obs=pts,
                                anchor_obs=anch_obs,
                                anchor_weight=3.0,
                                pose_prior=(rv_pr.reshape(3), wp),
                                fx_prior=(prior_fx, wf))
                            if not np.isfinite(rms_n):
                                continue
                            R_new, _ = cv2.Rodrigues(rvec_n)
                            fx_new = float(fx_n)
                        d = _angle_between(
                            np.asarray(per_frame_R[f]), np.asarray(R_new))
                        max_delta = max(max_delta, d)
                        per_frame_K[f] = np.array(
                            [[fx_new, 0.0, gp_cx], [0.0, fx_new, gp_cy],
                             [0.0, 0.0, 1.0]])
                        per_frame_R[f] = np.asarray(R_new)
                        per_frame_t[f] = -np.asarray(R_new) @ C
                        gp_touched.add(f)
                        n_polished += 1
                    if max_delta < 0.05:
                        break
                gp_ran = True
                logger.info(
                    "static line solve: global polish — %d sweep(s) over %d "
                    "frame(s) (%.0f%% sparse), final max step %.2f deg",
                    sweep + 1, len(gp_covered), 100 * gp_frac_sparse,
                    max_delta)

        _anchor_click_checkpoint("post-polish+lens")

        # Outlier rejection: replace bad single-frame solves with a SLERP/LERP
        # interpolation from their nearest good neighbours, BEFORE smoothing —
        # so the smoother never spreads a spike across its window (a single 60deg
        # solve spike was being smeared into ~18 mediocre frames). Two kinds:
        #   * rotation-jump — rotation deviates from the SLERP of its neighbours
        #     by > rot_deg (a solve spike that still fits its few lines, so
        #     line-RMS misses it — this was the real kroupi/origi culprit).
        #   * line-RMS — reprojection RMS >> the clip median (wrong-line lock).
        # gberch has neither (max jump 0.33deg, max RMS ~5px) -> no-op there.
        if bool(cfg.get("line_extraction_outlier_rejection", True)):
            from scipy.spatial.transform import Rotation, Slerp

            from src.utils.camera_projection import project_world_to_image
            _odist = tuple(float(x) for x in sol.distortion[:2])

            def _geo(A: np.ndarray, B: np.ndarray) -> float:
                c = (np.trace(np.asarray(A).T @ np.asarray(B)) - 1.0) / 2.0
                return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))

            def _slerp(a: int, b: int, w: float) -> np.ndarray:
                return Slerp([0.0, 1.0], Rotation.from_matrix(
                    [per_frame_R[a], per_frame_R[b]]))([w]).as_matrix()[0]

            def _frame_rms(fid: int) -> float | None:
                lines = detected_lines_by_frame.get(fid)
                if not lines:
                    return None
                K = per_frame_K[fid]; R = per_frame_R[fid]; t = per_frame_t[fid]
                rs: list[float] = []
                for ln in lines:
                    proj = project_world_to_image(
                        K, R, t, _odist, np.array(ln["world_segment"]))
                    pa, pb = proj[0], proj[1]
                    d = pb - pa
                    nrm = np.array([-d[1], d[0]])
                    if np.linalg.norm(nrm) < 1e-6:
                        continue
                    nrm = nrm / np.linalg.norm(nrm)
                    for ip in ln["image_segment"]:
                        rs.append(abs(float(np.dot(np.array(ip) - pa, nrm))))
                if not rs:
                    return None
                # Circle-bearing frames: fat-tail detector outliers dominate
                # the raw rms of an honest lock (the recurring lesson) — use
                # the median so lens-limited circle solves aren't mass-
                # rejected and stripped of their constraints (208 frames on
                # origi01 fell to this even with the rot criterion off).
                if any("circle" in ln["name"] for ln in lines):
                    return float(np.median(np.abs(rs)))
                return float(np.sqrt(np.mean(np.square(rs))))

            from src.utils.static_c_profile import _solve_frame_at_fixed_c
            from src.utils.static_line_solver import _dist5 as _dist5_fn
            cx_o, cy_o = sol.principal_point
            _dist5_o = _dist5_fn(sol.distortion)

            def _resolve(i: int, R_seed: np.ndarray, fx_seed: float):
                """Re-detect + line-solve a rejected frame from a CLEAN seed.
                Returns (R, fx, lines) when it fits and stays near the seed, else
                None — so a frame rejected for a transient bad solve recovers its
                line-accurate camera instead of a flickery pure interpolation."""
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, img = cap.read()
                if not ok:
                    return None
                seed_K = np.array(
                    [[fx_seed, 0.0, cx_o], [0.0, fx_seed, cy_o], [0.0, 0.0, 1.0]])
                det = detect_lines_for_frames(
                    {i: img}, {i: {"K": seed_K, "R": R_seed, "t": -R_seed @ C}},
                    _odist, det_cfg, min_confidence=det_min_confidence,
                    min_n_samples=det_min_n_samples, min_lines=1)
                lines = det.get(i, [])
                if len(lines) < min_lines:
                    return None
                rv, _ = cv2.Rodrigues(R_seed)
                rvec, fx, rms = _solve_frame_at_fixed_c(
                    lines, cx_o, cy_o, _dist5_o, C, rv.reshape(3), fx_seed,
                    fx_rel=0.05 if len(lines) < 4 else None)
                if not np.isfinite(rms) or rms > 4.0:
                    return None
                R_e, _ = cv2.Rodrigues(rvec)
                # only accept if it stays near the clean interp seed (a far jump
                # would be a wrong-line lock re-admitting the rejected error)
                if _geo(R_e, R_seed) > rot_thr:
                    return None
                return R_e, float(fx), lines

            # Iterate: a multi-frame seam block masks itself under immediate-
            # neighbour comparison (both neighbours are in the block). Each pass
            # bridges the most-deviant frame; once replaced it reads as good, so
            # the next pass exposes the rest of the block. Converges in a few
            # passes. The residual-vs-SLERP metric is fast-pan-safe (a linear
            # pan has ~0 residual), so the low threshold never eats real motion.
            rot_thr = float(cfg.get("line_extraction_outlier_rot_deg", 2.0))
            # After the global polish, continuity was already optimised
            # JOINTLY with each polished frame's constraints — a residual
            # deviation from the neighbour SLERP there is evidence overruling
            # the prior, not a defect, and the rot-jump criterion degenerates
            # into mass rejection (188-212 frames on origi01) + constraint
            # DELETION + blind interp. But frames the polish PINNED (>=3
            # lines) were never balanced against continuity — rot-jump keeps
            # its protective job for their wrong-line spikes (disabling it
            # globally let a 7 deg spike survive on origi02). Exempt only
            # the polished frames.
            rel = float(cfg.get("line_extraction_outlier_rel", 3.0))
            abs_rms = float(cfg.get("line_extraction_outlier_max_rms", 8.0))
            max_passes = int(cfg.get("line_extraction_outlier_passes", 6))
            total_rejected, last_pass = 0, 0
            for _pass in range(max_passes):
                sset = [
                    i for i in range(len(per_frame_R)) if per_frame_R[i] is not None
                ]
                if len(sset) < 3:
                    break
                rot_res = {}
                for k in range(1, len(sset) - 1):
                    i, a, b = sset[k], sset[k - 1], sset[k + 1]
                    if gp_ran and i in gp_touched:
                        continue
                    w = (i - a) / (b - a) if b > a else 0.5
                    rot_res[i] = _geo(per_frame_R[i], _slerp(a, b, w))
                rms_map = {i: _frame_rms(i) for i in sset}
                rms_vals = np.array([v for v in rms_map.values() if v is not None])
                rms_med = float(np.median(rms_vals)) if rms_vals.size else 0.0
                rms_thr = max(abs_rms, rel * rms_med)
                bad = sorted(
                    {i for i, r in rot_res.items() if r > rot_thr}
                    | {i for i, v in rms_map.items() if v is not None and v > rms_thr}
                )
                good = np.array([i for i in sset if i not in set(bad)])
                if not bad or good.size < 2:
                    break
                for i in bad:
                    lo = good[good < i]; hi = good[good > i]
                    if lo.size and hi.size:
                        a, b = int(lo[-1]), int(hi[0])
                        w = (i - a) / (b - a)
                        R = _slerp(a, b, w)
                        fx = (1 - w) * per_frame_K[a][0, 0] + w * per_frame_K[b][0, 0]
                    else:
                        j = int(lo[-1]) if lo.size else int(hi[0])
                        R = per_frame_R[j]; fx = float(per_frame_K[j][0, 0])
                    # try to recover a line-accurate camera from the clean seed
                    rs = _resolve(i, R, float(fx))
                    used_lines = None
                    if rs is not None:
                        R, fx, used_lines = rs
                    K = per_frame_K[i].copy()
                    K[0, 0] = fx; K[1, 1] = fx
                    per_frame_K[i] = K
                    per_frame_R[i] = R
                    per_frame_t[i] = -R @ C
                    per_frame_conf[i] = 0.6 if used_lines else 0.4
                    if used_lines:
                        detected_lines_by_frame[i] = [
                            {
                                "name": ln.name,
                                "image_segment": [list(ln.image_segment[0]),
                                                  list(ln.image_segment[1])],
                                "world_segment": [list(ln.world_segment[0]),
                                                  list(ln.world_segment[1])],
                            }
                            for ln in used_lines
                        ]
                    else:
                        detected_lines_by_frame.pop(i, None)
                total_rejected += len(bad)
                last_pass = _pass + 1
            if total_rejected:
                logger.info(
                    "static line solve: outlier-rejected %d frame(s) over %d "
                    "pass(es) (rot > %.1f deg or line-RMS > clip-median x %.1f), "
                    "neighbour interp", total_rejected, last_pass, rot_thr, rel,
                )

        _anchor_click_checkpoint("post-outlier")

        # Pin-and-smooth temporal smoothing of the per-frame rotation. The
        # dominant jitter is the seam step where a line-solved frame meets an
        # interpolated gap frame (the SLERP fill is velocity-discontinuous). A
        # uniform smooth removes it but drags the *correct* line-solved frames
        # off their painted lines. Instead we PIN the line-solved frames and let
        # only the interpolated frames relax onto a smooth path between them — so
        # seam jitter drops at zero cost to the solved frames, and an already-
        # solved/smooth clip (gberch) is untouched. Re-derives t = -R @ C so the
        # camera body stays fixed. window < 3 disables.
        smooth_window = int(cfg.get("line_extraction_smooth_window", 9))
        smooth_iters = int(cfg.get("line_extraction_smooth_iters", 4))
        ordered = [i for i in range(len(per_frame_R)) if per_frame_R[i] is not None]
        if smooth_window >= 3 and len(ordered) >= smooth_window:
            from src.utils.temporal_smoothing import pin_and_smooth_quat
            Rs = np.stack([per_frame_R[i] for i in ordered])

            # "solved" = trustworthy enough to pin: >=2 detected straight
            # lines, a detected centre circle (circle-aided solves pass the
            # same rms gate), or a point-re-solved anchor frame. Without
            # pinning, the smoother would drag the only real solves in a
            # line-sparse span toward interpolation.
            def _pinned(i: int) -> bool:
                if i in anchor_resolved_frames:
                    return True
                entries = detected_lines_by_frame.get(i, [])
                n_straight = sum(
                    1 for ln in entries if "circle" not in ln["name"])
                has_circle = any("circle" in ln["name"] for ln in entries)
                return n_straight >= 2 or has_circle

            solved_mask = [_pinned(i) for i in ordered]
            Rs_s = pin_and_smooth_quat(
                Rs, solved_mask, window=smooth_window, iters=smooth_iters)
            n_moved = 0
            for j, i in enumerate(ordered):
                if not np.allclose(Rs_s[j], per_frame_R[i], atol=1e-9):
                    n_moved += 1
                per_frame_R[i] = Rs_s[j]
                per_frame_t[i] = -Rs_s[j] @ C
            logger.info(
                "static line solve: pin-and-smoothed %d/%d interp frame(s) "
                "(window=%d, %d solved pinned)", n_moved, len(ordered),
                smooth_window, sum(solved_mask))

        _anchor_click_checkpoint("final")

        rms_arr = np.array(
            [v for v in sol.per_frame_line_rms.values() if np.isfinite(v)]
        )
        if rms_arr.size:
            logger.info(
                "static line solve: locked C=%s across %d frames — line RMS "
                "mean=%.3f median=%.3f max=%.3f frac<1px=%.2f",
                np.round(C, 3).tolist(), len(sol.per_frame_KRt),
                float(rms_arr.mean()), float(np.median(rms_arr)),
                float(rms_arr.max()), float((rms_arr < 1.0).mean()),
            )
        return sol

    def _pnlcalib_bootstrap_cameras(
        self, frames_bgr: dict[int, np.ndarray], cfg: dict,
    ) -> dict[int, dict]:
        """Per-frame PnLCalib cameras (our frame) for detection bootstrap.

        Used only as the clip-adaptive fallback in the static-line solve when
        the anchor-interpolated bootstrap has poor detection coverage. PnLCalib
        projects the catalogue accurately per frame even where anchor
        interpolation is off, so it makes a strong detection bootstrap. Returns
        ``{frame: {"K","R","t"}}`` for frames that calibrate to a physically
        plausible camera position (implausible / off-pitch solves are dropped,
        so they don't poison the C seed). Empty dict if PnLCalib is unavailable.
        """
        try:
            from src.utils.auto_anchor import is_plausible_position
            from src.utils.neural_calibrator import PnLCalibrator
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("PnLCalib bootstrap unavailable (%s)", exc)
            return {}

        aa = cfg.get("auto_anchors", {})
        mc = aa.get("model", {})
        bounds = aa.get("plausibility_bounds", {
            "x": (-30.0, 135.0), "y": (-60.0, 130.0), "z": (3.0, 80.0),
        })
        try:
            cal = PnLCalibrator(
                device=mc.get("device", "auto"),
                kp_threshold=float(mc.get("kp_threshold", 0.3434)),
                line_threshold=float(mc.get("line_threshold", 0.7867)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PnLCalib bootstrap init failed (%s)", exc)
            return {}

        out: dict[int, dict] = {}
        for fid, img in frames_bgr.items():
            try:
                r = cal.calibrate(img)
            except Exception:  # noqa: BLE001 - PnLCalib optimiser can raise
                continue
            if r is None:
                continue
            if not is_plausible_position(np.asarray(r.world_position), bounds):
                continue
            R, _ = cv2.Rodrigues(np.asarray(r.rvec).reshape(3))
            out[fid] = {
                "K": np.asarray(r.K),
                "R": R,
                "t": np.asarray(r.tvec).reshape(3),
            }
        logger.info(
            "PnLCalib per-frame bootstrap: %d/%d covered frames calibrated "
            "plausibly", len(out), len(frames_bgr),
        )
        return out

    def _propagate_pair(
        self,
        cap: cv2.VideoCapture,
        a: int,
        b: int,
        anchor_solutions: dict[int, tuple[np.ndarray, np.ndarray]],
        per_frame_K: list,
        per_frame_R: list,
        per_frame_conf: list,
        is_anchor: list,
        cfg: dict,
    ) -> None:
        max_features = int(cfg.get("max_features_per_frame", 1000))
        inlier_min = float(cfg.get("ransac_inlier_min_ratio", 0.4))

        # Read frames a..b inclusive into memory (small per-anchor span).
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        frames = []
        for _ in range(b - a + 1):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        if len(frames) < 2:
            return

        # Forward propagation
        Ks_fwd = [anchor_solutions[a][0]]
        Rs_fwd = [anchor_solutions[a][1]]
        inlier_ratios: list[float] = [1.0]
        for i in range(1, len(frames)):
            res = propagate_one_frame(
                frames[i - 1], frames[i], Ks_fwd[-1], Rs_fwd[-1],
                max_features=max_features, ransac_inlier_min_ratio=inlier_min,
            )
            if res is None:
                Ks_fwd.append(Ks_fwd[-1])
                Rs_fwd.append(Rs_fwd[-1])
                inlier_ratios.append(0.0)
            else:
                Ks_fwd.append(res.K)
                Rs_fwd.append(res.R)
                inlier_ratios.append(res.inlier_ratio)

        # Backward propagation
        Ks_bwd = [anchor_solutions[b][0]]
        Rs_bwd = [anchor_solutions[b][1]]
        for i in range(len(frames) - 2, -1, -1):
            res = propagate_one_frame(
                frames[i + 1], frames[i], Ks_bwd[0], Rs_bwd[0],
                max_features=max_features, ransac_inlier_min_ratio=inlier_min,
            )
            if res is None:
                Ks_bwd.insert(0, Ks_bwd[0])
                Rs_bwd.insert(0, Rs_bwd[0])
            else:
                Ks_bwd.insert(0, res.K)
                Rs_bwd.insert(0, res.R)

        # Bidirectional smooth
        Ks_s, Rs_s = smooth_between_anchors(Ks_fwd, Rs_fwd, Ks_bwd, Rs_bwd)

        for offset, (K, R) in enumerate(zip(Ks_s, Rs_s)):
            global_idx = a + offset
            # Anchor frames keep their exact solver-derived K, R.
            if is_anchor[global_idx]:
                continue
            disagreement = _angle_between(Rs_fwd[offset], Rs_bwd[offset])
            signals = FrameSignals(
                inlier_ratio=inlier_ratios[offset],
                fwd_bwd_disagreement_deg=disagreement,
                pitch_line_residual_px=None,
            )
            per_frame_K[global_idx] = K
            per_frame_R[global_idx] = R
            per_frame_conf[global_idx] = confidence_from_signals(signals)
        # Endpoints stay exact (already set as anchors).

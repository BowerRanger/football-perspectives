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

logger = logging.getLogger(__name__)


def _angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    cos_t = (np.trace(R1.T @ R2) - 1) / 2
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(np.degrees(np.arccos(cos_t)))


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
            if not anchors_path.exists():
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

    def _run_shot(
        self,
        shot_id: str,
        anchors_path: Path,
        clip_path: Path,
        cfg: dict,
    ) -> None:
        """Single-shot camera solve. The body is the original run() logic
        with file paths parameterised on shot_id."""
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
        static_line_centre: np.ndarray | None = None
        if bool(cfg.get("line_extraction", False)):
            if static_camera:
                static_line_centre = self._refine_with_static_line_solve(
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

        track = CameraTrack(
            clip_id=anchors.clip_id,
            fps=float(fps),
            image_size=(w, h),
            t_world=list(t_world_median),
            frames=tuple(frames_out),
            principal_point=(float(principal_point[0]), float(principal_point[1])),
            camera_centre=(
                tuple(float(x) for x in static_line_centre)
                if static_line_centre is not None
                else (
                    tuple(float(x) for x in sol.camera_centre)
                    if sol.camera_centre is not None
                    else None
                )
            ),
            distortion=tuple(float(x) for x in sol.distortion),
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
    ) -> np.ndarray | None:
        """Static-camera line solve: detect painted lines on every
        propagated frame, profile the camera centre, bundle-adjust one
        shared centre, then iteratively re-detect under the coherent
        cameras. Writes per-frame ``(K, R, t)`` back in place and returns
        the single locked camera centre (or ``None`` if it bailed and
        left the propagated cameras untouched).
        """
        from src.utils.anchor_solver import _is_rich
        from src.utils.line_detector import DetectorConfig
        from src.utils.line_camera_refine import detect_lines_for_frames
        from src.utils.static_c_profile import make_c_grid, profile_camera_centre
        from src.utils.static_line_solver import solve_static_camera_from_lines

        det_cfg = DetectorConfig(
            search_strip_px=int(cfg.get("line_extraction_strip_px", 25)),
            min_gradient=float(cfg.get("line_extraction_min_gradient", 10.0)),
        )
        lens_model = str(cfg.get("line_extraction_lens_model", "pinhole_k1k2"))
        n_rounds = int(cfg.get("line_extraction_static_rounds", 3))
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

        # Step 0 — detect under the propagated bootstrap cameras.
        per_frame_lines = detect_lines_for_frames(
            frames_bgr, _cameras_from_arrays(), dist2, det_cfg,
        )
        if len(per_frame_lines) < 2:
            logger.warning(
                "static line solve: only %d frame(s) yielded detected lines; "
                "keeping the propagated cameras unchanged",
                len(per_frame_lines),
            )
            return None

        # Per-frame (rvec, fx) bootstrap seeds from the propagated cameras.
        bootstrap: dict[int, tuple[np.ndarray, float]] = {}
        for fid in per_frame_lines:
            rv, _ = cv2.Rodrigues(per_frame_R[fid])
            bootstrap[fid] = (rv.reshape(3), float(per_frame_K[fid][0, 0]))

        # Seed C from the propagated centres (rich-anchor frames preferred).
        rich = {a.frame for a in anchors.anchors if _is_rich(a)}
        seed_cs = [
            -per_frame_R[f].T @ per_frame_t[f]
            for f in per_frame_lines if f in rich
        ] or [
            -per_frame_R[f].T @ per_frame_t[f] for f in per_frame_lines
        ]
        c_center = np.median(np.stack(seed_cs), axis=0)
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
        coarse = profile_camera_centre(
            per_frame_lines, anchors.image_size,
            c_grid=make_c_grid(c_center, extent_m=7.5, n_steps=5),
            lens_seed=lens_seed, per_frame_bootstrap=bootstrap,
        )
        fine = profile_camera_centre(
            per_frame_lines, anchors.image_size,
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
        seeds = fine.per_frame_seeds
        sol = None
        for round_idx in range(max(1, n_rounds)):
            sol = solve_static_camera_from_lines(
                per_frame_lines, anchors.image_size,
                c_seed=c_seed, lens_seed=lens_seed,
                per_frame_seeds=seeds, point_hints=anchor_landmarks,
                lens_model=lens_model, point_hint_weight=point_hint_weight,
            )
            if round_idx < n_rounds - 1:
                cams = {
                    fid: {"K": K, "R": R, "t": t}
                    for fid, (K, R, t) in sol.per_frame_KRt.items()
                }
                redet = detect_lines_for_frames(
                    frames_bgr, cams, tuple(sol.distortion[:2]), det_cfg,
                )
                if len(redet) >= 2:
                    per_frame_lines = redet
                c_seed = sol.camera_centre
                seeds = {
                    fid: (cv2.Rodrigues(R)[0].reshape(3), float(K[0, 0]))
                    for fid, (K, R, _t) in sol.per_frame_KRt.items()
                }

        assert sol is not None
        C = sol.camera_centre

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

        # One-C consistency: frames the solve skipped still share C.
        for i in covered:
            if i not in sol.per_frame_KRt and per_frame_R[i] is not None:
                per_frame_t[i] = -per_frame_R[i] @ C

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
        return C

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

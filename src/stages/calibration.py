"""Stage 2: Camera calibration via PnLCalib neural inference.

Broadcast cameras are static within a shot — they sit on a fixed tripod and
only pan, tilt, and zoom.  This stage:

1. Samples keyframes evenly through each shot's clip.
2. Runs PnLCalib on each keyframe to recover ``(K, R, t)``.
3. Computes a robust median camera position across keyframes.
4. Re-anchors each keyframe's translation to the shared median position,
   keeping the per-frame rotation and focal length (which reflect panning
   and zoom within the shot).
5. Writes ``CalibrationResult``.

If PnLCalib fails on every keyframe of a shot (e.g., extreme behind-goal
replay angles), the shot is given an empty calibration and a warning is
logged.  Downstream stages handle empty calibrations gracefully.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CalibrationResult, CameraFrame
from src.schemas.shots import ShotsManifest
from src.utils.calibration_debug import render_shot_overlays
from src.utils.calibration_refine import refine_shot_calibration
from src.utils.camera import camera_world_position
from src.utils.manual_calibration import solve_from_annotations
from src.utils.neural_calibrator import NeuralCalibration, PnLCalibrator

logger = logging.getLogger(__name__)


# Physical plausibility bounds for a broadcast camera, used to filter noisy
# PnLCalib outputs before fusing.  PnLCalib occasionally produces numerically
# valid but physically impossible results (camera underground, fx of 5 or
# 51943, etc.); those samples must not feed into the median.
_DEFAULT_BOUNDS = {
    "x": (-30.0, 135.0),   # 30 m beyond either touchline
    "y": (-60.0, 130.0),   # 60 m behind the near / far touchlines
    "z": (3.0, 80.0),      # typical broadcast elevation range
    "fx": (500.0, 15000.0),
}


def _is_plausible(
    calibration: NeuralCalibration,
    bounds: dict[str, tuple[float, float]],
) -> bool:
    """Return True if a :class:`NeuralCalibration` looks physically plausible.

    Checks camera world position, focal length and optical-axis direction
    against configurable bounds.
    """
    pos = calibration.world_position
    x_lo, x_hi = bounds["x"]
    y_lo, y_hi = bounds["y"]
    z_lo, z_hi = bounds["z"]
    fx_lo, fx_hi = bounds["fx"]

    if not (x_lo <= pos[0] <= x_hi):
        return False
    if not (y_lo <= pos[1] <= y_hi):
        return False
    if not (z_lo <= pos[2] <= z_hi):
        return False

    fx = float(calibration.K[0, 0])
    if not (fx_lo <= fx <= fx_hi):
        return False

    # Optical axis must point downward in world z (camera looks at pitch).
    R, _ = cv2.Rodrigues(np.asarray(calibration.rvec, dtype=np.float64))
    if R[:, 2][2] > 0:
        return False

    return True


def _median_absolute_deviation(arr: np.ndarray) -> np.ndarray:
    """Column-wise median absolute deviation."""
    med = np.median(arr, axis=0)
    return np.median(np.abs(arr - med), axis=0)


def _robust_median_position(
    positions: list[np.ndarray],
) -> np.ndarray | None:
    """Median camera world position across keyframes, filtering MAD outliers.

    Returns ``None`` if the input list is empty.  For ≤ 2 samples the plain
    median is returned.  Otherwise samples more than 3 MADs from the median
    on any axis are dropped before taking a second median.
    """
    if not positions:
        return None
    arr = np.asarray(positions, dtype=np.float64)
    if len(arr) <= 2:
        return np.median(arr, axis=0)

    med = np.median(arr, axis=0)
    mad = _median_absolute_deviation(arr)
    # A MAD of zero means all samples agree on that axis; clip to a tiny
    # positive value so the division below keeps those samples.
    mad_clipped = np.where(mad > 1e-6, mad, 1e-6)
    deviation = np.abs(arr - med) / mad_clipped
    mask = np.all(deviation < 3.0, axis=1)
    if not np.any(mask):
        return med
    return np.median(arr[mask], axis=0)


def _neural_to_cam_frame(
    calibration: NeuralCalibration,
    frame_idx: int,
    override_position: np.ndarray | None = None,
) -> CameraFrame:
    """Convert a :class:`NeuralCalibration` to a :class:`CameraFrame`.

    When ``override_position`` is provided, the translation is recomputed so
    the camera sits exactly at that world position while keeping the
    per-frame rotation and intrinsics from PnLCalib.  This is how the static
    camera fuser enforces a shared position across keyframes.
    """
    K = np.asarray(calibration.K, dtype=np.float64)
    rvec = np.asarray(calibration.rvec, dtype=np.float64).reshape(3)
    if override_position is not None:
        R, _ = cv2.Rodrigues(rvec)
        position = np.asarray(override_position, dtype=np.float64).reshape(3)
        tvec = (-R @ position).astype(np.float64)
    else:
        tvec = np.asarray(calibration.tvec, dtype=np.float64).reshape(3)

    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.tolist(),
        translation_vector=tvec.tolist(),
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
        tracked_landmark_types=[],
    )


def _compute_keyframes(
    total_frames: int,
    keyframe_interval: int,
    max_keyframes: int,
) -> list[int]:
    """Sample keyframes from ``[0, total_frames)``.

    Chooses an interval that is at least ``keyframe_interval`` but never
    produces more than ``max_keyframes`` samples.  Short shots fall back to
    the configured interval.
    """
    if total_frames <= 0:
        return []
    effective = max(keyframe_interval, 1)
    if max_keyframes > 0:
        # Ceil division so (total_frames / max_keyframes) is an upper bound
        # on the resulting sample count.
        interval_from_cap = -(-total_frames // max_keyframes)
        effective = max(effective, interval_from_cap)
    return list(range(0, total_frames, effective))


class CameraCalibrationStage(BaseStage):
    name = "calibration"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        neural_calibrator: PnLCalibrator | None = None,
        **_: object,
    ) -> None:
        super().__init__(config, output_dir)
        self._calibrator_ext = neural_calibrator
        self._calibrator_cache: PnLCalibrator | None = None

    def _calibrator(self) -> PnLCalibrator:
        """Return the :class:`PnLCalibrator` instance, constructing one lazily
        if none was injected.
        """
        if self._calibrator_ext is not None:
            return self._calibrator_ext
        if self._calibrator_cache is None:
            cfg = self.config.get("calibration", {})
            self._calibrator_cache = PnLCalibrator(
                device=str(cfg.get("device", "auto")),
                kp_threshold=float(cfg.get("kp_threshold", 0.3434)),
                line_threshold=float(cfg.get("line_threshold", 0.7867)),
                pnl_refine=bool(cfg.get("pnl_refine", True)),
            )
        return self._calibrator_cache

    def is_complete(self) -> bool:
        cal_dir = self.output_dir / "calibration"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = ShotsManifest.load(manifest_path)
        except Exception:
            return False
        return all(
            (cal_dir / f"{shot.id}_calibration.json").exists()
            for shot in manifest.shots
        )

    def run(self) -> None:
        cal_dir = self.output_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("calibration", {})
        keyframe_interval = int(cfg.get("keyframe_interval", 30))
        max_keyframes = int(cfg.get("max_keyframes_per_shot", 10))
        bounds = self._resolve_bounds(cfg.get("plausibility_bounds"))

        shots_dir = self.output_dir / "shots"
        manifest_path = shots_dir / "shots_manifest.json"
        if not manifest_path.exists():
            print("  -> inferred shots_manifest.json from prepared clips")
        manifest = ShotsManifest.load_or_infer(shots_dir, persist=True)

        debug_overlay_enabled = bool(cfg.get("debug_overlay", True))
        debug_overlay_n_frames = int(cfg.get("debug_overlay_n_frames", 6))
        line_refine_enabled = bool(cfg.get("line_refine", True))

        for shot in manifest.shots:
            result = self._calibrate_shot(
                shot.id,
                shot.clip_file,
                keyframe_interval=keyframe_interval,
                max_keyframes=max_keyframes,
                bounds=bounds,
            )
            # ── Apply manual landmark annotations (if any) ──
            # Each annotated frame becomes a high-trust keyframe that
            # replaces (or supplements) PnLCalib's automatic output at
            # that frame index.  See src/utils/manual_calibration.py.
            n_anchors, anchor_modes = self._apply_manual_annotations(shot.id, result)
            if n_anchors:
                mode_desc = ", ".join(f"{k}:{v}" for k, v in sorted(anchor_modes.items()))
                print(f"     manual anchors: {n_anchors} keyframe(s) injected ({mode_desc})")
            if line_refine_enabled and result.frames:
                clip_path = self.output_dir / shot.clip_file
                try:
                    result, diagnostics = refine_shot_calibration(result, clip_path)
                    n_accepted = sum(1 for d in diagnostics if d.accepted)
                    n_icl = sum(1 for d in diagnostics if d.icl_accepted)
                    if diagnostics:
                        before = float(np.mean([d.initial_residual_px for d in diagnostics if np.isfinite(d.initial_residual_px)]) or 0.0)
                        after = float(np.mean([d.refined_residual_px for d in diagnostics if np.isfinite(d.refined_residual_px)]) or 0.0)
                        icl_before = [d.icl_initial_residual_px for d in diagnostics if d.icl_accepted and np.isfinite(d.icl_initial_residual_px)]
                        icl_after = [d.icl_refined_residual_px for d in diagnostics if d.icl_accepted and np.isfinite(d.icl_refined_residual_px)]
                        print(
                            f"     line refine: {n_accepted}/{len(diagnostics)} keyframes "
                            f"improved VP residual {before:.1f}→{after:.1f}°"
                        )
                        if icl_before:
                            fx_factors = [d.icl_focal_change_factor for d in diagnostics if d.icl_accepted]
                            print(
                                f"     ICL refine: {n_icl}/{len(diagnostics)} keyframes "
                                f"improved line-distance {float(np.mean(icl_before)):.1f}→{float(np.mean(icl_after)):.1f}px "
                                f"(fx ×{float(np.mean(fx_factors)):.2f})"
                            )
                    pf_diags = getattr(diagnostics, "per_frame_diagnostics", None) or []
                    if pf_diags:
                        n_pf_accepted = sum(1 for d in pf_diags if d.accepted)
                        residuals = [d.refined_residual_px for d in pf_diags
                                     if d.accepted and np.isfinite(d.refined_residual_px)]
                        mean_res = float(np.mean(residuals)) if residuals else 0.0
                        print(
                            f"     per-frame ICL: {n_pf_accepted}/{len(pf_diags)} frames "
                            f"refined (mean line residual {mean_res:.1f} px)"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("line refine failed for %s: %s", shot.id, exc)
            result.save(cal_dir / f"{shot.id}_calibration.json")
            flag = " (no calibration frames — PnLCalib failed)" if not result.frames else ""
            print(f"  -> {shot.id}: {len(result.frames)} frames calibrated{flag}")
            if debug_overlay_enabled and result.frames:
                try:
                    written = render_shot_overlays(
                        self.output_dir, shot.id, n_frames=debug_overlay_n_frames,
                    )
                    if written:
                        print(f"     debug overlay: {len(written)} frame(s) → calibration/debug/{shot.id}/")
                except Exception as exc:  # noqa: BLE001 — non-fatal diagnostic
                    logger.warning("debug overlay failed for %s: %s", shot.id, exc)

    def _apply_manual_annotations(
        self, shot_id: str, result: CalibrationResult,
    ) -> tuple[int, dict[str, int]]:
        """Inject manual-annotation keyframes into ``result`` (in place).

        Reads ``output/calibration/annotations/<shot_id>.json``, solves
        each annotated frame via :func:`solve_from_annotations`, and:

        - **Replaces** any existing PnLCalib keyframe at the same
          frame index with the manual one.
        - **Appends** new keyframes for annotated frames PnLCalib
          didn't sample.

        The result is sorted by frame index and the camera world
        position is re-anchored to the median of all keyframes (so
        the manual anchors pull the static-camera position toward the
        truth when PnLCalib's median was off).

        Returns the number of manual anchors injected.
        """
        ann_path = self.output_dir / "calibration" / "annotations" / f"{shot_id}.json"
        if not ann_path.exists():
            return 0, {}
        try:
            ann_data: dict = json.loads(ann_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("annotations: %s unreadable: %s", shot_id, exc)
            return 0, {}
        if not ann_data:
            return 0, {}

        # Look up clip dimensions for the solver
        clip_path = self.output_dir / "shots" / f"{shot_id}.mp4"
        if not clip_path.exists():
            logger.warning("annotations: %s clip missing", shot_id)
            return 0, {}
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return 0, {}
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        # fx_init from the median of existing PnLCalib outputs (if any),
        # else a sensible default.
        existing_fxs = [
            float(cf.intrinsic_matrix[0][0]) for cf in result.frames
        ]
        fx_init = float(np.median(existing_fxs)) if existing_fxs else 3500.0

        # Shared camera world position from the existing PnLCalib
        # keyframes.  Every keyframe has the same position baked in
        # (the static-camera fuser enforces that), but compute the
        # median as a defensive measure in case earlier stages
        # modified a subset.  When no keyframes exist yet, fall back
        # to the free-position solver — the first manual anchor has
        # to carry the position itself.
        shared_position: np.ndarray | None = None
        if result.frames:
            positions = np.array([
                camera_world_position(
                    np.asarray(cf.rotation_vector, dtype=np.float64),
                    np.asarray(cf.translation_vector, dtype=np.float64),
                )
                for cf in result.frames
            ], dtype=np.float64)
            shared_position = np.median(positions, axis=0)

        manual_frames: list[CameraFrame] = []
        mode_counts: dict[str, int] = {}
        for frame_str, landmarks in ann_data.items():
            try:
                frame_idx = int(frame_str)
            except ValueError:
                continue
            if not isinstance(landmarks, dict) or not landmarks:
                continue
            res = solve_from_annotations(
                landmarks,
                image_size=(width, height),
                fx_init=fx_init,
                frame_idx=frame_idx,
                camera_position_world=shared_position,
            )
            if res is None:
                min_needed = 3 if shared_position is not None else 4
                logger.warning(
                    "annotations: %s frame %d failed (need ≥%d valid landmarks)",
                    shot_id, frame_idx, min_needed,
                )
                continue
            manual_frames.append(res.camera_frame)
            mode_counts[res.mode] = mode_counts.get(res.mode, 0) + 1

        if not manual_frames:
            return 0, {}

        # Replace any existing keyframe at the same frame index.
        manual_indices = {cf.frame for cf in manual_frames}
        kept = [cf for cf in result.frames if cf.frame not in manual_indices]
        merged = sorted(kept + manual_frames, key=lambda cf: cf.frame)

        # Mutate the input CalibrationResult in place so the caller
        # sees the augmented keyframe set.
        result.frames.clear()
        result.frames.extend(merged)
        return len(manual_frames), mode_counts

    @staticmethod
    def _resolve_bounds(
        override: dict | None,
    ) -> dict[str, tuple[float, float]]:
        """Merge user-provided plausibility bounds on top of the defaults."""
        resolved: dict[str, tuple[float, float]] = {
            k: tuple(v) for k, v in _DEFAULT_BOUNDS.items()
        }
        if override:
            for key, value in override.items():
                if key in resolved and isinstance(value, (list, tuple)) and len(value) == 2:
                    resolved[key] = (float(value[0]), float(value[1]))
        return resolved

    def _calibrate_shot(
        self,
        shot_id: str,
        clip_file: str,
        keyframe_interval: int,
        max_keyframes: int,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> CalibrationResult:
        clip_path = self.output_dir / clip_file
        if bounds is None:
            bounds = self._resolve_bounds(None)

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            logger.warning("%s: failed to open clip %s", shot_id, clip_path)
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            keyframes = _compute_keyframes(total_frames, keyframe_interval, max_keyframes)
            if not keyframes:
                logger.warning("%s: clip has no frames", shot_id)
                return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

            calibrator = self._calibrator()
            per_frame: list[tuple[int, NeuralCalibration]] = []
            rejected_count = 0
            for frame_idx in keyframes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    logger.debug("%s: could not read frame %d", shot_id, frame_idx)
                    continue
                result = calibrator.calibrate(frame)
                if result is None:
                    logger.debug(
                        "%s: PnLCalib returned None for frame %d", shot_id, frame_idx,
                    )
                    continue
                if not _is_plausible(result, bounds):
                    rejected_count += 1
                    logger.debug(
                        "%s: implausible calibration at frame %d "
                        "(pos=%s, fx=%.0f) — dropping",
                        shot_id,
                        frame_idx,
                        result.world_position.tolist(),
                        float(result.K[0, 0]),
                    )
                    continue
                per_frame.append((frame_idx, result))
        finally:
            cap.release()

        if not per_frame:
            logger.warning(
                "%s: PnLCalib produced no plausible calibrations across %d "
                "keyframes (%d rejected on plausibility) — shot will have "
                "an empty calibration",
                shot_id,
                len(keyframes),
                rejected_count,
            )
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        positions = [c.world_position for _, c in per_frame]
        shared_position = _robust_median_position(positions)
        if shared_position is None:
            logger.warning("%s: could not compute median camera position", shot_id)
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        # Keep only frames whose per-frame world position is within the
        # consensus cluster.  PnLCalib sometimes returns plausible-but-wrong
        # results where the rotation + focal length were computed against a
        # different camera location, so naively overriding the position
        # leaves an inconsistent projection.  Drop those frames.
        cluster_tolerance = float(
            self.config.get("calibration", {}).get("cluster_tolerance", 5.0)
        )
        consistent: list[tuple[int, NeuralCalibration]] = []
        for frame_idx, cal in per_frame:
            if np.linalg.norm(cal.world_position - shared_position) <= cluster_tolerance:
                consistent.append((frame_idx, cal))
            else:
                rejected_count += 1
                logger.debug(
                    "%s: frame %d dropped — position %s differs from consensus %s",
                    shot_id,
                    frame_idx,
                    cal.world_position.tolist(),
                    shared_position.tolist(),
                )

        if not consistent:
            logger.warning(
                "%s: no keyframes survived the consensus cluster filter", shot_id,
            )
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

        logger.info(
            "%s: static camera at (%.2f, %.2f, %.2f) across %d consensus "
            "keyframes (%d rejected total)",
            shot_id,
            shared_position[0],
            shared_position[1],
            shared_position[2],
            len(consistent),
            rejected_count,
        )

        frames = [
            _neural_to_cam_frame(
                calibration=cal,
                frame_idx=frame_idx,
                override_position=shared_position,
            )
            for frame_idx, cal in consistent
        ]
        return CalibrationResult(shot_id=shot_id, camera_type="static", frames=frames)

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
        return (self.output_dir / "camera" / "camera_track.json").exists()

    def run(self) -> None:
        cfg = self.config.get("camera", {})
        anchors_path = self.output_dir / "camera" / "anchors.json"
        if not anchors_path.exists():
            raise FileNotFoundError(
                f"camera stage requires user-supplied anchors at {anchors_path}. "
                "Open the web viewer (recon.py serve) and place keyframes."
            )
        anchors = AnchorSet.load(anchors_path)

        clip_dir = self.output_dir / "shots"
        clips = list(clip_dir.glob("*.mp4"))
        if not clips:
            raise FileNotFoundError(f"no clip in {clip_dir}")
        clip_path = clips[0]

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

        try:
            sol = solve_anchors_jointly(
                tuple(qualifying), image_size=anchors.image_size,
            )
        except AnchorSolveError as exc:
            raise RuntimeError(f"camera stage failed: {exc}") from exc

        t_world_median = sol.t_world
        principal_point = sol.principal_point
        # Per-anchor (K, R, t) — each anchor has its own translation.
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
        # Pass anchor_KR_only to the propagator (it doesn't need t for the
        # feature-tracking homography — t enters only via interpolation
        # below, after propagation).
        anchor_KR_only = {f: (K, R) for f, (K, R, _) in anchor_solutions.items()}
        for a, b in zip(anchor_frames, anchor_frames[1:]):
            self._propagate_pair(
                cap, a, b, anchor_KR_only,
                per_frame_K, per_frame_R, per_frame_conf, is_anchor, cfg,
            )
            # Linear interpolation of t between adjacent anchors.
            t_a = per_frame_t[a]
            t_b = per_frame_t[b]
            for offset in range(b - a + 1):
                idx = a + offset
                if is_anchor[idx]:
                    continue
                w = offset / (b - a)
                per_frame_t[idx] = (1.0 - w) * t_a + w * t_b

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
        )
        track.save(self.output_dir / "camera" / "camera_track.json")

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

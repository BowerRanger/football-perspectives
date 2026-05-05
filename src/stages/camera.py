"""Camera-tracking stage: anchors + propagation + smoothing → camera_track.json."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.anchor import AnchorSet
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.utils.anchor_solver import solve_first_anchor, solve_subsequent_anchor
from src.utils.bidirectional_smoother import smooth_between_anchors
from src.utils.camera_confidence import FrameSignals, confidence_from_signals
from src.utils.feature_propagator import propagate_one_frame

logger = logging.getLogger(__name__)


def _angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    cos_t = (np.trace(R1.T @ R2) - 1) / 2
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(np.degrees(np.arccos(cos_t)))


def _reprojection_residual(
    landmarks: tuple, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> float:
    """Mean reprojection error (px) of landmark world points under (K, R, t).

    Returns a large sentinel (1e9) if any landmark is behind the camera.
    """
    residuals: list[float] = []
    for lm in landmarks:
        cam = R @ np.array(lm.world_xyz) + t
        if cam[2] <= 0:
            return 1e9
        pix = K @ cam
        proj = pix[:2] / pix[2]
        residuals.append(float(np.linalg.norm(np.array(lm.image_xy) - proj)))
    return float(np.mean(residuals)) if residuals else 0.0


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

        # Step 1: solve anchor frames.
        first_anchor = anchors.anchors[0]
        K0, R0, t_world = solve_first_anchor(first_anchor.landmarks)
        anchor_solutions: dict[int, tuple[np.ndarray, np.ndarray]] = {
            first_anchor.frame: (K0, R0)
        }
        anchor_confidence_override: dict[int, float] = {}
        for a in anchors.anchors[1:]:
            K, R = solve_subsequent_anchor(
                a.landmarks, t_world, image_size=anchors.image_size
            )
            anchor_solutions[a.frame] = (K, R)
            residual = _reprojection_residual(a.landmarks, K, R, t_world)
            if residual > anchor_max_residual:
                logger.warning(
                    "anchor at frame %d has reprojection residual %.2f px > "
                    "%.2f px threshold; flagging as low-confidence",
                    a.frame, residual, anchor_max_residual,
                )
                anchor_confidence_override[a.frame] = 0.5

        # Step 2: per-frame propagate forward and backward between consecutive anchor pairs.
        per_frame_K: list[np.ndarray | None] = [None] * n_frames
        per_frame_R: list[np.ndarray | None] = [None] * n_frames
        per_frame_conf: list[float] = [0.0] * n_frames
        is_anchor: list[bool] = [False] * n_frames

        for af in anchor_solutions:
            per_frame_K[af] = anchor_solutions[af][0]
            per_frame_R[af] = anchor_solutions[af][1]
            per_frame_conf[af] = anchor_confidence_override.get(af, 1.0)
            is_anchor[af] = True

        anchor_frames = sorted(anchor_solutions.keys())
        # Frames before first anchor / after last anchor are propagated one-way.
        for a, b in zip(anchor_frames, anchor_frames[1:]):
            self._propagate_pair(
                cap, a, b, anchor_solutions,
                per_frame_K, per_frame_R, per_frame_conf, is_anchor, cfg,
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
            if K is None or R is None:
                continue  # frames outside any anchor span are skipped in v1
            frames_out.append(
                CameraFrame(
                    frame=i,
                    K=K.tolist(),
                    R=R.tolist(),
                    confidence=per_frame_conf[i],
                    is_anchor=is_anchor[i],
                )
            )

        track = CameraTrack(
            clip_id=anchors.clip_id,
            fps=float(fps),
            image_size=(w, h),
            t_world=list(t_world),
            frames=tuple(frames_out),
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

"""Per-frame calibration interpolation for triangulation.

The calibration stage produces ~5–10 keyframes per shot (PnLCalib inference
is expensive, so we sample sparsely).  The triangulation stage needs the
camera parameters at every pose frame.  This module interpolates between
the nearest calibrated keyframes:

- **Rotation**: SLERP (spherical linear interpolation) between adjacent
  keyframe rotation vectors.
- **Focal length**: linear interpolation of ``fx`` (and ``fy``).
- **Translation**: kept constant per-shot because the calibration stage
  enforces a shared world position across all keyframes in a shot.  We
  simply take the first keyframe's translation.

Outside the keyframe range we clamp to the nearest keyframe (no
extrapolation).  When only one keyframe exists we return its parameters
unchanged for any target frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.schemas.calibration import CalibrationResult, CameraFrame


@dataclass(frozen=True)
class InterpolatedCalibration:
    """Interpolated camera parameters for a single pose frame."""

    K: np.ndarray     # 3x3 intrinsic
    rvec: np.ndarray  # 3-vector Rodrigues rotation
    tvec: np.ndarray  # 3-vector translation


class CalibrationInterpolator:
    """Sample a shot's calibration at arbitrary pose frames.

    Build once per shot, then call :meth:`at` per pose frame.  Returns
    ``None`` if the shot has no keyframes (empty calibration).
    """

    def __init__(self, calibration: CalibrationResult):
        self._frames: list[CameraFrame] = sorted(
            calibration.frames, key=lambda cf: cf.frame,
        )
        if not self._frames:
            self._slerp = None
            self._keyframe_indices = np.empty(0, dtype=np.int64)
            self._fx_values = np.empty(0, dtype=np.float64)
            self._fy_values = np.empty(0, dtype=np.float64)
            self._principal_points = np.empty((0, 2), dtype=np.float64)
            return

        self._keyframe_indices = np.array(
            [cf.frame for cf in self._frames], dtype=np.int64,
        )
        self._fx_values = np.array(
            [cf.intrinsic_matrix[0][0] for cf in self._frames], dtype=np.float64,
        )
        self._fy_values = np.array(
            [cf.intrinsic_matrix[1][1] for cf in self._frames], dtype=np.float64,
        )
        self._principal_points = np.array(
            [[cf.intrinsic_matrix[0][2], cf.intrinsic_matrix[1][2]] for cf in self._frames],
            dtype=np.float64,
        )
        # Translation is constant per shot under the static-camera model, but
        # in case someone hands us a panning calibration we still store all
        # translations and pick the nearest.
        self._translations = np.array(
            [cf.translation_vector for cf in self._frames], dtype=np.float64,
        )

        if len(self._frames) >= 2:
            rotations = Rotation.from_rotvec(
                np.array([cf.rotation_vector for cf in self._frames], dtype=np.float64),
            )
            self._slerp = Slerp(self._keyframe_indices.astype(np.float64), rotations)
        else:
            self._slerp = None  # single keyframe — no interpolation needed

    @property
    def is_empty(self) -> bool:
        return len(self._frames) == 0

    def at(self, frame_idx: int) -> InterpolatedCalibration | None:
        """Return interpolated calibration at ``frame_idx``.

        Returns ``None`` if there are no keyframes.
        """
        if self.is_empty:
            return None

        clamped = int(
            np.clip(frame_idx, self._keyframe_indices[0], self._keyframe_indices[-1]),
        )

        if self._slerp is None:
            # Single keyframe: return it directly regardless of frame_idx.
            cf = self._frames[0]
            rvec = np.asarray(cf.rotation_vector, dtype=np.float64)
            tvec = np.asarray(cf.translation_vector, dtype=np.float64)
            K = np.asarray(cf.intrinsic_matrix, dtype=np.float64)
            return InterpolatedCalibration(K=K, rvec=rvec, tvec=tvec)

        # Rotation via SLERP (scipy handles the bracketing internally)
        rotation = self._slerp([float(clamped)])[0]
        rvec = rotation.as_rotvec()

        # Focal length via linear interp
        fx = float(np.interp(clamped, self._keyframe_indices, self._fx_values))
        fy = float(np.interp(clamped, self._keyframe_indices, self._fy_values))
        # Principal point (usually constant but interpolate for safety)
        cx = float(np.interp(clamped, self._keyframe_indices, self._principal_points[:, 0]))
        cy = float(np.interp(clamped, self._keyframe_indices, self._principal_points[:, 1]))

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # Translation: recompute from the world position implied by the
        # nearest keyframe so we stay consistent with the static-camera
        # guarantee from the calibration stage.  Position C = -R_key^T @ t_key.
        nearest_idx = int(np.argmin(np.abs(self._keyframe_indices - clamped)))
        nearest = self._frames[nearest_idx]
        R_key, _ = _rodrigues(np.asarray(nearest.rotation_vector, dtype=np.float64))
        t_key = np.asarray(nearest.translation_vector, dtype=np.float64)
        world_position = -R_key.T @ t_key

        # New translation from the interpolated rotation but the same world
        # position: t_new = -R_new @ C.
        R_new, _ = _rodrigues(rvec)
        tvec = (-R_new @ world_position).astype(np.float64)

        return InterpolatedCalibration(K=K, rvec=rvec.astype(np.float64), tvec=tvec)


def _rodrigues(rvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around cv2.Rodrigues to avoid importing cv2 at module load."""
    import cv2

    return cv2.Rodrigues(rvec.reshape(3, 1))

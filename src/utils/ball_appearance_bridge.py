"""Layer 4: bridge short WASB detection gaps via normalised cross-
correlation against a rolling template.

The bridge is *guided* by the IMM tracker's prediction — it only
searches inside a ``search_radius_px`` window around the predicted
pixel. A high NCC peak there is accepted as a bridged detection (with
discounted confidence) so the IMM can continue updating instead of
gap-filling for several frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AppearanceBridgeCfg:
    enabled: bool
    max_gap_frames: int
    template_size_px: int
    search_radius_px: int
    min_ncc: float
    template_max_age_frames: int
    template_update_confidence: float


class AppearanceBridge:
    """Stateful holder for a rolling ball template plus an NCC bridger.

    Not thread-safe. One instance per shot.
    """

    def __init__(self, cfg: AppearanceBridgeCfg) -> None:
        self._cfg = cfg
        self._template: np.ndarray | None = None
        self._template_frame: int | None = None

    def update_template(
        self,
        *,
        frame: int,
        frame_image: np.ndarray,
        uv: tuple[float, float],
        confidence: float,
    ) -> None:
        if not self._cfg.enabled:
            return
        if confidence < self._cfg.template_update_confidence:
            return
        half = self._cfg.template_size_px // 2
        u, v = int(round(uv[0])), int(round(uv[1]))
        h, w = frame_image.shape[:2]
        if u - half < 0 or v - half < 0 or u + half > w or v + half > h:
            return
        crop = frame_image[v - half:v + half, u - half:u + half]
        if crop.shape[:2] != (self._cfg.template_size_px, self._cfg.template_size_px):
            return
        self._template = crop.copy()
        self._template_frame = frame

    def try_bridge(
        self,
        *,
        frame: int,
        frame_image: np.ndarray,
        predicted_uv: tuple[float, float] | None,
        consecutive_misses: int,
    ) -> tuple[tuple[float, float], float] | None:
        """Try to find the ball near ``predicted_uv``. Returns
        ``((u, v), confidence)`` on success, or ``None`` if disabled,
        out of gap budget, template stale, or NCC below threshold."""
        if not self._cfg.enabled:
            return None
        if self._template is None or self._template_frame is None:
            return None
        if consecutive_misses > self._cfg.max_gap_frames:
            return None
        if predicted_uv is None:
            return None
        age = frame - self._template_frame
        if age > self._cfg.template_max_age_frames:
            return None

        h, w = frame_image.shape[:2]
        r = self._cfg.search_radius_px
        u, v = int(round(predicted_uv[0])), int(round(predicted_uv[1]))
        u0, v0 = max(0, u - r), max(0, v - r)
        u1, v1 = min(w, u + r), min(h, v + r)
        if u1 - u0 <= self._cfg.template_size_px or v1 - v0 <= self._cfg.template_size_px:
            return None
        window = frame_image[v0:v1, u0:u1]
        result = cv2.matchTemplate(window, self._template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, peak_loc = cv2.minMaxLoc(result)
        if peak < self._cfg.min_ncc:
            return None
        half = self._cfg.template_size_px // 2
        peak_u = u0 + peak_loc[0] + half
        peak_v = v0 + peak_loc[1] + half
        # Discount confidence so the IMM weighs real WASB hits higher.
        return (float(peak_u), float(peak_v)), float(peak) * 0.5

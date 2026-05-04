"""Per-frame confidence aggregation for the camera stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameSignals:
    inlier_ratio: float          # in [0, 1]
    fwd_bwd_disagreement_deg: float
    pitch_line_residual_px: float | None  # None if pitch lines not detected


def confidence_from_signals(
    signals: FrameSignals,
    *,
    pitch_line_residual_max_px: float = 5.0,
    fwd_bwd_disagreement_warn_deg: float = 0.5,
) -> float:
    """Returns confidence in [0, 1].

    All three signals are clipped and combined multiplicatively so that any
    one being bad drives the overall score down.
    """
    inlier = max(0.0, min(1.0, signals.inlier_ratio))
    disagreement = max(0.0, 1.0 - signals.fwd_bwd_disagreement_deg / (3 * fwd_bwd_disagreement_warn_deg))
    if signals.pitch_line_residual_px is None:
        line_score = 1.0
    else:
        line_score = max(0.0, 1.0 - signals.pitch_line_residual_px / (3 * pitch_line_residual_max_px))
    return float(inlier * disagreement * line_score)

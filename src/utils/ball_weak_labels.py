"""Weak training labels from the solved ball track (spec §4.3 step 1).

The 145 gold labels (operator-clicked pixels) are too thin to fine-tune on
alone; WASB also trains best on labelled RUNS (3-frame stacks want all three
frames labelled). Near a manual anchor the piecewise/events solve is
operator-anchored and physically constrained, so its per-frame world
positions are trustworthy exactly where the raw detector failed — the
hard examples. This module projects those positions back to pixels inside
±window of each gold frame. Pure and torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping
from xml.sax.saxutils import escape

import numpy as np

from src.utils.camera_projection import project_world_to_image

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.schemas.ball_track import BallTrack

_WEAK_STATES = frozenset({"grounded", "flight"})


def weak_labels_from_track(
    track: "BallTrack",
    *,
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    image_size: tuple[int, int],
    gold_frames: set[int],
    window: int = 20,
    min_conf: float = 0.5,
    edge_margin_px: float = 4.0,
) -> dict[int, tuple[float, float]]:
    """Solved-track pixels usable as weak labels around gold anchors."""
    w, h = float(image_size[0]), float(image_size[1])
    out: dict[int, tuple[float, float]] = {}
    for bf in track.frames:
        f = bf.frame
        if f in gold_frames:
            continue
        if not any(abs(f - g) <= window for g in gold_frames):
            continue
        if bf.world_xyz is None or bf.state not in _WEAK_STATES:
            continue
        if bf.confidence < min_conf:
            continue
        K, R, t = per_frame_K.get(f), per_frame_R.get(f), per_frame_t.get(f)
        if K is None or R is None or t is None:
            continue
        uv = project_world_to_image(
            K, R, t, distortion,
            np.asarray([bf.world_xyz], dtype=float),
        )[0]
        u, v = float(uv[0]), float(uv[1])
        if not (edge_margin_px <= u <= w - edge_margin_px
                and edge_margin_px <= v <= h - edge_margin_px):
            continue
        out[f] = (u, v)
    return out


def merge_labels(
    gold: Mapping[int, tuple[float, float]],
    weak: Mapping[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Union of label maps; gold wins on frame collision."""
    merged: dict[int, tuple[float, float]] = dict(weak)
    merged.update(gold)
    return merged


def labels_to_cvat_xml(
    clip_id: str, labels: Mapping[int, tuple[float, float]],
) -> str:
    """Render a label map in the same CVAT dialect as anchors_to_cvat_xml
    (validated against the vendored WASB soccer loader)."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotations>",
        f'  <track id="0" label="ball" source="{escape(str(clip_id))}">',
    ]
    for frame in sorted(labels):
        u, v = labels[frame]
        lines.append(
            f'    <points frame="{int(frame)}" outside="0" occluded="0" '
            f'points="{u:.2f},{v:.2f}">'
        )
        lines.append('      <attribute name="used_in_game">1</attribute>')
        lines.append("    </points>")
    lines.append("  </track>")
    lines.append("</annotations>")
    return "\n".join(lines) + "\n"

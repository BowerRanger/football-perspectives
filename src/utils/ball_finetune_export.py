"""Export manual ball anchors as WASB training annotations + frames.

The investigation showed the ball *detector* is the wall: WASB confidently
misdetects at touches and can't disambiguate the ball, and no inference-only
trick works around it. The fix is to fine-tune WASB — and our **manual ball
anchors are gold labels** (the operator clicked the true ball pixel).

WASB's soccer loader reads per-video CVAT-style XML
(``<track><points frame= outside= occluded= points="x,y">
<attribute name="used_in_game">1</attribute></points></track>``) plus frames
extracted as ``{fid:05d}.png``. The format is identical to soccer, so the
exported data trains via ``dataset=soccer dataset.root_dir=<export>`` with no
WASB-repo changes. See docs/superpowers/specs/2026-06-15-ball-finetune-README.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape

from src.schemas.ball_anchor import BallAnchor


def anchors_to_cvat_xml(clip_id: str, anchors: Sequence[BallAnchor]) -> str:
    """Render ball anchors as a WASB/CVAT annotation XML string.

    A labelled anchor (has ``image_xy``) becomes a visible, used-in-game ball
    point; an ``off_screen_flight`` anchor (no pixel) is marked ``outside=1``.
    """
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotations>",
        f'  <track id="0" label="ball" source="{escape(clip_id)}">',
    ]
    for a in sorted(anchors, key=lambda x: x.frame):
        outside = "1" if a.image_xy is None else "0"
        x, y = (a.image_xy if a.image_xy is not None else (-1.0, -1.0))
        lines.append(
            f'    <points frame="{int(a.frame)}" outside="{outside}" '
            f'occluded="0" points="{x:.2f},{y:.2f}">'
        )
        lines.append('      <attribute name="used_in_game">1</attribute>')
        lines.append("    </points>")
    lines.append("  </track>")
    lines.append("</annotations>")
    return "\n".join(lines)


def export_dataset(
    *,
    clip_path: Path,
    anchors: Sequence[BallAnchor],
    out_root: Path,
    video_id: str,
) -> dict:
    """Write a WASB soccer-format dataset for one clip: extract frames to
    ``frames/{video_id}/{fid:05d}.png`` and the anno to ``annos/{video_id}.xml``.

    Returns ``{n_frames, n_labels, anno_path, frames_dir}``. Frame extraction
    uses OpenCV; kept import-local so the pure XML helper stays dependency-free.
    """
    import cv2

    frames_dir = out_root / "frames" / video_id
    annos_dir = out_root / "annos"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annos_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {clip_path}")
    n = 0
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frames_dir / f"{idx:05d}.png"), frame)
            idx += 1
        n = idx
    finally:
        cap.release()

    anno_path = annos_dir / f"{video_id}.xml"
    anno_path.write_text(anchors_to_cvat_xml(video_id, anchors))
    return {
        "n_frames": n,
        "n_labels": sum(1 for a in anchors if a.image_xy is not None),
        "anno_path": str(anno_path),
        "frames_dir": str(frames_dir),
    }


__all__ = ["anchors_to_cvat_xml", "export_dataset"]

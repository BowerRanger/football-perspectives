"""Auto-accumulating ball-label corpus — the fine-tuning data flywheel.

Every time the operator saves manual ball anchors (the web ball editor), the
clicked ball pixels are gold labels *targeted at exactly the frames the
detector fails* (touches). ``record_labels`` appends them to a growing
WASB soccer-format corpus as a cheap byproduct of normal use (label XML +
manifest only — no frame extraction on the hot path).

Before training, ``materialize_frames`` extracts the frames for every clip in
the corpus. The result trains directly via
``dataset=soccer dataset.root_dir=<corpus>`` (see the fine-tune README).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_finetune_export import anchors_to_cvat_xml, extract_frames


def load_manifest(corpus_root: str | Path) -> dict:
    p = Path(corpus_root) / "manifest.json"
    if not p.exists():
        return {"clips": {}}
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001 — a corrupt manifest shouldn't crash a save
        return {"clips": {}}
    data.setdefault("clips", {})
    return data


def record_labels(
    corpus_root: str | Path,
    clip_id: str,
    clip_rel_path: str,
    anchors: Sequence[BallAnchor],
) -> dict | None:
    """Append/refresh one clip's gold labels in the corpus.

    Writes ``annos/{clip_id}.xml`` and upserts the manifest entry. Returns the
    entry, or ``None`` (a no-op) when there are no labelled anchors. Cheap: no
    frame extraction here — that's :func:`materialize_frames`.
    """
    corpus_root = Path(corpus_root)
    n_labels = sum(1 for a in anchors if a.image_xy is not None)
    if n_labels == 0:
        return None
    (corpus_root / "annos").mkdir(parents=True, exist_ok=True)
    anno_rel = f"annos/{clip_id}.xml"
    (corpus_root / anno_rel).write_text(anchors_to_cvat_xml(clip_id, anchors))

    manifest = load_manifest(corpus_root)
    prev = manifest["clips"].get(clip_id, {})
    entry = {
        "clip_path": clip_rel_path,
        "anno": anno_rel,
        "n_labels": n_labels,
        "saves": int(prev.get("saves", 0)) + 1,
    }
    manifest["clips"][clip_id] = entry
    (corpus_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return entry


def materialize_frames(
    corpus_root: str | Path,
    resolve_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Extract frames for every clip in the corpus manifest (idempotent).

    ``clip_path`` entries are resolved relative to ``resolve_root`` (the
    pipeline output dir). Returns ``{clip_id: status}``.
    """
    corpus_root = Path(corpus_root)
    resolve_root = Path(resolve_root)
    out: dict[str, str] = {}
    for clip_id, entry in load_manifest(corpus_root)["clips"].items():
        frames_dir = corpus_root / "frames" / clip_id
        if not force and frames_dir.exists() and any(frames_dir.iterdir()):
            out[clip_id] = "cached"
            continue
        clip = resolve_root / entry["clip_path"]
        if not clip.exists():
            out[clip_id] = "missing_video"
            continue
        n = extract_frames(clip, frames_dir)
        out[clip_id] = f"extracted:{n}"
    return out


__all__ = ["load_manifest", "record_labels", "materialize_frames"]

"""Read a CapCut draft's shot separation as ground truth.

CapCut (Mac) stores each project as JSON under
``~/Movies/CapCut/User Data/Projects/com.lveditor.draft/<name>/draft_info.json``.
The operator separates shots (and fades) on the timeline; this script
extracts every video segment's exact source time range — no exports, no
timestamps written down.

Ground-truth semantics:
- each remaining segment = one expected shot (or fade, if very short —
  flagged for confirmation)
- segments DELETED from the timeline = content that must be dropped
  (gaps in source coverage)
- grouping is annotated afterwards against the numbered contact sheet
  this script renders.

Usage:
    python scripts/capcut_ground_truth.py DRAFT_NAME \
        [--out gt.json] [--sheet sheet.png] [--annotate ann.json]

``--annotate`` merges a ``{"groups": {"g01": [2,5,7], ...},
"drop": [0,1], "fades": [4,6]}`` file (indices from the sheet) into the
output manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DRAFT_ROOT = Path.home() / "Movies/CapCut/User Data/Projects/com.lveditor.draft"


def read_draft_segments(draft_name: str) -> dict:
    """Extract video-track segments (source-time order) from a draft."""
    info_path = DRAFT_ROOT / draft_name / "draft_info.json"
    draft = json.loads(info_path.read_text())
    videos = {v["id"]: v for v in draft.get("materials", {}).get("videos", [])}

    segments = []
    sources = set()
    for track in draft.get("tracks", []):
        if track.get("type") != "video":
            continue
        for seg in track.get("segments", []):
            st = seg.get("source_timerange") or {}
            mat = videos.get(seg.get("material_id"), {})
            src = mat.get("path", "")
            sources.add(src)
            segments.append({
                "source_start_s": round(st.get("start", 0) / 1e6, 3),
                "source_end_s": round(
                    (st.get("start", 0) + st.get("duration", 0)) / 1e6, 3,
                ),
            })
    segments.sort(key=lambda s: s["source_start_s"])
    for i, s in enumerate(segments):
        s["index"] = i
        s["duration_s"] = round(s["source_end_s"] - s["source_start_s"], 3)
    if len(sources) > 1:
        raise SystemExit(f"draft uses multiple sources: {sources}")
    return {
        "draft": draft_name,
        "source": next(iter(sources)) if sources else "",
        "segments": segments,
    }


def render_sheet(gt: dict, out_path: Path, cols: int = 5) -> None:
    """Numbered midpoint-thumbnail contact sheet for annotation."""
    cap = cv2.VideoCapture(gt["source"])
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    tiles = []
    for s in gt["segments"]:
        mid = (s["source_start_s"] + s["source_end_s"]) / 2.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((180, 320, 3), np.uint8)
        img = cv2.resize(frame, (320, 180))
        label = (f"#{s['index']:02d}  {s['source_start_s']:.1f}-"
                 f"{s['source_end_s']:.1f}s ({s['duration_s']:.1f}s)")
        cv2.rectangle(img, (0, 0), (320, 22), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        tiles.append(img)
    cap.release()
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros((180, 320, 3), np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("draft_name")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--annotate", type=Path, default=None,
                    help="JSON with groups/drop/fades keyed by sheet index")
    args = ap.parse_args()

    gt = read_draft_segments(args.draft_name)
    print(f"draft {gt['draft']}: {len(gt['segments'])} segments of "
          f"{Path(gt['source']).name}")
    for s in gt["segments"]:
        note = " (fade?)" if s["duration_s"] < 1.0 else ""
        print(f"  #{s['index']:02d} {s['source_start_s']:7.2f}-"
              f"{s['source_end_s']:7.2f}s  {s['duration_s']:5.2f}s{note}")

    if args.annotate and args.annotate.exists():
        ann = json.loads(args.annotate.read_text())
        drop = set(ann.get("drop", []))
        fades = set(ann.get("fades", []))
        group_of = {}
        for gid, idxs in ann.get("groups", {}).items():
            for i in idxs:
                group_of[i] = gid
        for s in gt["segments"]:
            i = s["index"]
            s["role"] = ("fade" if i in fades
                         else "drop" if i in drop
                         else "keep")
            s["group"] = group_of.get(i, "")
        gt["annotated"] = True

    if args.sheet:
        render_sheet(gt, args.sheet)
        print(f"sheet -> {args.sheet}")
    if args.out:
        args.out.write_text(json.dumps(gt, indent=1))
        print(f"manifest -> {args.out}")


if __name__ == "__main__":
    main()

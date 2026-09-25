"""Build the self-contained Ball Truth Lab viewer HTML from one or more
per-clip ``results.json`` files (see ``prototypes/ball_hybrid_poc/CONTRACT.md``
for the shape).

Usage:
    .venv311/bin/python prototypes/ball_hybrid_poc/viewer/build_viewer.py \
        --results prototypes/ball_hybrid_poc/viewer/mock_data \
        --out prototypes/ball_hybrid_poc/viewer/ball_truth_lab.html

``--results`` accepts a directory (searched recursively for ``results.json``
or ``*_results.json`` files), a single file, or a list of files. Data is
rounded (1 cm for 3-D positions, 0.1 px for pixel-space values) before being
embedded, to keep the output file small.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TEMPLATE_PATH = Path(__file__).with_name("template.html")
DATA_PLACEHOLDER = "/*__BALL_TRUTH_LAB_DATA__*/"

# Field-name -> decimal-places rounding table. Applied to any bare float
# found under a matching key while walking the JSON tree. `None` keys use a
# length-based fallback (see `_precision_for`).
_PX_KEYS = {"uv", "image_xy", "u", "v", "broadcast_px_error", "side_px_error"}
_METRE_KEYS = {
    "xyz", "xyz_gt", "x", "y", "z", "centre_xyz_per_frame",
    "p50", "p95", "max", "contact_gap", "ground_float_sink",
    "anchor_heldout_err_m", "fix_err_m",
}
_FRACTION_KEYS = {"pct_le_20cm", "conf"}


def _precision_for(key: str | None, length: int | None = None) -> int:
    if key in _PX_KEYS:
        return 1  # 0.1 px
    if key in _METRE_KEYS:
        return 2  # 1 cm
    if key in _FRACTION_KEYS:
        return 3
    if length == 3:
        return 2  # unlabelled [x, y, z]-shaped triples -> treat as metres
    if length == 2:
        return 1  # unlabelled [u, v]-shaped pairs -> treat as pixels
    return 3


def _round_tree(obj: Any, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {k: _round_tree(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            prec = _precision_for(key, length=len(obj))
            return [round(v, prec) if isinstance(v, float) else v for v in obj]
        return [_round_tree(v, key) for v in obj]
    if isinstance(obj, float):
        return round(obj, _precision_for(key))
    return obj


def _find_results_files(results_arg: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in results_arg:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.rglob("results.json")) + sorted(p.rglob("*_results.json"))
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"--results path not found: {p}")
    # de-dupe while preserving order
    seen = set()
    unique = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            unique.append(f)
    return unique


def load_clip_results(results_arg: list[str]) -> dict[str, Any]:
    """Load and merge per-clip results.json files into {clip_id: results}."""
    files = _find_results_files(results_arg)
    if not files:
        raise FileNotFoundError(f"no results.json files found under {results_arg}")
    clips: dict[str, Any] = {}
    for f in files:
        data = json.loads(f.read_text())
        clip_id = data.get("clip_id") or f.parent.name
        clips[clip_id] = _round_tree(data)
    return clips


def build_html(clips: dict[str, Any]) -> str:
    template = TEMPLATE_PATH.read_text()
    if DATA_PLACEHOLDER not in template:
        raise ValueError(
            f"template.html is missing the data placeholder {DATA_PLACEHOLDER!r}"
        )
    payload = json.dumps(clips, separators=(",", ":"))
    # Guard against a stray "</script>" inside the JSON breaking the inline
    # <script> block the payload is embedded in.
    payload = payload.replace("</script", "<\\/script")
    return template.replace(DATA_PLACEHOLDER, payload, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results", nargs="+", required=True,
        help="Directory (searched recursively) or file(s) of per-clip results.json.",
    )
    ap.add_argument("--out", required=True, help="Output HTML path.")
    args = ap.parse_args()

    clips = load_clip_results(args.results)
    html = build_html(clips)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.2f} MB, {len(clips)} clip(s): {', '.join(sorted(clips))})")
    if size_mb > 8:
        print("WARNING: output exceeds the 8 MB artifact-page budget")


if __name__ == "__main__":
    main()

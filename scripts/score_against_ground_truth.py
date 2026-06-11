"""Score prepare_shots output against CapCut-derived ground truth.

Compares the pipeline's shot boundaries with the operator's hand-cut
segments (from ``capcut_ground_truth.py``):

- **boundary recall**: every GT cut (segment start, excluding t=0)
  should be matched by a pipeline span edge within ``--tol`` frames.
  GT fade segments are handled as intervals: the fade's start and end
  each count as one expected boundary (the pipeline excludes blend
  frames, so its span edges should bracket the fade).
- **boundary precision**: pipeline span edges not near any GT cut are
  over-splits.
- **fade exclusion**: no pipeline span may overlap a GT fade interval.

Usage:
    python scripts/score_against_ground_truth.py GT.json OUTPUT_DIR \
        [--tol 3] [--fade-max-s 1.0]

GT fades default to "segments shorter than --fade-max-s"; an annotated
GT (``role: fade``) overrides that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_gt(path: Path, fade_max_s: float) -> tuple[list[dict], float]:
    gt = json.loads(path.read_text())
    segments = gt["segments"]
    for s in segments:
        if "role" not in s:
            s["role"] = "fade" if s["duration_s"] < fade_max_s else "shot"
    return segments, gt.get("fps", 25.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--fade-max-s", type=float, default=1.0)
    args = ap.parse_args()

    segments, fps_hint = load_gt(args.gt, args.fade_max_s)
    manifest = json.loads(
        (args.output_dir / "shots" / "shots_manifest.json").read_text())
    fps = manifest.get("fps") or fps_hint

    def f(t_s: float) -> int:
        return int(round(t_s * fps))

    # Pipeline span edges on the source timeline.
    edges: set[int] = set()
    spans = []
    for s in manifest["shots"]:
        a, b = f(s["source_start_s"]), f(s["source_end_s"])
        spans.append((s["id"], a, b))
        edges.add(a)
        edges.add(b)

    def matched(frame: int) -> bool:
        return any(abs(frame - e) <= args.tol for e in edges)

    # Expected boundaries: every transition between GT segments. For a
    # shot|shot transition that's one cut; fade segments contribute
    # their start and end (the pipeline should bracket the blend).
    expected: list[tuple[str, int]] = []
    for i, s in enumerate(segments):
        if i > 0:
            expected.append((f"#{i:02d} start", f(s["source_start_s"])))
    hits = [(name, fr) for name, fr in expected if matched(fr)]
    misses = [(name, fr) for name, fr in expected if not matched(fr)]

    # Precision: pipeline edges near any GT segment edge.
    gt_edges = {f(s["source_start_s"]) for s in segments}
    gt_edges |= {f(s["source_end_s"]) for s in segments}
    extra = sorted(
        e for e in edges
        if not any(abs(e - g) <= args.tol for g in gt_edges)
    )

    # Fade exclusion: pipeline spans overlapping GT fade intervals.
    fade_violations = []
    for s in segments:
        if s["role"] != "fade":
            continue
        lo, hi = f(s["source_start_s"]) + args.tol, f(s["source_end_s"]) - args.tol
        for sid, a, b in spans:
            if a < hi and b > lo:
                fade_violations.append((sid, s["index"]))

    n_exp = len(expected)
    print(f"GT: {len(segments)} segments "
          f"({sum(1 for s in segments if s['role'] == 'fade')} fades) | "
          f"pipeline: {len(spans)} shots")
    print(f"boundary recall   : {len(hits)}/{n_exp} "
          f"({100 * len(hits) / max(1, n_exp):.0f}%)")
    print(f"over-split edges  : {len(extra)}")
    print(f"fade overlaps     : {len(fade_violations)}")
    if misses:
        print("\nmissed boundaries:")
        for name, fr in misses:
            print(f"  {name} @ {fr / fps:7.2f}s (frame {fr})")
    if extra:
        print("\nover-split edges (pipeline edge, no GT edge nearby):")
        for e in extra[:20]:
            print(f"  {e / fps:7.2f}s (frame {e})")
    if fade_violations:
        print("\nspans overlapping GT fades:")
        for sid, idx in fade_violations[:10]:
            print(f"  {sid} overlaps fade #{idx:02d}")


if __name__ == "__main__":
    main()


def score_keeps(gt_keeps_path: Path, output_dir: Path,
                early_tol: int = 2, late_tol: int = 12) -> None:
    """Keep-centric reproduction score ("leads to those clips").

    For each ground-truth keep clip, the pipeline must produce exactly
    one shot covering it: start within [GT_start - early_tol,
    GT_start + late_tol] (early = fade ghosting, strictly bounded;
    late = safe over-trim, tolerated), end within the mirrored window,
    and no extra pipeline cut strictly inside the keep.
    """
    keeps = json.loads(gt_keeps_path.read_text())
    manifest = json.loads(
        (output_dir / "shots" / "shots_manifest.json").read_text())
    fps = manifest.get("fps") or keeps.get("fps", 25.0)

    def f(t_s: float) -> int:
        return int(round(t_s * fps))

    spans = [(s["id"], f(s["source_start_s"]), f(s["source_end_s"]))
             for s in manifest["shots"]]

    passed, problems = 0, []
    for seg in keeps["segments"]:
        a, b = f(seg["source_start_s"]), f(seg["source_end_s"])
        overlapping = [(sid, x, y) for sid, x, y in spans
                       if x < b and y > a]
        label = f"keep#{seg['index']:02d} {seg['source_start_s']:.1f}s"
        if not overlapping:
            problems.append(f"{label}: NO pipeline shot")
            continue
        if len(overlapping) > 1:
            problems.append(
                f"{label}: SPLIT across {len(overlapping)} shots "
                f"({', '.join(o[0] for o in overlapping)})")
            continue
        sid, x, y = overlapping[0]
        issues = []
        if x < a - early_tol:
            issues.append(f"starts {a - x}f early (ghost risk)")
        if x > a + late_tol:
            issues.append(f"starts {x - a}f late")
        if y > b + early_tol:
            issues.append(f"ends {y - b}f late (ghost risk)")
        if y < b - late_tol:
            issues.append(f"ends {b - y}f early")
        if issues:
            problems.append(f"{label} ({sid}): " + "; ".join(issues))
        else:
            passed += 1
    n = len(keeps["segments"])
    print(f"\nkeep reproduction: {passed}/{n} "
          f"({100 * passed / max(1, n):.0f}%)")
    for p in problems:
        print(f"  {p}")

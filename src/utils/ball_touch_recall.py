"""Touch-detection recall/precision against a manual anchor set used as
pseudo-ground-truth (e.g. gberch's 59 hand-placed anchors).

Pure and torch-free: ``(frame, player_id, bone)`` triples in, metrics out.
Lets every detection-improvement phase be measured without new labelling
(see docs/superpowers/specs/2026-06-15-ball-detection-direction-changes-design.md §7).
"""

from __future__ import annotations

import json
from pathlib import Path

Touch = tuple[int, str, str]


def touches_from_anchor_set(path: str | Path) -> list[Touch]:
    """Load a BallAnchorSet JSON and return the ``player_touch`` triples
    ``(frame, player_id, bone)`` in frame order."""
    data = json.loads(Path(path).read_text())
    out: list[Touch] = []
    for a in data.get("anchors", []):
        if a.get("state") == "player_touch":
            out.append((
                int(a["frame"]),
                str(a.get("player_id") or ""),
                str(a.get("bone") or ""),
            ))
    return sorted(out, key=lambda t: t[0])


def match_touches(
    manual: list[Touch],
    auto: list[Touch],
    *,
    frame_tol: int = 2,
    require_bone: bool = True,
) -> dict:
    """Greedy 1:1 matching of auto touches to manual touches.

    An auto touch matches a manual one when ``|Δframe| <= frame_tol`` and
    (when ``require_bone``) the bone agrees. Each manual touch can be
    claimed at most once (nearest unclaimed auto wins). Returns counts and
    recall/precision.
    """
    manual_sorted = sorted(manual, key=lambda t: t[0])
    auto_sorted = sorted(auto, key=lambda t: t[0])
    claimed = [False] * len(manual_sorted)
    tp = 0
    for af, ap, ab in auto_sorted:
        best_j = -1
        best_d = frame_tol + 1
        for j, (mf, mp, mb) in enumerate(manual_sorted):
            if claimed[j]:
                continue
            d = abs(af - mf)
            if d > frame_tol:
                continue
            if require_bone and ab != mb:
                continue
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            claimed[best_j] = True
            tp += 1
    n_manual = len(manual_sorted)
    n_auto = len(auto_sorted)
    fp = n_auto - tp
    return {
        "n_manual": n_manual,
        "n_auto": n_auto,
        "true_positive": tp,
        "false_positive": fp,
        "recall": (tp / n_manual) if n_manual else 0.0,
        "precision": (tp / n_auto) if n_auto else 0.0,
    }


def dismissed_touches_from_anchor_set(path: str | Path) -> list[Touch]:
    """The dismissed ``player_touch`` triples from a manual sidecar."""
    data = json.loads(Path(path).read_text())
    out: list[Touch] = []
    for d in data.get("dismissed_auto", []):
        if d.get("state") == "player_touch":
            out.append((int(d["frame"]), str(d.get("player_id") or ""),
                        str(d.get("bone") or "")))
    return sorted(out, key=lambda t: t[0])


def fp_breakdown(
    auto: list[Touch],
    manual: list[Touch],
    dismissed: list[Touch],
    *,
    frame_tol: int = 2,
) -> dict:
    """Partition strict-match false positives into operator-dismissed
    (reviewed, confirmed wrong) vs unreviewed (unknown — possibly real
    touches the manual set never annotated)."""
    manual_sorted = sorted(manual, key=lambda t: t[0])
    claimed = [False] * len(manual_sorted)
    fps: list[Touch] = []
    for af, ap, ab in sorted(auto, key=lambda t: t[0]):
        best_j = -1
        best_d = frame_tol + 1
        for j, (mf, _mp, mb) in enumerate(manual_sorted):
            if claimed[j]:
                continue
            d = abs(af - mf)
            if d > frame_tol or ab != mb:
                continue
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            claimed[best_j] = True
        else:
            fps.append((af, ap, ab))
    dismissed_set = set(dismissed)
    n_dismissed = sum(1 for fp in fps if fp in dismissed_set)
    return {
        "fp_total": len(fps),
        "fp_dismissed": n_dismissed,
        "fp_unreviewed": len(fps) - n_dismissed,
    }


__all__ = [
    "match_touches",
    "touches_from_anchor_set",
    "dismissed_touches_from_anchor_set",
    "fp_breakdown",
    "Touch",
]

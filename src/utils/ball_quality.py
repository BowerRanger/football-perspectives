"""Ball-stage quality payload for the dashboard timeline (spec §5.1).

Aggregates the three per-shot sidecars the ball stage already writes
(observations / diag / keyframes) into one compact payload the ball
anchor editor renders as a per-frame quality strip plus a ranked
"annotate here next" list. Pure and torch-free: the web endpoint only
parses JSON and delegates here.
"""

from __future__ import annotations

DEFAULT_MIN_GAP_FRAMES = 12
_MAX_ANNOTATE_ITEMS = 10
# Underconstrained flight spans are the operator's highest-value fix (one
# bracketing anchor resolves the whole arc); plain detection gaps rank below.
_GAP_SEVERITY_WEIGHT = 0.5


def detection_gaps(
    frames: list[dict], min_gap_frames: int,
) -> list[tuple[int, int]]:
    """Maximal runs of >= ``min_gap_frames`` consecutive frames with no
    accepted detection (zero confidence or IMM gap-fill)."""
    gaps: list[tuple[int, int]] = []
    run_start: int | None = None
    prev: int | None = None
    for rec in sorted(frames, key=lambda r: int(r["frame"])):
        f = int(rec["frame"])
        missing = (
            float(rec.get("confidence", 0.0)) <= 0.0
            or bool(rec.get("gap_fill", False))
        )
        contiguous = prev is not None and f == prev + 1
        if missing and run_start is not None and contiguous:
            pass  # run continues
        elif missing:
            if run_start is not None and prev is not None \
                    and prev - run_start + 1 >= min_gap_frames:
                gaps.append((run_start, prev))
            run_start = f
        else:
            if run_start is not None and prev is not None \
                    and prev - run_start + 1 >= min_gap_frames:
                gaps.append((run_start, prev))
            run_start = None
        prev = f
    if run_start is not None and prev is not None \
            and prev - run_start + 1 >= min_gap_frames:
        gaps.append((run_start, prev))
    return gaps


def rank_annotate_next(
    underconstrained_spans: list[dict], gaps: list[tuple[int, int]],
) -> list[dict]:
    """Ranked "annotate here next" items, most valuable first."""
    items: list[dict] = []
    for span in underconstrained_spans:
        start, end = int(span["start"]), int(span["end"])
        residual = float(span.get("residual_px") or 0.0)
        items.append({
            "start": start, "end": end,
            "reason": "underconstrained_flight",
            "severity": (end - start + 1) * (1.0 + residual / 10.0),
        })
    for start, end in gaps:
        items.append({
            "start": start, "end": end,
            "reason": "detection_gap",
            "severity": _GAP_SEVERITY_WEIGHT * (end - start + 1),
        })
    items.sort(key=lambda it: (-it["severity"], it["start"]))
    return items[:_MAX_ANNOTATE_ITEMS]


def build_quality_payload(
    observations: dict | None,
    diag: dict | None,
    keyframes: dict | None,
    *,
    min_gap_frames: int = DEFAULT_MIN_GAP_FRAMES,
) -> dict:
    """One payload for GET /ball-quality/{shot_id}; every input optional."""
    obs = observations or {}
    dg = diag or {}
    kf = keyframes or {}
    obs_frames = list(obs.get("frames", []))
    spans = list(dg.get("underconstrained_spans", []))
    gaps = detection_gaps(obs_frames, min_gap_frames)
    return {
        "n_frames": len(obs_frames),
        "fps": obs.get("fps"),
        "frames": [
            {
                "frame": int(r["frame"]),
                "confidence": float(r.get("confidence", 0.0)),
                "gap_fill": bool(r.get("gap_fill", False)),
                "source": r.get("source", "none"),
            }
            for r in obs_frames
        ],
        "events": list(dg.get("events", [])),
        "underconstrained_spans": spans,
        "segments": [
            {
                "start_frame": s.get("start_frame"),
                "end_frame": s.get("end_frame"),
                "kind": s.get("kind"),
            }
            for s in kf.get("segments", [])
        ],
        "detection_coverage": dg.get("detection_coverage"),
        "annotate_next": rank_annotate_next(spans, gaps),
    }

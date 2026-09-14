"""Static/moving camera-model decision for one shot.

The camera stage's ``static_camera`` contract assumes ONE shared
world-frame camera centre for the whole clip — correct for a fixed
stadium-mounted PTZ rig, but wrong for a spidercam/wirecam replay angle
that genuinely translates. This module decides, from evidence, whether
a shot's anchors are consistent with a single shared centre
(:func:`evaluate_static_gate`), and provides the reporting-only helpers
used alongside that decision: leave-one-out click triage
(:func:`find_loo_click_culprit`) and weak-support gap surfacing
(:func:`find_weak_support_gaps`). :func:`parse_static_camera_mode`
handles the tri-state ``camera.static_camera`` config value.

Diagnosed and built against gberch-2 (2026-09-09): a spidercam replay
angle whose camera translates ~10m across the shot while zooming
fx≈990→2680. See
``docs/superpowers/specs/2026-09-09-moving-camera-support.md`` for the
full evidence write-up.
"""

from __future__ import annotations

import logging
from typing import Callable, NamedTuple

import numpy as np

from src.schemas.anchor import Anchor
from src.utils.anchor_solver import JointSolution, _is_degenerate_solo, _is_rich

logger = logging.getLogger(__name__)


# ── Tri-state config parse ────────────────────────────────────────────────


def parse_static_camera_mode(value: object) -> str:
    """Parse the tri-state ``camera.static_camera`` config value.

    Accepts the new string form (``"auto"``, ``"true"``/``"static"``,
    ``"false"``/``"moving"``, case-insensitive) and the legacy boolean
    form every config before this feature used: ``True`` forces today's
    unconditional static path (the back-compat escape hatch), ``False``
    forces today's unconditional moving/free-translation path. ``None``
    (key absent) is ``"auto"``.

    Returns one of ``"auto"``, ``"static"``, ``"moving"``. Raises
    ``ValueError`` on any other value so a typo'd config fails loudly
    rather than silently picking a mode.
    """
    if isinstance(value, bool):
        return "static" if value else "moving"
    if value is None:
        return "auto"
    s = str(value).strip().lower()
    if s == "auto":
        return "auto"
    if s in ("true", "static"):
        return "static"
    if s in ("false", "moving"):
        return "moving"
    raise ValueError(
        f"camera.static_camera must be 'auto', 'true', or 'false' "
        f"(or a legacy bool); got {value!r}"
    )


# ── Consistency gate ───────────────────────────────────────────────────────


class StaticGateResult(NamedTuple):
    """Outcome of testing whether a shot's anchors share one camera centre.

    ``per_anchor`` maps frame -> (solo_residual_px, clamped_residual_px)
    for every rich, non-degenerate anchor that was actually evaluated —
    empty when there was fewer than one such anchor to compare (in which
    case ``holds`` defaults to ``True``: nothing contradicts staticness).
    """

    holds: bool
    worst_frame: int | None
    worst_solo_px: float
    worst_clamped_px: float
    centre_spread_m: float
    per_anchor: dict[int, tuple[float, float]]


def _rich_nondegenerate_frames(
    anchors: tuple[Anchor, ...], sol: JointSolution,
) -> list[int]:
    """Frames that are both a "rich" anchor (enough non-coplanar
    landmarks to solo-solve) AND whose CURRENT (K, R, t) in ``sol`` is
    physically plausible — i.e. a trustworthy solo-quality baseline for
    the gate. Excludes anchors Task A's hardening already dropped from
    solo seeding (they fall through to a t-fixed fallback instead, which
    isn't a comparable "solo" residual)."""
    out = []
    for a in anchors:
        if not _is_rich(a):
            continue
        got = sol.per_anchor_KRt.get(a.frame)
        if got is None:
            continue
        K, _R, t = got
        fx = float(K[0, 0])
        if _is_degenerate_solo(np.asarray(t, dtype=np.float64), fx):
            continue
        out.append(a.frame)
    return out


def evaluate_static_gate(
    anchors: tuple[Anchor, ...],
    sol: JointSolution,
    relocked: JointSolution,
    *,
    residual_ratio: float = 3.0,
    residual_floor_px: float = 8.0,
) -> StaticGateResult:
    """Does a single shared camera centre explain every rich anchor?

    Compares, for each rich non-degenerate anchor, its own free (solo)
    reprojection residual (``sol.per_anchor_residual_px`` — computed
    before any shared-centre relock) against its residual once C is
    clamped to the candidate shared centre (``relocked``, typically
    ``refine_with_shared_translation``'s output on the same anchors).
    The static-camera model holds only if EVERY such anchor stays under
    ``max(residual_ratio * solo_residual, residual_floor_px)``.

    An anchor that only fits well with its OWN free translation is
    evidence the camera body actually moved — not evidence of a bad
    click (:func:`find_loo_click_culprit` is the tool for that).
    """
    frames = _rich_nondegenerate_frames(anchors, sol)
    per_anchor: dict[int, tuple[float, float]] = {}
    for f in frames:
        solo_px = sol.per_anchor_residual_px.get(f)
        clamped_px = relocked.per_anchor_residual_px.get(f)
        if solo_px is None or clamped_px is None:
            continue
        if not np.isfinite(solo_px) or not np.isfinite(clamped_px):
            continue
        per_anchor[f] = (float(solo_px), float(clamped_px))

    if not per_anchor:
        return StaticGateResult(
            holds=True, worst_frame=None, worst_solo_px=0.0,
            worst_clamped_px=0.0, centre_spread_m=0.0, per_anchor={},
        )

    holds = True
    worst_frame: int | None = None
    worst_margin = float("-inf")
    for f, (solo_px, clamped_px) in per_anchor.items():
        thresh = max(residual_ratio * solo_px, residual_floor_px)
        if clamped_px > thresh:
            holds = False
        margin = clamped_px - thresh
        if margin > worst_margin:
            worst_margin = margin
            worst_frame = f

    assert worst_frame is not None  # per_anchor is non-empty here
    worst_solo_px, worst_clamped_px = per_anchor[worst_frame]

    centre_spread_m = 0.0
    if relocked.camera_centre is not None:
        C_shared = np.asarray(relocked.camera_centre, dtype=np.float64)
        dists = []
        for f in per_anchor:
            K, R, t = sol.per_anchor_KRt[f]
            C_f = -np.asarray(R, dtype=np.float64).T @ np.asarray(t, dtype=np.float64)
            dists.append(float(np.linalg.norm(C_f - C_shared)))
        centre_spread_m = max(dists) if dists else 0.0

    return StaticGateResult(
        holds=holds, worst_frame=worst_frame,
        worst_solo_px=worst_solo_px, worst_clamped_px=worst_clamped_px,
        centre_spread_m=centre_spread_m, per_anchor=per_anchor,
    )


# ── Leave-one-out click triage (reporting only) ───────────────────────────


class ClickTriageResult(NamedTuple):
    anchor_frame: int
    culprit_name: str
    residual_with_px: float
    residual_without_px: float


def find_loo_click_culprit(
    anchor: Anchor,
    solve_fn: Callable[[Anchor], float],
    *,
    baseline_residual_px: float | None = None,
    min_collapse_ratio: float = 5.0,
    accept_below_px: float = 15.0,
) -> ClickTriageResult | None:
    """Leave-one-out triage over ``anchor``'s point landmarks.

    ``solve_fn(anchor_without_one_landmark) -> residual_px`` is supplied
    by the caller so this stays model-agnostic: pass a closure that
    re-solves under whichever model the shot actually uses (a static
    C-fixed solve, a moving free solve, whatever the camera stage
    chose). Drops each landmark in turn and reports the one whose
    removal both collapses the residual by ``min_collapse_ratio``x AND
    leaves a trustworthy fit (``accept_below_px``) — a modest ratio or a
    still-bad residual isn't a confident single-click diagnosis, it's a
    genuine multi-anchor inconsistency (e.g. gberch-2's frame 0, where no
    single dropped click gets anywhere close: 37 -> 33.5px best).

    This function is flag-only: it never mutates ``anchor`` or drops
    anything itself. Manual clicks are operator data and always win —
    the caller reports the finding and leaves fixing it to a human in
    the anchor editor.

    ``baseline_residual_px``, if supplied, MUST come from the same
    ``solve_fn`` model as the leave-one-out candidates (an apples-to-
    apples comparison) — passing a residual computed under a DIFFERENT,
    more-constrained model (e.g. a joint bounded-motion fit) as the
    baseline while leave-one-out re-solves each candidate freely will
    always "collapse" the ratio for reasons having nothing to do with
    any click, since removing the extra model constraint alone already
    improves the fit. Leave it ``None`` to have this function derive it
    itself via ``solve_fn(anchor)``, which is always consistent.

    Returns ``None`` when there are <2 landmarks (nothing to leave out),
    the baseline residual is non-positive/non-finite, the baseline is
    already at or under ``accept_below_px`` (nothing to fix), or no drop
    meets both gates.
    """
    if len(anchor.landmarks) < 2:
        return None
    if baseline_residual_px is None:
        baseline_residual_px = solve_fn(anchor)
    if not np.isfinite(baseline_residual_px) or baseline_residual_px <= 0:
        return None
    if baseline_residual_px <= accept_below_px:
        # Already an acceptable fit with every click — nothing to
        # diagnose (this also protects against a caller-supplied
        # baseline from a stricter model than solve_fn, which would
        # otherwise look like an artificially huge collapse).
        return None

    best: ClickTriageResult | None = None
    for i, lm in enumerate(anchor.landmarks):
        reduced = Anchor(
            frame=anchor.frame,
            landmarks=tuple(
                l for j, l in enumerate(anchor.landmarks) if j != i
            ),
            lines=anchor.lines,
        )
        res = solve_fn(reduced)
        if not np.isfinite(res) or res <= 0:
            continue
        ratio = baseline_residual_px / res
        if ratio < min_collapse_ratio or res > accept_below_px:
            continue
        if best is None or res < best.residual_without_px:
            best = ClickTriageResult(
                anchor_frame=anchor.frame,
                culprit_name=lm.name,
                residual_with_px=float(baseline_residual_px),
                residual_without_px=float(res),
            )
    return best


def anchors_needing_click_triage(
    per_anchor_residual_px: dict[int, float],
    *,
    flag_threshold_px: float,
    neighbour_ratio: float = 2.0,
) -> list[int]:
    """Frames whose residual exceeds ``flag_threshold_px`` AND stands far
    above the other anchors' (``neighbour_ratio``x their median) —
    candidates for :func:`find_loo_click_culprit`.

    A uniformly-bad clip (every anchor high, e.g. a genuinely-moving
    camera solved under a forced static model) is a model problem, not a
    single-click problem, so it's deliberately excluded: the "far above
    its neighbours" test only fires on an outlier among otherwise-good
    anchors.
    """
    out: list[int] = []
    for f, r in per_anchor_residual_px.items():
        others = [v for k, v in per_anchor_residual_px.items() if k != f]
        if not others:
            continue
        med_others = float(np.median(others))
        if r > flag_threshold_px and r > neighbour_ratio * max(med_others, 1e-6):
            out.append(f)
    return sorted(out)


# ── Weak-support gap surfacing ─────────────────────────────────────────────


class WeakSupportGap(NamedTuple):
    start_frame: int
    end_frame: int
    suggested_frame: int
    gap_frames: int
    mean_confidence: float


def find_weak_support_gaps(
    anchor_frames: list[int],
    per_frame_confidence: list[float],
    *,
    max_gap_frames: int = 60,
    weak_confidence_below: float = 0.6,
) -> list[WeakSupportGap]:
    """Between-anchor spans, in a moving-camera shot, too wide and too
    unsupported to trust the smooth centre-interpolation alone.

    ``per_frame_confidence`` is indexed by absolute frame number (same
    length/order as the camera stage's own per-frame confidence array).
    A span between consecutive anchors is flagged when it is longer than
    ``max_gap_frames`` AND its mean confidence stays below
    ``weak_confidence_below`` — a span that line-extraction/propagation
    already rescued to high per-frame confidence doesn't need a new
    anchor even if it's long.
    """
    gaps: list[WeakSupportGap] = []
    for a, b in zip(anchor_frames, anchor_frames[1:]):
        gap = b - a - 1
        if gap <= max_gap_frames:
            continue
        span = per_frame_confidence[a + 1 : b]
        mean_conf = float(np.mean(span)) if span else 0.0
        if mean_conf >= weak_confidence_below:
            continue
        gaps.append(WeakSupportGap(
            start_frame=a, end_frame=b, suggested_frame=(a + b) // 2,
            gap_frames=gap, mean_confidence=mean_conf,
        ))
    return gaps

"""Ball pixel-track outlier rejection (cleanup item 1).

Diagnostics on gberch showed the track feeding touch detection is polluted
by (a) out-of-image positions (negative / beyond frame, e.g. IMM gap-fill
extrapolations of a fast ball, or mis-mapped detections) and (b) isolated
teleport spikes. Both create spurious direction-change corners far from any
player. This removes them before segmentation/event detection.

NOTE (honest scope): a *stable, self-consistent* run of wrong-object
detections (e.g. a confident WASB false positive pinned to the frame top
across several frames) is NOT a spike and survives this filter — that needs a
spatial/pose prior or a better detector (re-review / fine-tuning).
"""

from __future__ import annotations

from math import hypot

UV = tuple[float, float]


def clean_pixel_track(
    uvs: dict[int, UV],
    *,
    image_size: tuple[int, int] | None = None,
    max_jump_px: float = 250.0,
) -> dict[int, UV]:
    """Return ``uvs`` with out-of-image positions and isolated teleport
    spikes removed.

    ``max_jump_px`` is the per-frame ball-speed budget; the budget between two
    observations scales with their frame gap. A point is a *spike* when it
    exceeds the budget from BOTH neighbours while the neighbours are mutually
    consistent (so it's the point, not the neighbours, that's wrong).
    """
    items = [
        (f, (float(uv[0]), float(uv[1])))
        for f, uv in sorted(uvs.items()) if uv is not None
    ]
    if image_size is not None:
        w, h = image_size
        items = [
            (f, uv) for f, uv in items
            if 0.0 <= uv[0] < w and 0.0 <= uv[1] < h
        ]

    kept: dict[int, UV] = {}
    for i, (f, uv) in enumerate(items):
        prev = items[i - 1] if i > 0 else None
        nxt = items[i + 1] if i < len(items) - 1 else None
        if prev is not None and nxt is not None:
            pf, puv = prev
            nf, nuv = nxt
            budget_p = max_jump_px * max(1, f - pf)
            budget_n = max_jump_px * max(1, nf - f)
            budget_pn = max_jump_px * max(1, nf - pf)
            d_prev = hypot(uv[0] - puv[0], uv[1] - puv[1])
            d_next = hypot(uv[0] - nuv[0], uv[1] - nuv[1])
            d_pn = hypot(puv[0] - nuv[0], puv[1] - nuv[1])
            if d_prev > budget_p and d_next > budget_n and d_pn <= budget_pn:
                continue  # isolated spike — neighbours agree, this point doesn't
        kept[f] = uv
    return kept


__all__ = ["clean_pixel_track"]


def veto_click_contradicted(
    uvs: dict[int, UV],
    clicks: dict[int, UV],
    *,
    max_px: float = 60.0,
    max_gap_frames: int = 6,
) -> dict[int, UV]:
    """Remove observations the operator's clicks prove false.

    Between two clicks ≤ ``max_gap_frames`` apart the true ball lies near
    the interpolated click path; a detection farther than ``max_px`` from
    it is a known false positive (e.g. a detector locked on a static
    object at the ball's old position) and must not feed the IMM, the
    event search, or any fit. Frames without a close click bracket are
    untouched — the operator has not spoken there. (Sub-20cm campaign
    W5s; the eval harness applies the same rule to grading GT.)
    """
    ordered = sorted(clicks.items())
    expected: dict[int, UV] = {f: uv for f, uv in ordered}
    for (fa, ua), (fb, ub) in zip(ordered, ordered[1:]):
        if 0 < fb - fa <= max_gap_frames:
            for f in range(fa + 1, fb):
                s = (f - fa) / (fb - fa)
                expected[f] = (ua[0] + (ub[0] - ua[0]) * s,
                               ua[1] + (ub[1] - ua[1]) * s)
    out: dict[int, UV] = {}
    for f, uv in uvs.items():
        exp = expected.get(f)
        if (exp is not None and uv is not None
                and (uv[0] - exp[0]) ** 2 + (uv[1] - exp[1]) ** 2
                > max_px ** 2):
            continue
        out[f] = uv
    return out


def veto_flight_contradicted(
    uvs: dict[int, UV],
    arc_px: dict[int, UV],
    *,
    mpp_by_frame: dict[int, float],
    far_px: float = 60.0,
    static_px_per_frame: float = 1.5,
    static_window: int = 2,
    max_ball_speed_m_s: float = 40.0,
    fps: float = 25.0,
    max_island_frames: int = 2,
    context_uvs: dict[int, UV] | None = None,
) -> list[int]:
    """Frames whose detection a BALLISTIC span proves non-ball (W9).

    Interior detections never shape a two-knot arc, so the arc's
    reprojection can honestly judge them. A detection ``far_px`` off the
    arc is vetoed when it is physically impossible as a ball:

    - a **teleport island**: a run of ≤ ``max_island_frames`` frames whose
      pixel steps to every neighbouring detection exceed what a ball at
      that depth can travel per frame (oscillating goal-mouth lock-ons);
    - a **static lock-on**: local drift ≤ ``static_px_per_frame`` while
      the arc moves away (net corners, furniture).

    A far-but-coherent run is NEVER vetoed — it may be a real deflection
    the solve missed, and the metric must keep charging the track for it.
    Returns the sorted list of vetoed frames.
    """
    frames = sorted(f for f in uvs if uvs[f] is not None)
    if not frames:
        return []

    def _step_feasible(fa: int, fb: int) -> bool:
        ua, ub = uvs[fa], uvs[fb]
        gap = abs(fb - fa)
        mpp = max(mpp_by_frame.get(fa, 0.02), 1e-6)
        budget = (max_ball_speed_m_s / fps) * gap / mpp
        return ((ua[0] - ub[0]) ** 2 + (ua[1] - ub[1]) ** 2) <= budget ** 2

    # Connected components under feasible-step adjacency.
    components: list[list[int]] = [[frames[0]]]
    for fa, fb in zip(frames, frames[1:]):
        if fb - fa <= static_window + 1 and _step_feasible(fa, fb):
            components[-1].append(fb)
        else:
            components.append([fb])

    def _far(f: int) -> bool:
        uv, arc = uvs.get(f), arc_px.get(f)
        if uv is None or arc is None:
            return False
        return ((uv[0] - arc[0]) ** 2 + (uv[1] - arc[1]) ** 2) > far_px ** 2

    def _context_reachable(comp: list[int]) -> bool:
        """A run feasibly connected to evidence OUTSIDE the vetoable set
        (span-boundary anchors/detections) is reachable by a real ball —
        never an island, whatever its arc distance."""
        for f in (comp[0], comp[-1]):
            for g, guv in (context_uvs or {}).items():
                gap = abs(g - f)
                if not 0 < gap <= static_window + 1:
                    continue
                mpp = max(mpp_by_frame.get(f, 0.02), 1e-6)
                budget = (max_ball_speed_m_s / fps) * gap / mpp
                uv = uvs[f]
                if ((uv[0] - guv[0]) ** 2 + (uv[1] - guv[1]) ** 2) \
                        <= budget ** 2:
                    return True
        return False

    vetoed: set[int] = set()
    for comp in components:
        if any(not _far(f) for f in comp):
            continue   # touches the arc somewhere — could be the ball
        if (len(comp) <= max_island_frames and len(comp) < len(frames)
                and not _context_reachable(comp)):
            vetoed.update(comp)   # teleport island, wholly off the arc
            continue
        internal_static = all(
            ((uvs[fa][0] - uvs[fb][0]) ** 2
             + (uvs[fa][1] - uvs[fb][1]) ** 2)
            <= (static_px_per_frame * (fb - fa)) ** 2
            for fa, fb in zip(comp, comp[1:])
        ) if len(comp) >= 2 else False
        if internal_static:
            vetoed.update(comp)   # static lock-on while the arc moves away
    return sorted(vetoed)

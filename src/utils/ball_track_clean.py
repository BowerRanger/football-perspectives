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

"""Review proposals for cross-replay groups whose partner geometry left a
shot with no usable fixes.

Sub-20cm campaign W6: s013's replay group g02 had 6/6 partner cameras that
were GLOBALLY WRONG (3-6px per-frame reprojection residual, but fixes
6-8m underground — error purely along the reference ray, invisible to
any lateral/pixel gate). ``ball_cross_replay.py``'s ``pair_geometry_trusted``
/``filter_physical_fixes`` gates correctly reject that geometry per-pair,
but the stage's own diagnostics otherwise go silent: a shot with replay
partners in its group just quietly gets zero fixes, indistinguishable
from "no partner had overlapping evidence". That silence cost a manual
~30-minute anchor-editor investigation (2026-08-20 operator handoff doc)
to even find the cause. This module turns the "every partner gated out"
outcome into an actionable diag/quality_report entry instead.

Pure module: no file I/O, no new gates — it only reads the per-partner
metadata ``ball.py``'s ``_triangulate_groups`` already assembles.
"""

from __future__ import annotations

_SUGGESTION = "review partner camera anchors in the anchor editor"


def build_replay_review_proposals(
    shot_id: str,
    partners_meta: dict[str, dict],
) -> list[dict]:
    """One proposal per partner that left ``shot_id`` with zero usable
    fixes from that pairing.

    ``partners_meta`` is shaped like ``_triangulate_groups``'s own
    per-reference ``a_partners`` accumulator (also reusable per-pair on
    the non-reference side — see the wiring note): ``partner_shot_id ->
    {...}``, where a geometry-rejected pair (``pair_geometry_trusted``
    returned ``False``) carries ``"rejected": "implausible_geometry"`` +
    ``"n_impossible"``, and a pair whose geometry passed but whose fixes
    were individually filtered down to zero
    (``filter_moving_fixes``/``filter_physical_fixes``, or simply no
    synced detections) carries no ``"rejected"`` key and should instead
    carry ``"n_fixes": 0`` (see the wiring note for the one-line addition
    ``_triangulate_groups`` needs to record that case instead of a bare
    ``continue``). Any entry with neither ``"rejected"`` truthy nor
    ``n_inlier_fixes``/``n_fixes`` > 0 is treated as "nothing usable
    survived" and proposed for review.

    Call this once the group's triangulation for ``shot_id`` produced NO
    usable fixes overall (mirrors ``_triangulate_groups``'s
    ``if not productive:`` branch) — every entry in ``partners_meta`` is
    assumed to be a non-productive partner at the call site; this
    function does not itself filter out a partner that DID produce
    fixes, so callers must pass only the gated-out subset when a group
    has a mix of productive and non-productive partners.
    """
    proposals: list[dict] = []
    for partner in sorted(partners_meta):
        meta = partners_meta[partner]
        reason = meta.get("rejected") or "no_inlier_fixes"
        # Prefer the most specific count available; `or`-chaining would
        # skip a genuine 0 (e.g. n_fixes=0 for a pair whose geometry
        # passed but every individual fix was filtered out), so check
        # presence explicitly rather than truthiness.
        n_gated = 0
        for key in ("n_impossible", "n_fixes", "n_pairs"):
            val = meta.get(key)
            if val is not None:
                n_gated = int(val)
                break
        proposals.append({
            "shot": shot_id,
            "partner": partner,
            "reason": reason,
            "n_gated": n_gated,
            "suggestion": _SUGGESTION,
        })
    return proposals


def summarize_replay_reviews(
    proposals_by_shot: dict[str, list[dict]],
) -> dict:
    """Compact ``quality_report`` block: total proposal count + per-shot
    proposal lists, dropping shots with nothing to review."""
    nonempty = {shot: props for shot, props in proposals_by_shot.items()
                if props}
    return {
        "n_proposals": sum(len(props) for props in nonempty.values()),
        "shots": nonempty,
    }

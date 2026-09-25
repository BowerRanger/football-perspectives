"""Combine per-cue ``CueEvidence`` candidate lists into fused event
proposals.

Three fusion policies, all sharing the same no-single-unsupported-cue
principle -- a hard gate, never a soft score that just leaks a low
confidence through:

- ``fuse_cues`` ("any2", the original IC-D-brief policy): a frame
  cluster becomes a fused event when it draws evidence from >= 2
  distinct cues within +-``frame_tol`` frames of each other, OR from
  exactly 1 cue whose cluster frame lands within +-``frame_tol`` of an
  existing auto/kinematic anchor frame (see
  ``ball_cue_common.load_auto_event_frames``).
- ``fuse_cues_combo`` ("net_blur_combo", added in the 2026-09-25 tuning
  round): only mints from the specific pairings
  ``{net, blur}``, ``{blur, auto}``, or ``{audio, blur}`` -- i.e. blur
  must be present in every accepted cluster, since it was the strongest
  single signal in the tuning-round ablation; {audio, net} alone (no
  blur) is never enough under this policy.
- ``fuse_cues_weighted`` ("weighted"): a per-cue reliability score
  (learned from a tuning fold's raw-cue precision -- see
  ``CueReliability`` and ``scripts/tune_ball_event_cues.py``) summed
  over a cluster's distinct cues (+ an auto-corroboration bonus), minted
  only when the sum clears a tuned ``threshold``.

``fuse_cues_with_cfg`` dispatches on ``CueCfg.fusion_policy`` so callers
(the eval/tuning scripts, and eventually ``ball.py``) don't need to know
which concrete function backs the configured policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.ball_cue_config import CueCfg, DEFAULT_CUE_CFG
from src.utils.ball_hybrid_types import CueEvidence

_FRAME_TOL = 2

_AUDIO_CUE = "audio_onset"
_NET_CUE = "net_energy"
_BLUR_CUE = "blur_direction_change"


@dataclass(frozen=True)
class FusedEvent:
    frame: int
    kind: str
    cues: tuple[str, ...]
    conf: float
    xyz: tuple[float, float, float] | None
    uv: tuple[float, float] | None
    support: str  # "cues" (>=2 distinct cues) or "cue+auto" (1 cue + existing auto anchor)


def _cluster(evidence: list[CueEvidence], *, frame_tol: int) -> list[list[CueEvidence]]:
    """Greedy 1-D clustering by frame: sorted, start a new cluster
    whenever the gap to the cluster's *last* member exceeds
    ``frame_tol`` (a chain-linked window, not a fixed-width bucket)."""
    clusters: list[list[CueEvidence]] = []
    for e in sorted(evidence, key=lambda e: e.frame):
        if clusters and e.frame - clusters[-1][-1].frame <= frame_tol:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return clusters


def _near_auto(frame: int, auto_frames: list[int], *, frame_tol: int) -> bool:
    return any(abs(frame - af) <= frame_tol for af in auto_frames)


def _pick_kind(members: list[CueEvidence]) -> str:
    # The net-energy cue is unambiguous about "goal_impact"; prefer it.
    # Otherwise take the most common non-generic kind, else fall back to
    # the generic "contact" label single-camera audio/blur cues use.
    kinds = [m.kind for m in members]
    if "goal_impact" in kinds:
        return "goal_impact"
    non_generic = [k for k in kinds if k != "contact"]
    if non_generic:
        return max(set(non_generic), key=non_generic.count)
    return "contact"


def _fuse_conf(members: list[CueEvidence]) -> float:
    # Noisy-OR across member confidences: 1 - prod(1 - c_i).
    p = 1.0
    for m in members:
        p *= (1.0 - max(0.0, min(1.0, m.conf)))
    return float(1.0 - p)


def fuse_cues(
    cue_evidence: dict[str, list[CueEvidence]],
    auto_event_frames: list[int],
    *,
    frame_tol: int = _FRAME_TOL,
) -> list[FusedEvent]:
    """``cue_evidence`` maps a cue-module name (e.g. ``"audio"``,
    ``"net"``, ``"blur"``) to its candidate ``CueEvidence`` list;
    ``auto_event_frames`` is the existing auto/kinematic anchor frame
    set. Returns fused events sorted by frame; a cluster with only one
    distinct ``CueEvidence.cue`` value and no nearby auto anchor is
    dropped entirely (not returned with a low confidence -- the policy
    is a hard gate, not a soft score)."""
    all_evidence: list[CueEvidence] = [
        e for items in cue_evidence.values() for e in items]
    clusters = _cluster(all_evidence, frame_tol=frame_tol)

    out: list[FusedEvent] = []
    for members in clusters:
        distinct_cues = sorted({m.cue for m in members})
        center_frame = int(round(sum(m.frame for m in members) / len(members)))
        if len(distinct_cues) >= 2:
            support = "cues"
        elif _near_auto(center_frame, auto_event_frames, frame_tol=frame_tol):
            support = "cue+auto"
        else:
            continue
        xyz = next((m.xyz for m in members if m.xyz is not None), None)
        uv = next((m.uv for m in members if m.uv is not None), None)
        out.append(FusedEvent(
            frame=center_frame, kind=_pick_kind(members),
            cues=tuple(distinct_cues), conf=_fuse_conf(members),
            xyz=xyz, uv=uv, support=support,
        ))
    return sorted(out, key=lambda f: f.frame)


def fuse_cues_combo(
    cue_evidence: dict[str, list[CueEvidence]],
    auto_event_frames: list[int],
    *,
    frame_tol: int = _FRAME_TOL,
) -> list[FusedEvent]:
    """"net_blur_combo" policy: mint only from ``{net, blur}``,
    ``{blur, auto}``, or ``{audio, blur}`` -- blur is required in every
    accepted cluster; ``{audio, net}`` alone is never enough."""
    all_evidence: list[CueEvidence] = [
        e for items in cue_evidence.values() for e in items]
    clusters = _cluster(all_evidence, frame_tol=frame_tol)

    out: list[FusedEvent] = []
    for members in clusters:
        distinct_cues = {m.cue for m in members}
        center_frame = int(round(sum(m.frame for m in members) / len(members)))
        has_blur = _BLUR_CUE in distinct_cues
        has_net = _NET_CUE in distinct_cues
        has_audio = _AUDIO_CUE in distinct_cues
        near_auto = _near_auto(center_frame, auto_event_frames, frame_tol=frame_tol)
        if has_blur and has_net:
            support = "cues"
        elif has_blur and has_audio:
            support = "cues"
        elif has_blur and near_auto:
            support = "cue+auto"
        else:
            continue
        xyz = next((m.xyz for m in members if m.xyz is not None), None)
        uv = next((m.uv for m in members if m.uv is not None), None)
        out.append(FusedEvent(
            frame=center_frame, kind=_pick_kind(members),
            cues=tuple(sorted(distinct_cues)), conf=_fuse_conf(members),
            xyz=xyz, uv=uv, support=support,
        ))
    return sorted(out, key=lambda f: f.frame)


@dataclass(frozen=True)
class CueReliability:
    """Per-cue weights for ``fuse_cues_weighted``, learned once on a
    TUNING fold (see ``scripts/tune_ball_event_cues.py``) and frozen for
    application to a held-out fold. ``weights`` maps a ``CueEvidence.cue``
    name to its reliability (conventionally that cue's raw-candidate
    precision on the tuning fold); ``auto_weight`` is the fixed bonus for
    landing near an existing auto anchor; ``threshold`` is the minimum
    summed score (over a cluster's distinct cues, + the auto bonus if
    applicable) required to mint a fused event."""

    weights: dict[str, float]
    auto_weight: float
    threshold: float


def _cluster_score(
    members: list[CueEvidence], near_auto: bool, reliability: CueReliability,
) -> float:
    distinct_cues = {m.cue for m in members}
    score = sum(reliability.weights.get(c, 0.0) for c in distinct_cues)
    if near_auto:
        score += reliability.auto_weight
    return score


def fuse_cues_weighted(
    cue_evidence: dict[str, list[CueEvidence]],
    auto_event_frames: list[int],
    reliability: CueReliability,
    *,
    frame_tol: int = _FRAME_TOL,
) -> list[FusedEvent]:
    """"weighted" policy: sum ``reliability.weights`` over a cluster's
    distinct cues (+ ``reliability.auto_weight`` if an existing auto
    anchor is nearby) and mint only when the sum clears
    ``reliability.threshold``. A cluster's own summed score becomes its
    ``FusedEvent.conf`` (NOT the noisy-OR ``_fuse_conf`` the other two
    policies use), so it's directly comparable to ``threshold``."""
    all_evidence: list[CueEvidence] = [
        e for items in cue_evidence.values() for e in items]
    clusters = _cluster(all_evidence, frame_tol=frame_tol)

    out: list[FusedEvent] = []
    for members in clusters:
        center_frame = int(round(sum(m.frame for m in members) / len(members)))
        near_auto = _near_auto(center_frame, auto_event_frames, frame_tol=frame_tol)
        score = _cluster_score(members, near_auto, reliability)
        if score < reliability.threshold:
            continue
        distinct_cues = sorted({m.cue for m in members})
        xyz = next((m.xyz for m in members if m.xyz is not None), None)
        uv = next((m.uv for m in members if m.uv is not None), None)
        support = "cues" if len(distinct_cues) >= 2 else "cue+auto"
        out.append(FusedEvent(
            frame=center_frame, kind=_pick_kind(members),
            cues=tuple(distinct_cues), conf=score,
            xyz=xyz, uv=uv, support=support,
        ))
    return sorted(out, key=lambda f: f.frame)


def fuse_cues_with_cfg(
    cue_evidence: dict[str, list[CueEvidence]],
    auto_event_frames: list[int],
    cfg: CueCfg = DEFAULT_CUE_CFG,
) -> list[FusedEvent]:
    """Dispatch on ``cfg.fusion_policy`` ("any2" | "net_blur_combo" |
    "weighted") so callers don't need to know which concrete fuse_cues*
    function backs the configured policy."""
    if cfg.fusion_policy == "any2":
        return fuse_cues(cue_evidence, auto_event_frames, frame_tol=cfg.fusion_frame_tol)
    if cfg.fusion_policy == "net_blur_combo":
        return fuse_cues_combo(cue_evidence, auto_event_frames,
                                frame_tol=cfg.fusion_frame_tol)
    if cfg.fusion_policy == "weighted":
        reliability = CueReliability(
            weights=dict(cfg.fusion_weights), auto_weight=cfg.fusion_auto_weight,
            threshold=cfg.fusion_threshold)
        return fuse_cues_weighted(cue_evidence, auto_event_frames, reliability,
                                   frame_tol=cfg.fusion_frame_tol)
    raise ValueError(f"unknown fusion_policy {cfg.fusion_policy!r}")


__all__ = [
    "FusedEvent", "fuse_cues", "fuse_cues_combo", "CueReliability",
    "fuse_cues_weighted", "fuse_cues_with_cfg",
]

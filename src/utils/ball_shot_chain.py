"""Shot chains (spec §6): strike -> [deflections...] -> terminal impact.

A chain is a grouping of ordinary anchors — no new solve path. Segments
between flight-implying members are already ballistic via
ball_segments._implies_flight; this module adds (a) auto-proposal pairing
(each detected goal_impact <- the last preceding touch within a window)
and (b) per-chain validation warnings against the resolved keyframes
(missing members, unresolved worlds, implied launch speed outside the
shot envelope — which catches a mis-clicked frame immediately).
Pure and torch-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.schemas.ball_keyframes import BallKeyframeSet
    from src.utils.ball_auto_events import BallEvent


@dataclass(frozen=True)
class ShotChainCfg:
    enabled: bool = True
    pair_window_frames: int = 75
    launch_speed_warn_min_m_s: float = 8.0
    launch_speed_warn_max_m_s: float = 45.0


def propose_shot_chains(
    events: "Sequence[BallEvent]", cfg: ShotChainCfg,
) -> tuple[tuple[int, int], ...]:
    """Strike->impact pairs: each goal_impact event claims the LAST touch
    event strictly before it within ``pair_window_frames``."""
    if not cfg.enabled:
        return ()
    touches = sorted(e.frame for e in events if e.kind == "touch")
    chains: list[tuple[int, int]] = []
    for impact in sorted(e.frame for e in events if e.kind == "goal_impact"):
        candidates = [
            f for f in touches
            if f < impact and impact - f <= cfg.pair_window_frames
        ]
        if candidates:
            chains.append((candidates[-1], impact))
    return tuple(chains)


def chain_warnings(
    chain: Sequence[int],
    keyframes: "BallKeyframeSet | None",
    fps: float,
    cfg: ShotChainCfg,
) -> list[dict]:
    """Validation warnings for one chain; empty list means the chain is
    consistent with the resolved keyframes."""
    frames = [int(f) for f in chain]
    by_frame = {
        kf.frame: kf for kf in (keyframes.keyframes if keyframes else ())
    }
    warnings: list[dict] = []
    missing = [f for f in frames if f not in by_frame]
    if missing:
        warnings.append({
            "kind": "missing_keyframe", "frames": missing,
            "detail": "chain member frame(s) have no resolved keyframe "
                      "— place an anchor there",
        })
    unresolved = [
        f for f in frames
        if f in by_frame and by_frame[f].world_xyz is None
    ]
    if unresolved:
        warnings.append({
            "kind": "unresolved_world", "frames": unresolved,
            "detail": "chain member keyframe(s) have no 3-D position",
        })
    usable = [f for f in frames if by_frame.get(f) is not None
              and by_frame[f].world_xyz is not None]
    for a, b in zip(usable, usable[1:]):
        wa, wb = by_frame[a].world_xyz, by_frame[b].world_xyz
        dt = (b - a) / fps
        if dt <= 0:
            continue
        speed = math.dist(wa, wb) / dt
        if not (cfg.launch_speed_warn_min_m_s
                <= speed <= cfg.launch_speed_warn_max_m_s):
            warnings.append({
                "kind": "launch_speed", "frames": [a, b],
                "speed_m_s": speed,
                "detail": f"implied speed {speed:.1f} m/s outside "
                          f"[{cfg.launch_speed_warn_min_m_s:.0f}, "
                          f"{cfg.launch_speed_warn_max_m_s:.0f}] m/s "
                          "— check the clicked frames",
            })
    return warnings

"""Mode-sequence beam search for the ball solver (Phase 2).

Exposes ``solve_modes(...)`` with the same call signature and ``SolveResult``
return type as ``solve_piecewise`` in ``ball_piecewise_solver``.  The search
explores partitions of the shot timeline into labelled ``Mode`` segments and
selects the lowest-cost complete hypothesis via a left-to-right beam.

Tasks implemented so far
------------------------
T2  — Mode enum, frozen dataclasses (Breakpoint / Segment / Hypothesis),
      ModeSearchCfg, ``_mode_search_cfg`` config mapper.
T3–T7 — to follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

# Re-export solver result shapes so callers can import from one place.
from src.utils.ball_piecewise_solver import FlightSegment, SolveResult  # noqa: F401


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class Mode(IntEnum):
    """Ball trajectory mode.

    Integer values are the deterministic tie-break key: when two hypotheses
    have equal cost, the one whose current mode has the *lower* integer value
    wins.  The ordering encodes prior likelihood: physical rolling > ballistic
    flight > ball at a player's foot > stopped > completely hidden.
    """

    ROLLING = 0
    FLIGHT = 1
    POSSESSED = 2
    STATIONARY = 3
    OUT_OF_VIEW = 4


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Breakpoint:
    """A candidate partition point on the shot timeline.

    Attributes
    ----------
    frame:
        Frame index at which the break occurs.
    kind:
        Semantic label: e.g. ``"touch"``, ``"bounce"``, ``"boundary"``,
        ``"goal_impact"``.
    event_score:
        Confidence score in [0, 1].  Synthetic clip-boundary breakpoints
        carry ``event_score=0.0`` (design note F17).
    is_manual:
        ``True`` when the operator explicitly placed this anchor; manual
        breakpoints become FORCED partition points during the beam search.
    player_branch:
        When this break candidates a POSSESSED transition, identifies the
        player the branch tethers to.  ``None`` for all other kinds.
    """

    frame: int
    kind: str
    event_score: float
    is_manual: bool
    player_branch: str | None = None


@dataclass(frozen=True, eq=False)
class Segment:
    """One scored segment in a beam hypothesis.

    Attributes
    ----------
    fa, fb:
        Inclusive frame range ``[fa, fb]``.
    mode:
        Assigned mode for this segment.
    worlds:
        Sparse map of ``frame → world_xyz`` (np.ndarray shape ``(3,)``)
        produced by the segment fitter.  May be empty for ``OUT_OF_VIEW``.
    residual_px:
        Mean reprojection residual in raw pixels (design note F5 — not
        gate-normalised).
    underconstrained:
        ``True`` when the fitter could not uniquely determine depth.
    kind:
        SolveResult kind string matching ``_KIND_STATE`` in the assembler:
        ``"rolling"`` / ``"ballistic"`` / ``"possessed"`` / ``"stationary"``
        / ``"out_of_view"``.
    player_id:
        Possessing player identifier; ``None`` unless ``mode == POSSESSED``.
    boundary_vel:
        ``(start_vel | None, end_vel | None)`` — velocities at the segment
        endpoints.  Non-``None`` for all modes where a velocity can be
        computed (see design note F4).
    """

    fa: int
    fb: int
    mode: Mode
    worlds: dict[int, np.ndarray]
    residual_px: float
    underconstrained: bool
    kind: str
    player_id: str | None
    boundary_vel: tuple[np.ndarray | None, np.ndarray | None]


@dataclass(frozen=True, eq=False)
class Hypothesis:
    """One live beam state: the partial solution up to ``last_bp_idx``.

    Attributes
    ----------
    last_bp_idx:
        Index into the sorted breakpoint list of the last breakpoint
        consumed.  ``-1`` for the initial (empty) hypothesis.
    cur_mode:
        Mode of the last committed segment, or ``None`` for the seed state.
    segments:
        Ordered tuple of committed ``Segment`` objects.
    cost:
        Cumulative cost (lower = better).
    end_velocity:
        Velocity at the end of the last segment; ``None`` if unknown.
    player_id:
        Possessing player at the end of the last segment; ``None`` otherwise.
    """

    last_bp_idx: int
    cur_mode: Mode | None
    segments: tuple[Segment, ...]
    cost: float
    end_velocity: np.ndarray | None
    player_id: str | None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeSearchCfg:
    """Tunable parameters for the beam mode-sequence search.

    All values are read from ``ball.mode_search.*`` in the pipeline config
    via ``_mode_search_cfg``; the field defaults are the production values.
    """

    beam_width: int = 8
    """Number of hypotheses retained at each breakpoint column."""

    segment_cost_constant: float = 6.0
    """BIC-style parsimony term added per segment regardless of fit quality."""

    unexplained_break_penalty: float = 10.0
    """Cost for inserting a mode transition with no supporting event."""

    ignored_event_penalty: float = 8.0
    """Cost for skipping a high-score breakpoint without a matching transition."""

    out_of_view_frame_penalty: float = 1.5
    """Extra cost per frame spent in OUT_OF_VIEW mode."""

    possessed_tether_px: float = 40.0
    """Pixel-space tolerance for the soft tether to the possessing player's foot."""

    velocity_discontinuity_weight: float = 2.0
    """Weight applied to ‖Δv‖ when scoring velocity continuity across a break."""

    max_segment_fit_calls: int = 20000
    """Hard budget: raise BudgetExceeded once the fitter cache hits this limit."""


def _mode_search_cfg(cfg: dict[str, Any]) -> ModeSearchCfg:
    """Map ``ball.mode_search.*`` config keys onto :class:`ModeSearchCfg`.

    Pattern: identical to ``_second_pass_cfg`` / ``_cross_replay_cfg`` in
    ``src/stages/ball.py``.  Each field falls back to the dataclass default
    when the key is absent.

    Parameters
    ----------
    cfg:
        The *outer* pipeline config dict (i.e. the full ``ball.*`` sub-dict
        from ``config/default.yaml``, or the raw top-level dict — only the
        ``"mode_search"`` key is inspected).
    """
    ms = cfg.get("mode_search", {})
    base = ModeSearchCfg()
    return ModeSearchCfg(
        beam_width=int(ms.get("beam_width", base.beam_width)),
        segment_cost_constant=float(ms.get("segment_cost_constant", base.segment_cost_constant)),
        unexplained_break_penalty=float(
            ms.get("unexplained_break_penalty", base.unexplained_break_penalty)
        ),
        ignored_event_penalty=float(ms.get("ignored_event_penalty", base.ignored_event_penalty)),
        out_of_view_frame_penalty=float(
            ms.get("out_of_view_frame_penalty", base.out_of_view_frame_penalty)
        ),
        possessed_tether_px=float(ms.get("possessed_tether_px", base.possessed_tether_px)),
        velocity_discontinuity_weight=float(
            ms.get("velocity_discontinuity_weight", base.velocity_discontinuity_weight)
        ),
        max_segment_fit_calls=int(ms.get("max_segment_fit_calls", base.max_segment_fit_calls)),
    )

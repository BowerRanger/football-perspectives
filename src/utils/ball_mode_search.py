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
from typing import TYPE_CHECKING, Any

import numpy as np

# Re-export solver result shapes so callers can import from one place.
from src.utils.ball_piecewise_solver import (  # noqa: F401
    FlightSegment,
    SolveResult,
    SolverCfg,
    TrajectoryNode,
    _Arc,
    _CommitState,
    _Solver,
    _SpanOutcome,
    apply_node_authority,
    apply_restitution_flags,
    commit_span,
)
from src.utils.ball_physics import (
    G_VEC,
    eval_parabola,
    fit_rolling_segment,
    parabola_end_velocity,
)
from src.utils.bundle_adjust import fit_parabola_to_image_observations

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.utils.ball_auto_events import BallEvent
    from src.utils.ball_player_context import PlayerContext


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
    """Extra cost per *observed* frame gapped out by OUT_OF_VIEW mode (fix C1c).

    Charged per frame in the span that carried a confident ball detection.
    Gapping a detection-rich span is therefore expensive; OUT_OF_VIEW is a
    last resort that wins only where the ball has no usable evidence."""

    out_of_view_missing_frame_penalty: float = 0.05
    """Tiny per-frame cost for *missing* (un-observed) frames in an OUT_OF_VIEW
    span (fix C1c).

    Keeps a long gap over genuinely-missing frames from being totally free
    (so the beam still prefers a short out-of-view span over a long one),
    while staying far below ``out_of_view_frame_penalty`` so the dominant cost
    is the observed-frame term."""

    possessed_tether_px: float = 40.0
    """Pixel-space tolerance for the soft tether to the possessing player's foot."""

    velocity_discontinuity_weight: float = 2.0
    """Weight applied to ‖Δv‖ when scoring velocity continuity across a break."""

    max_segment_fit_calls: int = 20000
    """Hard budget: raise BudgetExceeded once the fitter cache hits this limit."""

    max_skip: int = 3
    """Max intermediate breakpoints a single segment may absorb (skip).

    A span from ``bp_i`` may extend directly to ``bp_j`` with
    ``j <= i + 1 + max_skip``; the ``j - i - 1`` intermediate breakpoints are
    *skipped* and each is charged ``ignored_event_cost``.  Bounds beam
    branching while letting the search merge spurious velocity breaks (the
    40-break case) into a single physical segment."""


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
        out_of_view_missing_frame_penalty=float(
            ms.get(
                "out_of_view_missing_frame_penalty",
                base.out_of_view_missing_frame_penalty,
            )
        ),
        possessed_tether_px=float(ms.get("possessed_tether_px", base.possessed_tether_px)),
        velocity_discontinuity_weight=float(
            ms.get("velocity_discontinuity_weight", base.velocity_discontinuity_weight)
        ),
        max_segment_fit_calls=int(ms.get("max_segment_fit_calls", base.max_segment_fit_calls)),
        max_skip=int(ms.get("max_skip", base.max_skip)),
    )


# ---------------------------------------------------------------------------
# Task 3 — fitter-reuse layer + sound segment-fit cache
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when the segment fitter exceeds ``max_segment_fit_calls``.

    The beam search treats this as the signal to abandon the global solve
    and fall back to the whole-shot piecewise solver (design note F8).
    """


@dataclass(frozen=True, eq=False)
class SegmentFit:
    """One scored segment fit produced by :class:`_SegmentSolver`.

    ``eq=False`` so the search may use identity for cache-hit assertions
    (numpy arrays inside ``worlds`` are not hashable / comparable anyway).

    Attributes
    ----------
    worlds:
        Sparse ``frame -> world_xyz`` (np.ndarray shape ``(3,)``) over the
        full inclusive segment range ``[fa, fb]``.  Empty for OUT_OF_VIEW.
    residual_px:
        Mean reprojection residual in *raw* pixels (design note F5 — not
        gate-normalised).
    underconstrained:
        ``True`` when the fitter's residual gate flags the fit (depth not
        uniquely determined).  Always ``False`` for OUT_OF_VIEW.
    kind:
        SolveResult kind string for the assembler (``_KIND_STATE``):
        ``"ballistic"`` / ``"rolling"`` / ``"stationary"`` / ``"out_of_view"``
        / ``"possessed"``.
    boundary_vel:
        ``(v_at_fa, v_at_fb)`` — world velocities at the two endpoints
        (design note F4).  ``(None, None)`` for OUT_OF_VIEW.
    n_observed:
        Number of frames in ``[fa, fb]`` that carry a confident ball
        observation (a real detection).  Populated only for OUT_OF_VIEW
        fits (fix C1c) — it is the count of detections the gap would discard,
        and drives the OUT_OF_VIEW cost so gapping a detection-rich span is
        expensive.  ``0`` (default) for every other mode.
    """

    worlds: dict[int, np.ndarray]
    residual_px: float
    underconstrained: bool
    kind: str
    boundary_vel: tuple[np.ndarray | None, np.ndarray | None]
    n_observed: int = 0


class _SegmentSolver:
    """Endpoint-free segment fitter wrapping a real :class:`_Solver`.

    Built from the *exact* kwargs ``solve_modes`` receives, so it reuses
    the solver's observation collection (``_interior_obs`` — gap-fill /
    only-frames skip rules), world-fix lookup (``_fixes_in``), ground
    ray-casting (``_ground_raycast``) and pixel-RMS scoring (``_pixel_rms``)
    *unchanged*.  No skip-rule logic is reimplemented here.

    Why fit the primitives directly rather than through ``_fit_arc`` /
    ``_rolling_span``
    -----------------------------------------------------------------
    ``_Solver._fit_arc(a, b, fa, fb)`` *requires* node worlds ``a, b`` and
    pins them as hard knots — exactly the endpoint dependence design note
    F7 forbids for free-floating breakpoints (it would make the cache key
    depend on a non-frame-determined world).  ``_rolling_span`` likewise
    pins ``a[:2]`` / ``b[:2]`` as exact roll endpoints.  So for a free-floating
    breakpoint FLIGHT calls ``fit_parabola_to_image_observations`` directly with
    NO endpoint knot, and ROLLING seeds its endpoints from the first/last
    interior ground ray-cast rather than a pinned node endpoint.  When ``fa`` /
    ``fb`` lands on a *resolved anchor node* (manual OR auto — its world is
    deterministic from the frame, not the search path) that world IS passed as a
    knot: a FLIGHT knot (the depth-critical pin) and a ROLLING endpoint XY.  This
    keeps the fit a pure function of ``(fa, fb, mode, interior obs, in-range
    fixes, anchor-present?)`` — all frame-determined — so a frame-keyed cache is
    sound.

    Cache soundness (F7)
    --------------------
    Key = ``(fa, fb, mode, player_id, fa_anchor?, fb_anchor?, wfixes_sig)``.
    Endpoint-free fits depend only on frame-determined inputs; the *only*
    endpoint dependence is when ``fa``/``fb`` is a resolved anchor node (then its
    world is a knot/endpoint), and that world is deterministic from the frame, so
    the boolean flag plus the frame fully capture it.  ``wfixes_sig`` hashes only
    the in-range fix *frames* (fix values are deterministic from frame).
    """

    def __init__(
        self,
        *,
        mode_cfg: ModeSearchCfg | None = None,
        player_ctx: "PlayerContext | None" = None,
        **solver_kwargs: Any,
    ) -> None:
        self._solver = _Solver(**solver_kwargs)
        self._mode_cfg = mode_cfg or ModeSearchCfg()
        self._player_ctx = player_ctx
        self._cache: dict[tuple, SegmentFit] = {}
        self._fit_calls: int = 0

    # ------------------------------------------------------------------
    # Public API

    def fit_segment(
        self,
        fa: int,
        fb: int,
        mode: Mode,
        player_id: str | None = None,
        fa_anchor_world: np.ndarray | None = None,
        fb_anchor_world: np.ndarray | None = None,
    ) -> SegmentFit:
        """Fit ``[fa, fb]`` under ``mode``; memoized + budget-gated.

        **Precondition (cache soundness — design note F7):**
        ``fa_anchor_world`` and ``fb_anchor_world`` MUST be *resolved-node*
        worlds — i.e. worlds whose value is deterministically derived from the
        frame index alone (manual operator anchors OR resolved auto-anchors
        looked up from the fixed node set; both are a function of the frame, not
        of the search path).  Do NOT pass a path-dependent world such as the
        endpoint of a previously solved segment: the cache key records only
        whether an anchor is present (``True``/``False``), not its value, so
        a different world for the same frame produces a silent cache hit with
        stale results and corrupts the beam search.
        """
        key = (
            int(fa), int(fb), int(mode), player_id,
            fa_anchor_world is not None, fb_anchor_world is not None,
            self._wfixes_sig(fa, fb),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # A real (non-cached) fit is about to run — charge it to the budget.
        self._fit_calls += 1
        if self._fit_calls > self._mode_cfg.max_segment_fit_calls:
            raise BudgetExceeded(
                f"segment fit budget {self._mode_cfg.max_segment_fit_calls} "
                f"exceeded at span {fa}-{fb} mode {mode.name}"
            )

        fit = self._dispatch(fa, fb, mode, player_id,
                             fa_anchor_world, fb_anchor_world)
        self._cache[key] = fit
        return fit

    # ------------------------------------------------------------------
    # Dispatch

    def _dispatch(
        self,
        fa: int,
        fb: int,
        mode: Mode,
        player_id: str | None,
        fa_anchor_world: np.ndarray | None,
        fb_anchor_world: np.ndarray | None,
    ) -> SegmentFit:
        if mode is Mode.FLIGHT:
            return self._fit_flight(fa, fb, fa_anchor_world, fb_anchor_world)
        if mode is Mode.ROLLING:
            return self._fit_rolling(fa, fb, fa_anchor_world, fb_anchor_world)
        if mode is Mode.STATIONARY:
            return self._fit_stationary(fa, fb)
        if mode is Mode.OUT_OF_VIEW:
            return self._fit_out_of_view(fa, fb)
        if mode is Mode.POSSESSED:
            return self._fit_possessed(fa, fb, player_id)
        raise ValueError(f"unknown mode {mode!r}")

    # ------------------------------------------------------------------
    # Helpers

    def _wfixes_sig(self, fa: int, fb: int) -> tuple[int, ...]:
        """Frame-only signature of the in-range world fixes (F7).

        Fix *values* are deterministic from the frame, so the sorted tuple
        of in-range fix frames fully captures the cache dependency.
        """
        return tuple(sorted(f for f, _xyz, _w in self._solver._fixes_in(fa, fb)))

    def _n_confident_obs(self, fa: int, fb: int) -> int:
        """Count frames in ``[fa, fb]`` (inclusive) with a confident ball
        observation — a real detection the solver would consume.

        A frame counts when it has a ball pixel (``uvs[f] is not None``), a
        valid camera, and is NOT a gap-fill (synthesised) frame — the same
        admission rule ``_interior_obs`` applies, extended to the inclusive
        endpoints.  Drives the OUT_OF_VIEW cost (fix C1c).
        """
        solver = self._solver
        n = 0
        for f in range(fa, fb + 1):
            uv = solver.uvs.get(f)
            if uv is None or not solver._has_cam(f) or f in solver.gap_fill:
                continue
            n += 1
        return n

    # ------------------------------------------------------------------
    # OUT_OF_VIEW

    def _fit_out_of_view(self, fa: int, fb: int) -> SegmentFit:
        """Empty-worlds fit; carries the confident-observation count (C1c).

        ``n_observed`` is the number of frames in ``[fa, fb]`` with a real ball
        detection.  ``segment_cost`` charges per observed frame, so gapping a
        detection-rich span is expensive while gapping a genuinely-missing span
        stays cheap (correct LAST-RESORT semantics).
        """
        return SegmentFit(
            worlds={}, residual_px=0.0, underconstrained=False,
            kind="out_of_view", boundary_vel=(None, None),
            n_observed=self._n_confident_obs(fa, fb),
        )

    # ------------------------------------------------------------------
    # FLIGHT

    def _fit_flight(
        self,
        fa: int,
        fb: int,
        fa_anchor_world: np.ndarray | None,
        fb_anchor_world: np.ndarray | None,
    ) -> SegmentFit:
        """Endpoint-free gravity-arc fit from interior pixels + world fixes.

        No endpoint knot is added for a free-floating breakpoint (F7).  A
        resolved anchor node (manual OR auto) at ``fa``/``fb`` is passed as a
        soft knot — its world is deterministic from the frame, so the cache
        key's anchor-bool captures it.
        """
        solver = self._solver
        fps = solver.fps
        cfg: SolverCfg = solver.cfg
        obs, Ks, Rs, ts = solver._interior_obs(fa, fb)
        wfixes = solver._fixes_in(fa, fb)

        if len(obs) < cfg.min_obs_for_lm_fit:
            # Too few observations to determine an arc — flag and bail with
            # whatever ground ray-casts exist so the renderer is not empty.
            worlds = {}
            for f in range(fa, fb + 1):
                g = solver._ground_raycast(f)
                if g is not None:
                    worlds[f] = g
            resid = solver._pixel_rms(worlds, list(worlds))
            return SegmentFit(
                worlds=worlds,
                residual_px=float(resid) if resid is not None else 0.0,
                underconstrained=True,
                kind="ballistic",
                boundary_vel=(None, None),
            )

        first_obs_frame = obs[0][0]
        knots: dict[int, np.ndarray] = {}
        if fa_anchor_world is not None:
            knots[fa - first_obs_frame] = np.asarray(fa_anchor_world, float)
        if fb_anchor_world is not None:
            knots[fb - first_obs_frame] = np.asarray(fb_anchor_world, float)

        try:
            p0_fit, v0_fit, mean_resid = fit_parabola_to_image_observations(
                obs, Ks=Ks, Rs=Rs, t_world=ts,
                fps=fps, distortion=solver.distortion,
                p0_fixed=None,
                knot_frames=knots or None,
                world_fixes=wfixes or None,
            )
        except Exception:
            worlds = {}
            for f in range(fa, fb + 1):
                g = solver._ground_raycast(f)
                if g is not None:
                    worlds[f] = g
            resid = solver._pixel_rms(worlds, list(worlds))
            return SegmentFit(
                worlds=worlds,
                residual_px=float(resid) if resid is not None else 0.0,
                underconstrained=True,
                kind="ballistic",
                boundary_vel=(None, None),
            )

        # Re-base (p0, v0) — fitted at first_obs_frame — onto the span start.
        dt0 = (fa - first_obs_frame) / fps
        p0 = p0_fit + v0_fit * dt0 + 0.5 * G_VEC * dt0**2
        v0 = v0_fit + G_VEC * dt0

        worlds: dict[int, np.ndarray] = {}
        for f in range(fa, fb + 1):
            t_rel = (f - fa) / fps
            worlds[f] = eval_parabola(p0, v0, np.array([t_rel]))[0]

        # Boundary velocities (F4): v at fa is v0; v at fb is v0 + g·T.
        v_a = v0
        v_b = parabola_end_velocity(v0, (fb - fa) / fps)

        # Fix F5: recompute residual over obs frames only (raw pixels).
        # ``mean_resid`` from fit_parabola_to_image_observations includes the
        # 1e3-weighted knot residual block when knot_frames is non-empty,
        # inflating the value when a manual-anchor knot is present.  Mirror
        # ``_fit_arc`` in ball_piecewise_solver: call ``_pixel_rms`` over the
        # actual observation frames used in this fit.
        obs_frames_only = [f for f, _ in obs]
        raw_resid = solver._pixel_rms(worlds, obs_frames_only)
        resid = float(raw_resid) if (raw_resid is not None and np.isfinite(raw_resid)) else 0.0
        underconstrained = resid > cfg.flight_max_residual_px

        # Fix C1: flight monocular-degeneracy guard.  A ground roll's pixel
        # track can be fit perfectly (residual ≈ 0) by a far-away fast parabola
        # whose recovered worlds are wildly off-pitch / far below ground — the
        # FLIGHT analogue of the ROLLING lob guard in ``_fit_rolling``.  Such a
        # fit is physically impossible, so flag it underconstrained: it must not
        # masquerade as a low-cost FLIGHT explanation for a roll.
        if not underconstrained and self._flight_worlds_implausible(worlds):
            underconstrained = True

        return SegmentFit(
            worlds=worlds,
            residual_px=resid,
            underconstrained=underconstrained,
            kind="ballistic",
            boundary_vel=(np.asarray(v_a, float), np.asarray(v_b, float)),
        )

    def _flight_worlds_implausible(self, worlds: dict[int, np.ndarray]) -> bool:
        """True when a fitted flight arc's worlds are non-physical.

        A flight arc must stay within a generous envelope around the pitch and
        above a small sub-ground margin.  We reuse the same clamp magnitude
        ``_ground_raycast`` uses (``2× max pitch dimension``, floor 50 m) for
        the horizontal extent, and reject any world more than a few metres
        below the ground plane.  Catches the monocular degeneracy where a roll
        is explained by a far-away parabola at absurd depth.
        """
        solver = self._solver
        if not worlds:
            return False
        clamp = max(50.0, 2.0 * max(solver.pitch.length_m, solver.pitch.width_m))
        z_floor = -2.0  # a couple of metres of slack below the ground plane
        for w in worlds.values():
            if not np.all(np.isfinite(w)):
                return True
            if abs(float(w[0])) > clamp or abs(float(w[1])) > clamp:
                return True
            if float(w[2]) < z_floor:
                return True
        return False

    # ------------------------------------------------------------------
    # ROLLING

    def _fit_rolling(
        self,
        fa: int,
        fb: int,
        fa_anchor_world: np.ndarray | None = None,
        fb_anchor_world: np.ndarray | None = None,
    ) -> SegmentFit:
        """Grounded fit: a clean constant-decel roll when it applies, else a
        per-frame ground ray-cast fallback (fix F16).

        When ``fa``/``fb`` lands on a resolved anchor node its world XY pins
        the corresponding roll endpoint (mirroring piecewise's ``_rolling_span``
        which takes the node worlds ``a``/``b`` directly); the ground plane
        already fixes z.  A non-anchor endpoint is seeded from the first/last
        in-pitch ground ray-cast.  Anchor worlds are frame-deterministic, so the
        fit stays frame-determined and the cache stays sound (F7).

        Fix F16 — grounded ray-cast fallback
        ------------------------------------
        The constant-decel roll model is the *preferred* grounded explanation —
        a real roll is smoother than raw ray-casts.  But where the grounded ball
        is NOT a tidy roll (it jitters, changes speed, or there are too few
        interior obs for a roll), the roll fit comes out high-residual /
        speed-violating → underconstrained, which loses to OUT_OF_VIEW (cost ∝
        observed frames) and the ball goes MISSING despite being detected on the
        ground.  Piecewise never has this gap: its open-span fallback ray-casts
        every non-flight observed frame to the pitch plane (z = ball radius),
        which reprojects EXACTLY to the observed pixel.  When the clean roll does
        not apply we mirror that fallback here — per-frame ground ray-casts with
        the SAME guards (skip ``_ground_raycast``→None, gap-fill, and off-pitch
        landings) — so a grounded observed span costs ≈0 residual and stays
        GROUNDED instead of collapsing to OUT_OF_VIEW.  This depends only on
        ``(fa, fb, frozen obs/cameras)`` — frame-determined — so the cache key
        stays sound (F7).
        """
        solver = self._solver
        fps = solver.fps
        cfg: SolverCfg = solver.cfg
        T = (fb - fa) / fps

        obs: list[tuple[float, np.ndarray]] = []
        obs_frames: list[int] = []
        for f in range(fa + 1, fb):
            ground = self._inpitch_raycast(f)
            if ground is None:
                continue
            obs.append(((f - fa) / fps, ground[:2]))
            obs_frames.append(f)

        a_xy = (
            np.asarray(fa_anchor_world, float)[:2]
            if fa_anchor_world is not None
            else self._endpoint_xy(fa, obs)
        )
        b_xy = (
            np.asarray(fb_anchor_world, float)[:2]
            if fb_anchor_world is not None
            else self._endpoint_xy(fb, obs, prefer_last=True)
        )

        fit = fit_rolling_segment(
            a_xy, b_xy, max(T, 1e-6), obs, cfg.rolling_decel_max_m_s2,
        )
        z = cfg.ball_radius_m
        worlds: dict[int, np.ndarray] = {}
        times = np.array([(f - fa) / fps for f in range(fa, fb + 1)])
        pts = fit.eval(times, z)
        for i, f in enumerate(range(fa, fb + 1)):
            worlds[f] = pts[i]

        resid = solver._pixel_rms(worlds, obs_frames)
        residual_px = float(resid) if resid is not None else 0.0
        v_a = self._roll_velocity(fit, 0.0, z, fps)
        v_b = self._roll_velocity(fit, T, z, fps)

        # Fix: lob-degeneracy guard from ``_rolling_span``.  A roll faster than
        # any realistic ground pass is physically impossible — the usual culprit
        # is a LOB whose ground projection happens to fit the rolling model in
        # pixels (monocular degeneracy).  Such a span is airborne, so its ground
        # ray-casts are a knowingly-wrong depth; flag the roll underconstrained
        # (ceding to FLIGHT) and do NOT engage the grounded fallback — emitting
        # pixel-exact ground worlds would re-create the very degeneracy this
        # guard exists to prevent (mirror in ball_piecewise_solver._rolling_span
        # line ~896–899; the open-span loop likewise refuses ground ray-casts on
        # flight-posterior frames, lines ~1260–1264).
        speed_violation = False
        if len(pts) >= 2:
            speeds = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1) * fps
            if float(np.max(speeds)) > cfg.rolling_max_speed_m_s:
                speed_violation = True
        if speed_violation:
            return SegmentFit(
                worlds=worlds,
                residual_px=residual_px,
                underconstrained=True,
                kind="rolling",
                boundary_vel=(v_a, v_b),
            )

        # Clean roll preferred when it applies — a real roll is smoother than
        # raw ray-casts.
        clean_roll = (
            resid is not None
            and resid <= cfg.rolling_max_residual_px
            and len(obs) >= cfg.min_obs_for_lm_fit
        )
        if clean_roll:
            return SegmentFit(
                worlds=worlds,
                residual_px=residual_px,
                underconstrained=False,
                kind="rolling",
                boundary_vel=(v_a, v_b),
            )

        # Airborne-anchor gate: when a resolved anchor node at either endpoint
        # places the ball clearly above the pitch plane (z > ground tol), the
        # span has DEPTH evidence that it is a flight.  Its ground ray-casts are
        # then a knowingly-wrong depth, so the grounded fallback must not engage
        # (it would undercut the anchor-backed FLIGHT with a pixel-exact but
        # depth-wrong roll).  Cede to FLIGHT by flagging the roll underconstrained
        # — exactly what happened before F16 on anchored airborne spans.  Anchor
        # worlds are frame-deterministic → cache-sound (F7).
        airborne_anchor = (
            (fa_anchor_world is not None
             and float(np.asarray(fa_anchor_world, float)[2]) > cfg.ground_z_tol_m)
            or (fb_anchor_world is not None
                and float(np.asarray(fb_anchor_world, float)[2]) > cfg.ground_z_tol_m)
        )
        if airborne_anchor:
            return SegmentFit(
                worlds=worlds,
                residual_px=residual_px,
                underconstrained=True,
                kind="rolling",
                boundary_vel=(v_a, v_b),
            )

        # Clean-flight gate: a high gravity arc's ground ray-casts fit a "roll"
        # in pixels just as well as the true flight (monocular degeneracy), so
        # the grounded fallback would tie a clean FLIGHT on residual and steal
        # the span on the mode tie-break.  Mirror piecewise's flight promotion +
        # apex test (``open_span_min_apex_m``): if a FLIGHT fit over the SAME
        # span is clean (not underconstrained) AND rises to a real apex, the ball
        # is airborne — cede to FLIGHT and do not emit depth-wrong ground worlds.
        # ``_fit_flight`` is memoized on the same frame-determined key, so this is
        # cheap and cache-sound (F7).
        if self._clean_flight_exists(fa, fb, fa_anchor_world, fb_anchor_world):
            return SegmentFit(
                worlds=worlds,
                residual_px=residual_px,
                underconstrained=True,
                kind="rolling",
                boundary_vel=(v_a, v_b),
            )

        # Raw-raycast plausibility gate: a high gravity arc (a bounce, a long
        # lob) projects to a ground track that SWEEPS implausibly fast — its
        # frame-to-frame ray-cast speed blows past any real ground pass.  The
        # smoothed roll-fit speed guard above under-reports this (the LM fit
        # damps it), so test the RAW ray-cast speeds directly: if the grounded
        # track is physically impossible the span is airborne and its ground
        # ray-casts are a wrong-depth teleport — cede to FLIGHT rather than emit
        # them.  This is what keeps the fallback from explaining a bounce arc or
        # a roll+flight span as one cheap grounded roll (mirror of piecewise's
        # rolling speed / open-span apex guards).
        if self._raw_raycast_implausibly_fast(fa, fb):
            return SegmentFit(
                worlds=worlds,
                residual_px=residual_px,
                underconstrained=True,
                kind="rolling",
                boundary_vel=(v_a, v_b),
            )

        # Fallback (F16): the clean roll does not apply (high residual or too
        # few obs), no airborne anchor, no clean flight, and the grounded track
        # is physically plausible — the ball IS detected on the ground, so cover
        # it with per-frame ground ray-casts and keep it grounded instead of
        # collapsing to OUT_OF_VIEW.
        return self._grounded_raycast_fallback(fa, fb, roll_fit=fit, T=T)

    def _raw_raycast_implausibly_fast(self, fa: int, fb: int) -> bool:
        """True when the span's RAW ground ray-cast track moves faster than any
        real ground pass (``rolling_max_speed_m_s``) — the tell-tale of an arc
        whose projection only *looks* grounded (monocular degeneracy).

        Speed is measured between consecutive in-pitch ray-cast frames (the same
        admission rules as the fallback), normalised by their frame gap so a
        single dropped frame does not inflate the estimate.  Frame-determined →
        cache-sound (F7).
        """
        solver = self._solver
        cfg: SolverCfg = solver.cfg
        fps = solver.fps
        prev_f: int | None = None
        prev_xy: np.ndarray | None = None
        for f in range(fa, fb + 1):
            ground = self._inpitch_raycast(f)
            if ground is None:
                continue
            xy = np.asarray(ground, float)[:2]
            if prev_f is not None and prev_xy is not None:
                speed = float(np.linalg.norm(xy - prev_xy)) * fps / (f - prev_f)
                if speed > cfg.rolling_max_speed_m_s:
                    return True
            prev_f, prev_xy = f, xy
        return False

    def _clean_flight_exists(
        self,
        fa: int,
        fb: int,
        fa_anchor_world: np.ndarray | None,
        fb_anchor_world: np.ndarray | None,
    ) -> bool:
        """True when a FLIGHT fit over ``[fa, fb]`` is a genuinely airborne
        explanation of the span's pixels — used to keep the grounded ray-cast
        fallback from stealing a true flight span on the residual tie (mirror of
        piecewise's ``open_span_min_apex_m`` flight test).

        We test the flight's PIXEL residual and its apex, NOT its
        ``underconstrained`` flag: a single long arc with no bounce overshoots
        the ground at its tail and is flagged underconstrained for that reason
        alone, even though it explains the pixels perfectly and is unmistakably
        airborne (the beam would emit it as a shorter sub-arc).  Distinguishing
        the two cases needs the residual+apex pair:

        - true airborne arc → low pixel residual AND a real apex
          (``>= open_span_min_apex_m``);
        - grounded jitter    → the flight cannot fit the pixels (high residual)
          and its recovered arc has an absurd / sub-ground apex.

        Delegates to the memoized :meth:`_fit_flight` so the cost is
        cache-amortised and frame-determined (cache-sound, F7).
        """
        cfg: SolverCfg = self._solver.cfg
        flight = self.fit_segment(
            fa, fb, Mode.FLIGHT,
            fa_anchor_world=fa_anchor_world, fb_anchor_world=fb_anchor_world,
        )
        if not flight.worlds:
            return False
        if flight.residual_px > cfg.flight_max_residual_px:
            return False
        apex = max(float(w[2]) for w in flight.worlds.values())
        return apex >= cfg.open_span_min_apex_m

    def _inpitch_raycast(self, f: int) -> np.ndarray | None:
        """Ground ray-cast for frame ``f``, gated by the same admission rules
        the roll/open-span loops use: skip gap-fill (synthesised) frames, frames
        with no valid ground ray-cast, and ray-casts landing outside the pitch +
        margin (avoids near-horizon teleports).  ``None`` when excluded.

        Mirrors ``ball_piecewise_solver._rolling_span`` /
        ``_solve_open_span``'s grounded guard (fix a181dd5).
        """
        solver = self._solver
        if f in solver.gap_fill:
            return None
        ground = solver._ground_raycast(f)
        if ground is None:
            return None
        m = solver.cfg.pitch_margin_m
        if not (
            -m <= ground[0] <= solver.pitch.length_m + m
            and -m <= ground[1] <= solver.pitch.width_m + m
        ):
            return None
        return ground

    def _grounded_raycast_fallback(
        self, fa: int, fb: int, roll_fit, T: float,
    ) -> SegmentFit:
        """Per-frame ground ray-cast coverage for a grounded-but-not-cleanly-
        rolling span (fix F16) — the global-solver mirror of piecewise's
        open-span grounded fallback.

        Ray-casts every frame in the inclusive range that survives
        :meth:`_inpitch_raycast`'s guards (no gap-fill, no off-pitch landing)
        AND does not carry a high IMM flight posterior (an airborne frame's
        ground ray-cast is a knowingly-wrong depth — piecewise's open-span loop
        refuses it too, lines ~1260–1264).  The accepted ray-casts ARE the
        worlds; ``_pixel_rms`` over them is ≈0 because each ray-cast reprojects
        to its own observed pixel, so the span decisively beats OUT_OF_VIEW.
        ``underconstrained`` is False when at least ``min_obs_for_lm_fit`` frames
        ray-cast cleanly (enough grounded evidence); otherwise the span has too
        little evidence and stays flagged.  Boundary velocities are a
        finite-difference of the ray-cast worlds.
        """
        solver = self._solver
        fps = solver.fps
        cfg: SolverCfg = solver.cfg

        worlds: dict[int, np.ndarray] = {}
        for f in range(fa, fb + 1):
            if solver.p_flight.get(f, 0.0) >= 0.5:
                # Airborne posterior: a grounded ray-cast would teleport the
                # ball to the wrong depth.  Leave the frame for FLIGHT/missing.
                continue
            ground = self._inpitch_raycast(f)
            if ground is None:
                continue
            worlds[f] = np.asarray(ground, dtype=float)

        if not worlds:
            # No grounded evidence at all — keep the roll fit's worlds so the
            # renderer is not empty, but flag it (same posture as the old path).
            z = cfg.ball_radius_m
            roll_worlds: dict[int, np.ndarray] = {}
            times = np.array([(f - fa) / fps for f in range(fa, fb + 1)])
            pts = roll_fit.eval(times, z)
            for i, f in enumerate(range(fa, fb + 1)):
                roll_worlds[f] = pts[i]
            return SegmentFit(
                worlds=roll_worlds,
                residual_px=0.0,
                underconstrained=True,
                kind="rolling",
                boundary_vel=(
                    self._roll_velocity(roll_fit, 0.0, z, fps),
                    self._roll_velocity(roll_fit, T, z, fps),
                ),
            )

        frames_sorted = sorted(worlds)
        resid = solver._pixel_rms(worlds, frames_sorted)
        residual_px = float(resid) if resid is not None else 0.0
        # Enough clean ray-casts → the grounded explanation is well-supported.
        underconstrained = len(worlds) < cfg.min_obs_for_lm_fit

        v_a = self._fd_world_velocity(worlds, frames_sorted, frames_sorted[0])
        v_b = self._fd_world_velocity(worlds, frames_sorted, frames_sorted[-1])
        return SegmentFit(
            worlds=worlds,
            residual_px=residual_px,
            underconstrained=underconstrained,
            kind="rolling",
            boundary_vel=(v_a, v_b),
        )

    def _fd_world_velocity(
        self,
        worlds: dict[int, np.ndarray],
        frames_sorted: list[int],
        centre: int,
    ) -> np.ndarray:
        """Finite-difference world velocity (m/s) at ``centre`` over the
        ray-cast worlds — uses the neighbouring covered frames."""
        fps = self._solver.fps
        idx = frames_sorted.index(centre)
        lo = frames_sorted[max(0, idx - 1)]
        hi = frames_sorted[min(len(frames_sorted) - 1, idx + 1)]
        if hi == lo:
            return np.zeros(3)
        return (worlds[hi] - worlds[lo]) / ((hi - lo) / fps)

    def _endpoint_xy(
        self, frame: int, obs: list[tuple[float, np.ndarray]],
        prefer_last: bool = False,
    ) -> np.ndarray:
        """Best xy estimate for a roll endpoint from frame-determined data.

        Uses that frame's ground ray-cast when available; falls back to the
        nearest interior observation, then to the origin (a degenerate
        zero-length roll the residual gate will flag).
        """
        g = self._solver._ground_raycast(frame)
        if g is not None:
            return g[:2]
        if obs:
            return np.asarray(obs[-1][1] if prefer_last else obs[0][1], float)
        return np.zeros(2)

    @staticmethod
    def _roll_velocity(fit, t_s: float, z: float, fps: float) -> np.ndarray:
        """Finite-difference velocity of the roll fit at ``t_s`` (m/s)."""
        h = 1.0 / fps
        t0 = max(0.0, t_s - h)
        t1 = min(fit.duration_s, t_s + h)
        if t1 <= t0:
            t0, t1 = 0.0, max(h, fit.duration_s)
        p = fit.eval(np.array([t0, t1]), z)
        return (p[1] - p[0]) / (t1 - t0)

    # ------------------------------------------------------------------
    # STATIONARY

    def _fit_stationary(self, fa: int, fb: int) -> SegmentFit:
        """Constant world = mean ground ray-cast of the interior obs."""
        solver = self._solver
        cfg: SolverCfg = solver.cfg
        pts: list[np.ndarray] = []
        for f in range(fa + 1, fb):
            g = solver._ground_raycast(f)
            if g is None or f in solver.gap_fill:
                continue
            pts.append(g)
        if not pts:
            # No grounded evidence: still anchor at the ball radius height
            # over a zero point so the renderer is non-empty; flag it.
            const = np.array([0.0, 0.0, cfg.ball_radius_m])
            worlds = {f: const.copy() for f in range(fa, fb + 1)}
            return SegmentFit(
                worlds=worlds, residual_px=0.0, underconstrained=True,
                kind="stationary",
                boundary_vel=(np.zeros(3), np.zeros(3)),
            )
        const = np.mean(np.stack(pts), axis=0)
        worlds = {f: const.copy() for f in range(fa, fb + 1)}
        resid = solver._pixel_rms(worlds, list(range(fa, fb + 1)))
        return SegmentFit(
            worlds=worlds,
            residual_px=float(resid) if resid is not None else 0.0,
            underconstrained=False,
            kind="stationary",
            boundary_vel=(np.zeros(3), np.zeros(3)),
        )

    # ------------------------------------------------------------------
    # POSSESSED

    # Contact bone used to tether the ball to the possessing player.
    # We query l_foot (the "lower" foot) as the primary contact bone;
    # r_foot is used as a fallback when l_foot is not available.
    _CONTACT_BONES: tuple[str, ...] = ("l_foot", "r_foot")

    def _fit_possessed(
        self,
        fa: int,
        fb: int,
        player_id: str | None,
    ) -> SegmentFit:
        """Tether the ball to a player's foot via PlayerContext FK.

        For each frame in ``[fa, fb]``:
        - ``worlds[f]`` is the possessing player's contact-joint world position
          from PlayerContext FK (``l_foot`` preferred, ``r_foot`` fallback).
        - Frames where FK is unavailable: hold the last known foot world
          (forward extrapolation); if no prior known world exists, interpolate
          toward the next available frame.  Held frames are counted; too many
          marks the segment underconstrained.
        - ``residual_px``: mean reprojection error between the projected foot
          world and the ball's observed pixel, computed via ``_pixel_rms``.
        - ``boundary_vel``: finite-difference of the foot world at ``fa`` and
          ``fb`` (design note F4).
        - ``kind`` is always ``"possessed"`` (state maps to ``"grounded"`` —
          design note F13).

        Parameters
        ----------
        fa, fb:
            Inclusive segment range.
        player_id:
            ID of the possessing player.  Must be non-None.

        Raises
        ------
        ValueError
            If ``player_id`` is None, or if this solver has no
            ``PlayerContext`` (was constructed without ``player_ctx``).
        """
        if self._player_ctx is None:
            raise ValueError(
                "POSSESSED mode requires a PlayerContext — pass "
                "player_ctx=<PlayerContext> to _SegmentSolver.__init__."
            )
        if player_id is None:
            raise ValueError(
                "POSSESSED mode requires player_id to be specified."
            )

        ctx = self._player_ctx
        solver = self._solver
        fps = solver.fps

        # --- 1. Collect FK foot worlds for all frames in [fa, fb] ---------
        # Primary bone: l_foot; fallback: r_foot.
        raw_worlds: dict[int, np.ndarray] = {}
        for f in range(fa, fb + 1):
            for bone in self._CONTACT_BONES:
                world = ctx.joint_world(f, player_id, bone)
                if world is not None:
                    raw_worlds[f] = np.asarray(world, dtype=float)
                    break

        # --- 2. Fill gaps: forward-hold then backward-hold ----------------
        # Forward pass: hold last known.
        worlds: dict[int, np.ndarray] = {}
        last_known: np.ndarray | None = None
        for f in range(fa, fb + 1):
            if f in raw_worlds:
                last_known = raw_worlds[f]
                worlds[f] = raw_worlds[f].copy()
            elif last_known is not None:
                worlds[f] = last_known.copy()

        # Backward pass: fill any remaining holes (frames before first known)
        # from the first available frame.
        first_known: np.ndarray | None = None
        for f in range(fa, fb + 1):
            if f in raw_worlds:
                first_known = raw_worlds[f]
                break
        if first_known is not None:
            for f in range(fa, fb + 1):
                if f not in worlds:
                    worlds[f] = first_known.copy()

        # --- 3. Underconstrained check ------------------------------------
        n_total = fb - fa + 1
        n_held = sum(1 for f in range(fa, fb + 1) if f not in raw_worlds)
        # More than half the frames held → underconstrained.
        underconstrained = n_held > n_total // 2

        # Edge-case: no FK at all for this player in this span.
        if not worlds:
            cfg: SolverCfg = solver.cfg
            fallback = np.array([0.0, 0.0, cfg.ball_radius_m])
            worlds = {f: fallback.copy() for f in range(fa, fb + 1)}
            return SegmentFit(
                worlds=worlds,
                residual_px=0.0,
                underconstrained=True,
                kind="possessed",
                boundary_vel=(np.zeros(3), np.zeros(3)),
            )

        # --- 4. Residual: pixel RMS between projected foot and ball obs ----
        obs_frames = [
            f for f in range(fa, fb + 1)
            if solver._pixel_rms({f: worlds[f]}, [f]) is not None
        ]
        resid = solver._pixel_rms(worlds, obs_frames)
        residual_px = float(resid) if (resid is not None and np.isfinite(resid)) else 0.0

        # --- 5. Boundary velocities (F4): finite-diff of foot world --------
        fps_f = float(fps)
        h_frames = max(1, min(2, (fb - fa) // 4))

        def _fd_vel(centre: int) -> np.ndarray:
            f0 = max(fa, centre - h_frames)
            f1 = min(fb, centre + h_frames)
            if f1 == f0:
                return np.zeros(3)
            return (worlds[f1] - worlds[f0]) / ((f1 - f0) / fps_f)

        v_a = _fd_vel(fa)
        v_b = _fd_vel(fb)

        return SegmentFit(
            worlds=worlds,
            residual_px=residual_px,
            underconstrained=underconstrained,
            kind="possessed",
            boundary_vel=(v_a, v_b),
        )


# ---------------------------------------------------------------------------
# Module-level helper: nearest players by projected foot pixel
# ---------------------------------------------------------------------------

def nearest_players(
    ball_uv: tuple[float, float],
    frame: int,
    player_ctx: "PlayerContext",
    k: int = 2,
) -> list[str]:
    """Return the ``k`` player IDs whose projected foot pixel is nearest to
    ``ball_uv`` at ``frame``, sorted nearest-first and deduplicated by player.

    Used by Task 6's possessed-player branching to identify which players to
    try as possessor candidates.

    Parameters
    ----------
    ball_uv:
        Ball pixel at ``frame`` (u, v) in image coordinates.
    frame:
        Frame index to query.
    player_ctx:
        A :class:`~src.utils.ball_player_context.PlayerContext` (or
        duck-typed substitute providing ``joints_at``).
    k:
        Maximum number of distinct player IDs to return.

    Returns
    -------
    list[str]
        Up to ``k`` player IDs, nearest projected foot pixel first.
        Empty if no player data is available at ``frame``.
    """
    u, v = float(ball_uv[0]), float(ball_uv[1])
    samples = player_ctx.joints_at(int(frame))

    # Accumulate best (nearest) distance per player_id.
    best: dict[str, float] = {}
    for s in samples:
        if s.uv is None:
            continue
        dist = ((s.uv[0] - u) ** 2 + (s.uv[1] - v) ** 2) ** 0.5
        if s.player_id not in best or dist < best[s.player_id]:
            best[s.player_id] = dist

    ordered = sorted(best.items(), key=lambda kv: kv[1])
    return [pid for pid, _ in ordered[:k]]


# ---------------------------------------------------------------------------
# Task 5 — Scoring functions
# ---------------------------------------------------------------------------

def segment_cost(
    fit: SegmentFit,
    mode: Mode,
    n_frames_in_segment: int,
    cfg: ModeSearchCfg,
) -> float:
    """Cost of assigning ``mode`` to a segment whose fitter returned ``fit``.

    Formula (all terms are additive):

    1. **Raw-pixel residual** — ``fit.residual_px`` used directly (design note
       F5: NOT gate-normalised; a 4 px FLIGHT fit must beat a 7 px ROLLING fit
       regardless of their respective gate thresholds).

    2. **Parsimony constant** — ``cfg.segment_cost_constant`` (BIC-style; one
       fixed cost per segment to penalise over-segmentation).

    3. **OUT_OF_VIEW penalty (fix C1c)** — when ``mode is Mode.OUT_OF_VIEW``:
       ``cfg.out_of_view_frame_penalty * fit.n_observed`` (the dominant term —
       one charge per frame in the span that carried a confident ball
       detection) PLUS ``cfg.out_of_view_missing_frame_penalty *
       (n_frames_in_segment − fit.n_observed)`` (a tiny per-frame term over the
       genuinely-missing frames so a long empty gap is not totally free).  Net
       effect: gapping a detection-rich span is expensive; gapping a span with
       no usable evidence is cheap — OUT_OF_VIEW becomes the LAST RESORT it
       should be.

    4. **Underconstrained flag penalty** — when ``fit.underconstrained`` is
       ``True``, add ``cfg.unexplained_break_penalty`` as a cost.  We reuse the
       ``unexplained_break_penalty`` magnitude (10.0) because both signal
       "something went wrong with this segment that a better partition would
       avoid": an underconstrained fit and an unexplained break are comparably
       undesirable.  A dedicated constant (e.g. ``underconstrained_penalty``)
       was considered but would bloat the config for no modelling benefit in v1.

    **Kind weight (v1)** — intentionally flat (1.0) for all modes (design note
    F6).  This avoids over-fitting the weight vector to the 3 validation clips.
    A non-uniform kind_weight would require per-mode calibration data; until
    that data exists, raw-pixel residual provides the only cross-mode
    discriminator.

    Parameters
    ----------
    fit:
        The :class:`SegmentFit` returned by the segment fitter.
    mode:
        The mode being scored (used for the OUT_OF_VIEW branch).
    n_frames_in_segment:
        Inclusive frame count ``fb - fa + 1`` for the OUT_OF_VIEW penalty.
    cfg:
        Scoring hyper-parameters.

    Returns
    -------
    float
        Non-negative cost (lower = better fit for this partition).
    """
    # --- Term 1: raw-px residual (F5) ---
    cost = float(fit.residual_px)

    # --- Term 2: BIC parsimony ---
    cost += cfg.segment_cost_constant

    # --- Term 3: OUT_OF_VIEW penalty (fix C1c) ---
    # Dominant cost: one charge per OBSERVED frame the gap would discard.
    # Tiny cost: per genuinely-missing frame, so an empty gap is not free.
    if mode is Mode.OUT_OF_VIEW:
        n_observed = int(fit.n_observed)
        n_missing = max(0, n_frames_in_segment - n_observed)
        cost += cfg.out_of_view_frame_penalty * n_observed
        cost += cfg.out_of_view_missing_frame_penalty * n_missing

    # --- Term 4: underconstrained flag ---
    if fit.underconstrained:
        # Reuse unexplained_break_penalty magnitude — see docstring.
        cost += cfg.unexplained_break_penalty

    return cost


def _velocity_continuity_delta(
    prev_mode: "Mode | None",
    next_mode: Mode,
    prev_end_vel: np.ndarray,
    next_start_vel: np.ndarray,
) -> float | None:
    """Δv magnitude to charge for velocity continuity across a transition.

    Mode-pair aware (fix C1a).  Returns the magnitude the caller multiplies by
    ``velocity_discontinuity_weight``, or ``None`` when no continuity penalty
    applies (a legitimate physical event whose velocity change is expected).

    Both velocities are assumed defined (the caller guards on that).

    Rules
    -----
    - flight→flight            : FULL ‖Δv‖ (mid-air change is suspicious).
    - flight→rolling/stationary: horizontal ‖Δv_xy‖ only (vertical kill is
      physical at a landing).
    - rolling/stationary→flight: ``None`` (kick/bounce-up launch — the velocity
      gain is the event itself).
    - any pair involving POSSESSED: ``None`` (ball velocity is player-driven).
    - rolling↔rolling / stationary↔rolling on the ground: horizontal ‖Δv_xy‖.
    - any other pair: FULL ‖Δv‖ (conservative default).
    """
    v_prev = np.asarray(prev_end_vel, float)
    v_next = np.asarray(next_start_vel, float)
    dv = v_next - v_prev
    full = float(np.linalg.norm(dv))
    horiz = float(np.linalg.norm(dv[:2]))

    _ground = (Mode.ROLLING, Mode.STATIONARY)

    # Possession: ball velocity is player-controlled — never penalise.
    if prev_mode is Mode.POSSESSED or next_mode is Mode.POSSESSED:
        return None

    # Launch off the ground (kick / bounce-up): the velocity gain is the event.
    if prev_mode in _ground and next_mode is Mode.FLIGHT:
        return None

    # Landing: vertical velocity is killed — penalise only horizontal slip.
    if prev_mode is Mode.FLIGHT and next_mode in _ground:
        return horiz

    # Mid-air continuation: a velocity change here is suspicious — full Δv.
    if prev_mode is Mode.FLIGHT and next_mode is Mode.FLIGHT:
        return full

    # Ground-to-ground: motion should stay continuous — horizontal Δv.
    if prev_mode in _ground and next_mode in _ground:
        return horiz

    # Conservative default (e.g. seed transition with prev_mode None): full Δv.
    return full


def transition_cost(
    prev_seg: "Segment | None",
    bp: Breakpoint,
    next_mode: Mode,
    prev_end_vel: "np.ndarray | None",
    next_start_vel: "np.ndarray | None",
    cfg: ModeSearchCfg,
) -> float:
    """Cost of inserting a mode transition at breakpoint ``bp``.

    This function is the **only** place where inter-segment structure is
    penalised; segment_cost handles intra-segment fit quality.

    Formula:

    1. **Event agreement** — depends on ``bp``:

       - ``bp.is_manual = True``: the operator explicitly placed this break;
         it is always legitimate.  No penalty, no bonus.
       - ``bp.kind == "boundary"``: synthetic clip-boundary breakpoints are
         neither events nor surprises (design note F17).  No penalty, no bonus.
       - ``bp.event_score`` high: transition coincides with a detected event →
         a **bonus** (negative cost) of ``−cfg.unexplained_break_penalty *
         bp.event_score``.  The bonus is scaled by event_score so a
         high-confidence touch/bounce is more rewarded than a marginal one.
       - ``bp.event_score ≈ 0`` (non-manual, non-boundary): the transition has
         no supporting event → ``+ cfg.unexplained_break_penalty``.

    2. **Velocity continuity (mode-pair aware — fix C1a)** — penalised by
       ``cfg.velocity_discontinuity_weight`` times a Δv magnitude that depends
       on the mode pair, ONLY when both velocities are non-None (design note
       F4).  The component charged differs because not every velocity change is
       a physics violation:

       - **flight→flight** (mid-air mode continuation): penalise the FULL
         ‖Δv‖ — a mid-air velocity change with no real event is suspicious.
       - **flight→rolling / flight→stationary** (a landing): penalise ONLY the
         horizontal component ‖Δv_xy‖.  The vertical velocity is killed by the
         landing — that is physical, not a discontinuity to be punished.
       - **rolling/stationary→flight** (a kick / bounce-up launch): NO velocity
         penalty — the launch is an event and the velocity gain is physical.
       - **any transition involving POSSESSED**: NO velocity penalty — ball
         velocity is player-controlled.
       - **rolling→rolling / stationary↔rolling on the ground**: penalise the
         horizontal ‖Δv_xy‖ — ground motion should stay continuous.
       - any other pair: full ‖Δv‖ (conservative default).

    3. **Restitution awareness (fix C1b)** — applied ONLY at a true bounce,
       i.e. a flight→flight transition where BOTH boundary velocities have a
       defined z-component: call ``ball_physics.restitution(prev_end_vel,
       next_start_vel)``.
       - If it returns ``None`` (ball not arriving downward) → no penalty.
       - If it returns a value outside ``[restitution_min, restitution_max]``
         (defaults 0.5–0.85 from SolverCfg) → a modest out-of-envelope
         penalty equal to ``cfg.velocity_discontinuity_weight * 5.0`` (one
         "bad" m/s Δv equivalent).
       A flight→rolling / flight→stationary *landing* has restitution ≈ 0
       (inelastic — the ball stops bouncing and rolls).  That is PHYSICAL, not
       a violation, so the restitution envelope is NOT checked there.

    Parameters
    ----------
    prev_seg:
        The :class:`Segment` ending at this breakpoint, or ``None`` for the
        very first (seed) transition.
    bp:
        The :class:`Breakpoint` at the transition frame.
    next_mode:
        The mode of the incoming segment.
    prev_end_vel:
        Velocity at the *end* of the previous segment (m/s, world coords).
        May be ``None`` when the previous segment is ``OUT_OF_VIEW`` or the
        fitter could not determine velocity.
    next_start_vel:
        Velocity at the *start* of the next segment.  Same caveats.
    cfg:
        Scoring hyper-parameters.

    Returns
    -------
    float
        Signed cost delta (may be negative = bonus).
    """
    # --- Term 1: event agreement ---
    cost = 0.0

    if bp.is_manual:
        # Forced break — always legitimate; neither penalty nor bonus.
        pass
    elif bp.kind == "boundary":
        # Synthetic clip-boundary — neutral (design note F17).
        pass
    elif bp.event_score > 0.0:
        # Event-backed transition: bonus proportional to confidence.
        # A high-confidence touch should be cheaper than a no-event break.
        cost -= cfg.unexplained_break_penalty * bp.event_score
    else:
        # No supporting event and not manual/boundary → unexplained break.
        cost += cfg.unexplained_break_penalty

    # --- Term 2: velocity continuity — mode-pair aware (fix C1a, F4) ---
    if prev_end_vel is not None and next_start_vel is not None:
        prev_mode = prev_seg.mode if prev_seg is not None else None
        delta_v = _velocity_continuity_delta(prev_mode, next_mode,
                                             prev_end_vel, next_start_vel)
        if delta_v is not None:
            cost += cfg.velocity_discontinuity_weight * float(delta_v)

        # --- Term 3: restitution check — flight→flight bounce ONLY (fix C1b) ---
        # A true bounce is flight→flight; restitution() returns None when the
        # ball is not arriving downward (v_in_z > -0.1).  A flight→ground
        # landing has restitution ≈ 0 (inelastic) which is physical, so the
        # envelope is NOT checked there.
        if prev_mode is Mode.FLIGHT and next_mode is Mode.FLIGHT:
            from src.utils.ball_physics import restitution as _restitution
            e = _restitution(prev_end_vel, next_start_vel)
            if e is not None:
                # Import SolverCfg defaults for the physical envelope.
                e_min = SolverCfg().restitution_min   # 0.5
                e_max = SolverCfg().restitution_max   # 0.85
                if not (e_min <= e <= e_max):
                    # Modest penalty: equivalent to ~5 m/s Δv anomaly.
                    cost += cfg.velocity_discontinuity_weight * 5.0

    return cost


def ignored_event_cost(bp: Breakpoint, cfg: ModeSearchCfg) -> float:
    """Cost the beam adds when it SKIPS a breakpoint without a matching transition.

    The beam loop calls this helper for every breakpoint it does *not*
    consume when extending a hypothesis — it charges the hypothesis for
    ignoring a potentially important event.

    Formula: ``cfg.ignored_event_penalty * bp.event_score``

    A low-confidence or zero-score breakpoint contributes little; a
    high-confidence event that the beam skips incurs the full
    ``ignored_event_penalty``.

    Note: Manual breakpoints are never skipped (they are forced partition
    points in the beam), so in practice the beam never calls this for them.
    The function is stateless and returns a non-zero value for any
    ``bp.event_score > 0``, regardless of ``is_manual``, to keep the helper
    simple and composable.

    Parameters
    ----------
    bp:
        The :class:`Breakpoint` being skipped.
    cfg:
        Scoring hyper-parameters.

    Returns
    -------
    float
        Non-negative cost to be added to the skipping hypothesis.
    """
    return cfg.ignored_event_penalty * float(bp.event_score)


# ---------------------------------------------------------------------------
# Task 6 — left-to-right beam search loop
# ---------------------------------------------------------------------------

# Cost-quantization precision for the deterministic tie-break key (F10).
# Float noise below this magnitude must NOT flip hypothesis ordering.
_COST_QUANT_DP = 6


@dataclass(frozen=True, eq=False)
class BeamResult:
    """Outcome of :func:`run_beam`.

    Attributes
    ----------
    winner:
        Lowest-cost COMPLETE hypothesis (spans frame 0 .. n_frames-1).
    runner_up:
        Lowest-cost complete hypothesis whose segment partition (breakpoint
        set OR any segment's mode) differs from the winner (design note F11);
        ``None`` when only one partition reached the end.
    hypotheses_explored:
        Total number of candidate hypotheses generated (pre-pruning) — a
        coarse search-effort metric for diagnostics.
    fit_calls:
        Number of real (non-cached) segment fits charged to the solver's
        budget over the whole search.
    """

    winner: Hypothesis
    runner_up: Hypothesis | None
    hypotheses_explored: int
    fit_calls: int


def _eligible_modes(
    seg_solver: "_SegmentSolver",
    fa: int,
    ball_uv_at,
) -> list[tuple[Mode, str | None]]:
    """Return ``(mode, player_id)`` extension candidates for a span at ``fa``.

    ROLLING / FLIGHT / STATIONARY / OUT_OF_VIEW are always candidates (each
    with ``player_id=None``).  POSSESSED is added — one branch per player
    returned by :func:`nearest_players` over the last confident ball pixel at
    ``fa`` — only when a :class:`PlayerContext` is available AND a ball pixel
    exists at ``fa``.  OUT_OF_VIEW is always present as a fallback so a span
    with no usable evidence can still be explained.
    """
    cands: list[tuple[Mode, str | None]] = [
        (Mode.ROLLING, None),
        (Mode.FLIGHT, None),
        (Mode.STATIONARY, None),
        (Mode.OUT_OF_VIEW, None),
    ]
    ctx = seg_solver._player_ctx
    if ctx is not None and ball_uv_at is not None:
        uv = ball_uv_at(fa)
        if uv is not None:
            for pid in nearest_players(uv, fa, ctx, k=2):
                cands.append((Mode.POSSESSED, pid))
    return cands


def _comparable_partition_sig(
    hyp: Hypothesis,
) -> tuple[tuple[int, int, int, str], ...]:
    """Totally-orderable partition signature (fix C2).

    Same content as :func:`_partition_sig` but ``player_id`` is normalised to
    ``""`` (never ``None``) so the tuple is comparable with ``<`` — ``None`` is
    not orderable against ``str`` inside a tuple.  Used as the final tie-break
    field in :func:`_quant_key`.
    """
    return tuple(
        (s.fa, s.fb, int(s.mode), s.player_id or "") for s in hyp.segments
    )


def _quant_key(
    hyp: Hypothesis,
) -> tuple[float, int, int, tuple[tuple[int, int, int, str], ...]]:
    """Deterministic, TOTAL ordering key (design notes F10 + fix C2).

    Cost is quantized to ``_COST_QUANT_DP`` decimals so float noise below that
    magnitude cannot flip the ordering; ties then break by the last consumed
    frame, then by the current mode's integer value (Mode ordering), then —
    as a final STRUCTURAL tie-break — by the full partition signature
    (per-segment ``(fa, fb, mode_int, player_id)``).

    Without the partition-signature field two genuinely-different partitions
    could tie on ``(cost, last_frame, mode_int)`` and resolve by Python list
    append order, making the winner depend on hypothesis-insertion order rather
    than on structure.  The signature makes the order total and reproducible.
    """
    last_frame = hyp.segments[-1].fb if hyp.segments else -1
    mode_int = int(hyp.cur_mode) if hyp.cur_mode is not None else -1
    return (
        round(float(hyp.cost), _COST_QUANT_DP),
        last_frame,
        mode_int,
        _comparable_partition_sig(hyp),
    )


def _collapse_dominated(hyps: list[Hypothesis]) -> list[Hypothesis]:
    """Dominance collapse: among hypotheses ending at the same breakpoint with
    the same ``(cur_mode, player_id)`` keep only the lowest-cost one.

    HEURISTIC — beam approximation, NOT a strict dominance relation (fix C3).
    The next extension's :func:`transition_cost` reads ``end_velocity`` (for the
    velocity-continuity and restitution terms), so two hypotheses with the same
    ``(last_bp_idx, cur_mode, player_id)`` but DIFFERENT ``end_velocity`` are
    NOT strictly interchangeable — a higher-cost-so-far hypothesis with an
    end-velocity that better matches the next segment could, in principle, win
    overall.  Collapsing them by dropping ``end_velocity`` is a deliberate beam
    pruning trade-off (it bounds branching at the cost of occasionally pruning a
    hypothesis that a velocity-aware exact search would keep).  In practice the
    surviving lowest-cost-so-far hypothesis is a strong proxy and the beam width
    plus the breakpoint structure recover the right partition; the multi-impact
    and determinism tests guard against regressions.

    Deterministic: when costs tie within the quantization, the total
    ``_quant_key`` ordering (including the structural partition signature, fix
    C2) picks the survivor — never list append order.
    """
    best: dict[tuple[int, int, str | None], Hypothesis] = {}
    for h in hyps:
        mode_int = int(h.cur_mode) if h.cur_mode is not None else -1
        key = (h.last_bp_idx, mode_int, h.player_id)
        cur = best.get(key)
        if cur is None or _quant_key(h) < _quant_key(cur):
            best[key] = h
    return list(best.values())


def _partition_sig(hyp: Hypothesis) -> tuple[tuple[int, int, int, str | None], ...]:
    """Signature distinguishing partitions (F11): per-segment frame range +
    mode + possessing player.  Two hypotheses with the same signature are the
    same partition even if their float costs differ infinitesimally."""
    return tuple(
        (s.fa, s.fb, int(s.mode), s.player_id) for s in hyp.segments
    )


def run_beam(
    seg_solver: "_SegmentSolver",
    breakpoints: list[Breakpoint],
    n_frames: int,
    cfg: ModeSearchCfg,
    *,
    ball_uv_at,
    node_world_by_frame: "Mapping[int, np.ndarray] | None" = None,
) -> BeamResult:
    """Left-to-right beam search over timeline partitions.

    Algorithm
    ---------
    The sorted ``breakpoints`` list (frame-0 boundary + manual anchors +
    event candidates + final boundary, all SORTED by frame, supplied by the
    caller) defines the partition lattice.  Hypotheses are extended one
    *segment* at a time between consecutive *chosen* breakpoints:

    - The seed hypothesis sits at breakpoint index 0 (the frame-0 boundary)
      with no committed segment and zero cost.
    - A hypothesis ending at breakpoint ``i`` may extend to breakpoint ``j``
      with ``i < j <= i + 1 + cfg.max_skip``.  The segment covers
      ``[bp_i.frame, bp_j.frame]``; the ``j - i - 1`` intermediate
      breakpoints are *absorbed* (skipped) and each is charged
      :func:`ignored_event_cost`.  This bounds branching (the span absorbs at
      most ``max_skip`` breaks) while letting the beam merge spurious velocity
      breaks into one physical segment.
    - For each extension, every ``(mode, player_id)`` from
      :func:`_eligible_modes` is tried.  Cost increment =
      ``segment_cost(fit) + Σ ignored_event_cost(skipped) +
      transition_cost(at bp_j)``.

    Manual-anchor forcing (pruning, not penalty)
    --------------------------------------------
    A segment may NOT span across a manual-anchor frame: any extension whose
    span ``[bp_i.frame, bp_j.frame]`` strictly contains a breakpoint with
    ``is_manual=True`` is pruned outright.  Because manual anchors appear in
    the breakpoint list, this also forces them to be chosen as boundaries:
    the only way past a manual anchor is to stop a segment exactly on it.

    Beam pruning + dominance
    ------------------------
    Per breakpoint column the live hypotheses are dominance-collapsed (same
    ``(last_bp_idx, cur_mode, player_id)`` → keep lowest cost), then sorted by
    the quantized key ``(round(cost, 6), last_frame, cur_mode_int)`` (F10) and
    truncated to ``cfg.beam_width``.

    Completion + runner-up
    ----------------------
    A hypothesis is *complete* when it ends at the final breakpoint index
    (frame ``n_frames - 1``).  The winner is the lowest-cost complete
    hypothesis; the runner-up is the lowest-cost complete hypothesis whose
    partition signature (breakpoint set OR any segment's mode/player) differs
    from the winner's (F11), or ``None`` if only one partition reached the end.

    Determinism: no randomness; every ordering uses the quantized key.

    Resolved-anchor depth (regression fix)
    --------------------------------------
    ``node_world_by_frame`` maps a resolved trajectory-node frame (manual
    anchor OR auto-anchor) to its resolved world ``xyz``.  When a segment
    boundary ``fa``/``fb`` lands on such a node, that node's world is passed
    to ``fit_segment`` as ``fa_anchor_world``/``fb_anchor_world`` — pinning the
    flight arc's endpoint exactly as piecewise's ``_fit_arc`` does, so the
    anchor DEPTH constraint is honoured across the span (without it, anchored
    flight spans fit endpoint-free → underconstrained/high-residual → lose to
    OUT_OF_VIEW).  NON-node breakpoints (events / velocity-breaks) get no knot
    (``.get`` returns ``None``) and stay endpoint-free.  A node's world is
    *frame-deterministic* (a function of the frame index, not of the search
    path), so the cache key's ``fa_anchor``/``fb_anchor`` booleans plus the
    frame fully determine the knot — the cache stays sound (design note F7).

    Design decision — skip ∩ manual-anchor interaction
    --------------------------------------------------
    When a manual anchor is one of the intermediate breakpoints of a candidate
    span, the span is pruned (it would strictly contain the manual frame).
    The manual anchor therefore can never be "absorbed" — it always ends a
    segment.  ``max_skip`` only ever absorbs non-manual candidate breaks.

    Raises
    ------
    ValueError
        If fewer than two breakpoints are supplied, or if the breakpoint
        frames are not STRICTLY increasing (fix C4) — duplicate or
        out-of-order frames produce zero/negative-length spans that the
        ``fb <= fa`` skip silently drops, which can dead-end a column and
        leave the clip unexplainable.  The caller must dedup/sort first.
    BudgetExceeded
        Propagated unchanged from ``seg_solver.fit_segment`` so Task 8 can
        catch it and fall back to the whole-shot piecewise solver (F8).
    """
    if len(breakpoints) < 2:
        raise ValueError(
            "run_beam requires at least two breakpoints (frame-0 + final "
            "boundary)."
        )

    # Resolved-node worlds keyed by frame (manual + auto anchors).  Each value
    # is frame-deterministic, so passing it as a flight knot is cache-sound.
    node_worlds: "Mapping[int, np.ndarray]" = node_world_by_frame or {}

    # Fix C4: breakpoint frames must be strictly increasing.  A duplicate or
    # out-of-order frame yields a zero/negative-length span (fb <= fa) that the
    # extension loop silently skips, which can dead-end a column and produce no
    # complete hypothesis.  Fail fast with a clear message instead.
    frames = [bp.frame for bp in breakpoints]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError(
            "run_beam requires breakpoints with strictly increasing frames; "
            f"got {frames} (duplicate or out-of-order). Dedup/sort first."
        )

    n_bp = len(breakpoints)
    final_idx = n_bp - 1

    # Precompute manual-frame set for fast strict-containment pruning.
    manual_frames = sorted(bp.frame for bp in breakpoints if bp.is_manual)

    def _spans_manual(fa: int, fb: int) -> bool:
        """True if any manual anchor lies strictly inside (fa, fb)."""
        for mf in manual_frames:
            if fa < mf < fb:
                return True
        return False

    # Seed: empty hypothesis at breakpoint index 0.
    seed = Hypothesis(
        last_bp_idx=0,
        cur_mode=None,
        segments=(),
        cost=0.0,
        end_velocity=None,
        player_id=None,
    )

    # Column-keyed live beams: column index -> list[Hypothesis] ending there.
    # We process columns in increasing order; extensions only ever move
    # forward, so a single forward sweep suffices.
    columns: dict[int, list[Hypothesis]] = {0: [seed]}
    completed: list[Hypothesis] = []
    hypotheses_explored = 0

    for i in range(0, final_idx):
        live = columns.get(i)
        if not live:
            continue
        # Dominance-collapse + beam-prune this column before expanding it.
        live = _collapse_dominated(live)
        live.sort(key=_quant_key)
        live = live[: cfg.beam_width]
        columns[i] = live

        bp_i = breakpoints[i]
        fa = bp_i.frame
        modes = _eligible_modes(seg_solver, fa, ball_uv_at)

        j_max = min(final_idx, i + 1 + cfg.max_skip)
        for j in range(i + 1, j_max + 1):
            bp_j = breakpoints[j]
            fb = bp_j.frame
            if fb <= fa:
                continue
            if _spans_manual(fa, fb):
                # Strictly contains a manual anchor — forbidden span; and any
                # j beyond also would, so stop extending from this i.
                break
            # Skipped (absorbed) intermediate breakpoints i+1 .. j-1.
            skip_cost = sum(
                ignored_event_cost(breakpoints[s], cfg)
                for s in range(i + 1, j)
            )

            # Pin endpoints to resolved anchor worlds where a segment boundary
            # lands on a node (regression fix).  Non-node breakpoints return
            # None and stay endpoint-free.  The fitter only consumes these for
            # FLIGHT (depth-critical) and ROLLING (anchor XY); other modes
            # ignore them.  Worlds are frame-deterministic → cache-sound (F7).
            fa_world = node_worlds.get(fa)
            fb_world = node_worlds.get(fb)
            for mode, pid in modes:
                fit = seg_solver.fit_segment(
                    fa, fb, mode, player_id=pid,
                    fa_anchor_world=fa_world, fb_anchor_world=fb_world,
                )
                n_seg = fb - fa + 1
                s_cost = segment_cost(fit, mode, n_seg, cfg)

                new_seg = Segment(
                    fa=fa,
                    fb=fb,
                    mode=mode,
                    worlds=fit.worlds,
                    residual_px=fit.residual_px,
                    underconstrained=fit.underconstrained,
                    kind=fit.kind,
                    player_id=pid,
                    boundary_vel=fit.boundary_vel,
                )

                for h in live:
                    prev_seg = h.segments[-1] if h.segments else None
                    next_start_vel = fit.boundary_vel[0]
                    # The transition occurs where the NEW segment begins —
                    # breakpoint ``i`` (frame ``fa``) — between the previous
                    # segment and this one.  ``transition_cost`` scores that
                    # break's event-agreement and the velocity continuity from
                    # the previous segment's end velocity into this segment's
                    # start velocity.
                    t_cost = transition_cost(
                        prev_seg, bp_i, mode,
                        h.end_velocity, next_start_vel, cfg,
                    )
                    new_cost = h.cost + s_cost + skip_cost + t_cost
                    new_hyp = Hypothesis(
                        last_bp_idx=j,
                        cur_mode=mode,
                        segments=h.segments + (new_seg,),
                        cost=new_cost,
                        end_velocity=fit.boundary_vel[1],
                        player_id=pid,
                    )
                    hypotheses_explored += 1
                    if j == final_idx:
                        completed.append(new_hyp)
                    else:
                        columns.setdefault(j, []).append(new_hyp)

    if not completed:
        raise RuntimeError(
            "beam search produced no complete hypothesis spanning the clip — "
            "check that the breakpoint list includes both clip boundaries."
        )

    completed.sort(key=_quant_key)
    winner = completed[0]
    win_sig = _partition_sig(winner)
    runner_up: Hypothesis | None = None
    for h in completed[1:]:
        if _partition_sig(h) != win_sig:
            runner_up = h
            break

    return BeamResult(
        winner=winner,
        runner_up=runner_up,
        hypotheses_explored=hypotheses_explored,
        fit_calls=seg_solver._fit_calls,
    )


# ---------------------------------------------------------------------------
# Task 7 — render winner to SolveResult + public ``solve_modes`` entry point
# ---------------------------------------------------------------------------

# Default minimum length (frames) for an OUT_OF_VIEW span to be surfaced to the
# operator-feedback loop. A span shorter than this is normal detection jitter
# and is not worth flagging; a longer one is a real evidence gap the operator
# may want to anchor (F14). Read from ``cfg.mode_search.out_of_view_min_span``
# when present, else this constant.
_OUT_OF_VIEW_MIN_SPAN = 3


def _breakpoint_priority(bp: Breakpoint) -> int:
    """Dedup priority: manual (2) > event (1) > boundary (0).

    When two breakpoints land on the same frame the highest-priority one is
    kept (a manual anchor at an event frame stays manual/hard).
    """
    if bp.is_manual:
        return 2
    if bp.kind == "boundary":
        return 0
    return 1


def build_breakpoints(
    *,
    nodes: "Sequence[TrajectoryNode]",
    events: "Sequence[BallEvent]",
    n_frames: int,
) -> list[Breakpoint]:
    """Assemble the sorted, deduplicated breakpoint list for the beam (F2).

    NOT via ``_audit_auto_nodes`` (which self-references the greedy
    ``_solve_span``). Instead:

    * Synthetic clip-boundary breakpoints at frame 0 and ``n_frames - 1``
      (``kind="boundary"``, ``event_score=0.0``, ``is_manual=False`` — F17).
    * Manual anchors (``node.is_manual``) → HARD breakpoints
      (``is_manual=True``, ``event_score=1.0``). Manual anchor frames are the
      only forced partition points in the beam.
    * Event candidates from :func:`detect_event_candidates` (permissive) →
      ``kind = event.kind``, ``event_score = event.score``, ``is_manual=False``.

    Non-manual auto nodes are NOT added as breakpoints: per F2 they are
    *candidates* the beam ranks, and the event detector already surfaces the
    velocity breaks behind them; pinning every auto node would re-introduce the
    audit's hard-knot dependence. Only manual anchors hard-pin.

    Dedup is by frame, keeping the highest-priority breakpoint
    (manual > event > boundary). The result is STRICTLY increasing by frame —
    the precondition ``run_beam`` enforces.
    """
    by_frame: dict[int, Breakpoint] = {}

    def _add(bp: Breakpoint) -> None:
        f = int(bp.frame)
        if not (0 <= f < n_frames):
            return
        existing = by_frame.get(f)
        if existing is None or _breakpoint_priority(bp) > _breakpoint_priority(existing):
            by_frame[f] = bp

    # Boundaries first (lowest priority — any event/manual on these frames wins).
    _add(Breakpoint(frame=0, kind="boundary", event_score=0.0, is_manual=False))
    last = max(n_frames - 1, 0)
    _add(Breakpoint(frame=last, kind="boundary", event_score=0.0,
                    is_manual=False))

    # Manual anchors → hard breakpoints.
    for node in nodes:
        if node.is_manual:
            _add(Breakpoint(frame=int(node.frame), kind=node.state or "manual",
                            event_score=1.0, is_manual=True))

    # Event candidates → ranked breakpoints.
    for ev in events:
        score = float(getattr(ev, "score", 0.0))
        _add(Breakpoint(
            frame=int(ev.frame),
            kind=str(getattr(ev, "kind", "event")),
            event_score=max(0.0, min(1.0, score)),
            is_manual=False,
        ))

    return [by_frame[f] for f in sorted(by_frame)]


def _ball_uv_at(steps) -> "callable":
    """Closure returning the confident ball pixel at a frame (or None).

    Mirrors ``_Solver``'s ``uvs`` map: last confident uv per frame (a frame
    has at most one step, so this is just the step's uv).
    """
    uvs: dict[int, tuple[float, float]] = {}
    for s in steps:
        if getattr(s, "uv", None) is not None:
            uvs[int(s.frame)] = (float(s.uv[0]), float(s.uv[1]))

    def lookup(frame: int):
        return uvs.get(int(frame))

    return lookup


def _segment_to_outcome(seg: Segment, fps: float) -> _SpanOutcome:
    """Translate a winning beam :class:`Segment` into a piecewise
    :class:`_SpanOutcome` so the Task-0 :func:`commit_span` assembler can
    render it into the EXACT ``SolveResult`` shape ``solve_piecewise`` produces.

    Mode → kind → state mapping (the kind drives ``_KIND_STATE`` in
    ``commit_span``):

    * FLIGHT      → ``"ballistic"`` → state ``"flight"``; an ``_Arc`` is built
      from ``worlds[fa]`` (p0) and ``boundary_vel[0]`` (v0) so a FlightSegment
      is emitted and the frames get a non-null ``flight_segment_id``.
    * ROLLING     → ``"rolling"``    → state ``"grounded"``.
    * POSSESSED   → ``"possessed"``  → state ``"grounded"`` (F13). NO arc, so
      these frames are never ``"flight"``.
    * STATIONARY  → ``"stationary"`` → state ``"grounded"``.
    * OUT_OF_VIEW → ``"out_of_view"``→ state ``"missing"`` (F12); empty worlds,
      so no frames are written and they keep the default ``"missing"``.
    """
    out = _SpanOutcome(kind=seg.kind)
    out.worlds = {f: np.asarray(w, dtype=float) for f, w in seg.worlds.items()}
    out.underconstrained = bool(seg.underconstrained)
    out.residual_px = float(seg.residual_px) if seg.worlds else None

    if seg.mode is Mode.FLIGHT and seg.worlds:
        v0 = seg.boundary_vel[0]
        p0 = seg.worlds.get(seg.fa)
        if v0 is not None and p0 is not None:
            out.arcs = [_Arc(
                fa=seg.fa, fb=seg.fb,
                p0=np.asarray(p0, dtype=float),
                v0=np.asarray(v0, dtype=float),
                residual_px=float(seg.residual_px),
                n_obs=len(seg.worlds),
            )]
    return out


def _out_of_view_spans(
    winner: Hypothesis,
    min_span: int,
) -> list[dict[str, int]]:
    """OUT_OF_VIEW segments longer than ``min_span`` frames, as {start,end}.

    Surfaced for operator feedback (F12/F14): a long span the global solver
    could not explain with any physical mode is a cue for a manual anchor.
    """
    spans: list[dict[str, int]] = []
    for seg in winner.segments:
        if seg.mode is Mode.OUT_OF_VIEW and (seg.fb - seg.fa + 1) > min_span:
            spans.append({"start": int(seg.fa), "end": int(seg.fb)})
    return spans


def render(
    winner: Hypothesis,
    beam: BeamResult,
    *,
    nodes: "Sequence[TrajectoryNode]",
    fps: float,
    n_frames: int,
    cfg: SolverCfg,
    mode_cfg: ModeSearchCfg,
    confidences: "Mapping[int, float] | None" = None,
    gap_fill: set[int] | None = None,
    out_of_view_min_span: int = _OUT_OF_VIEW_MIN_SPAN,
) -> SolveResult:
    """Assemble the winning hypothesis into the EXACT ``SolveResult`` shape.

    Reuses the Task-0 free functions (:func:`commit_span`,
    :func:`apply_node_authority`, :func:`apply_restitution_flags`) so the dense
    ``world_by_frame``, full-frame ``state_by_frame`` (default ``"missing"``),
    ``flight_segments`` and ``diagnostics["segments"]`` come out byte-shape
    identical to ``solve_piecewise``.

    Adds the Phase-2 diagnostics: ``mode_search`` (search-effort metrics) and
    ``out_of_view_spans`` (also appended to ``underconstrained_spans`` — F14).
    """
    st = _CommitState(
        world_by_frame={},
        state_by_frame={f: "missing" for f in range(n_frames)},
        segments=[],
        diagnostics={
            "segments": [],
            "bounces": [],
            "underconstrained_spans": [],
            "splits": 0,
        },
        arcs_by_bound={},
    )

    # Commit each winning segment via the shared assembler.
    for seg in winner.segments:
        outcome = _segment_to_outcome(seg, fps)
        # Confidence: 0.9 weight on the segment's worlds (mirrors the
        # node-bracketed span confidence in solve_piecewise; OUT_OF_VIEW writes
        # no worlds so its conf is unused).
        commit_span(
            st, outcome, seg.fa, seg.fb, conf=0.9,
            n_frames=n_frames,
            confidences=confidences, gap_fill=gap_fill,
        )

    # Nodes are authoritative (manual anchors win their exact world + state).
    apply_node_authority(st, list(nodes), n_frames, cfg)
    apply_restitution_flags(st, list(nodes), fps, cfg)

    # Per-frame flight-segment id for the BallTrack writer / overlay.

    # Phase-2 search diagnostics.
    runner = beam.runner_up
    st.diagnostics["mode_search"] = {
        "hypotheses_explored": int(beam.hypotheses_explored),
        "beam_width": int(mode_cfg.beam_width),
        "winning_cost": float(winner.cost),
        "runner_up_cost": (float(runner.cost) if runner is not None else None),
        "fit_calls": int(beam.fit_calls),
        "budget_hit": False,
    }

    # Out-of-view spans (operator feedback). Long ones ALSO surface as
    # underconstrained spans so the feedback loop flags them (F14).
    oov = _out_of_view_spans(winner, out_of_view_min_span)
    st.diagnostics["out_of_view_spans"] = oov
    existing = {
        (u.get("start"), u.get("end"))
        for u in st.diagnostics["underconstrained_spans"]
    }
    for span in oov:
        key = (span["start"], span["end"])
        if key not in existing:
            st.diagnostics["underconstrained_spans"].append({
                "start": span["start"], "end": span["end"],
                "residual_px": None,
            })
            existing.add(key)

    return SolveResult(
        world_by_frame=st.world_by_frame,
        state_by_frame=st.state_by_frame,
        flight_segments=tuple(st.segments),
        diagnostics=st.diagnostics,
    )


def solve_modes(
    *,
    nodes: "Sequence[TrajectoryNode]",
    steps,
    confidences: "Mapping[int, float]",
    per_frame_K: "Mapping[int, np.ndarray]",
    per_frame_R: "Mapping[int, np.ndarray]",
    per_frame_t: "Mapping[int, np.ndarray]",
    distortion: tuple[float, float],
    fps: float,
    n_frames: int,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    split_hints: "Sequence[tuple[int, float]]" = (),
    z_hints: "Mapping[int, tuple[float, float]] | None" = None,
    manual_obs_frames: set[int] | None = None,
    world_fixes: "Mapping[int, tuple[np.ndarray, float]] | None" = None,
    cfg: SolverCfg | None = None,
    mode_search_cfg: ModeSearchCfg | None = None,
    player_ctx: "PlayerContext | None" = None,
    events: "Sequence[BallEvent] | None" = None,
) -> SolveResult:
    """Global mode-sequence ball solve. Mirrors ``solve_piecewise``'s signature
    and ``SolveResult`` return (the call-site swap is one line in ``ball.py``).

    Pipeline
    --------
    1. Build a :class:`_SegmentSolver` from the kwargs (same reader rules as
       ``_Solver``); ``player_ctx`` enables POSSESSED branches.
    2. :func:`build_breakpoints` — boundaries + manual-anchor hard breaks +
       permissive event candidates (F2; NOT ``_audit_auto_nodes``). When
       ``events`` is None and a ``player_ctx`` is available the candidates are
       computed here via :func:`detect_event_candidates`.
    3. :func:`run_beam` over the breakpoints (``BudgetExceeded`` propagates for
       Task 8's whole-shot piecewise fallback — F8).
    4. :func:`render` the winner into the EXACT ``SolveResult`` shape with the
       Task-0 assembler + Phase-2 diagnostics.

    Raises
    ------
    BudgetExceeded
        From the segment fitter past ``max_segment_fit_calls`` — caught by the
        stage seam (Task 8) which falls back to ``solve_piecewise``.
    """
    solver_cfg = cfg or SolverCfg()
    mode_cfg = mode_search_cfg or ModeSearchCfg()

    seg_solver = _SegmentSolver(
        mode_cfg=mode_cfg,
        player_ctx=player_ctx,
        nodes=nodes, steps=steps, confidences=confidences,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R,
        per_frame_t=per_frame_t, distortion=distortion,
        fps=fps, n_frames=n_frames,
        pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
        split_hints=split_hints, z_hints=z_hints,
        manual_obs_frames=manual_obs_frames,
        world_fixes=world_fixes,
        cfg=solver_cfg,
    )

    # Event candidates: prefer caller-supplied; else compute permissively when
    # a player context is available (the detector needs it for touch
    # classification). Without a player_ctx, fall back to no events — the beam
    # still resolves the partition from boundaries + manual anchors.
    if events is None and player_ctx is not None:
        from src.utils.ball_auto_events import detect_event_candidates
        events = detect_event_candidates(
            steps=steps,
            confidences=dict(confidences),
            player_ctx=player_ctx,
            per_frame_K=per_frame_K,
            per_frame_R=per_frame_R,
            per_frame_t=per_frame_t,
            distortion=distortion,
            profile="permissive",
        )
    events = events or ()

    breakpoints = build_breakpoints(
        nodes=nodes, events=events, n_frames=n_frames,
    )

    # Resolved-node worlds keyed by frame — manual anchors AND auto-anchors.
    # A node's resolved world is frame-deterministic (a function of the frame,
    # not the search path), so it is safe to pass as a flight/rolling knot:
    # the segment-fit cache key carries fa_anchor/fb_anchor booleans, and
    # frame + bool fully determine the knot (design note F7).  Threading these
    # pins anchored flight-segment endpoints so the depth constraint survives
    # the beam (without it, anchored spans fit endpoint-free and lose to
    # OUT_OF_VIEW — the coverage regression this fixes).
    node_world_by_frame: dict[int, np.ndarray] = {
        int(n.frame): np.asarray(n.world_xyz, float)
        for n in nodes
        if n.world_xyz is not None
    }

    ball_uv_at = _ball_uv_at(steps)
    beam = run_beam(
        seg_solver, breakpoints, n_frames, mode_cfg,
        ball_uv_at=ball_uv_at,
        node_world_by_frame=node_world_by_frame,
    )

    return render(
        beam.winner, beam,
        nodes=nodes, fps=fps, n_frames=n_frames,
        cfg=solver_cfg, mode_cfg=mode_cfg,
        confidences=dict(confidences),
        gap_fill=seg_solver._solver.gap_fill,
        out_of_view_min_span=getattr(
            mode_cfg, "out_of_view_min_span", _OUT_OF_VIEW_MIN_SPAN,
        ),
    )

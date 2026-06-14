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
from src.utils.ball_piecewise_solver import (  # noqa: F401
    FlightSegment,
    SolveResult,
    SolverCfg,
    _Solver,
)
from src.utils.ball_physics import (
    G_VEC,
    eval_parabola,
    fit_rolling_segment,
    parabola_end_velocity,
)
from src.utils.bundle_adjust import fit_parabola_to_image_observations


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
    """

    worlds: dict[int, np.ndarray]
    residual_px: float
    underconstrained: bool
    kind: str
    boundary_vel: tuple[np.ndarray | None, np.ndarray | None]


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
    pins ``a[:2]`` / ``b[:2]`` as exact roll endpoints.  So FLIGHT calls
    ``fit_parabola_to_image_observations`` directly with NO endpoint knot
    (unless ``fa``/``fb`` is a *manual* anchor, whose world is deterministic
    from its frame), and ROLLING calls ``fit_rolling_segment`` over ground
    ray-casts seeded from the first/last interior observation rather than
    pinned node endpoints.  This keeps the fit a pure function of
    ``(fa, fb, mode, interior obs, in-range fixes)`` — all frame-determined
    — so a frame-keyed cache is sound.

    Cache soundness (F7)
    --------------------
    Key = ``(fa, fb, mode, player_id, fa_anchor?, fb_anchor?, wfixes_sig)``.
    Endpoint-free fits depend only on frame-determined inputs; the *only*
    endpoint dependence is when ``fa``/``fb`` is a manual anchor (then its
    world is a knot), and the anchor world is deterministic from the frame,
    so the boolean flag captures it.  ``wfixes_sig`` hashes only the
    in-range fix *frames* (fix values are deterministic from frame).
    """

    def __init__(self, *, mode_cfg: ModeSearchCfg | None = None, **solver_kwargs: Any) -> None:
        self._solver = _Solver(**solver_kwargs)
        self._mode_cfg = mode_cfg or ModeSearchCfg()
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
        ``fa_anchor_world`` and ``fb_anchor_world`` MUST be *manual-anchor*
        worlds — i.e. worlds whose value is deterministically derived from
        the frame index alone (operator-placed anchors looked up from the
        ``BallAnchorSet``).  Do NOT pass a path-dependent world such as the
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
            return self._fit_rolling(fa, fb)
        if mode is Mode.STATIONARY:
            return self._fit_stationary(fa, fb)
        if mode is Mode.OUT_OF_VIEW:
            return SegmentFit(
                worlds={}, residual_px=0.0, underconstrained=False,
                kind="out_of_view", boundary_vel=(None, None),
            )
        if mode is Mode.POSSESSED:
            raise NotImplementedError(
                "POSSESSED mode is implemented in Task 4 (player FK tether)"
            )
        raise ValueError(f"unknown mode {mode!r}")

    # ------------------------------------------------------------------
    # Helpers

    def _wfixes_sig(self, fa: int, fb: int) -> tuple[int, ...]:
        """Frame-only signature of the in-range world fixes (F7).

        Fix *values* are deterministic from the frame, so the sorted tuple
        of in-range fix frames fully captures the cache dependency.
        """
        return tuple(sorted(f for f, _xyz, _w in self._solver._fixes_in(fa, fb)))

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
        manual anchor at ``fa``/``fb`` is passed as a soft knot — its world
        is deterministic from the frame, so the cache key's anchor-bool
        captures it.
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
        return SegmentFit(
            worlds=worlds,
            residual_px=resid,
            underconstrained=underconstrained,
            kind="ballistic",
            boundary_vel=(np.asarray(v_a, float), np.asarray(v_b, float)),
        )

    # ------------------------------------------------------------------
    # ROLLING

    def _fit_rolling(self, fa: int, fb: int) -> SegmentFit:
        """Endpoint-free constant-decel roll over ground ray-casts.

        Endpoints are seeded from the first/last in-pitch ground ray-cast
        (not pinned node worlds), so the fit stays frame-determined.
        """
        solver = self._solver
        fps = solver.fps
        cfg: SolverCfg = solver.cfg
        T = (fb - fa) / fps

        obs: list[tuple[float, np.ndarray]] = []
        obs_frames: list[int] = []
        m = cfg.pitch_margin_m
        for f in range(fa + 1, fb):
            ground = solver._ground_raycast(f)
            if ground is None or f in solver.gap_fill:
                continue
            if not (
                -m <= ground[0] <= solver.pitch.length_m + m
                and -m <= ground[1] <= solver.pitch.width_m + m
            ):
                continue
            obs.append(((f - fa) / fps, ground[:2]))
            obs_frames.append(f)

        a_xy = self._endpoint_xy(fa, obs)
        b_xy = self._endpoint_xy(fb, obs, prefer_last=True)

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
        underconstrained = (
            resid is not None and resid > cfg.rolling_max_residual_px
        )

        # Fix: restore lob-degeneracy guard from ``_rolling_span``.
        # A roll faster than any realistic ground pass is physically impossible
        # — the usual culprit is a lob whose ground projection happens to fit
        # the rolling model in pixels (monocular degeneracy).  Mirror the check
        # in ball_piecewise_solver._rolling_span (line ~896–899).
        if not underconstrained and len(pts) >= 2:
            speeds = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1) * fps
            if float(np.max(speeds)) > cfg.rolling_max_speed_m_s:
                underconstrained = True

        v_a = self._roll_velocity(fit, 0.0, z, fps)
        v_b = self._roll_velocity(fit, T, z, fps)
        return SegmentFit(
            worlds=worlds,
            residual_px=residual_px,
            underconstrained=underconstrained,
            kind="rolling",
            boundary_vel=(v_a, v_b),
        )

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

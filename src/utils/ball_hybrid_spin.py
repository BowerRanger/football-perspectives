"""Bounded monocular spin (Magnus) fit for one flight span.

``fit_span_spin`` is IC-E's contribution to the ball-hybrid trajectory
layer: given a flight span's two hard 3-D knots (``p_a``/``t_a`` ->
``p_b``/``t_b``, already resolved by ``ball_hybrid_types.Knot`` /
``ball_hybrid_physics.shoot_arc``) plus the span's interior pixel
evidence, decide whether a bounded rigid-body spin (Magnus) term
measurably improves the reprojection fit over the plain drag-only arc,
and if so return the fitted world-frame angular velocity.

## Why bounded, 2-dof, not a free 3-vector

``project_ball_v2_replay_triangulation`` (2026-06-12) found an
unconstrained 3-dof (or worse, a joint 9-dof rigid-body) Magnus fit from
a short monocular arc is ill-conditioned and diverges (one real run hit
512 km/s). Monocular depth and spin are jointly degenerate: a curving
2-D pixel path can be explained by depth changing OR by a lateral Magnus
force, and a free-standing 3-vector omega has a component along the
velocity direction (rifle spin) that produces exactly zero Magnus force
(``omega x v == 0`` when parallel) and is therefore completely
unobservable — leaving it free just gives the optimiser a direction to
run away in for no reprojection benefit.

This module fixes both problems:

- **2 dof only**, spanning exactly the two spin modes that DO produce an
  observable Magnus force: topspin/backspin about the horizontal axis
  perpendicular to the (baseline, no-spin) launch velocity's horizontal
  projection (vertical lift — dip/float), and sidespin about the world
  vertical axis (horizontal curl — the free-kick "banana"). The
  degenerate along-velocity component is never in the search space.
- **Bounded** to ``|omega| <= 62.8 rad/s`` (10 rev/s,
  ``ball_hybrid_physics.MAGNUS_MAX_OMEGA_REV_S``) by construction (each
  of the 2 components is box-bounded).
- **Both knots stay exact.** Every trial omega re-solves the launch
  velocity via ``shoot_arc``'s boundary-value shoot (endpoint-exact by
  construction), never a free 3-vector v0 alongside spin — so a spin fit
  can never trade knot accuracy for a lower-residual curve.
- **Accepted only when it earns it**: a coarse 2-D grid seed (cheap,
  using the baseline v0 rather than re-shooting) picks a good LM start,
  then a bounded ``least_squares`` refines it (re-shooting v0 every
  evaluation). The spun fit is accepted only when it clears BOTH a
  Bayesian-information-criterion bar (``min_delta_bic``, positive and
  reasonably large — Kass & Raftery's "strong evidence" convention,
  default 6.0) AND a minimum fractional reprojection-RSS improvement
  over the no-spin fit (``min_resid_gain``, default 10%). Either gate
  failing returns ``None`` — a span with too little/noisy evidence or a
  genuinely spin-free arc is left alone, matching ``SpinFit``'s contract
  that a non-positive ``delta_bic`` must never reach the trajectory
  layer's Magnus term (``ball_hybrid_types.SpinFit`` docstring).

## Wiring (for the production trajectory layer)

Call ``fit_span_spin`` once per resolved FLIGHT span (i.e. right where
``ball_hybrid_physics.shoot_arc``/``simulate`` is already used for that
span's endpoint-exact arc — the PoC's analogous call site is
``prototypes/ball_hybrid_poc/hybrid.py``'s flight branch of
``_solve_span``), passing:

- ``p_a``/``t_a``, ``p_b``/``t_b``: the span's two knots (seconds, e.g.
  frame/fps).
- ``obs_times``/``obs_uv``: the span's INTERIOR pixel evidence only
  (frames strictly between the knots — same filter as
  ``hybrid.py``'s ``_span_evidence``); this module does not itself
  decide inlier/outlier membership, so pass the already robust-gated
  inlier set for a clean fit. ``obs_times``, ``t_a`` and ``t_b`` must
  all share ONE consistent time base of the caller's choosing — e.g.
  absolute clip time (``frame / fps``, so ``t_a`` is the span's own
  start time, not 0) or span-relative (``t_a = 0.0``, ``obs_times`` and
  ``t_b`` measured from the span start). ``project_fn`` receives
  whichever base was used, verbatim, as its own ``t_s`` argument.
- ``project_fn``: a small closure over the shot's per-frame camera on
  the SAME time base as ``obs_times``/``t_a``/``t_b`` above — e.g., for
  absolute clip time, ``lambda t_s, xyz: ctx.project(round(t_s * fps),
  xyz)``; for span-relative time (``t_a=0.0``), ``lambda t_s, xyz:
  ctx.project(a_frame + round(t_s * fps), xyz)`` (``ctx`` =
  ``ball_hybrid_types.HybridShotCtx``).
- ``cd``/``magnus_coeff``: the SAME values the span's own drag-only
  ``shoot_arc``/``simulate`` calls already used, so the spin fit is an
  apples-to-apples refinement of that exact arc, not a different model.

On a non-``None`` result, write ``spin_axis_world`` (the unit vector,
``omega_world`` normalised) and ``spin_omega_rad_s`` (``rad_s``, signed
by the axis convention — i.e. store the *unnormalised* magnitude/sign
pair such that ``spin_axis_world * spin_omega_rad_s == omega_world``)
into the ``FlightSegment.parabola`` dict (see
``src/schemas/ball_track.py``'s ``FlightSegment`` docstring) — those are
exactly the two keys ``ball_orientation.py``'s ``_flight_omega`` reads
to rotate the exported ball. On ``None``, leave both keys absent/``None``
(``_flight_omega`` already treats that as "no spin", zero rotation from
this segment).

**Cost per span**: one coarse grid (default 5x5=25 cheap ``simulate``
calls, no inner optimisation) plus one bounded ``least_squares`` over 2
parameters (<= ``_OUTER_MAX_NFEV`` outer evaluations, each a full
``shoot_arc`` LM solve + one ``simulate`` call). In practice a few
hundred milliseconds per span on CPU; skip spans with fewer than
``min_obs`` interior observations outright (returns ``None`` for free).

No camera or IO dependency beyond the caller-supplied ``project_fn``
closure — pure numpy/scipy, deterministic given ``bounds``/config.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from src.utils.ball_hybrid_physics import (
    CD_DEFAULT,
    DEFAULT_MAGNUS_COEFF,
    MAGNUS_MAX_OMEGA_REV_S,
    shoot_arc,
    simulate,
)
from src.utils.ball_hybrid_types import SpinFit

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
# (t_s, xyz) -> predicted pixel (u, v). ``t_s`` is on the SAME time base
# the caller chose for ``obs_times``/``t_a``/``t_b`` (e.g. absolute clip
# time, frame / fps, OR span-relative with t_a=0.0 -- either is fine as
# long as all four agree); the closure is responsible for mapping it to
# a camera frame. See ``fit_span_spin``'s docstring for both conventions.
ProjectFn = Callable[[float, np.ndarray], np.ndarray]

# Bound on |omega|: 10 rev/s, matching
# ball_hybrid_physics.MAGNUS_MAX_OMEGA_REV_S (the same cap the PoC's
# Magnus refinement and the real ball_spin_presets module use as the
# plausible envelope for football spin).
MAX_OMEGA_RAD_S = MAGNUS_MAX_OMEGA_REV_S * 2.0 * math.pi
DEFAULT_BOUNDS: tuple[float, float] = (-MAX_OMEGA_RAD_S, MAX_OMEGA_RAD_S)

# Coarse seed grid over the 2 spin dof, as a fraction of the bound
# magnitude; deliberately sparse (5x5=25 points) since each point is a
# single cheap `simulate` call (no re-shooting), just meant to give the
# refinement a good-enough LM start and avoid an obviously-wrong local
# minimum near the box edges.
_GRID_FRACTIONS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
_OUTER_MAX_NFEV = 60

_UP = np.array([0.0, 0.0, 1.0])


def _spin_axes(v0_analytic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(axis_top, axis_side)`` unit vectors for the 2-dof parametrisation.

    ``axis_top`` is the horizontal axis perpendicular to ``v0_analytic``'s
    horizontal (xy) projection: spin about it (topspin/backspin) produces
    a Magnus force along the vertical (dip when topspin points the "right"
    way, float the other). ``axis_side`` is world +z: spin about it
    (sidespin) produces a horizontal Magnus force (curl). Together they
    span exactly the 2 dof of spin that are observable via reprojection
    (the third, along ``v0_analytic`` itself, produces zero Magnus force
    and is dropped — see module docstring).
    """
    horiz = np.array([float(v0_analytic[0]), float(v0_analytic[1]), 0.0])
    n = float(np.linalg.norm(horiz))
    if n < 1e-6:
        # Degenerate (near-vertical launch): world x is as good as any
        # horizontal reference.
        axis_top = np.array([1.0, 0.0, 0.0])
    else:
        h = horiz / n
        axis_top = np.array([-h[1], h[0], 0.0])
    return axis_top, _UP.copy()


def _omega_from_params(
    params: np.ndarray, axis_top: np.ndarray, axis_side: np.ndarray,
) -> np.ndarray:
    return params[0] * axis_top + params[1] * axis_side


def _px_residuals(
    positions: np.ndarray,
    obs_times: np.ndarray,
    obs_uv: np.ndarray,
    weights: np.ndarray,
    project_fn: ProjectFn,
) -> np.ndarray:
    """Flat ``(2*N,)`` weighted pixel-residual vector (u, v per point)."""
    n = len(obs_times)
    out = np.empty(2 * n)
    for i in range(n):
        uv_pred = project_fn(float(obs_times[i]), positions[i])
        out[2 * i] = weights[i] * (float(uv_pred[0]) - float(obs_uv[i, 0]))
        out[2 * i + 1] = weights[i] * (float(uv_pred[1]) - float(obs_uv[i, 1]))
    return out


def fit_span_spin(
    p_a: Vec3,
    t_a: float,
    p_b: Vec3,
    t_b: float,
    obs_times: Sequence[float],
    obs_uv: Sequence[Vec2],
    project_fn: ProjectFn,
    *,
    cd: float = CD_DEFAULT,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
    magnus_coeff: float = DEFAULT_MAGNUS_COEFF,
    obs_conf: Sequence[float] | None = None,
    min_obs: int = 8,
    min_delta_bic: float = 6.0,
    min_resid_gain: float = 0.10,
) -> SpinFit | None:
    """Fit a bounded 2-dof rigid-body spin for the flight span ``p_a`` (at
    ``t_a``) -> ``p_b`` (at ``t_b``), or return ``None`` when spin doesn't
    measurably earn its keep.

    ``obs_times``/``obs_uv`` are the span's interior pixel evidence
    (paired, same length); entries outside ``[t_a, t_b]`` are dropped.
    ``project_fn(t_s, xyz) -> (u, v)`` projects a world point at time
    ``t_s`` (same units as ``obs_times``) to pixels via the caller's
    per-frame camera. ``obs_conf`` (optional, same length) weights each
    observation's residual by ``sqrt(conf)``; defaults to all-1.0.

    Both knots stay exact for every trial: the launch velocity is
    re-solved via ``shoot_arc``'s boundary-value shoot for each candidate
    omega, never fit as a free parameter alongside spin.

    Returns ``None`` when: fewer than ``min_obs`` interior observations
    survive the ``[t_a, t_b]`` filter; the spun fit's BIC improvement
    over the no-spin baseline is below ``min_delta_bic``; its fractional
    reprojection-RSS improvement is below ``min_resid_gain``; or the
    fitted magnitude rounds to zero.
    """
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    duration_s = float(t_b - t_a)
    if duration_s <= 0:
        raise ValueError("fit_span_spin needs t_b > t_a")

    obs_times = np.asarray(obs_times, dtype=float)
    obs_uv = np.asarray(obs_uv, dtype=float).reshape(-1, 2)
    if len(obs_times) != len(obs_uv):
        raise ValueError("obs_times/obs_uv length mismatch")

    keep = (obs_times >= t_a) & (obs_times <= t_b)
    obs_times = obs_times[keep]
    obs_uv = obs_uv[keep]
    if obs_conf is not None:
        obs_conf = np.asarray(obs_conf, dtype=float)
        if len(obs_conf) != len(keep):
            raise ValueError("obs_conf length mismatch")
        obs_conf = obs_conf[keep]

    n = len(obs_times)
    if n < min_obs:
        return None

    weights = np.sqrt(obs_conf) if obs_conf is not None else np.ones(n)
    t_rel = obs_times - t_a

    # --- baseline (no-spin, drag-only) fit ------------------------------
    v0_base = shoot_arc(p_a, 0.0, p_b, duration_s, cd=cd)
    pos_base = simulate(p_a, v0_base, t_rel, cd=cd)
    resid_base = _px_residuals(pos_base, obs_times, obs_uv, weights, project_fn)
    rss_base = float(np.sum(resid_base ** 2))

    axis_top, axis_side = _spin_axes(v0_base)
    lo, hi = bounds

    def _residual(params: np.ndarray) -> np.ndarray:
        omega = _omega_from_params(params, axis_top, axis_side)
        v0 = shoot_arc(p_a, 0.0, p_b, duration_s, cd=cd, omega=omega,
                        magnus_coeff=magnus_coeff, v0_guess=v0_base)
        pos = simulate(p_a, v0, t_rel, cd=cd, omega=omega,
                        magnus_coeff=magnus_coeff)
        return _px_residuals(pos, obs_times, obs_uv, weights, project_fn)

    # Coarse seed grid, scored with the CHEAP approximation of holding v0
    # at the baseline (no re-shoot) — good enough to pick a basin for the
    # refinement below, at a fraction of the cost of re-shooting 25 times.
    best_x0 = np.zeros(2)
    best_cost = rss_base
    for ft in _GRID_FRACTIONS:
        for fs in _GRID_FRACTIONS:
            if ft == 0.0 and fs == 0.0:
                continue
            trial = np.array([ft * hi, fs * hi])
            omega = _omega_from_params(trial, axis_top, axis_side)
            pos = simulate(p_a, v0_base, t_rel, cd=cd, omega=omega,
                            magnus_coeff=magnus_coeff)
            r = _px_residuals(pos, obs_times, obs_uv, weights, project_fn)
            cost = float(np.sum(r ** 2))
            if cost < best_cost:
                best_cost = cost
                best_x0 = trial

    sol = least_squares(_residual, best_x0, method="trf",
                         bounds=([lo, lo], [hi, hi]), max_nfev=_OUTER_MAX_NFEV,
                         xtol=1e-8, ftol=1e-8)
    rss_spin = float(np.sum(sol.fun ** 2))

    if rss_base <= 0.0 or rss_spin <= 0.0:
        return None

    resid_gain = 1.0 - rss_spin / rss_base
    n_data = 2 * n  # (u, v) residual components, treated as independent
    k_extra = 2  # spin adds 2 dof over the 0-parameter baseline arc
    bic_base = n_data * math.log(rss_base / n_data)
    bic_spin = n_data * math.log(rss_spin / n_data) + k_extra * math.log(n_data)
    delta_bic = bic_base - bic_spin

    if delta_bic < min_delta_bic or resid_gain < min_resid_gain:
        return None

    omega_world = _omega_from_params(sol.x, axis_top, axis_side)
    rad_s = float(np.linalg.norm(omega_world))
    if rad_s < 1e-6:
        return None

    return SpinFit(
        omega_world=(float(omega_world[0]), float(omega_world[1]),
                     float(omega_world[2])),
        rad_s=rad_s,
        delta_bic=float(delta_bic),
    )

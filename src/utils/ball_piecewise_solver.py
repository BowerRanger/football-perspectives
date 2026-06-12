"""Piecewise-physical ball trajectory solver.

Consumes a timeline of resolved anchor *nodes* (world positions from
manual + auto anchors) plus the per-frame pixel track, and produces a
dense trajectory where every segment is a physical primitive:

  * **rolling** between ground-level nodes — endpoint-exact constant-
    acceleration roll (``ball_physics.fit_rolling_segment``), promoted
    to ballistic only when the observations cannot be explained by a
    roll;
  * **ballistic** otherwise — gravity arc through both endpoint nodes
    (free-p0 LM fit against interior pixels when enough observations
    exist, the analytic two-knot arc when not), with optional Magnus
    refinement, and split-and-retry at detected velocity breaks when a
    single arc cannot satisfy its residual gate.

Invariants:
  * position continuity at every node — segments start and end at their
    node positions by construction, so teleports cannot occur anywhere
    a node exists;
  * no silent bad fits — a span whose best fit still violates the
    residual gate is emitted *and* flagged in
    ``diagnostics["underconstrained_spans"]``;
  * bounce nodes between two ballistic arcs get a restitution check
    (``diagnostics["bounces"]``), flagged outside the physical range.

Frames outside any node bracket (leading/trailing spans, or a shot with
no anchors at all) fall back to per-frame grounded ray-casts plus
plausibility-gated flight fits — the pre-rework behaviour floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from src.schemas.ball_track import FlightSegment
from src.utils.ball_physics import (
    G_VEC,
    eval_parabola,
    fit_rolling_segment,
    parabola_end_velocity,
    restitution,
    two_knot_arc,
)
from src.utils.ball_plausibility import (
    PitchDims,
    PlausibilityCfg,
    is_plausible_trajectory,
)
from src.utils.bundle_adjust import (
    _integrate_magnus_positions,
    fit_magnus_trajectory,
    fit_parabola_to_image_observations,
)
from src.utils.ball_spin_presets import omega_seed_from_preset
from src.utils.camera_projection import project_world_to_image
from src.utils.foot_anchor import ankle_ray_to_pitch

logger = logging.getLogger(__name__)

# Node states that force the adjacent segment ballistic regardless of
# endpoint heights: a bounce/catch/goal hit is reached through the air;
# a kick / mid-air contact launches the ball into the air.
_BALLISTIC_END_STATES = frozenset({
    "bounce", "catch", "goal_impact", "header", "volley", "chest",
})
_BALLISTIC_START_STATES = frozenset({
    "kick", "header", "volley", "chest", "goal_impact",
})
# Node states whose own frame is ground contact.
_GROUND_NODE_STATES = frozenset({"grounded", "kick", "bounce"})
# Node states that imply a flight on their open side, used to offer the
# adjacent detection run to the arc fitter in leading/trailing spans.
_LAUNCH_STATES = frozenset({
    "kick", "bounce", "header", "volley", "chest", "goal_impact", "airborne",
    "player_touch",
})


@dataclass(frozen=True)
class TrajectoryNode:
    """A resolved hard-knot anchor on the shot timeline."""

    frame: int
    world_xyz: tuple[float, float, float]
    state: str
    confidence: float = 1.0
    spin: str | None = None
    is_manual: bool = False

    @property
    def z(self) -> float:
        return self.world_xyz[2]


@dataclass(frozen=True)
class SolverCfg:
    ball_radius_m: float = 0.11
    # Endpoints at or below this height count as ground level.
    ground_z_tol_m: float = 0.35
    rolling_max_residual_px: float = 8.0
    rolling_decel_max_m_s2: float = 6.0
    rolling_max_speed_m_s: float = 35.0
    flight_max_residual_px: float = 5.0
    min_obs_for_lm_fit: int = 3
    max_splits_per_span: int = 3
    min_flight_frames: int = 6
    restitution_min: float = 0.5
    restitution_max: float = 0.85
    # Longest credible single ballistic arc; bounds launch-adjacent runs.
    max_arc_seconds: float = 2.5
    # Open-span arcs whose apex stays below this are driven ground
    # passes, not flights — they keep their grounded ray-casts.
    open_span_min_apex_m: float = 0.5
    # Plausibility envelope (open-span fits only; node-bracketed arcs are
    # already constrained by their endpoints).
    z_max_m: float = 50.0
    horizontal_speed_max_m_s: float = 40.0
    pitch_margin_m: float = 5.0
    # Magnus refinement (same accept rules as the legacy stage, but a
    # tight node-violation cap so spin can never break continuity).
    spin_enabled: bool = True
    spin_min_seconds: float = 0.5
    spin_min_improve: float = 0.20
    spin_min_improve_hinted: float = 0.05
    spin_max_omega_rad_s: float = 200.0
    drag_k_over_m: float = 0.005
    spin_node_max_violation_m: float = 0.15


@dataclass(frozen=True)
class SolveResult:
    world_by_frame: dict[int, tuple[np.ndarray, float]]
    state_by_frame: dict[int, str]
    flight_segments: tuple[FlightSegment, ...]
    diagnostics: dict


def _magnus_end_velocity(
    v0: np.ndarray,
    omega: np.ndarray,
    drag_k_over_m: float,
    duration_s: float,
    substeps_per_frame_s: float = 1.0 / 100.0,
) -> np.ndarray:
    """Terminal velocity of a Magnus arc (RK4 on dv/dt = g + k·(ω × v)),
    mirroring ``_integrate_magnus_positions``'s dynamics."""
    v = np.asarray(v0, dtype=float).copy()
    omega = np.asarray(omega, dtype=float)
    n = max(1, int(np.ceil(duration_s / substeps_per_frame_s)))
    h = duration_s / n

    def accel(vv: np.ndarray) -> np.ndarray:
        return G_VEC + drag_k_over_m * np.cross(omega, vv)

    for _ in range(n):
        k1 = accel(v)
        k2 = accel(v + 0.5 * h * k1)
        k3 = accel(v + 0.5 * h * k2)
        k4 = accel(v + h * k3)
        v = v + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return v


@dataclass(frozen=True)
class _Arc:
    """One ballistic arc, parameterised at its own start frame."""

    fa: int
    fb: int
    p0: np.ndarray
    v0: np.ndarray
    residual_px: float
    n_obs: int
    omega_world: np.ndarray | None = None
    spin_axis: list | None = None
    spin_omega: float | None = None
    spin_confidence: float | None = None
    drag_k_over_m: float = 0.005

    def eval(self, frame: int, fps: float) -> np.ndarray:
        dt = (frame - self.fa) / fps
        if self.omega_world is not None:
            pts = _integrate_magnus_positions(
                self.p0, self.v0, self.omega_world, G_VEC,
                self.drag_k_over_m, np.array([0.0, max(dt, 0.0)]),
            )
            return pts[-1]
        return eval_parabola(self.p0, self.v0, np.array([dt]))[0]

    def end_velocity(self, fps: float) -> np.ndarray:
        duration_s = (self.fb - self.fa) / fps
        if self.omega_world is not None:
            return _magnus_end_velocity(
                self.v0, self.omega_world, self.drag_k_over_m, duration_s,
            )
        return parabola_end_velocity(self.v0, duration_s)


@dataclass
class _SpanOutcome:
    worlds: dict[int, np.ndarray] = field(default_factory=dict)
    kind: str = "rolling"
    arcs: list[_Arc] = field(default_factory=list)
    splits: int = 0
    underconstrained: bool = False
    residual_px: float | None = None
    physics_violation: bool = False
    fixes_used: int = 0


class _Solver:
    """One-shot solver; all inputs read-only after construction."""

    def __init__(
        self,
        *,
        nodes: Sequence[TrajectoryNode],
        steps,
        confidences: Mapping[int, float],
        per_frame_K: Mapping[int, np.ndarray],
        per_frame_R: Mapping[int, np.ndarray],
        per_frame_t: Mapping[int, np.ndarray],
        distortion: tuple[float, float],
        fps: float,
        n_frames: int,
        pitch_length_m: float,
        pitch_width_m: float,
        split_hints: Sequence[tuple[int, float]],
        z_hints: Mapping[int, tuple[float, float]] | None,
        manual_obs_frames: set[int] | None,
        world_fixes: Mapping[int, tuple[np.ndarray, float]] | None,
        cfg: SolverCfg,
    ) -> None:
        dedup: dict[int, TrajectoryNode] = {}
        for node in nodes:
            existing = dedup.get(node.frame)
            if existing is None or (node.is_manual and not existing.is_manual):
                dedup[node.frame] = node
        self.nodes = [dedup[f] for f in sorted(dedup)]
        self.steps = list(steps)
        self.uvs: dict[int, np.ndarray] = {
            s.frame: np.asarray(s.uv, dtype=float)
            for s in self.steps if s.uv is not None
        }
        self.p_flight: dict[int, float] = {
            s.frame: float(getattr(s, "p_flight", 0.0)) for s in self.steps
        }
        self.gap_fill: set[int] = {
            s.frame for s in self.steps if getattr(s, "is_gap_fill", False)
        }
        self.confidences = dict(confidences)
        self.K = per_frame_K
        self.R = per_frame_R
        self.t = per_frame_t
        self.distortion = distortion
        self.fps = float(fps)
        self.n_frames = int(n_frames)
        self.pitch = PitchDims(length_m=pitch_length_m, width_m=pitch_width_m)
        self.plaus = PlausibilityCfg(
            z_max_m=cfg.z_max_m,
            horizontal_speed_max_m_s=cfg.horizontal_speed_max_m_s,
            pitch_margin_m=cfg.pitch_margin_m,
        )
        self.split_hints = sorted(split_hints, key=lambda h: -h[1])
        self.z_hints = dict(z_hints or {})
        self.manual_obs_frames = set(manual_obs_frames or ())
        self.arc_contested_spans: set[tuple[int, int]] = set()
        self.world_fixes: Mapping[int, tuple[np.ndarray, float]] = dict(
            world_fixes or {}
        )
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Geometry helpers

    def _has_cam(self, f: int) -> bool:
        return f in self.K and f in self.R and f in self.t

    def _ground_raycast(self, f: int) -> np.ndarray | None:
        uv = self.uvs.get(f)
        if uv is None or not self._has_cam(f):
            return None
        try:
            world = np.asarray(ankle_ray_to_pitch(
                (float(uv[0]), float(uv[1])),
                K=self.K[f], R=self.R[f], t=self.t[f],
                plane_z=self.cfg.ball_radius_m, distortion=self.distortion,
            ), dtype=float)
        except Exception:
            return None
        clamp = max(50.0, 2.0 * max(self.pitch.length_m, self.pitch.width_m))
        if not np.all(np.isfinite(world)) or abs(world[0]) > clamp or abs(world[1]) > clamp:
            return None
        return world

    def _pixel_rms(self, worlds: Mapping[int, np.ndarray],
                   frames: Sequence[int]) -> float | None:
        errs = []
        for f in frames:
            uv = self.uvs.get(f)
            w = worlds.get(f)
            if uv is None or w is None or not self._has_cam(f):
                continue
            proj = project_world_to_image(
                self.K[f], self.R[f], self.t[f], self.distortion,
                np.asarray(w, dtype=float).reshape(1, 3),
            )[0]
            errs.append(float(np.linalg.norm(proj - uv)))
        if not errs:
            return None
        return float(np.sqrt(np.mean(np.square(errs))))

    def _interior_obs(self, fa: int, fb: int, only_frames=None):
        obs, Ks, Rs, ts = [], [], [], []
        for f in range(fa + 1, fb):
            uv = self.uvs.get(f)
            if uv is None or not self._has_cam(f) or f in self.gap_fill:
                continue
            if only_frames is not None and f not in only_frames:
                continue
            obs.append((f, (float(uv[0]), float(uv[1]))))
            Ks.append(self.K[f])
            Rs.append(self.R[f])
            ts.append(self.t[f])
        return obs, Ks, Rs, ts

    def _fixes_in(self, fa: int, fb: int) -> list[tuple[int, np.ndarray, float]]:
        """World fixes whose frame is in [fa, fb], in fitter list form."""
        result = []
        for f, (xyz, weight) in self.world_fixes.items():
            if fa <= f <= fb:
                result.append((f, np.asarray(xyz, dtype=float), float(weight)))
        return result

    # ------------------------------------------------------------------
    # Ballistic fitting

    def _analytic_arc(self, a: np.ndarray, b: np.ndarray,
                      fa: int, fb: int) -> _Arc:
        T = (fb - fa) / self.fps
        p0, v0 = two_knot_arc(a, b, T)
        return _Arc(fa=fa, fb=fb, p0=p0, v0=v0, residual_px=0.0, n_obs=0)

    def _fit_arc(self, a: np.ndarray, b: np.ndarray,
                 fa: int, fb: int, only_frames=None,
                 wfixes: list[tuple[int, np.ndarray, float]] | None = None,
                 ) -> _Arc:
        """Best single gravity arc through (fa, a) and (fb, b)."""
        obs, Ks, Rs, ts = self._interior_obs(fa, fb, only_frames)
        analytic = self._analytic_arc(a, b, fa, fb)
        if len(obs) < self.cfg.min_obs_for_lm_fit:
            worlds = {
                f: analytic.eval(f, self.fps) for f, _ in obs
            }
            resid = self._pixel_rms(worlds, [f for f, _ in obs])
            return _Arc(
                fa=fa, fb=fb, p0=analytic.p0, v0=analytic.v0,
                residual_px=float(resid) if resid is not None else 0.0,
                n_obs=len(obs),
            )
        first_obs_frame = obs[0][0]
        rel = lambda f: f - first_obs_frame  # noqa: E731
        knots = {rel(fa): np.asarray(a, float), rel(fb): np.asarray(b, float)}
        z_ranges = {
            rel(f): rng for f, rng in self.z_hints.items() if fa < f < fb
        }
        # Seed from the analytic two-knot arc: monocular pixel data has
        # depth-flipped local minima and the fitter's own seeding
        # heuristic regularly lands in one.
        dt_seed = (first_obs_frame - fa) / self.fps
        seed = (
            analytic.eval(first_obs_frame, self.fps),
            analytic.v0 + G_VEC * dt_seed,
        )
        try:
            p0_fit, v0_fit, _ = fit_parabola_to_image_observations(
                obs, Ks=Ks, Rs=Rs, t_world=ts,
                fps=self.fps, distortion=self.distortion,
                p0_fixed=None, knot_frames=knots,
                seed=seed,
                z_range_frames=z_ranges or None,
                # Both endpoints are hard knots here, so the arc's depth
                # is already determined; the coarse airborne buckets are
                # demoted to a light hinge that cannot fight the arc
                # when the operator picked the wrong bucket (C2 rule).
                z_range_weight=20.0,
                world_fixes=wfixes or None,
            )
        except Exception as exc:
            logger.debug("arc fit failed on %d-%d: %s — using analytic",
                         fa, fb, exc)
            return analytic
        # Re-base to span start.
        dt0 = (fa - first_obs_frame) / self.fps
        p0 = p0_fit + v0_fit * dt0 + 0.5 * G_VEC * dt0**2
        v0 = v0_fit + G_VEC * dt0
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(v0))):
            return analytic
        arc = _Arc(fa=fa, fb=fb, p0=p0, v0=v0, residual_px=0.0, n_obs=len(obs))
        worlds = {f: arc.eval(f, self.fps) for f, _ in obs}
        resid = self._pixel_rms(worlds, [f for f, _ in obs])
        arc = _Arc(
            fa=fa, fb=fb, p0=p0, v0=v0,
            residual_px=float(resid) if resid is not None else 0.0,
            n_obs=len(obs),
        )
        # The LM's soft knots can drift off the nodes when observations
        # disagree with the endpoints; the analytic arc is then both more
        # physical and continuity-exact. Keep whichever explains pixels
        # better once the LM stops beating the gate.
        end_err = float(np.linalg.norm(arc.eval(fb, self.fps) - b))
        start_err = float(np.linalg.norm(arc.eval(fa, self.fps) - a))
        if max(end_err, start_err) > 0.05:
            analytic_worlds = {
                f: analytic.eval(f, self.fps) for f, _ in obs
            }
            analytic_resid = self._pixel_rms(
                analytic_worlds, [f for f, _ in obs]
            )
            return _Arc(
                fa=fa, fb=fb, p0=analytic.p0, v0=analytic.v0,
                residual_px=(
                    float(analytic_resid) if analytic_resid is not None else 0.0
                ),
                n_obs=len(obs),
            )
        return arc

    def _try_magnus(self, arc: _Arc, a: np.ndarray, b: np.ndarray,
                    spin_preset: str | None,
                    wfixes: list[tuple[int, np.ndarray, float]] | None = None,
                    ) -> _Arc:
        cfg = self.cfg
        duration_s = (arc.fb - arc.fa) / self.fps
        if (
            not cfg.spin_enabled
            or duration_s < cfg.spin_min_seconds
            or arc.n_obs < cfg.min_obs_for_lm_fit
        ):
            return arc
        hint = bool(spin_preset and spin_preset not in ("none", "knuckle"))
        if spin_preset == "knuckle":
            return arc
        omega_seed = (
            omega_seed_from_preset(spin_preset, arc.v0)
            if hint else np.zeros(3)
        )
        obs, Ks, Rs, ts = self._interior_obs(arc.fa, arc.fb)
        if len(obs) < cfg.min_obs_for_lm_fit:
            return arc
        seed_norm = float(np.linalg.norm(omega_seed))
        try:
            if hint and seed_norm > 1e-9:
                mp0, mv0, momega, _ = fit_magnus_trajectory(
                    obs, Ks=Ks, Rs=Rs, t_world=ts,
                    fps=self.fps, drag_k_over_m=cfg.drag_k_over_m,
                    p0_seed=arc.p0, v0_seed=arc.v0, omega_seed=omega_seed,
                    p0_fixed=np.asarray(a, float),
                    omega_axis_fixed=omega_seed / seed_norm,
                    omega_mag_bound=cfg.spin_max_omega_rad_s,
                    v0_abs_bound=max(cfg.horizontal_speed_max_m_s * 1.5, 40.0),
                    distortion=self.distortion,
                    world_fixes=wfixes or None,
                )
            else:
                mp0, mv0, momega, _ = fit_magnus_trajectory(
                    obs, Ks=Ks, Rs=Rs, t_world=ts,
                    fps=self.fps, drag_k_over_m=cfg.drag_k_over_m,
                    p0_seed=arc.p0, v0_seed=arc.v0, omega_seed=omega_seed,
                    p0_fixed=np.asarray(a, float),
                    omega_abs_bound=cfg.spin_max_omega_rad_s / np.sqrt(3.0),
                    distortion=self.distortion,
                    world_fixes=wfixes or None,
                )
        except Exception as exc:
            logger.debug("magnus fit failed on %d-%d: %s", arc.fa, arc.fb, exc)
            return arc
        omega_mag = float(np.linalg.norm(momega))
        if omega_mag <= 0 or omega_mag > cfg.spin_max_omega_rad_s:
            return arc
        candidate = _Arc(
            fa=arc.fa, fb=arc.fb, p0=np.asarray(mp0, float),
            v0=np.asarray(mv0, float), residual_px=0.0, n_obs=arc.n_obs,
            omega_world=np.asarray(momega, float),
            drag_k_over_m=cfg.drag_k_over_m,
        )
        # Continuity guard: spin must not pull the arc off its end node.
        end_err = float(np.linalg.norm(candidate.eval(arc.fb, self.fps) - b))
        if end_err > cfg.spin_node_max_violation_m:
            return arc
        worlds = {f: candidate.eval(f, self.fps) for f, _ in obs}
        resid = self._pixel_rms(worlds, [f for f, _ in obs])
        if resid is None:
            return arc
        threshold = (
            cfg.spin_min_improve_hinted if hint else cfg.spin_min_improve
        )
        base = arc.residual_px if arc.residual_px > 0 else 1e-9
        improvement = (arc.residual_px - resid) / base
        if improvement < threshold:
            return arc
        duration_factor = min(1.0, duration_s / 1.0)
        return _Arc(
            fa=arc.fa, fb=arc.fb, p0=candidate.p0, v0=candidate.v0,
            residual_px=float(resid), n_obs=arc.n_obs,
            omega_world=candidate.omega_world,
            drag_k_over_m=cfg.drag_k_over_m,
            spin_axis=list((momega / omega_mag).astype(float)),
            spin_omega=omega_mag,
            spin_confidence=float(
                min(1.0, (improvement / 0.5) * duration_factor)
            ),
        )

    def _split_frame_for(self, fa: int, fb: int,
                         used: set[int]) -> int | None:
        for f, _score in self.split_hints:
            if f in used:
                continue
            if fa + 2 <= f <= fb - 2:
                ground = self._ground_raycast(f)
                if ground is None:
                    continue
                m = self.cfg.pitch_margin_m
                if (
                    -m <= ground[0] <= self.pitch.length_m + m
                    and -m <= ground[1] <= self.pitch.width_m + m
                ):
                    return f
        return None

    def _ballistic_span(self, a: np.ndarray, b: np.ndarray,
                        fa: int, fb: int, spin_preset: str | None,
                        splits_left: int, used_splits: set[int],
                        manual_span: bool = False) -> _SpanOutcome:
        wfixes = self._fixes_in(fa, fb)
        arc = self._fit_arc(a, b, fa, fb, wfixes=wfixes)
        if (
            manual_span
            and arc.n_obs > 0
            and arc.residual_px > self.cfg.flight_max_residual_px
        ):
            # Operator-anchored span whose detections fight the anchors:
            # the operator anchored it precisely because the detector
            # lost the plot. Refit against the anchor pixels alone and
            # trust the result — flagged if even those disagree, but
            # never split: inventing ground contacts inside a span the
            # operator has annotated would contradict their ground truth.
            logger.info(
                "ball solver: span %d-%d refit on anchor pixels only "
                "(detections at %.1f px vs gate)",
                fa, fb, arc.residual_px,
            )
            arc = self._fit_arc(
                a, b, fa, fb, only_frames=self.manual_obs_frames,
                wfixes=wfixes,
            )
            splits_left = 0
        if (
            arc.n_obs > 0
            and arc.residual_px > self.cfg.flight_max_residual_px
            and splits_left > 0
        ):
            split = self._split_frame_for(fa, fb, used_splits)
            ground = self._ground_raycast(split) if split is not None else None
            if split is not None and ground is not None:
                left = self._ballistic_span(
                    a, ground, fa, split, spin_preset,
                    splits_left - 1, used_splits | {split},
                    manual_span=manual_span,
                )
                right = self._ballistic_span(
                    ground, b, split, fb, spin_preset,
                    splits_left - 1, used_splits | {split},
                    manual_span=manual_span,
                )
                combined = _SpanOutcome(kind="ballistic")
                combined.worlds.update(left.worlds)
                combined.worlds.update(right.worlds)
                combined.arcs = left.arcs + right.arcs
                combined.splits = 1 + left.splits + right.splits
                combined.underconstrained = (
                    left.underconstrained or right.underconstrained
                )
                resids = [
                    s.residual_px for s in (left, right)
                    if s.residual_px is not None
                ]
                combined.residual_px = max(resids) if resids else None
                combined.fixes_used = left.fixes_used + right.fixes_used
                return combined
        arc = self._try_magnus(arc, a, b, spin_preset, wfixes=wfixes)
        out = _SpanOutcome(kind="ballistic")
        out.arcs = [arc]
        out.residual_px = arc.residual_px if arc.n_obs else None
        out.underconstrained = (
            arc.n_obs > 0
            and arc.residual_px > self.cfg.flight_max_residual_px
        )
        out.fixes_used = len(wfixes)
        for f in range(fa, fb + 1):
            out.worlds[f] = arc.eval(f, self.fps)
        return out

    # ------------------------------------------------------------------
    # Span solving

    def _solve_span(self, node_a: TrajectoryNode,
                    node_b: TrajectoryNode) -> _SpanOutcome:
        fa, fb = node_a.frame, node_b.frame
        a = np.asarray(node_a.world_xyz, dtype=float)
        b = np.asarray(node_b.world_xyz, dtype=float)
        if fb - fa < 1:
            return _SpanOutcome()
        tol = self.cfg.ground_z_tol_m
        ballistic = (
            node_a.z > tol
            or node_b.z > tol
            or node_a.state in _BALLISTIC_START_STATES
            or node_b.state in _BALLISTIC_END_STATES
            or any(fa < f < fb for f in self.z_hints)
        )
        manual_span = node_a.is_manual or node_b.is_manual
        if ballistic:
            return self._ballistic_span(
                a, b, fa, fb, node_a.spin, self.cfg.max_splits_per_span,
                set(), manual_span=manual_span,
            )
        rolling = self._rolling_span(a, b, fa, fb)
        rolling_ok = not rolling.physics_violation and (
            rolling.residual_px is None
            or rolling.residual_px <= self.cfg.rolling_max_residual_px
        )
        # A sustained IMM flight posterior cannot force a flight (driven
        # ground passes trip it too), but it does earn the arc a hearing
        # even when the roll explains the pixels: a low lob's ground
        # projection often fits the rolling model, and the residual
        # comparison is then the only discriminator.
        contested = rolling_ok and (
            self._has_sustained_flight(fa, fb)
            or (fa, fb) in self.arc_contested_spans
        )
        if rolling_ok and not contested:
            return rolling
        promoted = self._ballistic_span(
            a, b, fa, fb, node_a.spin, self.cfg.max_splits_per_span,
            set(), manual_span=manual_span,
        )
        promoted_ok = (
            promoted.residual_px is not None
            and promoted.residual_px <= self.cfg.flight_max_residual_px
        )
        if rolling_ok and promoted_ok:
            roll_resid = rolling.residual_px if rolling.residual_px is not None else 0.0
            if promoted.residual_px < roll_resid:
                logger.info(
                    "ball solver: contested span %d-%d resolved to flight "
                    "(roll %.1f px vs arc %.1f px)",
                    fa, fb, roll_resid, promoted.residual_px,
                )
                return promoted
            return rolling
        if promoted_ok:
            logger.info(
                "ball solver: span %d-%d promoted to flight "
                "(roll %s px -> arc %.1f px)",
                fa, fb,
                f"{rolling.residual_px:.1f}" if rolling.residual_px is not None else "n/a",
                promoted.residual_px,
            )
            return promoted
        if rolling_ok:
            return rolling
        rolling.underconstrained = True
        return rolling

    def _has_sustained_flight(self, fa: int, fb: int) -> bool:
        """A run of >= min_flight_frames interior frames with the IMM
        flight posterior high — direct evidence the span is airborne."""
        run = 0
        for f in range(fa + 1, fb):
            if self.p_flight.get(f, 0.0) >= 0.5:
                run += 1
                if run >= self.cfg.min_flight_frames:
                    return True
            else:
                run = 0
        return False

    def _rolling_span(self, a: np.ndarray, b: np.ndarray,
                      fa: int, fb: int) -> _SpanOutcome:
        T = (fb - fa) / self.fps
        obs: list[tuple[float, np.ndarray]] = []
        obs_frames: list[int] = []
        m = self.cfg.pitch_margin_m
        for f in range(fa + 1, fb):
            ground = self._ground_raycast(f)
            if ground is None or f in self.gap_fill:
                continue
            if not (
                -m <= ground[0] <= self.pitch.length_m + m
                and -m <= ground[1] <= self.pitch.width_m + m
            ):
                continue
            obs.append(((f - fa) / self.fps, ground[:2]))
            obs_frames.append(f)
        fit = fit_rolling_segment(
            a[:2], b[:2], T, obs, self.cfg.rolling_decel_max_m_s2,
        )
        out = _SpanOutcome(kind="rolling")
        times = np.array([(f - fa) / self.fps for f in range(fa, fb + 1)])
        pts = fit.eval(times, self.cfg.ball_radius_m)
        for i, f in enumerate(range(fa, fb + 1)):
            out.worlds[f] = pts[i]
        out.residual_px = self._pixel_rms(out.worlds, obs_frames)
        # A roll faster than any ground pass is physically impossible —
        # the usual culprit is a lob whose ground projection happens to
        # fit the rolling model in pixels (monocular degeneracy).
        if len(pts) >= 2:
            speeds = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1) * self.fps
            if float(np.max(speeds)) > self.cfg.rolling_max_speed_m_s:
                out.physics_violation = True
        return out

    def _free_end_arc(self, node_a: TrajectoryNode, fb: int) -> _Arc | None:
        """Launch-pinned arc with a free end over (node_a.frame, fb).

        Used to test whether a failing span is better explained by a
        flight that simply does not end at the (auto) end node.
        """
        obs, Ks, Rs, ts = self._interior_obs(node_a.frame, fb + 1)
        if len(obs) < self.cfg.min_obs_for_lm_fit:
            return None
        a = np.asarray(node_a.world_xyz, dtype=float)
        wfixes = self._fixes_in(node_a.frame, fb)
        try:
            p0, v0, _ = fit_parabola_to_image_observations(
                obs, Ks=Ks, Rs=Rs, t_world=ts,
                fps=self.fps, distortion=self.distortion,
                p0_fixed=None,
                knot_frames={node_a.frame - obs[0][0]: a},
                world_fixes=wfixes or None,
            )
        except Exception:
            return None
        dt0 = (node_a.frame - obs[0][0]) / self.fps
        p0a = p0 + v0 * dt0 + 0.5 * G_VEC * dt0**2
        v0a = v0 + G_VEC * dt0
        arc = _Arc(fa=node_a.frame, fb=fb, p0=p0a, v0=v0a,
                   residual_px=0.0, n_obs=len(obs))
        worlds = {f: arc.eval(f, self.fps) for f, _ in obs}
        resid = self._pixel_rms(worlds, [f for f, _ in obs])
        if resid is None or resid > self.cfg.flight_max_residual_px:
            return None
        duration = (fb - node_a.frame) / self.fps
        if not is_plausible_trajectory(
            p0a, v0a, omega=None, duration_s=max(duration, 1e-3),
            fps=self.fps, cfg=self.plaus, pitch=self.pitch,
        ):
            return None
        apex = max(
            float(arc.eval(f, self.fps)[2])
            for f in range(node_a.frame, fb + 1)
        )
        if apex < self.cfg.open_span_min_apex_m:
            return None
        return _Arc(fa=node_a.frame, fb=fb, p0=p0a, v0=v0a,
                    residual_px=float(resid), n_obs=len(obs))

    def _longest_free_arc(
        self, node_a: TrajectoryNode, fb_max: int
    ) -> _Arc | None:
        """Longest accepted free-end arc from a launch node.

        Shrinks the window until the fit passes its gates — the true arc
        end (a landing, a catch) rarely coincides with a node frame.
        """
        fb = fb_max
        while fb - node_a.frame >= self.cfg.min_flight_frames:
            arc = self._free_end_arc(node_a, fb)
            if arc is not None:
                return arc
            fb -= max(2, (fb - node_a.frame) // 8)
        return None

    def _audit_auto_nodes(self) -> None:
        """Launch-arc audit: drop auto nodes a pinned flight contradicts.

        Auto anchors are hypotheses, not ground truth. From every launch
        node, fit the longest accepted free-end arc over the following
        detections (residual gate + plausibility + a real apex). Auto
        nodes the arc passes >1 m away from were mis-sampled airborne
        moments (slow lobs never trip the IMM) and are demoted; the span
        the arc lands on is recorded as contested so the span solve
        compares ballistic against rolling. Manual nodes are never
        touched, and an end node only falls when its bracketed span
        cannot be explained physically.
        """
        horizon_frames = int(self.cfg.max_arc_seconds * self.fps)
        last_uv_frame = max(self.uvs) if self.uvs else 0
        i = 0
        while i < len(self.nodes) - 1:
            node_a = self.nodes[i]
            if (
                node_a.state not in _LAUNCH_STATES
                or self.nodes[i + 1].is_manual
            ):
                i += 1
                continue
            fb_max = min(node_a.frame + horizon_frames, last_uv_frame)
            for node in self.nodes[i + 1:]:
                if node.is_manual:
                    fb_max = min(fb_max, node.frame)
                    break
            best_arc = self._longest_free_arc(node_a, fb_max)
            if best_arc is None:
                i += 1
                continue
            # Demote contradicted auto nodes inside the accepted arc.
            j = i + 1
            demoted_any = False
            while j < len(self.nodes):
                node = self.nodes[j]
                if node.is_manual or node.frame >= best_arc.fb:
                    break
                err = float(np.linalg.norm(
                    best_arc.eval(node.frame, self.fps)
                    - np.asarray(node.world_xyz, dtype=float)
                ))
                # Only sampled grounded nodes are demotable: touch /
                # bounce / goal nodes carry their own event evidence.
                if err > 1.0 and node.state == "grounded":
                    logger.info(
                        "ball solver: demoting auto node at frame %d (%s) "
                        "— a launch arc from frame %d (%.1f px) passes "
                        "%.1f m away",
                        node.frame, node.state, node_a.frame,
                        best_arc.residual_px, err,
                    )
                    self.nodes.pop(j)
                    demoted_any = True
                else:
                    j += 1
            if i + 1 >= len(self.nodes):
                break
            node_end = self.nodes[i + 1]
            # The arc's free endpoint is monocularly uncertain, so it
            # only overrules the end node when the node-bracketed span
            # itself cannot be explained physically.
            outcome = self._solve_span(node_a, node_end)
            arc_reaches_end = (
                node_end.frame <= best_arc.fb + self.cfg.min_flight_frames
            )
            if (
                (outcome.underconstrained or outcome.physics_violation)
                and arc_reaches_end
            ):
                err = float(np.linalg.norm(
                    best_arc.eval(node_end.frame, self.fps)
                    - np.asarray(node_end.world_xyz, dtype=float)
                ))
                if (
                    err > 1.0
                    and not node_end.is_manual
                    and node_end.state == "grounded"
                ):
                    logger.info(
                        "ball solver: demoting auto node at frame %d (%s) "
                        "— the launch arc from frame %d passes %.1f m away",
                        node_end.frame, node_end.state, node_a.frame, err,
                    )
                    self.nodes.pop(i + 1)
                    continue
            if demoted_any or outcome.underconstrained:
                self.arc_contested_spans.add((node_a.frame, node_end.frame))
            i += 1

    # ------------------------------------------------------------------
    # Open (unbracketed) spans

    def _flight_runs(self, frames: Sequence[int]) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        prev: int | None = None
        for f in frames:
            in_flight = (
                self.p_flight.get(f, 0.0) >= 0.5 and f in self.uvs
            )
            contiguous = prev is not None and f == prev + 1
            if in_flight and (start is None or not contiguous):
                if start is not None and prev is not None:
                    runs.append((start, prev))
                start = f
            elif not in_flight and start is not None:
                runs.append((start, prev))
                start = None
            prev = f
        if start is not None and prev is not None:
            runs.append((start, prev))
        return [
            (s, e) for s, e in runs
            if e - s + 1 >= self.cfg.min_flight_frames
        ]

    def _launch_adjacent_run(
        self,
        frames: Sequence[int],
        boundary: TrajectoryNode,
        boundary_at_start: bool,
    ) -> tuple[int, int] | None:
        """Detection run hugging a launch/landing node, IMM opinion aside.

        The IMM flight posterior misses slow lobs entirely (gentle arcs
        fit its grounded model), but a kick/bounce/header node *implies*
        a flight on its open side — so the adjacent contiguous detection
        run is offered to the arc fitter and the residual/plausibility
        gates decide.
        """
        ordered = list(frames) if boundary_at_start else list(reversed(frames))
        run: list[int] = []
        gap = 0
        for f in ordered:
            if self.uvs.get(f) is None or f in self.gap_fill:
                if not run:
                    continue
                gap += 1
                if gap > 2:
                    break
                continue
            gap = 0
            run.append(f)
        if len(run) < self.cfg.min_flight_frames:
            return None
        lo, hi = min(run), max(run)
        # Stay within a single-ballistic-arc horizon of the node.
        horizon = int(self.cfg.max_arc_seconds * self.fps)
        if boundary_at_start:
            hi = min(hi, boundary.frame + horizon)
        else:
            lo = max(lo, boundary.frame - horizon)
        if hi - lo + 1 < self.cfg.min_flight_frames:
            return None
        return lo, hi

    def _solve_open_span(
        self,
        f_lo: int,
        f_hi: int,
        boundary: TrajectoryNode | None,
        boundary_at_start: bool,
    ) -> _SpanOutcome:
        """Frames in [f_lo, f_hi) with at most one adjacent node."""
        out = _SpanOutcome(kind="open")
        frames = [f for f in range(f_lo, f_hi)]
        flight_frames: set[int] = set()
        runs = self._flight_runs(frames)
        if boundary is not None and boundary.state in _LAUNCH_STATES:
            extra = self._launch_adjacent_run(
                frames, boundary, boundary_at_start,
            )
            if extra is not None and not any(
                not (extra[1] < ra or extra[0] > rb) for ra, rb in runs
            ):
                runs.insert(0, extra)
        for (ra, rb) in runs:
            obs, Ks, Rs, ts = [], [], [], []
            for f in range(ra, rb + 1):
                uv = self.uvs.get(f)
                if uv is None or not self._has_cam(f) or f in self.gap_fill:
                    continue
                obs.append((f, (float(uv[0]), float(uv[1]))))
                Ks.append(self.K[f])
                Rs.append(self.R[f])
                ts.append(self.t[f])
            if len(obs) < self.cfg.min_obs_for_lm_fit:
                continue
            # A run launching off (or landing on) a nearby node extends
            # to that node: the kick/bounce/impact IS the arc endpoint,
            # even when the IMM took a few frames to flag the flight.
            knots: dict[int, np.ndarray] = {}
            arc_fa, arc_fb = obs[0][0], rb
            if boundary is not None:
                bf = boundary.frame
                near_window = max(3, self.cfg.min_flight_frames)
                if boundary_at_start and 0 <= ra - bf <= near_window:
                    knots[bf - obs[0][0]] = np.asarray(
                        boundary.world_xyz, dtype=float
                    )
                    arc_fa = bf
                elif not boundary_at_start and 0 <= bf - rb <= near_window:
                    knots[bf - obs[0][0]] = np.asarray(
                        boundary.world_xyz, dtype=float
                    )
                    arc_fb = bf
            # Seed from a sane basin: a two-knot arc between the span's
            # best-known endpoints (boundary node where extended, ground
            # ray-cast otherwise). The fitter's own seeding heuristic
            # regularly lands in a depth-flipped local minimum.
            seed = None
            node_world = (
                np.asarray(boundary.world_xyz, dtype=float)
                if boundary is not None else None
            )
            start_world = (
                node_world if arc_fa != obs[0][0]
                else self._ground_raycast(obs[0][0])
            )
            end_world = (
                node_world if arc_fb != rb
                else self._ground_raycast(obs[-1][0])
            )
            end_frame = arc_fb if arc_fb != rb else obs[-1][0]
            if (
                start_world is not None and end_world is not None
                and end_frame > arc_fa
            ):
                p0_s, v0_s = two_knot_arc(
                    start_world, end_world, (end_frame - arc_fa) / self.fps,
                )
                dt_s = (obs[0][0] - arc_fa) / self.fps
                seed = (
                    p0_s + v0_s * dt_s + 0.5 * G_VEC * dt_s**2,
                    v0_s + G_VEC * dt_s,
                )
            wfixes_run = self._fixes_in(arc_fa, arc_fb)
            try:
                p0, v0, _ = fit_parabola_to_image_observations(
                    obs, Ks=Ks, Rs=Rs, t_world=ts,
                    fps=self.fps, distortion=self.distortion,
                    knot_frames=knots or None,
                    seed=seed,
                    world_fixes=wfixes_run or None,
                )
            except Exception:
                continue
            # Re-base (p0, v0) from the first-observation frame onto the
            # (possibly node-extended) arc start.
            dt0 = (arc_fa - obs[0][0]) / self.fps
            p0_arc = p0 + v0 * dt0 + 0.5 * G_VEC * dt0**2
            v0_arc = v0 + G_VEC * dt0
            arc = _Arc(
                fa=arc_fa, fb=arc_fb, p0=p0_arc, v0=v0_arc,
                residual_px=0.0, n_obs=len(obs),
            )
            worlds = {f: arc.eval(f, self.fps) for f, _ in obs}
            resid = self._pixel_rms(worlds, [f for f, _ in obs])
            duration = (arc_fb - arc_fa) / self.fps
            if resid is None or resid > self.cfg.flight_max_residual_px:
                continue
            if not is_plausible_trajectory(
                p0_arc, v0_arc, omega=None, duration_s=max(duration, 1e-3),
                fps=self.fps, cfg=self.plaus, pitch=self.pitch,
            ):
                continue
            # An "arc" that never leaves the ground is a driven pass the
            # IMM mistook for flight — the grounded ray-cast explanation
            # is the honest one, so don't emit a flight segment for it.
            arc_zs = [
                float(arc.eval(f, self.fps)[2])
                for f in range(arc_fa, arc_fb + 1)
            ]
            if max(arc_zs) < self.cfg.open_span_min_apex_m:
                continue
            arc = _Arc(
                fa=arc_fa, fb=arc_fb, p0=p0_arc, v0=v0_arc,
                residual_px=float(resid), n_obs=len(obs),
            )
            out.arcs.append(arc)
            out.fixes_used += len(wfixes_run)
            for f in range(arc_fa, arc_fb + 1):
                if f_lo <= f < f_hi:
                    out.worlds[f] = arc.eval(f, self.fps)
                    flight_frames.add(f)
        for f in frames:
            if f in flight_frames:
                continue
            if f in self.gap_fill:
                # IMM-extrapolated pixel with no real observation behind it.
                # Other call sites (_rolling_span, _interior_obs) already skip
                # gap-fill; be consistent — leave missing rather than emit a
                # teleport from an extrapolated pixel.
                continue
            if self.p_flight.get(f, 0.0) >= 0.5:
                # Flight posterior but no accepted arc: a grounded
                # ray-cast would be a knowingly-wrong depth. Leave
                # missing rather than emit a teleport.
                continue
            ground = self._ground_raycast(f)
            if ground is None:
                continue
            m = self.cfg.pitch_margin_m
            if not (
                -m <= ground[0] <= self.pitch.length_m + m
                and -m <= ground[1] <= self.pitch.width_m + m
            ):
                # Near-horizon or off-image pixel projects outside the pitch
                # envelope — emitting it would cause a multi-metre teleport.
                # Leave missing, consistent with _rolling_span's own guard.
                continue
            out.worlds[f] = ground
        return out

    # ------------------------------------------------------------------

    def solve(self) -> SolveResult:
        cfg = self.cfg
        world_by_frame: dict[int, tuple[np.ndarray, float]] = {}
        state_by_frame: dict[int, str] = {
            f: "missing" for f in range(self.n_frames)
        }
        segments: list[FlightSegment] = []
        diagnostics: dict = {
            "segments": [],
            "bounces": [],
            "underconstrained_spans": [],
            "splits": 0,
        }
        arcs_by_bound: dict[tuple[int, str], _Arc] = {}

        def _commit_span(outcome: _SpanOutcome, fa: int, fb: int,
                         conf: float) -> None:
            for f, w in outcome.worlds.items():
                if 0 <= f < self.n_frames:
                    base_conf = conf
                    if outcome.kind == "open":
                        det = self.confidences.get(f, 0.5)
                        base_conf = det * (0.3 if f in self.gap_fill else 1.0)
                    world_by_frame[f] = (np.asarray(w, dtype=float), base_conf)
                    if outcome.kind == "rolling":
                        state_by_frame[f] = "grounded"
                    elif outcome.kind == "ballistic":
                        state_by_frame[f] = "flight"
                    else:
                        state_by_frame[f] = (
                            "flight"
                            if any(a.fa <= f <= a.fb for a in outcome.arcs)
                            else "grounded"
                        )
            for arc in outcome.arcs:
                sid = len(segments)
                segments.append(FlightSegment(
                    id=sid,
                    frame_range=(arc.fa, arc.fb),
                    parabola={
                        "p0": [float(x) for x in arc.p0],
                        "v0": [float(x) for x in arc.v0],
                        "g": -9.81,
                        "spin_axis_world": arc.spin_axis,
                        "spin_omega_rad_s": arc.spin_omega,
                        "spin_confidence": arc.spin_confidence,
                    },
                    fit_residual_px=float(arc.residual_px),
                ))
                arcs_by_bound[(arc.fa, "out")] = arc
                arcs_by_bound[(arc.fb, "in")] = arc
            diagnostics["splits"] += outcome.splits
            diagnostics["segments"].append({
                "start": fa, "end": fb, "kind": outcome.kind,
                "residual_px": outcome.residual_px,
                "underconstrained": outcome.underconstrained,
                "fixes_used": outcome.fixes_used,
            })
            if outcome.underconstrained:
                diagnostics["underconstrained_spans"].append({
                    "start": fa, "end": fb,
                    "residual_px": outcome.residual_px,
                })

        if self.nodes:
            self._audit_auto_nodes()
        if self.nodes:
            for node_a, node_b in zip(self.nodes, self.nodes[1:]):
                outcome = self._solve_span(node_a, node_b)
                _commit_span(
                    outcome, node_a.frame, node_b.frame,
                    conf=0.9 * min(node_a.confidence, node_b.confidence),
                )
            first, last = self.nodes[0], self.nodes[-1]
            if first.frame > 0:
                _commit_span(
                    self._solve_open_span(0, first.frame, first, False),
                    0, first.frame, conf=0.6,
                )
            if last.frame < self.n_frames - 1:
                _commit_span(
                    self._solve_open_span(
                        last.frame + 1, self.n_frames, last, True,
                    ),
                    last.frame + 1, self.n_frames - 1, conf=0.6,
                )
        else:
            _commit_span(
                self._solve_open_span(0, self.n_frames, None, False),
                0, max(self.n_frames - 1, 0), conf=0.6,
            )

        # Nodes are authoritative: exact world, state from node semantics.
        # A kick/bounce/contact frame that bookends an emitted arc belongs
        # to the flight; an explicit grounded (or ground-touch) anchor
        # stays grounded no matter what surrounds it.
        ballistic_frames: set[int] = set()
        for arc in arcs_by_bound.values():
            ballistic_frames.update(range(arc.fa, arc.fb + 1))
        for node in self.nodes:
            f = node.frame
            if not (0 <= f < self.n_frames):
                continue
            world_by_frame[f] = (
                np.asarray(node.world_xyz, dtype=float), node.confidence,
            )
            ground_pinned = node.state == "grounded" or (
                node.state == "player_touch" and node.z <= cfg.ground_z_tol_m
            )
            if ground_pinned:
                state_by_frame[f] = "grounded"
            elif f in ballistic_frames or node.z > cfg.ground_z_tol_m:
                state_by_frame[f] = "flight"
            else:
                state_by_frame[f] = "grounded"

        # Restitution at ground nodes between two ballistic arcs.
        for node in self.nodes[1:-1] if len(self.nodes) > 2 else []:
            arc_in = arcs_by_bound.get((node.frame, "in"))
            arc_out = arcs_by_bound.get((node.frame, "out"))
            if arc_in is None or arc_out is None:
                continue
            if node.z > cfg.ground_z_tol_m and node.state != "bounce":
                continue
            e = restitution(arc_in.end_velocity(self.fps), arc_out.v0)
            if e is None:
                continue
            flagged = not (cfg.restitution_min <= e <= cfg.restitution_max)
            diagnostics["bounces"].append({
                "frame": node.frame,
                "restitution": float(e),
                "flagged": bool(flagged),
            })
            if flagged:
                logger.warning(
                    "ball solver: bounce at frame %d has restitution %.2f "
                    "outside [%.2f, %.2f] — check the bracketing anchors",
                    node.frame, e, cfg.restitution_min, cfg.restitution_max,
                )

        return SolveResult(
            world_by_frame=world_by_frame,
            state_by_frame=state_by_frame,
            flight_segments=tuple(segments),
            diagnostics=diagnostics,
        )


def solve_piecewise(
    *,
    nodes: Sequence[TrajectoryNode],
    steps,
    confidences: Mapping[int, float],
    per_frame_K: Mapping[int, np.ndarray],
    per_frame_R: Mapping[int, np.ndarray],
    per_frame_t: Mapping[int, np.ndarray],
    distortion: tuple[float, float],
    fps: float,
    n_frames: int,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    split_hints: Sequence[tuple[int, float]] = (),
    z_hints: Mapping[int, tuple[float, float]] | None = None,
    manual_obs_frames: set[int] | None = None,
    world_fixes: Mapping[int, tuple[np.ndarray, float]] | None = None,
    cfg: SolverCfg | None = None,
) -> SolveResult:
    """Solve one shot's dense ball trajectory. See module docstring."""
    solver = _Solver(
        nodes=nodes, steps=steps, confidences=confidences,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R,
        per_frame_t=per_frame_t, distortion=distortion,
        fps=fps, n_frames=n_frames,
        pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
        split_hints=split_hints, z_hints=z_hints,
        manual_obs_frames=manual_obs_frames,
        world_fixes=world_fixes,
        cfg=cfg or SolverCfg(),
    )
    return solver.solve()

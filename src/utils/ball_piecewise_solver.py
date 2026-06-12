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
    flight_max_residual_px: float = 5.0
    min_obs_for_lm_fit: int = 3
    max_splits_per_span: int = 3
    min_flight_frames: int = 6
    restitution_min: float = 0.5
    restitution_max: float = 0.85
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

    def eval(self, frame: int, fps: float) -> np.ndarray:
        dt = (frame - self.fa) / fps
        if self.omega_world is not None:
            pts = _integrate_magnus_positions(
                self.p0, self.v0, self.omega_world, G_VEC,
                0.005, np.array([0.0, max(dt, 0.0)]),
            )
            return pts[-1]
        return eval_parabola(self.p0, self.v0, np.array([dt]))[0]

    def end_velocity(self, fps: float) -> np.ndarray:
        return parabola_end_velocity(self.v0, (self.fb - self.fa) / fps)


@dataclass
class _SpanOutcome:
    worlds: dict[int, np.ndarray] = field(default_factory=dict)
    kind: str = "rolling"
    arcs: list[_Arc] = field(default_factory=list)
    splits: int = 0
    underconstrained: bool = False
    residual_px: float | None = None


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

    def _interior_obs(self, fa: int, fb: int):
        obs, Ks, Rs, ts = [], [], [], []
        for f in range(fa + 1, fb):
            uv = self.uvs.get(f)
            if uv is None or not self._has_cam(f) or f in self.gap_fill:
                continue
            obs.append((f, (float(uv[0]), float(uv[1]))))
            Ks.append(self.K[f])
            Rs.append(self.R[f])
            ts.append(self.t[f])
        return obs, Ks, Rs, ts

    # ------------------------------------------------------------------
    # Ballistic fitting

    def _analytic_arc(self, a: np.ndarray, b: np.ndarray,
                      fa: int, fb: int) -> _Arc:
        T = (fb - fa) / self.fps
        p0, v0 = two_knot_arc(a, b, T)
        return _Arc(fa=fa, fb=fb, p0=p0, v0=v0, residual_px=0.0, n_obs=0)

    def _fit_arc(self, a: np.ndarray, b: np.ndarray,
                 fa: int, fb: int) -> _Arc:
        """Best single gravity arc through (fa, a) and (fb, b)."""
        obs, Ks, Rs, ts = self._interior_obs(fa, fb)
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
        try:
            p0_fit, v0_fit, _ = fit_parabola_to_image_observations(
                obs, Ks=Ks, Rs=Rs, t_world=ts,
                fps=self.fps, distortion=self.distortion,
                p0_fixed=None, knot_frames=knots,
                z_range_frames=z_ranges or None,
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
                    spin_preset: str | None) -> _Arc:
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
                )
            else:
                mp0, mv0, momega, _ = fit_magnus_trajectory(
                    obs, Ks=Ks, Rs=Rs, t_world=ts,
                    fps=self.fps, drag_k_over_m=cfg.drag_k_over_m,
                    p0_seed=arc.p0, v0_seed=arc.v0, omega_seed=omega_seed,
                    p0_fixed=np.asarray(a, float),
                    omega_abs_bound=cfg.spin_max_omega_rad_s / np.sqrt(3.0),
                    distortion=self.distortion,
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
                        splits_left: int, used_splits: set[int]) -> _SpanOutcome:
        arc = self._fit_arc(a, b, fa, fb)
        if (
            arc.n_obs > 0
            and arc.residual_px > self.cfg.flight_max_residual_px
            and splits_left > 0
        ):
            split = self._split_frame_for(fa, fb, used_splits)
            if split is not None:
                ground = self._ground_raycast(split)
                left = self._ballistic_span(
                    a, ground, fa, split, spin_preset,
                    splits_left - 1, used_splits | {split},
                )
                right = self._ballistic_span(
                    ground, b, split, fb, spin_preset,
                    splits_left - 1, used_splits | {split},
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
                return combined
        arc = self._try_magnus(arc, a, b, spin_preset)
        out = _SpanOutcome(kind="ballistic")
        out.arcs = [arc]
        out.residual_px = arc.residual_px if arc.n_obs else None
        out.underconstrained = (
            arc.n_obs > 0
            and arc.residual_px > self.cfg.flight_max_residual_px
        )
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
        if not ballistic:
            rolling = self._rolling_span(a, b, fa, fb)
            if (
                rolling.residual_px is None
                or rolling.residual_px <= self.cfg.rolling_max_residual_px
            ):
                return rolling
            promoted = self._ballistic_span(
                a, b, fa, fb, node_a.spin,
                self.cfg.max_splits_per_span, set(),
            )
            if (
                promoted.residual_px is not None
                and promoted.residual_px <= self.cfg.flight_max_residual_px
            ):
                logger.info(
                    "ball solver: span %d-%d promoted to flight "
                    "(roll %.1f px -> arc %.1f px)",
                    fa, fb, rolling.residual_px, promoted.residual_px,
                )
                return promoted
            rolling.underconstrained = True
            return rolling
        return self._ballistic_span(
            a, b, fa, fb, node_a.spin, self.cfg.max_splits_per_span, set(),
        )

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
        return out

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
                runs.append((start, prev if contiguous else prev))
                start = None
            prev = f
        if start is not None and prev is not None:
            runs.append((start, prev))
        return [
            (s, e) for s, e in runs
            if e - s + 1 >= self.cfg.min_flight_frames
        ]

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
        for (ra, rb) in self._flight_runs(frames):
            obs, Ks, Rs, ts = [], [], [], []
            for f in range(ra, rb + 1):
                uv = self.uvs.get(f)
                if uv is None or not self._has_cam(f):
                    continue
                obs.append((f, (float(uv[0]), float(uv[1]))))
                Ks.append(self.K[f])
                Rs.append(self.R[f])
                ts.append(self.t[f])
            if len(obs) < self.cfg.min_obs_for_lm_fit:
                continue
            knots = {}
            if boundary is not None:
                bf = boundary.frame
                near_start = boundary_at_start and abs(ra - bf) <= 1
                near_end = (not boundary_at_start) and abs(rb - bf) <= 1
                if near_start or near_end:
                    knots[bf - obs[0][0]] = np.asarray(
                        boundary.world_xyz, dtype=float
                    )
            try:
                p0, v0, _ = fit_parabola_to_image_observations(
                    obs, Ks=Ks, Rs=Rs, t_world=ts,
                    fps=self.fps, distortion=self.distortion,
                    knot_frames=knots or None,
                )
            except Exception:
                continue
            arc = _Arc(
                fa=obs[0][0], fb=rb, p0=p0, v0=v0,
                residual_px=0.0, n_obs=len(obs),
            )
            worlds = {f: arc.eval(f, self.fps) for f, _ in obs}
            resid = self._pixel_rms(worlds, [f for f, _ in obs])
            duration = (rb - ra) / self.fps
            if resid is None or resid > self.cfg.flight_max_residual_px:
                continue
            if not is_plausible_trajectory(
                p0, v0, omega=None, duration_s=max(duration, 1e-3),
                fps=self.fps, cfg=self.plaus, pitch=self.pitch,
            ):
                continue
            arc = _Arc(
                fa=obs[0][0], fb=rb, p0=p0, v0=v0,
                residual_px=float(resid), n_obs=len(obs),
            )
            out.arcs.append(arc)
            for f in range(ra, rb + 1):
                out.worlds[f] = arc.eval(f, self.fps)
                flight_frames.add(f)
        for f in frames:
            if f in flight_frames:
                continue
            if self.p_flight.get(f, 0.0) >= 0.5:
                # Flight posterior but no accepted arc: a grounded
                # ray-cast would be a knowingly-wrong depth. Leave
                # missing rather than emit a teleport.
                continue
            ground = self._ground_raycast(f)
            if ground is None:
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
            })
            if outcome.underconstrained:
                diagnostics["underconstrained_spans"].append({
                    "start": fa, "end": fb,
                    "residual_px": outcome.residual_px,
                })

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
        for node in self.nodes:
            f = node.frame
            if not (0 <= f < self.n_frames):
                continue
            world_by_frame[f] = (
                np.asarray(node.world_xyz, dtype=float), node.confidence,
            )
            if node.state in _GROUND_NODE_STATES or (
                node.z <= cfg.ground_z_tol_m
            ):
                state_by_frame[f] = "grounded"
            else:
                state_by_frame[f] = "flight"

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
        cfg=cfg or SolverCfg(),
    )
    return solver.solve()

"""Bundle-adjustment helpers for the broadcast-mono pipeline.

Two fitters live here:

* :func:`fit_parabola_to_image_observations` -- Levenberg-Marquardt fit
  of a constant-gravity parabola (p(t) = p0 + v0*t + 0.5*g*t^2) to a
  sequence of per-frame ball pixel observations, projected through the
  per-frame camera-track ``(K_t, R_t)`` and the clip-shared ``t_world``.
* :func:`fit_magnus_trajectory` -- Levenberg-Marquardt fit of a
  Magnus-augmented trajectory, ``dv/dt = g + k * (ω × v)``, integrated
  with RK4 inside the residual loop.  Recovers ``(p0, v0, ω)``.  Warm-
  starts from the parabola fit if seeds are supplied.

The parabola seed is computed by ground-projecting the first/last image
points to a coarse plane (z = 0.5 m, mid-flight) using
:func:`src.utils.foot_anchor.ankle_ray_to_pitch`, and assuming a
symmetric vertical velocity that places apex at mid-flight.
"""

from __future__ import annotations

import numpy as np


def fit_parabola_to_image_observations(
    observations: list[tuple[int, tuple[float, float]]],
    *,
    Ks: list[np.ndarray],
    Rs: list[np.ndarray],
    t_world: np.ndarray | list[np.ndarray],
    fps: float,
    g: float = -9.81,
    max_iter: int = 100,
    distortion: tuple[float, float] = (0.0, 0.0),
    p0_fixed: np.ndarray | None = None,
    knot_frames: dict[int, np.ndarray] | None = None,
    z_range_frames: dict[int, tuple[float, float]] | None = None,
    z_range_weight: float = 200.0,
    seed: tuple[np.ndarray, np.ndarray] | None = None,
    world_fixes: list[tuple[int, np.ndarray, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a 3D parabola to per-frame image observations.

    Args:
        observations: list of ``(frame_index, (u, v))`` pairs ordered by
            time.  ``frame_index`` is the absolute clip frame, used only
            for time deltas; ``Ks`` and ``Rs`` are looked up positionally.
        Ks: position-parallel to ``observations`` — one entry per
            observation, in the same order.
        Rs: position-parallel to ``observations``.
        t_world: either a clip-shared (3,) translation or a list of
            per-frame (3,) translations parallel to ``observations``.
            Use the per-frame form for static-camera clips where ``t``
            varies with the SLERP'd ``R``.
        fps: frame rate.
        g: gravity along world-z (default -9.81 m/s^2).
        max_iter: LM iteration cap (passed through to scipy as
            ``max_nfev = max_iter * 50``).
        distortion: (k1, k2) radial distortion. Default ``(0, 0)``;
            non-zero values undistort each image observation before
            measuring reprojection residuals.
        p0_fixed: when not ``None``, the world-space starting position is
            pinned to this value and only ``v0`` (3 dof) is optimised,
            reducing the ill-conditioned monocular-depth ambiguity.
        knot_frames: optional mapping of ``{rel_frame_index: world_position}``
            that adds soft world-space constraints.  Each entry appends a
            3-row residual block weighted by ``1e3`` so the optimised
            parabola passes through the target within numerical tolerance.
            Frame indices are relative to the first observation frame.
            Composes with ``p0_fixed``: ``p0_fixed`` pins frame 0 exactly
            while knot entries act as soft constraints at other frames.
        z_range_frames: optional mapping of ``{rel_frame_index: (z_min, z_max)}``.
            For each entry the fit's z at that frame is constrained to lie
            in ``[z_min, z_max]`` via a one-sided hinge residual: zero
            penalty inside the bucket, ``z_range_weight * deviation``
            outside it. Used by Layer 5 to translate ``airborne_low/mid/high``
            anchors into bucket-range Z constraints without committing to
            a single bucket midpoint.
        z_range_weight: per-metre weight for the ``z_range_frames`` hinge.
            Defaults to 200 — strong enough to enforce typical bucket
            widths against pixel reprojection noise.
        seed: optional ``(p0, v0)`` warm start at the first observation
            frame, overriding the ground-projection seeding heuristic.
        world_fixes: optional list of ``(frame_index, xyz, weight_px_per_m)``
            triples. ``frame_index`` is in absolute clip-frame space (same
            as ``observations``). For each fix, a 3-residual block
            ``weight * (pos(t_fix) - xyz)`` is appended after the pixel
            residuals, where ``pos`` is evaluated closed-form from the
            parabola. Fixes whose frame falls outside the observation range
            are still valid (extrapolation). ``None`` or ``[]`` leaves
            the residual vector byte-identical to the unfixed call.

    Returns:
        ``(p0, v0, mean_residual_px)`` where ``mean_residual_px`` is
        the RMS reprojection error in pixels.
    """
    from scipy.optimize import least_squares

    obs_array = np.array([o[1] for o in observations], dtype=float)
    frame_idx = np.array([o[0] for o in observations], dtype=int)
    dt = (frame_idx - frame_idx[0]) / fps
    g_vec = np.array([0.0, 0.0, g])

    # Normalise t_world to per-observation form so the residual loop is uniform.
    n_obs = len(observations)
    if isinstance(t_world, list) or (
        isinstance(t_world, np.ndarray) and t_world.ndim == 2
    ):
        ts = [np.asarray(t, dtype=float) for t in t_world]
        if len(ts) != n_obs:
            raise ValueError(
                f"per-frame t_world has {len(ts)} entries, expected {n_obs}"
            )
    else:
        t_shared = np.asarray(t_world, dtype=float)
        ts = [t_shared] * n_obs

    def _residuals(params: np.ndarray) -> np.ndarray:
        p0 = params[:3]
        v0 = params[3:6]
        pts = p0 + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, g_vec)
        residuals = []
        for i in range(n_obs):
            cam = Rs[i] @ pts[i] + ts[i]
            pix = Ks[i] @ cam
            uv = pix[:2] / pix[2]
            residuals.append(uv - obs_array[i])
        if knot_frames:
            knot_weight = 1.0e3
            for rel_idx, target_world in knot_frames.items():
                dt_k = rel_idx / fps
                pos_k = p0 + v0 * dt_k + 0.5 * (dt_k ** 2) * g_vec
                target = np.asarray(target_world, dtype=float)
                residuals.append(knot_weight * (pos_k - target))
        if z_range_frames:
            for rel_idx, (z_min, z_max) in z_range_frames.items():
                dt_k = rel_idx / fps
                z_k = p0[2] + v0[2] * dt_k + 0.5 * (dt_k ** 2) * g_vec[2]
                below = max(0.0, z_min - z_k)
                above = max(0.0, z_k - z_max)
                residuals.append(np.array([z_range_weight * (below + above)]))
        if world_fixes:
            for fix_frame, fix_xyz, fix_weight in world_fixes:
                dt_f = (fix_frame - frame_idx[0]) / fps
                pos_f = p0 + v0 * dt_f + 0.5 * (dt_f ** 2) * g_vec
                target = np.asarray(fix_xyz, dtype=float)
                residuals.append(fix_weight * (pos_f - target))
        return np.concatenate(residuals)

    # Seed from start/end image points -> ground projection (rough).
    from src.utils.foot_anchor import ankle_ray_to_pitch

    p_start = ankle_ray_to_pitch(
        observations[0][1],
        K=Ks[0],
        R=Rs[0],
        t=ts[0],
        plane_z=0.5,
        distortion=distortion,
    )
    p_end = ankle_ray_to_pitch(
        observations[-1][1],
        K=Ks[-1],
        R=Rs[-1],
        t=ts[-1],
        plane_z=0.5,
        distortion=distortion,
    )
    duration = dt[-1] if dt[-1] > 0 else 1.0
    v_horiz = (p_end - p_start) / duration
    v0_seed = np.array([v_horiz[0], v_horiz[1], 0.5 * abs(g) * duration])
    p0_seed = p_start
    if seed is not None:
        # Caller-supplied warm start (e.g. the analytic two-knot arc).
        # The default ground-projection heuristic regularly lands in a
        # depth-flipped local minimum on monocular data; a seed in the
        # right basin is worth more than any weighting.
        p0_seed = np.asarray(seed[0], dtype=float)
        v0_seed = np.asarray(seed[1], dtype=float)

    if p0_fixed is None:
        result = least_squares(
            _residuals,
            np.concatenate([p0_seed, v0_seed]),
            method="lm",
            max_nfev=max_iter * 50,
        )
    else:
        p0_pin = np.asarray(p0_fixed, dtype=float).copy()

        def _residuals_v0only(params: np.ndarray) -> np.ndarray:
            v0 = params[:3]
            pts = p0_pin + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, g_vec)
            residuals = []
            for i in range(n_obs):
                cam = Rs[i] @ pts[i] + ts[i]
                pix = Ks[i] @ cam
                uv = pix[:2] / pix[2]
                residuals.append(uv - obs_array[i])
            if knot_frames:
                knot_weight = 1.0e3
                for rel_idx, target_world in knot_frames.items():
                    dt_k = rel_idx / fps
                    pos_k = p0_pin + v0 * dt_k + 0.5 * (dt_k ** 2) * g_vec
                    target = np.asarray(target_world, dtype=float)
                    residuals.append(knot_weight * (pos_k - target))
            if z_range_frames:
                for rel_idx, (z_min, z_max) in z_range_frames.items():
                    dt_k = rel_idx / fps
                    z_k = p0_pin[2] + v0[2] * dt_k + 0.5 * (dt_k ** 2) * g_vec[2]
                    below = max(0.0, z_min - z_k)
                    above = max(0.0, z_k - z_max)
                    residuals.append(np.array([z_range_weight * (below + above)]))
            if world_fixes:
                for fix_frame, fix_xyz, fix_weight in world_fixes:
                    dt_f = (fix_frame - frame_idx[0]) / fps
                    pos_f = p0_pin + v0 * dt_f + 0.5 * (dt_f ** 2) * g_vec
                    target = np.asarray(fix_xyz, dtype=float)
                    residuals.append(fix_weight * (pos_f - target))
            return np.concatenate(residuals)

        result = least_squares(
            _residuals_v0only,
            v0_seed,
            method="lm",
            max_nfev=max_iter * 50,
        )
    n = len(observations)
    mean_residual = float(np.linalg.norm(result.fun) / np.sqrt(n))
    if p0_fixed is None:
        p0_opt = result.x[:3]
        v0_opt = result.x[3:6]
    else:
        p0_opt = np.asarray(p0_fixed, dtype=float).copy()
        v0_opt = result.x[:3]
    return p0_opt, v0_opt, mean_residual


def _integrate_magnus_positions(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray,
    g_vec: np.ndarray,
    drag_k_over_m: float,
    sample_times: np.ndarray,
    substeps_per_interval: int = 4,
) -> np.ndarray:
    """RK4-integrate ``dv/dt = g + k * (ω × v)`` and sample at ``sample_times``.

    ``sample_times`` must start at 0 and be monotonically increasing.
    Returns positions of shape ``(len(sample_times), 3)``.
    """
    out = np.zeros((len(sample_times), 3))
    out[0] = p0

    def accel(v: np.ndarray) -> np.ndarray:
        return g_vec + drag_k_over_m * np.cross(omega, v)

    p, v = p0.astype(float).copy(), v0.astype(float).copy()
    for i in range(1, len(sample_times)):
        t_prev = sample_times[i - 1]
        t_next = sample_times[i]
        total = t_next - t_prev
        if total <= 0:
            out[i] = p
            continue
        h = total / substeps_per_interval
        for _ in range(substeps_per_interval):
            k1v = accel(v)
            k1p = v
            k2v = accel(v + 0.5 * h * k1v)
            k2p = v + 0.5 * h * k1v
            k3v = accel(v + 0.5 * h * k2v)
            k3p = v + 0.5 * h * k2v
            k4v = accel(v + h * k3v)
            k4p = v + h * k3v
            p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
            v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[i] = p
    return out


def _integrate_magnus_backward(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray,
    g_vec: np.ndarray,
    drag_k_over_m: float,
    neg_times: np.ndarray,
    substeps_per_interval: int = 4,
) -> np.ndarray:
    """RK4-integrate backward from (p0, v0) at t=0 to each time in neg_times.

    ``neg_times`` must be strictly negative and sorted in *descending* order
    (i.e. closest-to-zero first: e.g. [-0.04, -0.08, -0.12]).  The same ODE
    stage formulas work with a negative step h — the physics is time-reversible
    for this form of the equation.

    Returns positions of shape ``(len(neg_times), 3)``.
    """
    out = np.zeros((len(neg_times), 3))

    def accel(v: np.ndarray) -> np.ndarray:
        return g_vec + drag_k_over_m * np.cross(omega, v)

    p, v = p0.astype(float).copy(), v0.astype(float).copy()
    t_cur = 0.0
    for i, t_target in enumerate(neg_times):
        total = t_target - t_cur  # negative
        if total >= 0:
            out[i] = p
            continue
        h = total / substeps_per_interval  # negative step
        for _ in range(substeps_per_interval):
            k1v = accel(v)
            k1p = v
            k2v = accel(v + 0.5 * h * k1v)
            k2p = v + 0.5 * h * k1v
            k3v = accel(v + 0.5 * h * k2v)
            k3p = v + 0.5 * h * k2v
            k4v = accel(v + h * k3v)
            k4p = v + h * k3v
            p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
            v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        t_cur = t_target
        out[i] = p
    return out


def fit_magnus_trajectory(
    observations: list[tuple[int, tuple[float, float]]],
    *,
    Ks: list[np.ndarray],
    Rs: list[np.ndarray],
    t_world: np.ndarray | list[np.ndarray],
    fps: float,
    g: float = -9.81,
    drag_k_over_m: float = 0.005,
    p0_seed: np.ndarray | None = None,
    v0_seed: np.ndarray | None = None,
    omega_seed: np.ndarray | None = None,
    max_iter: int = 100,
    distortion: tuple[float, float] = (0.0, 0.0),
    p0_fixed: np.ndarray | None = None,
    omega_abs_bound: float | None = None,
    omega_axis_fixed: np.ndarray | None = None,
    omega_mag_bound: float | None = None,
    v0_abs_bound: float | None = None,
    world_fixes: list[tuple[int, np.ndarray, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a Magnus-augmented 3D trajectory to per-frame image observations.

    Optimises ``(p0, v0, ω)`` (9 dof) by minimising pixel reprojection
    residuals under ``dv/dt = g + drag_k_over_m * (ω × v)``, integrated
    with RK4 between observation times.

    Args:
        observations: ``(frame_index, (u, v))`` pairs ordered by time.
        Ks, Rs, t_world: position-parallel to ``observations`` (see
            :func:`fit_parabola_to_image_observations` for details).
        fps: clip frame rate.
        g: gravity along world-z (default -9.81 m/s^2).
        drag_k_over_m: lumped drag/Magnus coefficient (k / m).
        p0_seed, v0_seed: warm-start seeds. If both ``None``, seeds are
            derived from a parabola fit on the same observations.
        omega_seed: 3-vector seed for angular velocity (rad/s).  Default
            zeros — the LM finds a non-zero ω only if it improves the
            pixel residual.
        max_iter: LM iteration cap.
        distortion: unused here (kept for signature symmetry with the
            parabola fitter; image residuals are computed against raw
            observations).

    Returns:
        ``(p0, v0, ω, mean_residual_px)``.
    """
    from scipy.optimize import least_squares

    obs_array = np.array([o[1] for o in observations], dtype=float)
    frame_idx = np.array([o[0] for o in observations], dtype=int)
    dt = (frame_idx - frame_idx[0]) / fps
    g_vec = np.array([0.0, 0.0, g])

    n_obs = len(observations)
    if isinstance(t_world, list) or (
        isinstance(t_world, np.ndarray) and t_world.ndim == 2
    ):
        ts = [np.asarray(t, dtype=float) for t in t_world]
        if len(ts) != n_obs:
            raise ValueError(
                f"per-frame t_world has {len(ts)} entries, expected {n_obs}"
            )
    else:
        t_shared = np.asarray(t_world, dtype=float)
        ts = [t_shared] * n_obs

    if p0_seed is None or v0_seed is None:
        p0_seed, v0_seed, _ = fit_parabola_to_image_observations(
            observations,
            Ks=Ks,
            Rs=Rs,
            t_world=t_world,
            fps=fps,
            g=g,
            max_iter=max_iter,
            distortion=distortion,
        )
    if omega_seed is None:
        omega_seed = np.zeros(3)

    # Build an augmented sample-times grid that includes fix times so
    # the RK4 integrator visits each fix frame exactly once — no double
    # integration required.  For each augmented time we track whether it
    # is an observation slot (obs_indices) or a fix slot (fix_slots).
    _active_fixes: list[tuple[float, np.ndarray, float]] = []  # (dt, xyz, w)
    if world_fixes:
        for fix_frame, fix_xyz, fix_weight in world_fixes:
            _active_fixes.append((
                (fix_frame - frame_idx[0]) / fps,
                np.asarray(fix_xyz, dtype=float),
                float(fix_weight),
            ))

    if _active_fixes:
        # Merge obs dts and fix dts into a sorted unique grid.
        # Fixes with negative dt (before the first observation) are placed
        # in the grid as-is; fixes at or after the first obs are merged
        # into the forward integration grid normally.
        #
        # When negative fix times are present the grid spans
        # [min_neg_fix_dt, ..., 0.0, ..., max_obs_dt].  A helper
        # evaluates positions at negative slots by backward RK4 from
        # (p0, v0) at t=0; the forward integrator covers t ≥ 0.
        _fix_dts = [(fdt, fxyz, fw) for fdt, fxyz, fw in _active_fixes]
        _neg_fix_dts = [(fdt, fxyz, fw) for fdt, fxyz, fw in _fix_dts if fdt < 0.0]
        _pos_fix_dts = [(fdt, fxyz, fw) for fdt, fxyz, fw in _fix_dts if fdt >= 0.0]

        # Forward grid: obs times + non-negative fix times (must include 0.0).
        _pos_fix_times_only = [fdt for fdt, _, _ in _pos_fix_dts]
        _all_fwd_set: list[float] = sorted(
            set(list(dt)) | set(_pos_fix_times_only)
        )
        _aug_times = np.array(_all_fwd_set)
        # obs_indices[i] → position of dt[i] in the forward augmented grid.
        _obs_indices = np.searchsorted(_aug_times, dt)
        # fix_indices for non-negative fixes → position in forward grid.
        if _pos_fix_dts:
            _pos_fix_indices = np.searchsorted(
                _aug_times, np.array([fdt for fdt, _, _ in _pos_fix_dts])
            )
        else:
            _pos_fix_indices = np.empty(0, dtype=int)

        # Backward grid: negative fix times sorted descending (closest to 0 first)
        # so the backward RK4 steps away from t=0 one slot at a time.
        _neg_fix_sorted = sorted(_neg_fix_dts, key=lambda x: x[0], reverse=True)
        _neg_times_arr = np.array([fdt for fdt, _, _ in _neg_fix_sorted])
    else:
        _aug_times = dt
        _obs_indices = np.arange(len(dt), dtype=int)
        _pos_fix_indices = np.empty(0, dtype=int)
        _pos_fix_dts = []
        _neg_fix_sorted = []
        _neg_times_arr = np.empty(0)

    def _eval_fixes(fwd_pts: np.ndarray, p0_node: np.ndarray,
                    v0_node: np.ndarray, omega: np.ndarray) -> list:
        """Return residual blocks for all active fixes.

        fwd_pts: positions on the forward grid (t ≥ 0), shape (N, 3).
        p0_node, v0_node: state at t=0 (= first obs frame) used as the
            seed for backward extrapolation to negative fix times.
        """
        blocks: list = []
        for j, (_, fix_xyz, fix_weight) in enumerate(_pos_fix_dts):
            pos_f = fwd_pts[_pos_fix_indices[j]]
            blocks.append(fix_weight * (pos_f - fix_xyz))
        if _neg_fix_sorted:
            neg_pts = _integrate_magnus_backward(
                p0_node, v0_node, omega, g_vec, drag_k_over_m, _neg_times_arr,
            )
            for k, (_, fix_xyz, fix_weight) in enumerate(_neg_fix_sorted):
                blocks.append(fix_weight * (neg_pts[k] - fix_xyz))
        return blocks

    def _residuals(params: np.ndarray) -> np.ndarray:
        p0 = params[:3]
        v0 = params[3:6]
        omega = params[6:9]
        pts = _integrate_magnus_positions(
            p0, v0, omega, g_vec, drag_k_over_m, _aug_times,
        )
        residuals = []
        for i in range(n_obs):
            cam = Rs[i] @ pts[_obs_indices[i]] + ts[i]
            pix = Ks[i] @ cam
            uv = pix[:2] / pix[2]
            residuals.append(uv - obs_array[i])
        residuals.extend(_eval_fixes(pts, p0, v0, omega))
        return np.concatenate(residuals)

    # Three parametrizations of the spin DOF, from most-to-least free:
    #
    # 1. omega_axis_fixed set → axis is locked to a unit vector and the
    #    LM only adjusts the scalar magnitude (1 DOF for spin). Use this
    #    when the caller has a strong directional prior, e.g. a user
    #    spin preset on a kick anchor. Magnitude is non-negative and
    #    optionally bounded above by ``omega_mag_bound``.
    # 2. omega_abs_bound set → bounded TRF: caps each omega component
    #    to ±bound. Still 3 DOF for spin but prevents the LM from
    #    running to a degenerate high-spin local minimum.
    # 3. Neither set → unbounded LM (the original behaviour). Fastest,
    #    but produces |omega| > 700 rad/s on hard real-world data.
    if omega_axis_fixed is not None:
        axis_unit = np.asarray(omega_axis_fixed, dtype=float)
        axis_norm = float(np.linalg.norm(axis_unit))
        if axis_norm < 1e-9:
            raise ValueError("omega_axis_fixed must be a non-zero vector")
        axis_unit = axis_unit / axis_norm
        # Magnitude seed: project the provided omega_seed onto the
        # fixed axis. Defaults to ~zero if omega_seed itself was zero.
        scalar_seed = float(np.dot(omega_seed, axis_unit))
        if scalar_seed <= 0.0:
            scalar_seed = max(scalar_seed, 0.0)
        # When a magnitude bound is supplied, treat it as a one-sided
        # upper bound (scalar ∈ [0, bound]). The lower bound stays at 0
        # because the user already chose the axis direction; negative
        # scalar would flip the spin direction and contradict the prior.
        method = "trf"
    elif omega_abs_bound is not None:
        inf = np.inf
        lo = np.array([-inf] * 6 + [-omega_abs_bound] * 3) if p0_fixed is None \
            else np.array([-inf] * 3 + [-omega_abs_bound] * 3)
        hi = -lo
        method = "trf"
    else:
        lo = hi = None
        method = "lm"

    if p0_fixed is None and omega_axis_fixed is None:
        x0 = np.concatenate([p0_seed, v0_seed, omega_seed])
        if method == "trf":
            result = least_squares(_residuals, x0, method="trf",
                                   bounds=(lo, hi), max_nfev=max_iter * 50)
        else:
            result = least_squares(_residuals, x0, method="lm",
                                   max_nfev=max_iter * 50)
        p0_opt = result.x[:3]
        v0_opt = result.x[3:6]
        omega_opt = result.x[6:9]
    elif p0_fixed is not None and omega_axis_fixed is None:
        p0_pin = np.asarray(p0_fixed, dtype=float).copy()

        def _residuals_anchored(params: np.ndarray) -> np.ndarray:
            v0 = params[:3]
            omega = params[3:6]
            positions = _integrate_magnus_positions(
                p0_pin, v0, omega, g_vec, drag_k_over_m, _aug_times,
            )
            residuals = []
            for i in range(n_obs):
                cam = Rs[i] @ positions[_obs_indices[i]] + ts[i]
                pix = Ks[i] @ cam
                uv = pix[:2] / pix[2]
                residuals.append(uv - obs_array[i])
            residuals.extend(_eval_fixes(positions, p0_pin, v0, omega))
            return np.concatenate(residuals)

        x0 = np.concatenate([v0_seed, omega_seed])
        if method == "trf":
            result = least_squares(_residuals_anchored, x0, method="trf",
                                   bounds=(lo, hi), max_nfev=max_iter * 50)
        else:
            result = least_squares(_residuals_anchored, x0, method="lm",
                                   max_nfev=max_iter * 50)
        p0_opt = p0_pin
        v0_opt = result.x[:3]
        omega_opt = result.x[3:6]
    else:
        # omega_axis_fixed: spin direction locked to the unit axis;
        # only the scalar magnitude is optimised alongside (v0, p0?).
        mag_hi = float(omega_mag_bound) if omega_mag_bound is not None else np.inf
        if p0_fixed is None:
            def _residuals_axis(params: np.ndarray) -> np.ndarray:
                p0 = params[:3]
                v0 = params[3:6]
                omega = params[6] * axis_unit
                pts = _integrate_magnus_positions(
                    p0, v0, omega, g_vec, drag_k_over_m, _aug_times,
                )
                residuals = []
                for i in range(n_obs):
                    cam = Rs[i] @ pts[_obs_indices[i]] + ts[i]
                    pix = Ks[i] @ cam
                    uv = pix[:2] / pix[2]
                    residuals.append(uv - obs_array[i])
                residuals.extend(_eval_fixes(pts, p0, v0, omega))
                return np.concatenate(residuals)

            x0 = np.concatenate([p0_seed, v0_seed, [scalar_seed]])
            lo7 = np.array([-np.inf] * 6 + [0.0])
            hi7 = np.array([np.inf] * 6 + [mag_hi])
            result = least_squares(_residuals_axis, x0, method="trf",
                                   bounds=(lo7, hi7), max_nfev=max_iter * 50)
            p0_opt = result.x[:3]
            v0_opt = result.x[3:6]
            omega_opt = result.x[6] * axis_unit
        else:
            p0_pin = np.asarray(p0_fixed, dtype=float).copy()

            def _residuals_axis_anchored(params: np.ndarray) -> np.ndarray:
                v0 = params[:3]
                omega = params[3] * axis_unit
                pts = _integrate_magnus_positions(
                    p0_pin, v0, omega, g_vec, drag_k_over_m, _aug_times,
                )
                residuals = []
                for i in range(n_obs):
                    cam = Rs[i] @ pts[_obs_indices[i]] + ts[i]
                    pix = Ks[i] @ cam
                    uv = pix[:2] / pix[2]
                    residuals.append(uv - obs_array[i])
                residuals.extend(_eval_fixes(pts, p0_pin, v0, omega))
                return np.concatenate(residuals)

            x0 = np.concatenate([v0_seed, [scalar_seed]])
            v0_hi = float(v0_abs_bound) if v0_abs_bound is not None else np.inf
            lo4 = np.array([-v0_hi] * 3 + [0.0])
            hi4 = np.array([v0_hi] * 3 + [mag_hi])
            # Clip the seed into the new bounds so the TRF initial point
            # is feasible (scipy raises 'x0 infeasible' otherwise).
            x0 = np.clip(x0, lo4, hi4)
            result = least_squares(_residuals_axis_anchored, x0, method="trf",
                                   bounds=(lo4, hi4), max_nfev=max_iter * 50)
            p0_opt = p0_pin
            v0_opt = result.x[:3]
            omega_opt = result.x[3] * axis_unit

    mean_residual = float(np.linalg.norm(result.fun) / np.sqrt(n_obs))
    return p0_opt, v0_opt, omega_opt, mean_residual


def _integrate_magnus_state(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray,
    g_vec: np.ndarray,
    drag_k_over_m: float,
    duration_s: float,
    substeps: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """RK4-integrate ``dv/dt = g + k·(ω × v)`` to ``duration_s``.

    Returns ``(p, v)`` at the final time. Mirrors
    :func:`_integrate_magnus_positions` but also returns the terminal
    velocity (needed to seed the post-bounce arc).
    """
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    omega = np.asarray(omega, dtype=float)
    if duration_s <= 0:
        return p, v
    h = duration_s / substeps

    def accel(vv: np.ndarray) -> np.ndarray:
        return g_vec + drag_k_over_m * np.cross(omega, vv)

    for _ in range(substeps):
        k1v = accel(v)
        k1p = v
        k2v = accel(v + 0.5 * h * k1v)
        k2p = v + 0.5 * h * k1v
        k3v = accel(v + 0.5 * h * k2v)
        k3p = v + 0.5 * h * k2v
        k4v = accel(v + h * k3v)
        k4p = v + h * k3v
        p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return p, v


def fit_coupled_bounce(
    obs_a: list[tuple[int, tuple[float, float]]],
    obs_b: list[tuple[int, tuple[float, float]]],
    *,
    cams_a: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    cams_b: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    bounce_frame: int,
    fps: float,
    g: float = -9.81,
    drag_k_over_m: float = 0.005,
    ball_radius: float = 0.11,
    restitution_min: float = 0.5,
    restitution_max: float = 0.85,
    mu_max: float = 0.7,
    omega_max: float = 95.0,
    p0_seed: np.ndarray | None = None,
    v0_seed: np.ndarray | None = None,
    omega0_seed: np.ndarray | None = None,
    e_seed: float = 0.7,
    mu_seed: float = 0.3,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Jointly fit two bounce-adjacent Magnus arcs sharing a spin-coupled bounce.

    Free parameters: arc A's initial state ``(p0, v0, ω0)`` (9 DOF) plus
    the bounce parameters ``(e, μ)`` (2 DOF) — 11 DOF total.  Arc B's
    initial state is *derived*, not free: arc A is integrated (drag+Magnus)
    forward to ``bounce_frame`` to obtain the bounce position and inbound
    ``(v_in, ω_in)``, then :func:`ball_physics.bounce` produces
    ``(v_out, ω_out)``; arc B starts from the bounce position with that
    post-bounce state.  This is what makes spin *identifiable*: the same ω
    must explain the curvature of both arcs *and* the velocity change across
    the bounce.

    The residual is the combined pixel reprojection over BOTH arcs'
    observations, computed with the existing Magnus integrator.

    Args:
        obs_a / obs_b: ``(frame_index, (u, v))`` pairs for each arc,
            ordered by time.  ``bounce_frame`` is the shared node frame
            (arc A ends there, arc B begins there).
        cams_a / cams_b: ``(Ks, Rs, ts)`` position-parallel to ``obs_a`` /
            ``obs_b`` respectively.
        bounce_frame: absolute clip frame of the bounce node.
        fps: clip frame rate.
        g, drag_k_over_m, ball_radius: physics constants.
        restitution_min/max, mu_max, omega_max: parameter bounds.
        *_seed: optional warm starts.
        max_iter: LM iteration cap.

    Returns:
        ``(p0, v0, omega0, e, mu, residual_px)`` — arc A's fitted initial
        state, the fitted bounce params, and the combined RMS pixel residual.
    """
    from scipy.optimize import least_squares

    from src.utils.ball_physics import bounce as _bounce

    Ks_a, Rs_a, ts_a = cams_a
    Ks_b, Rs_b, ts_b = cams_b
    g_vec = np.array([0.0, 0.0, g])

    obs_a_arr = np.array([o[1] for o in obs_a], dtype=float)
    obs_b_arr = np.array([o[1] for o in obs_b], dtype=float)
    fa0 = obs_a[0][0] if obs_a else bounce_frame
    # Times for arc A relative to its first observation; arc B relative to
    # the bounce frame.
    dt_a = np.array([(o[0] - fa0) / fps for o in obs_a])
    dt_b = np.array([(o[0] - bounce_frame) / fps for o in obs_b])
    t_bounce = (bounce_frame - fa0) / fps

    n_a = len(obs_a)
    n_b = len(obs_b)

    if p0_seed is None:
        p0_seed = np.zeros(3)
    if v0_seed is None:
        v0_seed = np.zeros(3)
    if omega0_seed is None:
        omega0_seed = np.zeros(3)

    def _project(pts_world, Ks, Rs, ts, obs_arr, n):
        res = []
        for i in range(n):
            cam = Rs[i] @ pts_world[i] + ts[i]
            pix = Ks[i] @ cam
            uv = pix[:2] / pix[2]
            res.append(uv - obs_arr[i])
        return res

    def _residuals(params: np.ndarray) -> np.ndarray:
        p0 = params[0:3]
        v0 = params[3:6]
        omega0 = params[6:9]
        e = params[9]
        mu = params[10]

        residuals: list[np.ndarray] = []

        # Arc A: integrate Magnus from (p0, v0, omega0) and sample at obs A.
        if n_a:
            pts_a = _integrate_magnus_positions(
                p0, v0, omega0, g_vec, drag_k_over_m, dt_a,
            )
            residuals.extend(_project(pts_a, Ks_a, Rs_a, ts_a, obs_a_arr, n_a))

        # Integrate arc A to the bounce frame to get the bounce state.
        p_bounce, v_in = _integrate_magnus_state(
            p0, v0, omega0, g_vec, drag_k_over_m, t_bounce,
        )
        v_out, omega_out = _bounce(v_in, omega0, e, mu, ball_radius)

        # Arc B starts at the bounce position with the post-bounce state.
        if n_b:
            pts_b = _integrate_magnus_positions(
                p_bounce, v_out, omega_out, g_vec, drag_k_over_m, dt_b,
            )
            residuals.extend(_project(pts_b, Ks_b, Rs_b, ts_b, obs_b_arr, n_b))

        if not residuals:
            return np.zeros(1)
        return np.concatenate(residuals)

    x0 = np.concatenate([
        np.asarray(p0_seed, float),
        np.asarray(v0_seed, float),
        np.asarray(omega0_seed, float),
        [float(e_seed), float(mu_seed)],
    ])
    inf = np.inf
    lo = np.array(
        [-inf, -inf, -inf, -inf, -inf, -inf,
         -omega_max, -omega_max, -omega_max,
         restitution_min, 0.0]
    )
    hi = np.array(
        [inf, inf, inf, inf, inf, inf,
         omega_max, omega_max, omega_max,
         restitution_max, mu_max]
    )
    x0 = np.clip(x0, lo, hi)

    result = least_squares(
        _residuals, x0, method="trf", bounds=(lo, hi),
        max_nfev=max_iter * 50,
    )
    n_total = max(n_a + n_b, 1)
    residual_px = float(np.linalg.norm(result.fun) / np.sqrt(n_total))
    p0_opt = result.x[0:3]
    v0_opt = result.x[3:6]
    omega0_opt = result.x[6:9]
    e_opt = float(result.x[9])
    mu_opt = float(result.x[10])
    return p0_opt, v0_opt, omega0_opt, e_opt, mu_opt, residual_px

"""IMM Kalman tracker for 2D ball pixel observations.

Two motion modes share a constant-velocity dynamic model (state =
``[u, v, vu, vv]``) but use different process-noise covariances:

* **Grounded** — small ``sigma_a`` for smooth rolling motion.
* **Flight** — large ``sigma_a`` to accommodate sharp pixel-space
  accelerations driven by gravity acting on the ball during a kick.

The mode posterior ``p_flight`` is the primary output signal — BallStage
uses runs of ``p_flight >= 0.5`` to seed the parabola / Magnus fit.

Per-mode chi-squared gating against the predicted innovation covariance
protects the filter from spurious detections; gated frames are treated
as missing, with the IMM continuing to predict.  Consecutive misses up
to ``max_gap_frames`` are gap-filled from the blended prediction; longer
gaps emit ``uv=None`` so BallStage marks them ``state="missing"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackerStep:
    """One filtered step emitted by :class:`BallTracker`."""

    frame: int
    uv: tuple[float, float] | None
    p_flight: float
    is_outlier: bool
    is_gap_fill: bool


class BallTracker:
    def __init__(
        self,
        *,
        process_noise_grounded_px: float = 4.0,
        process_noise_flight_px: float = 12.0,
        measurement_noise_px: float = 2.0,
        gating_sigma: float = 4.0,
        mode_transition_prob: float = 0.05,
        max_gap_frames: int = 6,
        initial_p_flight: float = 0.1,
    ) -> None:
        self._q_g = float(process_noise_grounded_px)
        self._q_f = float(process_noise_flight_px)
        self._r = float(measurement_noise_px)
        # Chi-squared 2-DOF threshold at the requested sigma. We square
        # the gate so it can be compared to Mahalanobis ‖y‖² directly.
        self._gate_d2 = float(gating_sigma) ** 2 * 2.0
        self._max_gap = int(max_gap_frames)
        p = float(mode_transition_prob)
        self._Pi = np.array([[1 - p, p], [p, 1 - p]])
        self._F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self._R = (self._r ** 2) * np.eye(2)
        self._x: list[np.ndarray | None] = [None, None]
        self._P: list[np.ndarray | None] = [None, None]
        self._mu = np.array([1.0 - initial_p_flight, initial_p_flight])
        self._consecutive_gap = 0

    @staticmethod
    def _process_noise(sigma_a: float) -> np.ndarray:
        s = sigma_a ** 2
        return s * np.array(
            [
                [1 / 4, 0, 1 / 2, 0],
                [0, 1 / 4, 0, 1 / 2],
                [1 / 2, 0, 1, 0],
                [0, 1 / 2, 0, 1],
            ],
            dtype=float,
        )

    def _init_state(self, uv: tuple[float, float]) -> None:
        x0 = np.array([uv[0], uv[1], 0.0, 0.0], dtype=float)
        P0 = np.diag([self._r ** 2, self._r ** 2, 50.0, 50.0])
        self._x = [x0.copy(), x0.copy()]
        self._P = [P0.copy(), P0.copy()]

    def update(self, frame: int, uv: tuple[float, float] | None) -> TrackerStep:
        # Cold start — wait for the first detection to seed the filter.
        if self._x[0] is None:
            if uv is None:
                return TrackerStep(
                    frame=frame, uv=None, p_flight=float(self._mu[1]),
                    is_outlier=False, is_gap_fill=True,
                )
            self._init_state(uv)
            self._consecutive_gap = 0
            return TrackerStep(
                frame=frame, uv=uv, p_flight=float(self._mu[1]),
                is_outlier=False, is_gap_fill=False,
            )

        # IMM mixing — combine previous per-mode estimates weighted by the
        # transition prior so each mode starts the predict step from a
        # consistent prior.
        c = self._Pi.T @ self._mu
        x_mixed = [np.zeros(4), np.zeros(4)]
        P_mixed = [np.zeros((4, 4)), np.zeros((4, 4))]
        for j in range(2):
            if c[j] <= 0:
                x_mixed[j] = self._x[j].copy()
                P_mixed[j] = self._P[j].copy()
                continue
            for i in range(2):
                w = self._Pi[i, j] * self._mu[i] / c[j]
                x_mixed[j] += w * self._x[i]
            for i in range(2):
                w = self._Pi[i, j] * self._mu[i] / c[j]
                d = self._x[i] - x_mixed[j]
                P_mixed[j] += w * (self._P[i] + np.outer(d, d))

        Qs = [self._process_noise(self._q_g), self._process_noise(self._q_f)]
        likelihoods = np.zeros(2)
        x_post: list[np.ndarray] = [np.zeros(4), np.zeros(4)]
        P_post: list[np.ndarray] = [np.zeros((4, 4)), np.zeros((4, 4))]

        gated_per_mode = [False, False]
        for j in range(2):
            x_pred = self._F @ x_mixed[j]
            P_pred = self._F @ P_mixed[j] @ self._F.T + Qs[j]
            if uv is not None:
                y = np.array(uv, dtype=float) - self._H @ x_pred
                S = self._H @ P_pred @ self._H.T + self._R
                S_inv = np.linalg.inv(S)
                d2 = float(y @ S_inv @ y)
                if d2 > self._gate_d2:
                    gated_per_mode[j] = True
                    x_post[j] = x_pred
                    P_post[j] = P_pred
                    likelihoods[j] = 1e-12
                else:
                    K = P_pred @ self._H.T @ S_inv
                    x_post[j] = x_pred + K @ y
                    P_post[j] = (np.eye(4) - K @ self._H) @ P_pred
                    likelihoods[j] = float(
                        np.exp(-0.5 * d2)
                        / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(S))
                    )
            else:
                x_post[j] = x_pred
                P_post[j] = P_pred
                # No measurement — neutral likelihood preserves prior.
                likelihoods[j] = 1.0

        # Outlier if no mode accepted the measurement.
        is_outlier = uv is not None and all(gated_per_mode)

        if uv is not None and not is_outlier:
            mu_new = c * likelihoods
            total = mu_new.sum()
            if total > 0:
                self._mu = mu_new / total

        self._x = x_post
        self._P = P_post

        is_gap = uv is None or is_outlier
        if is_gap:
            self._consecutive_gap += 1
        else:
            self._consecutive_gap = 0

        if is_gap and self._consecutive_gap > self._max_gap:
            out_uv: tuple[float, float] | None = None
        else:
            blended = self._mu[0] * x_post[0] + self._mu[1] * x_post[1]
            out_uv = (float(blended[0]), float(blended[1]))

        return TrackerStep(
            frame=frame,
            uv=out_uv,
            p_flight=float(self._mu[1]),
            is_outlier=is_outlier,
            is_gap_fill=is_gap,
        )

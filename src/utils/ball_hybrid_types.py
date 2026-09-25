"""Shared dataclasses for the production ball-hybrid trajectory layer.

Ported from the ``prototypes/ball_hybrid_poc`` spike (see
``prototypes/ball_hybrid_poc/CONTRACT.md``) into ``src/`` proper. This
module owns only the small, dependency-light types shared across the
hybrid stack (``ball_hybrid_physics.py``, ``ball_hybrid_blend.py``,
``ball_hybrid_trajectory.py``, ``ball_hybrid_gating.py``, and the
``ball.py`` wiring) so those modules don't need to agree on ad-hoc dicts
or import each other just to share a shape.

- ``Knot``: one span-boundary point the trajectory layer must (or should)
  honour. ``depth_hard=True`` means the FULL 3-D position (``xyz``) is
  authoritative (ground/touch/bounce/net/goal/catch/fix knots — the
  clicked-or-solved pixel resolves to a real 3-D point via the ground
  plane, a joint ray-intersection, or goal geometry). ``depth_hard=False``
  means only the LATERAL constraint (the ray through ``uv``) is
  authoritative — depth is a free/soft parameter the trajectory fit must
  determine from physics + neighbouring evidence, never pulled hard onto
  a single ray-guessed depth. This is the fix for the origi01 held-out
  regression (airborne_mid/high anchors must never hard-pin depth — see
  ``ball_hybrid_trajectory.py``'s module docstring).
- ``HybridShotCtx``: the per-frame camera (K/R/t/distortion) for one
  shot, with the same ``project``/``ray`` helper shape the PoC's
  ``ctx.ClipContext`` used, built directly from what ``ball.py``'s
  ``_solve_shot`` already has in memory (no file IO).
- ``CueEvidence``: one piece of corroborating evidence for an auto-event
  knot, supplied by IC-D's cue modules (``src/utils/ball_cue_*.py``) to
  ``ball_hybrid_gating.py`` to relax the confidence floor for a
  auto-anchor candidate that several independent cues agree on.
- ``SpinFit``: IC-E's (``ball_hybrid_spin.py``) fitted rigid-body spin for
  a flight span, consumed by the trajectory layer's Magnus term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from src.utils.camera_projection import pixel_ray, project_world_to_image

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

KnotSource = Literal["manual", "auto", "fix"]


@dataclass(frozen=True)
class Knot:
    """One trajectory span-boundary point.

    ``kind`` is the anchor/event state that produced it (e.g.
    ``"player_touch"``, ``"kick"``, ``"bounce"``, ``"header"``,
    ``"volley"``, ``"chest"``, ``"catch"``, ``"goal_impact"``,
    ``"grounded"``, ``"fix"``, or ``"internal_split"`` for a
    data-discovered split point) — NOT a fit-mechanism label.

    ``depth_hard`` distinguishes a knot whose full ``xyz`` must be
    honoured (ground/touch/bounce/net/goal/catch/fix) from one that is
    only laterally constrained (an airborne ray with no independent depth
    signal): for the latter, ``xyz`` still carries the best available
    estimate (so every knot has a usable position), but the trajectory
    layer must treat it as a soft pull via ``uv``/the camera ray, never a
    hard 3-D re-snap.

    ``weight`` is the knot's relative weight in a chain/span fit (e.g. a
    manual click weighted far above a real detector observation) —
    distinct from whether the knot is span-boundary-hard at all.
    """

    frame: int
    xyz: Vec3
    kind: str
    depth_hard: bool
    source: KnotSource
    weight: float = 1.0
    uv: Optional[Vec2] = None

    def __post_init__(self) -> None:
        if len(self.xyz) != 3:
            raise ValueError(f"Knot.xyz must be a 3-vector, got {self.xyz!r}")
        if self.uv is not None and len(self.uv) != 2:
            raise ValueError(f"Knot.uv must be a 2-vector, got {self.uv!r}")


@dataclass(frozen=True)
class CueEvidence:
    """One corroborating cue for an auto-event candidate knot, supplied
    to ``ball_hybrid_gating.py``'s evidence-consistency gate. ``kind``
    matches the candidate's event kind (e.g. ``"bounce"``); ``cue`` names
    the signal (e.g. ``"audio_impact"``, ``"shadow_contact"``,
    ``"crowd_reaction"`` — IC-D-defined). ``xyz``/``uv`` are optional —
    a cue may only corroborate a frame, not a position."""

    frame: int
    kind: str
    cue: str
    conf: float
    xyz: Optional[Vec3] = None
    uv: Optional[Vec2] = None


@dataclass(frozen=True)
class SpinFit:
    """A fitted rigid-body angular velocity for one flight span (IC-E,
    ``ball_hybrid_spin.py``). ``omega_world`` is rad/s in world axes;
    ``delta_bic`` is the Bayesian-information-criterion improvement of
    the spun fit over a spin-free fit (positive = spin genuinely helps;
    the trajectory layer should ignore a non-positive ``delta_bic``)."""

    omega_world: Vec3
    rad_s: float
    delta_bic: float


@dataclass(frozen=True)
class HybridShotCtx:
    """Per-frame camera context for one shot, built directly from what
    ``ball.py``'s ``_solve_shot`` already holds (``per_frame_K/R/t``,
    ``distortion``) — no file IO, unlike the PoC's ``ctx.ClipContext``
    (which loaded a ``CameraTrack`` from disk). Frame keys are plain
    ``int``.
    """

    clip_id: str
    fps: float
    image_size: tuple[int, int]
    per_frame_K: dict
    per_frame_R: dict
    per_frame_t: dict
    distortion: tuple[float, float]

    def project(self, frame: int, xyz) -> np.ndarray:
        """Project one point (shape ``(3,)``) or many (``(N, 3)``) to
        image pixels for ``frame``'s camera. Mirrors a single-point input
        back out as a single ``(2,)`` array."""
        frame = int(frame)
        K = self.per_frame_K[frame]
        R = self.per_frame_R[frame]
        t = self.per_frame_t[frame]
        pts = np.asarray(xyz, dtype=np.float64)
        single = pts.ndim == 1
        out = project_world_to_image(K, R, t, self.distortion,
                                      pts.reshape(-1, 3))
        return out[0] if single else out

    def ray(self, frame: int, uv) -> tuple[np.ndarray, np.ndarray]:
        """Camera centre + unit world-space ray direction through pixel
        ``uv`` at ``frame``'s camera. ``(C, d_hat)``."""
        frame = int(frame)
        K = self.per_frame_K[frame]
        R = self.per_frame_R[frame]
        t = self.per_frame_t[frame]
        return pixel_ray(uv, K, R, t, self.distortion)

    def has_frame(self, frame: int) -> bool:
        return int(frame) in self.per_frame_K

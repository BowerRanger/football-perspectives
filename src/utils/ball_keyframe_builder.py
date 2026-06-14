"""Pure builder turning resolved ball anchors into a sparse
``BallKeyframeSet``. Kept out of the already-large ``ball.py`` so the stage
only needs a single call after it has saved the dense track.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet, DepthSource
from src.schemas.ball_track import FlightSegment
from src.utils.ball_orientation import _flight_omega

_AIRBORNE_STATES = frozenset(
    {"airborne_low", "airborne_mid", "airborne_high"}
)


def _camera_ray(
    uv: tuple[float, float],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (origin, unit-dir) of the camera ray through pixel ``uv``.

    Same construction as ``ball._project_point_onto_pixel_ray``:
    ``C = -R^T t`` and ``d_hat = normalize(R^T K^-1 [u, v, 1])``.
    """
    from src.utils.camera_projection import undistort_pixel

    uv_arr = np.asarray(uv, dtype=float)
    if distortion != (0.0, 0.0):
        uv_arr = undistort_pixel(uv_arr, K, distortion)
    C = -R.T @ t
    d_world = R.T @ (np.linalg.inv(K) @ np.array([uv_arr[0], uv_arr[1], 1.0]))
    d_hat = d_world / np.linalg.norm(d_world)
    return (
        (float(C[0]), float(C[1]), float(C[2])),
        (float(d_hat[0]), float(d_hat[1]), float(d_hat[2])),
    )


def _depth_source(
    anc: BallAnchor, ground_touch_frames: set[int],
) -> DepthSource:
    """Return the ``DepthSource`` bucket for a resolved anchor.

    The mapping is:

    * ``goal_impact`` → ``"goal_geometry"``: the dense stage pins the ball
      through the geometrically-known goal contact point.
    * ``player_touch`` (airborne, i.e. frame NOT in ``ground_touch_frames``) →
      ``"player_bone"``: the dense stage drives the ball through the SMPL
      bone world position for the named player/bone.
    * ``player_touch`` (ground-touch frame, i.e. frame IN
      ``ground_touch_frames``) → ``"ground"``: the dense stage resolves via a
      ray cast to z=ball_radius, the same family as other contact states.
    * ``airborne_low`` / ``airborne_mid`` / ``airborne_high`` →
      ``"ray_physics"``: depth is underdetermined from a single pixel so the
      dense stage uses ray constraints together with gravity/knot physics.
    * Everything else (``grounded``, ``kick``, ``catch``, ``bounce``,
      ``header``, ``volley``, ``chest``, ``off_screen_flight``) → ``"ground"``:
      the dense stage resolves these via a ray cast to a canonical-height
      horizontal plane (``ankle_ray_to_pitch`` at ``state_to_height(state)``).
      This is the "ray-to-known-plane" family.  The bucket is labelled
      ``"ground"`` even when the canonical height is non-zero (e.g. a
      ``"header"`` resolves to ~2.3 m) because the depth source is still an
      analytically-known horizontal plane, not a player bone or physics fit.
    """
    if anc.state == "goal_impact":
        return "goal_geometry"
    if anc.state == "player_touch":
        if anc.frame in ground_touch_frames:
            return "ground"
        return "player_bone"
    if anc.state in _AIRBORNE_STATES:
        return "ray_physics"
    return "ground"


def _omega_for_frame(
    frame: int, flight_segments: Sequence[FlightSegment],
) -> tuple[float, float, float] | None:
    """Fitted Magnus spin vector (rad/s) for a keyframe on a flight segment.

    Returns ``spin_omega_rad_s * unit(spin_axis_world)`` (reusing the same
    ``_flight_omega`` helper that drives orientation integration) when
    ``frame`` falls inside a flight segment that recovered a spin; ``None``
    otherwise (no covering flight, or a flight without spin).
    """
    for seg in flight_segments:
        lo, hi = seg.frame_range
        if lo <= frame <= hi:
            omega = _flight_omega(seg)
            if float(np.linalg.norm(omega)) <= 0.0:
                return None
            return (float(omega[0]), float(omega[1]), float(omega[2]))
    return None


def build_ball_keyframe_set(
    *,
    clip_id: str,
    fps: float,
    image_size: tuple[int, int],
    anchor_by_frame: dict[int, BallAnchor],
    world_by_frame: dict[int, tuple[float, float, float] | None],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    ground_touch_frames: set[int] | None = None,
    flight_segments: Sequence[FlightSegment] = (),
) -> BallKeyframeSet:
    """Collect one ``BallKeyframe`` per manual anchor.

    ``world_by_frame`` holds the *already-resolved* world position for each
    anchor frame (as the dense stage produced it), so emitted ``world_xyz``
    matches the dense track exactly. Airborne anchors additionally get the
    clicked camera ray; ``off_screen_flight`` anchors (no pixel) get neither
    ray nor world position.

    Emitted keyframes keep the schema default ``confidence=1.0`` by design:
    a manual anchor is user-supplied ground truth, so there is no per-anchor
    confidence to propagate in Phase A.
    """
    gtf = ground_touch_frames or set()
    keyframes: list[BallKeyframe] = []
    for fi in sorted(anchor_by_frame):
        anc = anchor_by_frame[fi]
        world = world_by_frame.get(fi)
        ray = None
        if (
            anc.state in _AIRBORNE_STATES
            and anc.image_xy is not None
            and fi in per_frame_K
            and fi in per_frame_R
            and fi in per_frame_t
        ):
            ray = _camera_ray(
                (float(anc.image_xy[0]), float(anc.image_xy[1])),
                per_frame_K[fi], per_frame_R[fi], per_frame_t[fi],
                distortion,
            )
        keyframes.append(
            BallKeyframe(
                frame=fi,
                state=anc.state,  # type: ignore[arg-type]
                depth_source=_depth_source(anc, gtf),
                world_xyz=(
                    (float(world[0]), float(world[1]), float(world[2]))
                    if world is not None else None
                ),
                image_xy=(
                    (float(anc.image_xy[0]), float(anc.image_xy[1]))
                    if anc.image_xy is not None else None
                ),
                ray=ray,
                player_id=anc.player_id,
                bone=anc.bone,
                goal_element=anc.goal_element,
                touch_type=anc.touch_type,
                spin=anc.spin,
                omega_rad_s=_omega_for_frame(fi, flight_segments),
            )
        )
    return BallKeyframeSet(
        clip_id=clip_id,
        fps=fps,
        image_size=(int(image_size[0]), int(image_size[1])),
        keyframes=tuple(keyframes),
    )

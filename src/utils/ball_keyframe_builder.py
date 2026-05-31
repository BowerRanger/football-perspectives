"""Pure builder turning resolved ball anchors into a sparse
``BallKeyframeSet``. Kept out of the already-large ``ball.py`` so the stage
only needs a single call after it has saved the dense track.
"""

from __future__ import annotations

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet

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
) -> str:
    if anc.state == "goal_impact":
        return "goal_geometry"
    if anc.state == "player_touch":
        if anc.frame in ground_touch_frames:
            return "ground"
        return "player_bone"
    if anc.state in _AIRBORNE_STATES:
        return "ray_physics"
    return "ground"


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
) -> BallKeyframeSet:
    """Collect one ``BallKeyframe`` per manual anchor.

    ``world_by_frame`` holds the *already-resolved* world position for each
    anchor frame (as the dense stage produced it), so emitted ``world_xyz``
    matches the dense track exactly. Airborne anchors additionally get the
    clicked camera ray; ``off_screen_flight`` anchors (no pixel) get neither
    ray nor world position.
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
                depth_source=_depth_source(anc, gtf),  # type: ignore[arg-type]
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
            )
        )
    return BallKeyframeSet(
        clip_id=clip_id,
        fps=fps,
        image_size=(int(image_size[0]), int(image_size[1])),
        keyframes=tuple(keyframes),
    )

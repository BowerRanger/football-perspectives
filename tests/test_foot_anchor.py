import numpy as np
import pytest

from src.utils.foot_anchor import ankle_ray_to_pitch, anchor_translation


def _project(K, R, t, p):
    cam = R @ p + t
    pix = K @ cam
    return pix[:2] / pix[2]


@pytest.mark.unit
def test_ankle_ray_to_pitch_recovers_known_world_point():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    R = np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )
    t = np.array([-52.5, 100.0, 22.0])
    pitch_pt = np.array([30.0, 40.0, 0.05])  # foot height
    uv = _project(K, R, t, pitch_pt)
    recovered = ankle_ray_to_pitch(uv, K=K, R=R, t=t, plane_z=0.05)
    assert np.allclose(recovered, pitch_pt, atol=1e-3)


@pytest.mark.unit
def test_anchor_translation_subtracts_foot_offset():
    foot_world = np.array([30.0, 40.0, 0.05])
    # Foot is 0.95 m below root, expressed in the root frame. Since
    # R_root_world = I here, the root frame is aligned with the pitch
    # frame (z-up), so "below" is -z.
    foot_in_root = np.array([0.0, 0.0, -0.95])
    R_root_world = np.eye(3)
    root_t = anchor_translation(foot_world, foot_in_root, R_root_world)
    # Root should be 0.95 m above the foot in world z.
    assert np.allclose(root_t, np.array([30.0, 40.0, 1.0]), atol=1e-3)


@pytest.mark.unit
def test_ankle_ray_to_pitch_undistorts_pixel_before_back_projection():
    """When the camera has non-zero radial distortion, the back-projection
    must undo it before applying K^-1, otherwise the recovered pitch point
    is offset by tens of centimetres at the far touchline."""
    from src.utils.camera_projection import project_world_to_image

    # Physical broadcast pose: camera at world (52.5, -30, 30) looking at
    # pitch centre. Build R from a normalised look-direction so it's
    # orthonormal to floating-point precision.
    look = np.array([0.0, 64.0, -30.0])
    look = look / np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    C_world = np.array([52.5, -30.0, 30.0])
    t = -R @ C_world
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]], dtype=float)
    pitch_pt = np.array([30.0, 40.0, 0.05])
    distortion = (0.10, -0.02)

    distorted_uv = project_world_to_image(
        K, R, t, distortion, pitch_pt.reshape(1, 3),
    )[0]
    # Sanity: ignoring distortion should produce a visibly biased recovery.
    biased = ankle_ray_to_pitch(distorted_uv, K=K, R=R, t=t, plane_z=0.05)
    bias_dist = float(np.linalg.norm(biased - pitch_pt))
    assert bias_dist > 0.1, (
        f"test geometry doesn't exercise distortion (bias only {bias_dist:.3f} m)"
    )

    recovered = ankle_ray_to_pitch(
        distorted_uv, K=K, R=R, t=t, plane_z=0.05, distortion=distortion,
    )
    assert np.allclose(recovered, pitch_pt, atol=0.05)

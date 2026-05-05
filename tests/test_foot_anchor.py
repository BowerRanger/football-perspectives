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

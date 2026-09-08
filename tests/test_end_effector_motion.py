import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.end_effector_motion import animate_end_effectors
from src.utils.foot_contact import ContactSpan, FootContacts


def _inputs(n=40):
    thetas = np.zeros((n, 24, 3))
    root_R = np.tile(np.eye(3), (n, 1, 1))
    root_t = np.tile([0.0, 0.0, 0.95], (n, 1))
    return thetas, root_R, root_t


def _contacts(n, spans):
    in_contact = np.zeros((n, 2), bool)
    for s in spans:
        in_contact[s.start:s.end, s.side] = True
    return FootContacts(in_contact=in_contact, quality=np.ones((n, 2)),
                        spans=tuple(spans), n_frames=n)


def test_hands_get_a_relaxed_curl_and_body_joints_stay_untouched():
    thetas, root_R, root_t = _inputs()
    out, stats = animate_end_effectors(
        thetas=thetas, root_R=root_R, root_t=root_t, betas=np.zeros(10),
        contacts=None, fps=30.0)
    assert np.abs(out[:, (22, 23)]).max() > 0.1          # hands no longer flat
    body = [j for j in range(24) if j not in (10, 11, 22, 23)]
    np.testing.assert_array_equal(out[:, body], thetas[:, body])
    np.testing.assert_array_equal(out[:, (10, 11)], thetas[:, (10, 11)])  # no contacts -> flat feet


def test_follow_through_echoes_wrist_swing():
    thetas, root_R, root_t = _inputs()
    thetas[18:24, 20] = np.deg2rad([0, 0, 0])            # quiet wrist elsewhere
    thetas[20, 20, 1] = np.deg2rad(40)                   # single sharp wrist swing
    out, _ = animate_end_effectors(
        thetas=thetas, root_R=root_R, root_t=root_t, betas=np.zeros(10),
        contacts=None, fps=30.0)
    static = Rotation.from_rotvec(out[5, 22])
    after = Rotation.from_rotvec(out[22, 22])
    delta = np.degrees(np.linalg.norm((static.inv() * after).as_rotvec()))
    assert delta > 2.0                                    # whip visible after the swing


def test_toe_roll_only_around_span_edges_and_stance_stays_flat():
    n = 40
    thetas, root_R, root_t = _inputs(n)
    span = ContactSpan(0, 10, 20, np.array([0.0, 0.0, 0.025]))
    out, stats = animate_end_effectors(
        thetas=thetas, root_R=root_R, root_t=root_t, betas=np.zeros(10),
        contacts=_contacts(n, [span]), fps=30.0,
        cfg={"min_toe_clearance_m": -1.0})               # disable height gate for the synthetic pose
    assert stats["toe_roll_frames"] > 0
    left = out[:, 10]
    assert np.abs(left[10:20]).max() == 0.0              # stance flat
    assert left[20, 0] > 0.1                             # plantarflex at lift-off
    assert left[9, 0] < -0.05                            # dorsiflex on approach
    assert np.abs(left[:5]).max() == 0.0                 # far from the span: untouched
    np.testing.assert_array_equal(out[:, 11], thetas[:, 11])  # other foot untouched

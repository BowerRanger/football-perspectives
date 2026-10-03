"""shorts_framing: the G19 bad drafts (found by eye in 7 re-renders) must be
rejected from camera-track arrays alone, and a sane chase cam must pass."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.shorts_framing import (
    CameraArrays, FramingLimits, camera_arrays_from_track, check_framing,
    project_points,
)
from src.utils.virtual_cameras import intrinsics_from_fov, look_at_view

FRAMES = np.arange(371, 403)
STRIKER = np.array([18.0, 24.4])


def ball_path(frames=FRAMES):
    out = {}
    for f in frames:
        s = (f - 371) / 23.0
        p = np.array([18.0, 24.4, 0.11]) + s * (np.array([0.0, 37.1, 1.8]) - np.array([18.0, 24.4, 0.11]))
        out[int(f)] = tuple(p if s <= 1.0 else np.array([-1.4, 37.0, 1.6]))
    return out


def players(extra=None):
    fr = np.arange(0, 450)
    def still(xy): return (fr, np.tile(np.array(xy, float), (len(fr), 1)))
    d = {"S": still(STRIKER), "K": still((1.0, 34.0)), "F": still((30.0, 40.0))}
    d.update(extra or {})
    return d


def cam_from(centres, targets, fov, frames=FRAMES):
    Rs, ts = [], []
    for c, t in zip(centres, targets):
        R, tt = look_at_view(np.asarray(c, float), np.asarray(t, float))
        Rs.append(R); ts.append(tt)
    return CameraArrays(frames=np.asarray(frames), R=np.array(Rs), t=np.array(ts), fov_deg=fov)


def test_40m_top_down_drone_is_rejected_as_too_small():
    ball = ball_path()
    cen = [(9.0, 30.0, 40.0)] * len(FRAMES)
    tgt = [ball[int(f)] for f in FRAMES]
    cam = cam_from([(c[0] + 1.0, c[1] - 1.0, c[2]) for c in cen], tgt, fov=48.0)
    res = check_framing(cam, ball, players(), 371, 402, subject_pid="S")
    assert not res.ok
    assert "subject_too_small" in {f.check for f in res.failures}


def test_otm_inside_the_striker_is_rejected():
    ball = ball_path()
    cen = [(STRIKER[0] - 0.3, STRIKER[1] - 0.1, 1.5)] * len(FRAMES)
    tgt = [ball[int(f)] for f in FRAMES]
    cam = cam_from(cen, tgt, fov=46.0)
    res = check_framing(cam, ball, players(), 371, 402, subject_pid="S")
    assert "camera_in_player" in {f.check for f in res.failures}


def test_chase_cam_the_ball_outruns_is_rejected():
    ball = ball_path()
    # camera barely pans (10% of the ball's travel), from the side
    b0 = np.array(ball[371])
    cen = [(9.0, 18.0, 1.5)] * len(FRAMES)
    tgt = [b0 + 0.1 * (np.array(ball[int(f)]) - b0) for f in FRAMES]
    cam = cam_from(cen, tgt, fov=45.0)
    res = check_framing(cam, ball, players(), 371, 402)
    assert "ball_out_of_safe_area" in {f.check for f in res.failures}


def test_eye_line_blocked_by_a_defender_is_rejected():
    ball = {int(f): (8.0, 31.0, 1.0) for f in FRAMES}      # ball hangs in the air
    eye = (1.0, 34.0, 1.7)
    # defender stands on the keeper->ball line for the whole window
    d_xy = (eye[0] + (8.0 - eye[0]) * 0.4, eye[1] + (31.0 - eye[1]) * 0.4)
    cam = cam_from([eye] * len(FRAMES), [ball[int(f)] for f in FRAMES], fov=50.0)
    res = check_framing(cam, ball, players({"D": (np.arange(450), np.tile(np.array(d_xy), (450, 1)))}),
                        371, 402, exclude_pids=["K"])
    assert "ball_occluded" in {f.check for f in res.failures}


def test_keeper_own_body_is_ignored_for_eyes_cam():
    ball = {int(f): (8.0, 31.0, 1.0) for f in FRAMES}
    eye = (1.0, 34.0, 1.7)
    cam = cam_from([eye] * len(FRAMES), [ball[int(f)] for f in FRAMES], fov=50.0)
    res = check_framing(cam, ball, players(), 371, 402, exclude_pids=["K"])
    assert res.ok, res.failures


def test_sane_chase_cam_passes():
    ball = ball_path()
    cen, tgt = [], []
    for f in FRAMES:
        b = np.array(ball[int(f)])
        nxt = np.array(ball[min(int(f) + 1, 402)])
        v = nxt - b if np.linalg.norm(nxt - b) > 1e-6 else np.array([-1.0, 0.7, 0.0])
        v = np.array([v[0], v[1], 0.0]); v = v / np.linalg.norm(v)
        cen.append(b - v * 6.0 + np.array([0, 0, 1.6]))
        tgt.append(b)
    cam = cam_from(cen, tgt, fov=45.0)
    res = check_framing(cam, ball, players(), 376, 398)
    assert res.ok, [f.detail for f in res.failures]
    assert res.metrics["ball_px_median"] > 12


def test_limits_override_and_unknown_key():
    lim = FramingLimits().merged({"min_subject_px": 10.0})
    assert lim.min_subject_px == 10.0
    with pytest.raises(ValueError, match="unknown framing limit"):
        FramingLimits().merged({"nope": 1})


def test_portrait_projection_uses_vertical_fov():
    cam = cam_from([(0, 0, 1.0)], [(10, 0, 1.0)], fov=40.0, frames=[0])
    # a point 10 m ahead, 3.64 m up -> top of a 40deg vertical FOV at 10 m
    uv, z = project_points(cam, np.array([[10.0, 0.0, 1.0 + 10 * np.tan(np.radians(20))]]), np.array([0]))
    assert uv[0, 1] == pytest.approx(0.0, abs=1.0) and z[0] == pytest.approx(10.0)


def test_camera_arrays_from_track_recovers_fov():
    class F:  # CameraFrame look-alike
        def __init__(self, fr):
            self.frame = fr; self.K = intrinsics_from_fov(48.0, (1920, 1080))
            self.R = np.eye(3).tolist(); self.t = [0.0, 0.0, 0.0]
    class T:
        image_size = (1920, 1080); t_world = [0, 0, 0]; frames = [F(1), F(2)]
    arr = camera_arrays_from_track(T())
    assert arr.fov_deg == pytest.approx(48.0) and list(arr.frames) == [1, 2]


# --- calibration against the hand-approved gberch passes ----------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "output-shorts"
_REAL = _OUT / "render_experiments/s1_chase/gberch/cameras/chase_camera_track.json"


def _real_check(exp, cam_file, a, b, **kw):
    from src.schemas.camera_track import CameraTrack
    from src.utils.shorts_moments import _load_root_xy
    t = json.loads((_OUT / "ball/gberch_ball_track.json").read_text())
    ball = {f["frame"]: f["world_xyz"] for f in t["frames"]}
    pids = [p.stem.split("_")[0] for p in (_OUT / "refined_poses").glob("P*_refined.npz")]
    cam = camera_arrays_from_track(CameraTrack.load(
        _OUT / f"render_experiments/{exp}/gberch/cameras/{cam_file}"))
    lim = FramingLimits().merged(kw.pop("limits", None))
    return check_framing(cam, ball, _load_root_xy(_OUT, pids), a, b, limits=lim, **kw)


@pytest.mark.skipif(not _REAL.exists(), reason="gberch render_experiments not linked")
@pytest.mark.parametrize("exp,cam,a,b,kw", [
    ("s1_chase", "chase_camera_track.json", 343, 396, {}),
    ("s1_orbit_slow", "orbit_camera_track.json", 362, 380, {"subject_pid": "P006"}),
    ("s1_goal_slow", "goal_left_camera_track.json", 374, 408, {}),
    ("s2_goalline", "goal_left_camera_track.json", 302, 338, {}),
    ("s2_eyes", "eyes_P005_camera_track.json", 330, 362,
     {"exclude_pids": ["P005"], "limits": {"max_occluded_frames": 30}}),
    ("s2_eyes_slow", "eyes_P005_camera_track.json", 362, 393, {"exclude_pids": ["P005"]}),
    ("s2_goal", "goal_left_camera_track.json", 390, 420, {}),
    ("s3_ots", "ots_P006_camera_track.json", 335, 375, {"subject_pid": "P006"}),
    ("s3_chase_slow", "chase_camera_track.json", 366, 396, {}),
    ("s3_net_orbit", "orbit_camera_track.json", 388, 410, {}),
    ("s1_drone", "drone_camera_track.json", 232, 343,
     {"subject_pid": "P006", "limits": {"min_subject_px": 100, "check_ball": False}}),
])
def test_hand_approved_gberch_passes_are_accepted(exp, cam, a, b, kw):
    res = _real_check(exp, cam, a, b, **kw)
    assert res.ok, [f.detail for f in res.failures]


@pytest.mark.skipif(not _REAL.exists(), reason="gberch render_experiments not linked")
def test_goalline_opener_draft_is_rejected_on_real_data():
    # G19: "goal-line opener showed a defender" - the goalline rig draft
    res = _real_check("s2_goalline", "goalline_left_camera_track.json", 302, 338)
    assert not res.ok
    assert {f.check for f in res.failures} & {"ball_out_of_safe_area", "ball_occluded"}

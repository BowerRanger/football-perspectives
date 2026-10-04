"""Ball Studio router against a tiny synthetic output dir."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.utils import ball_truth_solver as S
from src.web.server import create_app

FPS = 30.0
IMG = (1920, 1080)
OFF_B = -10


def look_at(centre, target, fx=2000.0):
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    fwd = target - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = np.array([[fx, 0, IMG[0] / 2], [0, fx, IMG[1] / 2], [0, 0, 1.0]])
    return K, R, -R @ centre


CAMS = {"a": look_at([52.5, -30, 20], [52.5, 30, 0]),
        "b": look_at([10, 34, 15], [52.5, 20, 0])}


def write_cam(out: Path, sid: str, n: int = 80):
    K, R, t = CAMS[sid]
    frames = tuple(
        CameraFrame(frame=i, K=K.tolist(), R=R.tolist(), confidence=0.9,
                    is_anchor=False, t=t.tolist()) for i in range(n))
    CameraTrack(clip_id=sid, fps=FPS, image_size=IMG, t_world=t.tolist(),
                frames=frames, distortion=(0.0, 0.0)).save(
        out / "camera" / f"{sid}_camera_track.json")


@pytest.fixture
def env(tmp_path: Path):
    out = tmp_path
    (out / "shots").mkdir()
    (out / "camera").mkdir()
    (out / "ball").mkdir()
    (out / "refined_poses").mkdir()
    shots = [Shot(id=s, start_frame=0, end_frame=80, start_time=0, end_time=3,
                  clip_file=f"{s}.mp4") for s in ("a", "b", "z")]
    ShotsManifest(source_file="x.mp4", fps=FPS, total_frames=80, shots=shots).save(
        out / "shots" / "shots_manifest.json")
    SyncMap(groups=[GroupSync("", "a", [Alignment("a", 0), Alignment("b", OFF_B)])]).save(
        out / "shots" / "sync_map.json")
    write_cam(out, "a")
    write_cam(out, "b")  # z has no camera -> not a studio group
    n = 60
    np.savez(
        out / "refined_poses" / "P001_refined.npz",
        player_id="P001", frames=np.arange(n), betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)), root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=np.tile([40.0, 20.0, 0.9], (n, 1)), confidence=np.ones(n),
        view_count=np.ones(n, int), contributing_shots=np.array(["a"]))
    (out / "ball" / "a_ball_anchors.json").write_text(json.dumps({
        "clip_id": "a", "image_size": [1920, 1080],
        "anchors": [{"frame": 20, "image_xy": [900.0, 500.0], "state": "grounded"}]}))
    app = create_app(output_dir=out, config_path=None)
    return TestClient(app), out


def uv(sid, p):
    K, R, t = CAMS[sid]
    cam = S.Cam(K, R, t, (0.0, 0.0), IMG)
    return [float(x) for x in cam.project(np.asarray(p))[0][0]]


def tri_key(kid, ref, p):
    return {"id": kid, "frame": ref, "xyz": [0, 0, 0], "source": "triangulated",
            "observations": [
                {"shot_id": "a", "shot_frame": ref, "uv": uv("a", p)},
                {"shot_id": "b", "shot_frame": ref + OFF_B, "uv": uv("b", p)}]}


def truth_doc(c):
    doc = c.get("/api/ball-studio/groups/a/truth").json()["truth"]
    doc["keys"] = [tri_key("k0", 20, [30, 5, 0.11]), tri_key("k1", 50, [36, 9, 0.11])]
    return doc


def test_groups_listing(env):
    c, _ = env
    body = c.get("/api/ball-studio/groups").json()
    assert [g["group_id"] for g in body["groups"]] == ["a"]
    g = body["groups"][0]
    assert g["reference_shot"] == "a" and not g["has_truth"]
    sb = next(s for s in g["shots"] if s["shot_id"] == "b")
    assert sb["frame_offset"] == OFF_B and sb["frame_range"] == [0, 79]
    assert sb["width"] == 1920 and sb["height"] == 1080
    assert g["ref_frame_range"] == [0, 89]


def test_scene_payload(env):
    c, _ = env
    r = c.get("/api/ball-studio/groups/a/scene")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    s = r.json()
    sh = s["shots"][0]
    assert len(sh["K"]) == len(sh["R"]) == len(sh["t"]) == len(sh["frames"]) == 80
    assert len(sh["K"][0]) == 4 and len(sh["R"][0]) == 9
    assert sh["frame_range"] == [0, 79] and sh["width"] == 1920
    assert s["goals"]["goal_planes"][1]["value"] == 105.0
    assert s["goals"]["goal_planes"][0]["mouth"]["z_range"] == [0.0, 2.44]
    p = s["players"][0]
    assert p["player_id"] == "P001" and set(p["joints"]) == set(s["bones"])
    assert len(p["joints"]["head"]) == len(p["frames"])
    assert s["pipeline_anchors"][0]["kind"] == "grounded"


def test_scene_unknown_group_404_and_bad_id_400(env):
    c, _ = env
    assert c.get("/api/ball-studio/groups/nope/scene").status_code == 404
    assert c.get("/api/ball-studio/groups/bad.id/scene").status_code == 400


def test_truth_empty_skeleton(env):
    c, _ = env
    b = c.get("/api/ball-studio/groups/a/truth").json()
    assert b["exists"] is False and b["truth"]["keys"] == []
    assert {s["shot_id"]: s["frame_offset"] for s in b["truth"]["shots"]} == {"a": 0, "b": OFF_B}


def test_put_roundtrip_history_and_dense(env):
    c, out = env
    doc = truth_doc(c)
    r = c.put("/api/ball-studio/groups/a/truth", json={"truth": doc, "expected_updated_at": None})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["solve_ok"] and body["history_file"] is None
    assert (out / "ball_truth" / "a_ball_truth.json").exists()
    assert (out / "ball_truth" / "a_ball_truth_dense.json").exists()
    got = c.get("/api/ball-studio/groups/a/truth").json()
    assert got["exists"] and got["truth"]["meta"]["updated_at"] == body["updated_at"]
    assert got["dense"]["dense"]["frames"][0] == 20
    doc2 = got["truth"]
    doc2["meta"]["status"] = "reviewed"
    r2 = c.put("/api/ball-studio/groups/a/truth",
               json={"truth": doc2, "expected_updated_at": body["updated_at"]})
    assert r2.status_code == 200 and r2.json()["history_file"]
    assert list((out / "ball_truth" / ".history").glob("*.json"))


def test_put_conflict_409(env):
    c, _ = env
    doc = truth_doc(c)
    assert c.put("/api/ball-studio/groups/a/truth", json={"truth": doc}).status_code == 200
    r = c.put("/api/ball-studio/groups/a/truth",
              json={"truth": doc, "expected_updated_at": "1999-01-01T00:00:00Z"})
    assert r.status_code == 409
    assert r.json()["detail"]["current_updated_at"]
    # null expected with an existing file is also a conflict
    assert c.put("/api/ball-studio/groups/a/truth",
                 json={"truth": doc, "expected_updated_at": None}).status_code == 409


def test_put_invalid_422_writes_nothing(env):
    c, out = env
    doc = truth_doc(c)
    doc["outcome"] = "maybe"
    r = c.put("/api/ball-studio/groups/a/truth", json={"truth": doc})
    assert r.status_code == 422 and r.json()["detail"]["errors"]
    assert not (out / "ball_truth").exists()
    assert c.put("/api/ball-studio/groups/a/truth", json=doc).status_code == 422


def test_solve_endpoint(env):
    c, _ = env
    r = c.post("/api/ball-studio/groups/a/solve", json=truth_doc(c))
    assert r.status_code == 200
    b = r.json()
    assert b["ok"] and b["dense"]["frames"][0] == 20 and b["dense"]["frames"][-1] == 50
    assert b["keys"][0]["xyz"] == pytest.approx([30, 5, 0.11], abs=1e-2)
    assert b["projections"]["b"]["shot_frames"][0] == 20 + OFF_B
    assert c.post("/api/ball-studio/groups/a/solve", json={"version": 1}).status_code == 422


def test_triangulate_two_views(env):
    c, _ = env
    p = [40, 25, 2.0]
    body = {"frame": 30, "observations": [
        {"shot_id": "a", "shot_frame": 30, "uv": uv("a", p)},
        {"shot_id": "b", "shot_frame": 30 + OFF_B, "uv": uv("b", p)}]}
    r = c.post("/api/ball-studio/groups/a/triangulate", json=body).json()
    assert r["ok"] and r["xyz"] == pytest.approx(p, abs=1e-2)
    assert r["max_residual_px"] < 0.05 and r["skew_gap_cm"] < 0.5
    assert set(r["reprojected_uv"]) == {"a", "b"}
    assert r["reprojected_uv"]["a"] == pytest.approx(uv("a", p), abs=0.05)


def test_triangulate_bad_click_exceeds_limit(env):
    c, _ = env
    p = [40, 25, 2.0]
    ub = uv("b", p)
    ub[0] += 120
    r = c.post("/api/ball-studio/groups/a/triangulate", json={"frame": 30, "observations": [
        {"shot_id": "a", "shot_frame": 30, "uv": uv("a", p)},
        {"shot_id": "b", "shot_frame": 20, "uv": ub}]}).json()
    assert not r["ok"] and r["reason"] == "residual_exceeds_limit"
    assert r["xyz"] is not None and r["max_residual_px"] > 15


def test_triangulate_offset_override_probe(env):
    c, _ = env
    p = [40, 25, 2.0]
    obs = [{"shot_id": "a", "shot_frame": 30, "uv": uv("a", p)},
           {"shot_id": "b", "shot_frame": 20, "uv": uv("b", p)}]
    base = c.post("/api/ball-studio/groups/a/triangulate",
                  json={"frame": 30, "observations": obs}).json()
    probe = c.post("/api/ball-studio/groups/a/triangulate",
                   json={"frame": 30, "observations": obs, "offsets": {"b": OFF_B + 2}}).json()
    assert probe["offsets_used"]["b"] == OFF_B + 2
    assert probe["observations_used"][1]["shot_frame"] == 22
    assert probe["ok"] and base["ok"]
    # nothing was persisted by the probe
    assert c.get("/api/ball-studio/groups").json()["groups"][0]["shots"][1]["frame_offset"] == OFF_B
    assert c.post("/api/ball-studio/groups/a/triangulate",
                  json={"frame": 30, "observations": obs, "offsets": {"b": "x"}}).status_code == 422


def test_triangulate_single_view_epipolar_and_constraint(env):
    c, _ = env
    p = [40, 25, 0.11]
    one = {"frame": 30, "observations": [{"shot_id": "a", "shot_frame": 30, "uv": uv("a", p)}]}
    r = c.post("/api/ball-studio/groups/a/triangulate", json=one).json()
    assert r["ok"] and r["xyz"] is None and r["source"] == "ray"
    assert r["epipolar"][0]["shot_id"] == "b" and r["epipolar"][0]["polyline_uv"]
    r = c.post("/api/ball-studio/groups/a/triangulate",
               json={**one, "constraint": {"mode": "ground"}}).json()
    assert r["ok"] and r["source"] == "ray_ground"
    assert r["xyz"] == pytest.approx(p, abs=1e-2)


def test_spa_page_route(env):
    c, _ = env
    r = c.get("/ball-studio?group=a")
    assert r.status_code in (200, 404)  # 404 only when the SPA build is absent


# --- frame cadence (25->30 pulldown) ----------------------------------------

def write_pulldown_video(path: Path, n: int, phase: int) -> list[int]:
    import cv2

    rng = np.random.default_rng(phase)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (96, 64))
    reps, prev = [], None
    for f in range(n):
        if prev is not None and f % 6 == phase:
            img = prev
            reps.append(f)
        else:
            img = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        vw.write(img)
        prev = img
    vw.release()
    return reps


@pytest.fixture
def cad_env(env):
    c, out = env
    # MJPG under an .mp4 name: cv2 sniffs the container; the name is all
    # the studio needs to find a shot's video
    reps_a = write_pulldown_video(out / "shots" / "a.mp4", 90, 5)
    reps_b = write_pulldown_video(out / "shots" / "b.mp4", 80, 2)
    return c, out, reps_a, reps_b


def tri_body(ref, p):
    return {"frame": ref, "observations": [
        {"shot_id": "a", "shot_frame": ref, "uv": uv("a", p)},
        {"shot_id": "b", "shot_frame": ref + OFF_B, "uv": uv("b", p)}]}


def test_scene_reports_repeat_frames(cad_env):
    c, _out, reps_a, reps_b = cad_env
    shots = {s["shot_id"]: s for s in c.get("/api/ball-studio/groups/a/scene").json()["shots"]}
    assert shots["a"]["repeat_frames"] == reps_a
    assert shots["b"]["repeat_frames"] == reps_b


def test_scene_without_video_has_no_repeat_frames(env):
    c, _ = env
    shots = c.get("/api/ball-studio/groups/a/scene").json()["shots"]
    assert all(s["repeat_frames"] == [] for s in shots)


def test_triangulate_flags_views_not_simultaneous(cad_env):
    from src.utils.frame_cadence import content_time_shift

    c, _out, reps_a, reps_b = cad_env
    sa, sb = content_time_shift(90, reps_a, FPS), content_time_shift(80, reps_b, FPS)
    # the reference frame where the two views' content instants differ most
    ref = max(range(20, 70), key=lambda r: abs(sa[r] - sb[r + OFF_B]))
    assert abs(sa[ref] - sb[ref + OFF_B]) > 0.3 / FPS
    r = c.post("/api/ball-studio/groups/a/triangulate", json=tri_body(ref, [40.0, 25.0, 0.11])).json()
    assert r["ok"]
    assert "views_not_simultaneous" in [f["code"] for f in r["flags"]]


def test_triangulate_marks_repeated_observation_frames(cad_env):
    c, _out, reps_a, reps_b = cad_env
    ref = next(r for r in range(20, 70) if (r + OFF_B) in reps_b and r not in reps_a)
    r = c.post("/api/ball-studio/groups/a/triangulate", json=tri_body(ref, [40.0, 25.0, 0.11])).json()
    used = {o["shot_id"]: o for o in r["observations_used"]}
    assert used["b"]["repeat"] is True and used["a"]["repeat"] is False


def test_triangulate_fresh_in_both_views_has_no_cadence_flag(cad_env):
    c, _out, reps_a, reps_b = cad_env
    ref = next(r for r in range(20, 70)
               if r not in reps_a and (r + OFF_B) not in reps_b
               and (r - 1) not in reps_a and (r - 1 + OFF_B) not in reps_b)
    r = c.post("/api/ball-studio/groups/a/triangulate", json=tri_body(ref, [40.0, 25.0, 0.11])).json()
    assert "views_not_simultaneous" not in [f["code"] for f in r["flags"]]

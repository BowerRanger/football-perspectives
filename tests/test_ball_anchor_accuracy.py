"""Acceptance gate: ball within 10cm at anchors on real clips.

Re-runs the ball stage with a no-op detector (anchored frames reconstruct
from anchors + camera + refined_poses, independent of WASB) and measures the
perpendicular distance from each emitted world_xyz to the camera ray through
the user's clicked anchor pixel. Skips when the output dirs are not present.

Also runnable directly for a per-state table:
    python tests/test_ball_anchor_accuracy.py
"""
from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.schemas.ball_anchor import BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.stages.ball import BallStage
from src.utils.ball_detector import BallDetector
from src.utils.camera_projection import project_world_to_image, undistort_pixel

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [
    ("gberch", "output", "gberch"),
    ("kroupi01", "output-kroupi", "kroupi01"),
    ("origi01", "output-origi", "origi01"),
]
# States the goal must guarantee within 10cm laterally (contact + ground).
CONTACT_GROUND = {
    "grounded", "kick", "bounce", "catch", "goal_impact", "player_touch",
}


class _NoopDetector(BallDetector):
    # A no-op detector cannot re-detect crops (no heatmap to zoom on).
    SUPPORTS_REDETECT = False

    def detect(self, frame):  # noqa: ANN001
        return None


def _ray(uv, K, R, t, dist):
    uv = np.asarray(uv, float)
    if dist != (0.0, 0.0):
        uv = undistort_pixel(uv, K, dist)
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    return C, d / np.linalg.norm(d)


def _lateral(P, C, d):
    v = P - C
    return float(np.linalg.norm(P - (C + float(np.dot(v, d)) * d)))


def _rerun_and_collect(clip_id, out_dir, shot_id):
    out = ROOT / out_dir
    cam_p = out / "camera" / f"{shot_id}_camera_track.json"
    clip = out / "shots" / f"{shot_id}.mp4"
    anc_p = out / "ball" / f"{shot_id}_ball_anchors.json"
    if not (cam_p.exists() and clip.exists() and anc_p.exists()):
        pytest.skip(f"{clip_id}: output dir not present")
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    tmp = Path(tempfile.mkdtemp(prefix="ball_acc_"))
    stage = BallStage(config=cfg, output_dir=out, ball_detector=_NoopDetector())
    stage.shot_filter = shot_id
    track_out = tmp / f"{shot_id}_ball_track.json"
    stage._run_shot(shot_id, clip, cam_p, track_out, cfg["ball"], _NoopDetector())

    anc = BallAnchorSet.load(anc_p)
    track = BallTrack.load(track_out)
    cam = CameraTrack.load(cam_p)
    dist = cam.distortion
    K = {f.frame: np.array(f.K) for f in cam.frames}
    R = {f.frame: np.array(f.R) for f in cam.frames}
    tfb = np.array(cam.t_world)
    T = {f.frame: (np.array(f.t) if f.t is not None else tfb) for f in cam.frames}
    f2w = {f.frame: f.world_xyz for f in track.frames}
    rows = []  # (state, lateral_m, reproj_px, frame)
    for a in anc.anchors:
        if a.image_xy is None or a.frame not in K:
            continue
        w = f2w.get(a.frame)
        if w is None:
            continue
        P = np.array(w, float)
        uvp = project_world_to_image(K[a.frame], R[a.frame], T[a.frame], dist,
                                     P.reshape(1, 3))[0]
        reproj = float(np.linalg.norm(uvp - np.array(a.image_xy)))
        C, d = _ray(a.image_xy, K[a.frame], R[a.frame], T[a.frame], dist)
        rows.append((a.state, _lateral(P, C, d), reproj, a.frame))
    return rows


@pytest.mark.integration
@pytest.mark.parametrize("clip_id,out_dir,shot_id", CLIPS)
def test_contact_ground_anchors_within_10cm(clip_id, out_dir, shot_id):
    rows = _rerun_and_collect(clip_id, out_dir, shot_id)
    bad = [(s, round(lat, 3), f) for (s, lat, _, f) in rows
           if s in CONTACT_GROUND and lat > 0.10]
    assert not bad, f"{clip_id}: contact/ground anchors >10cm lateral: {bad}"


def _print_table():
    for clip_id, out_dir, shot_id in CLIPS:
        try:
            rows = _rerun_and_collect(clip_id, out_dir, shot_id)
        except Exception as exc:  # pytest.skip raises outside a test
            print(f"\n### {clip_id}: {exc}")
            continue
        by = defaultdict(list)
        for s, lat, rp, _ in rows:
            by[s].append((lat, rp))
        print(f"\n### {clip_id}")
        allv = []
        for s in sorted(by):
            lats = np.array([v[0] for v in by[s]])
            rps = np.array([v[1] for v in by[s]])
            allv.extend(lats.tolist())
            print(f"  {s:<14} n={len(lats):<3} lat med={np.median(lats):.3f} "
                  f"max={lats.max():.3f} reproj max={rps.max():6.1f} "
                  f">10cm:{int((lats>0.10).sum())}/{len(lats)}")
        allv = np.array(allv)
        if len(allv):
            print(f"  OVERALL med={np.median(allv):.3f} max={allv.max():.3f} "
                  f">10cm:{int((allv>0.10).sum())}/{len(allv)}")


if __name__ == "__main__":
    _print_table()

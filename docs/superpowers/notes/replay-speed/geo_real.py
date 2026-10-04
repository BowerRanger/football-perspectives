"""Run src.utils.replay_speed on real output dirs: geo_real.py <out_dir> <live> <replay> [...pairs]."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/joebower/workplace/football-perspectives/.claude/worktrees/gberch-shorts")
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.utils import replay_speed as rs  # noqa: E402


def points(out: Path, shot: str):
    tr = CameraTrack.load(out / "camera" / f"{shot}_camera_track.json")
    by = {f.frame: f for f in tr.frames}

    def cam(fi):
        f = by.get(fi)
        if f is None:
            return None
        return f.K, f.R, (f.t if f.t is not None else tr.t_world), tuple(tr.distortion)

    return rs.feet_on_pitch(json.loads((out / "tracks" / f"{shot}_tracks.json").read_text()), cam)


out = Path(sys.argv[1])
args = sys.argv[2:]
for live_id, rep_id in zip(args[::2], args[1::2]):
    t0 = time.time()
    est = rs.estimate_speed(points(out, live_id), points(out, rep_id))
    dt = time.time() - t0
    if est is None:
        print(rep_id, "no estimate")
        continue
    print(f"{rep_id} vs {live_id}: rate {est.rate:.3f} offset {est.offset:.1f} cost {est.cost_m:.2f}m "
          f"margin {est.margin:.2f} conf {est.confidence:.2f} halves {est.rate_first:.3f}/{est.rate_second:.3f} "
          f"ramp {est.ramp} cover {est.coverage:.2f}  ({dt:.1f}s)")

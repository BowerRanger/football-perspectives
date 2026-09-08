"""Gated PhysPT takeover: physics-refined motion only on flagged spans.

Thin CLI over ``src.utils.physpt_hybrid`` (the same code path the
refined_poses stage's default ``physpt_takeover`` pass uses). Splices
an existing PhysPT post-pass into spans flagged as unrealistic in the
current animation; all other frames stay byte-identical.

Usage: experiment_physpt_hybrid.py --current output \\
    --physpt output/physpt_experiment/physpt_current \\
    --output output/physpt_experiment/physpt_hybrid
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from src.schemas.refined_pose import RefinedPose
from src.utils.physpt_hybrid import TakeoverConfig,build_hybrid,keypoint_confidence


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--current',type=Path,default=REPO/'output')
    ap.add_argument('--physpt',type=Path,default=REPO/'output/physpt_experiment/physpt_current')
    ap.add_argument('--output',type=Path,default=REPO/'output/physpt_experiment/physpt_hybrid')
    defaults=TakeoverConfig()
    ap.add_argument('--acc-hi',type=float,default=defaults.acc_hi,help='root acceleration spike, m/s^2')
    ap.add_argument('--step-hi',type=float,default=defaults.step_hi,help='root rotation step spike, deg/frame')
    ap.add_argument('--conf-lo',type=float,default=defaults.conf_lo,help='mean keypoint confidence occlusion cutoff')
    ap.add_argument('--dilate',type=int,default=defaults.dilate)
    ap.add_argument('--merge-gap',type=int,default=defaults.merge_gap)
    ap.add_argument('--min-len',type=int,default=defaults.min_len)
    ap.add_argument('--ease',type=int,default=defaults.ease,help='rotation blend ramp, frames')
    ap.add_argument('--smooth-window',type=int,default=defaults.smooth_window,help='savgol window over the spliced translation (odd; <=2 disables)')
    ap.add_argument('--verify-slack',type=float,default=defaults.verify_slack,help='per-span acceptance: takeover peak must stay within this factor of the current animation''s span peak')
    args=ap.parse_args()
    cfg=TakeoverConfig.from_mapping({k:getattr(args,k) for k in TakeoverConfig.__dataclass_fields__})
    dest=args.output.resolve();(dest/'refined_poses').mkdir(parents=True,exist_ok=True)
    for name in ('camera','shots','tracks','ball'):
        if (args.current/name).exists() and not (dest/name).exists():
            (dest/name).symlink_to((args.current/name).resolve(),target_is_directory=True)
    meta={'detector':{k:getattr(cfg,k) for k in TakeoverConfig.__dataclass_fields__},'players':{}}
    total=flagged=0
    for p in sorted((args.current/'refined_poses').glob('*_refined.npz')):
        other=args.physpt/'refined_poses'/p.name
        if not other.exists():continue
        cur,phys=RefinedPose.load(p),RefinedPose.load(other)
        assert np.array_equal(cur.frames,phys.frames),p.name
        sid=cur.contributing_shots[0]
        conf=keypoint_confidence(args.current/'hmr_world'/f'{sid}__{cur.player_id}_kp2d.json',cur.frames)
        hybrid,spans=build_hybrid(cur,phys,conf,30.,cfg)
        hybrid.save(dest/'refined_poses'/p.name)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        # compare_animation --derived-after provenance: recorded input is
        # the current animation this hybrid was spliced from.
        (dest/'refined_poses'/f'{cur.player_id}_physpt.json').write_text(json.dumps(
            {'input_sha256':digest,'mode':'hybrid_gated_takeover','spans':spans},indent=2))
        meta['players'][cur.player_id]={'input_sha256':digest,'spans':spans,
            'flagged_frames':sum(s['frames'] for s in spans),'track_frames':len(cur.frames)}
        total+=len(cur.frames);flagged+=sum(s['frames'] for s in spans)
        print(f"{cur.player_id}: {len(spans)} span(s), {sum(s['frames'] for s in spans)}/{len(cur.frames)} frames",flush=True)
    meta['flagged_fraction']=flagged/total if total else 0.
    (dest/'experiment.json').write_text(json.dumps(meta,indent=2))
    print(f'flagged {flagged}/{total} frames ({100*meta["flagged_fraction"]:.1f}%)')


if __name__=='__main__':main()

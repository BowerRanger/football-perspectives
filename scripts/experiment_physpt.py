"""Apply released PhysPT weights to saved world motion without camera re-solving.

Thin CLI over ``src.utils.physpt_refiner`` (the same code path the
refined_poses stage's gated takeover uses). Does not overwrite current
pipeline outputs. The author's GlobalTrajPredictor is bypassed because
calibrated world motion is already available; the pretrained PhysPT
network itself is unmodified. Runs in ``.venv311``.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from src.schemas.refined_pose import RefinedPose
from src.utils.physpt_refiner import DEFAULT_AUTHOR_DIR, PhysPTRefiner


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=REPO/'output/animation_comparison/before')
    ap.add_argument('--output',type=Path,default=REPO/'output/physpt_experiment/physpt')
    ap.add_argument('--players',default='P001,P005,P019,P020')
    ap.add_argument('--device',choices=['cpu','mps','auto'],default='auto')
    ap.add_argument('--batch-size',type=int,default=8)
    args=ap.parse_args();source=args.input.resolve();dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
    for name in ('camera','shots','tracks','ball'):
        if (source/name).exists() and not (dest/name).exists():(dest/name).symlink_to(source/name,target_is_directory=True)
    refiner=PhysPTRefiner(device=args.device,batch_size=args.batch_size)
    checkpoint=DEFAULT_AUTHOR_DIR/'assets/checkpoint/PhysPT.pt'
    metadata={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'author_commit':'40d869927e53a3cade8ba657cacdaeb993f085b8','device':str(refiner.device),
              'input_fps':30,'model_fps':20,'global_trajectory_predictor':False,
              'alignment':'One XY start anchor per contiguous run; integrate predicted XY increments; predicted absolute Z.',
              'players':{}}
    for pid in args.players.split(','):
        path=source/'refined_poses'/f'{pid}_refined.npz';track=RefinedPose.load(path)
        after,runs=refiner.refine(track)
        after.save(dest/'refined_poses'/path.name)
        meta={'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runs':runs}
        metadata['players'][pid]=meta
        (dest/'refined_poses'/f'{pid}_physpt.json').write_text(json.dumps(meta,indent=2))
        (dest/'experiment.json').write_text(json.dumps(metadata,indent=2))
        print(pid,'saved',flush=True)


if __name__=='__main__':main()

"""Repeatable refinement ablations using one fixed set of HMR/camera inputs."""
from __future__ import annotations
import argparse,copy,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.stages.refined_poses import RefinedPosesStage
from scripts.compare_animation import compare


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('output/animation_comparison'))
    ap.add_argument('--config',type=Path,default=Path('config/default.yaml'))
    ap.add_argument('--variants',default='after,residual_on,rotation_only')
    args=ap.parse_args()
    import torch
    torch.set_num_threads(2)
    root=args.root.resolve(); cfg=yaml.safe_load(args.config.read_text()); timings={}
    for variant in args.variants.split(','):
        if variant not in ('after','residual_on','no_residual','rotation_only'):
            raise ValueError('unknown variant '+variant)
        out=root/variant;out.mkdir(exist_ok=True)
        if variant!='after':
            for name in ('hmr_world','camera','shots','tracks','ball'):
                source=root/'after'/name
                if source.exists() and not (out/name).exists():
                    (out/name).symlink_to(source,target_is_directory=True)
        config=copy.deepcopy(cfg)
        if variant=='no_residual':
            config['refined_poses']['jitter']['residual_pass_enabled']=False
        if variant=='residual_on':
            config['refined_poses']['jitter']['residual_pass_enabled']=True
        if variant=='rotation_only':
            config['refined_poses']['kinematic_refinement']['enabled']=False
        (out/'refinement_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        start=time.monotonic()
        print('Refining',variant,flush=True)
        RefinedPosesStage(config,out).run()
        timings[variant]=time.monotonic()-start
        compare(root/'before',out,root/'metrics'/variant)
    (root/'refinement_timings.json').write_text(json.dumps(timings,indent=2))


if __name__=='__main__':main()

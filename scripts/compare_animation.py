"""Compare two saved animations on identical frames, observations and contacts.

Usage: python scripts/compare_animation.py --before ... --after ... --output ...
The BEFORE keypoints/contact labels are the common evaluation reference. They
are not ground truth; neither solver gets to improve its score by rejecting
more difficult contacts or changing the observations being scored.
"""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.schemas.refined_pose import RefinedPose
from src.schemas.camera_track import CameraTrack
from src.schemas.foot_contacts import load_foot_contacts
from src.utils.smpl_skeleton import load_smpl_neutral_model,beta_adjusted_rest_joints,compute_all_joint_worlds_batch
from src.utils.animation_quality import motion_metrics


def contact_mask(path,source_frames,target_frames):
    mask=np.zeros((len(target_frames),2),bool)
    if not path.exists(): return mask
    fc,_=load_foot_contacts(path)
    for span in fc.spans:
        # Exact source frame membership, not a numeric interval across gaps.
        mask[np.isin(target_frames,source_frames[span.start:span.end]),span.side]=True
    return mask


def aggregate(players):
    """Weighted means over scored samples; maxima over the full comparison."""
    out={}
    for side in ('before','after'):
        rows=[p[side] for p in players.values()]
        def stats(values):
            values=[v for v in values if v['count']]
            n=sum(v['count'] for v in values)
            return {'count':n,'mean':sum(v['mean']*v['count'] for v in values)/n if n else None,
                    'max':max(v['max'] for v in values) if n else None}
        data={k:sum(r[k] for r in rows) for k in ('samples','root_steps_over_90_deg',
              'anatomical_violating_joint_frames','ground_penetrating_frames')}
        for k in ('root_step_deg','joint_step_deg','root_acc_m_s2','root_xy_acc_m_s2',
                  'root_z_acc_m_s2','foot_speed_m_s','body_reprojection_px'):
            data[k]=stats([r[k] for r in rows])
        for k in ('candidate_contact','verified_contact'):
            data[k]={'frame_coverage':sum(r[k]['frame_coverage']*r['samples'] for r in rows)/data['samples'] if data['samples'] else None}
            for s in ('stance_speed_m_s','transition_speed_m_s','transition_joint_step_deg'):
                data[k][s]=stats([r[k][s] for r in rows])
        out[side]=data
    return out


def compare(before,after,output,derived=False,derived_input=None):
    output.mkdir(parents=True,exist_ok=True)
    model=load_smpl_neutral_model(); results={}; animation={}
    for p in sorted((before/'refined_poses').glob('*_refined.npz')):
        other=after/'refined_poses'/p.name
        if not other.exists(): continue
        old,new=RefinedPose.load(p),RefinedPose.load(other)
        sid=old.contributing_shots[0];pid=old.player_id
        if derived:
            # The after track was derived from a recorded input (e.g.
            # PhysPT); verify that digest instead of requiring a fresh
            # extraction. The input is the before track itself unless
            # --derived-input points at a different source directory.
            sidecar=after/'refined_poses'/f'{pid}_physpt.json'
            if not sidecar.exists(): continue
            source=(derived_input/'refined_poses'/p.name) if derived_input else p
            if not source.exists(): continue
            if json.loads(sidecar.read_text()).get('input_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest(): continue
        else:
            # Only score a freshly re-extracted after track.
            if not (after/'hmr_world'/f'{sid}__{pid}_raw.npz').exists(): continue
            provenance_path = after/'refined_poses'/f'{pid}_provenance.json'
            if not provenance_path.exists(): continue
            provenance = json.loads(provenance_path.read_text())
            source_path = after/'hmr_world'/f'{sid}__{pid}_smpl_world.npz'
            if provenance.get('sources',{}).get(source_path.name) != hashlib.sha256(source_path.read_bytes()).hexdigest(): continue
        common,oi,ni=np.intersect1d(old.frames,new.frames,return_indices=True)
        if len(common)<3:continue
        camera=CameraTrack.load(before/'camera'/f'{sid}_camera_track.json')
        cam={f.frame:f for f in camera.frames}
        kp={f['frame']:f['keypoints'] for f in json.loads((before/'hmr_world'/f'{sid}__{pid}_kp2d.json').read_text())['frames']}
        obs={'K':[],'R':[],'t':[],'kp2d':[],'distortion':camera.distortion}
        for f in common:
            c=cam.get(int(f))
            obs['K'].append(c.K if c else np.eye(3));obs['R'].append(c.R if c else np.eye(3))
            obs['t'].append((c.t if c.t is not None else camera.t_world) if c else np.zeros(3))
            obs['kp2d'].append(kp.get(int(f),np.zeros((17,3))) if c else np.zeros((17,3)))
        obs={k:np.asarray(v) for k,v in obs.items()}
        with np.load(before/'hmr_world'/f'{sid}__{pid}_smpl_world.npz') as z: raw_frames=z['frames']
        candidate=contact_mask(before/'hmr_world'/f'{sid}__{pid}_foot_contacts.json',raw_frames,common)
        results[pid]={}; animation[pid]={'frames':common.tolist(),'fps':camera.fps}
        for label,tr,idx,directory in (('before',old,oi,before),('after',new,ni,after)):
            rest=beta_adjusted_rest_joints(tr.betas,model)
            verified=contact_mask(directory/'refined_poses'/f'{pid}_resolved_contacts.json',tr.frames,common)
            args=dict(frames=common,thetas=tr.thetas[idx],root_R=tr.root_R[idx],root_t=tr.root_t[idx],rest_joints=rest,fps=camera.fps)
            results[pid][label]=motion_metrics(**args,candidate_contacts=candidate,verified_contacts=verified,observations=obs)
            w=compute_all_joint_worlds_batch(args['thetas'],args['root_R'],args['root_t'],rest)
            animation[pid][label]=np.round(w,4).tolist()
    report={'reference':'Same original keypoints and candidate-contact labels; neither is ground truth.',
            'players':results,'aggregate':aggregate(results)}
    (output/'comparison.json').write_text(json.dumps(report,indent=2))
    # JS assignment works from file://, unlike fetch() of local JSON.
    (output/'motion.js').write_text('window.ANIMATION='+json.dumps(animation,separators=(',',':'))+';\nwindow.METRICS='+json.dumps(report)+';')
    print('player  root max step before/after  root max acceleration before/after  reprojection before/after')
    for pid,v in results.items():
        vals=[]
        for k in ('root_step_deg','root_acc_m_s2','body_reprojection_px'):
            stat='mean' if k=='body_reprojection_px' else 'max'
            vals.append('/'.join(f"{v[x][k][stat]:.2f}" for x in ('before','after')))
        print(pid,*vals)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('before','after','output'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--derived-after',action='store_true',
        help='The after directory holds a derived refinement (e.g. PhysPT); verify '
             'each track\'s recorded input digest instead of requiring a fresh '
             'extraction.')
    ap.add_argument('--derived-input',type=Path,default=None,
        help='Directory whose refined_poses/ files were the derived run\'s input, '
             'when that input was not the before animation itself.')
    args=ap.parse_args();compare(args.before,args.after,args.output,derived=args.derived_after,derived_input=args.derived_input)

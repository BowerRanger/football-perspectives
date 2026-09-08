"""Motion-snap evaluation: does an animation keep the sharp movements?

Scores an animation against the OBSERVED 2D keypoints (projected limb
speed vs detected keypoint speed at wrists/ankles — direct image
evidence of how fast limbs really moved) plus world-space sharpness
measures. Built from the 2026-09-08 floaty-animation diagnosis; the
"sharp-move speed ratio" is the headline number (1.0 = matches the
video, lower = smoothing stole real motion).

Usage: eval_motion_snap.py --animation <dir> [--reference output]
       [--shot gberch] [--json out.json]
The reference supplies camera + kp2d evidence; the animation dir needs
only refined_poses/.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from src.schemas.refined_pose import RefinedPose
from src.schemas.camera_track import CameraTrack
from src.utils.pose_temporal import frame_runs
from src.utils.smpl_skeleton import load_smpl_neutral_model,beta_adjusted_rest_joints,compute_all_joint_worlds_batch

PAIRS=[(9,20),(10,21),(15,7),(16,8)]  # COCO wrist/ankle -> SMPL joint


def evaluate(animation:Path,reference:Path,shot:str,fps:float=30.):
    model=load_smpl_neutral_model()
    cam=CameraTrack.load(reference/'camera'/f'{shot}_camera_track.json')
    cam_by={f.frame:f for f in cam.frames}
    obs=[];pred=[];hf=[];ang={'knees':[], 'elbows':[]};foot=[];racc=[];rstep=[]
    for p in sorted((animation/'refined_poses').glob('*_refined.npz')):
        tr=RefinedPose.load(p)
        kp_path=reference/'hmr_world'/f'{shot}__{tr.player_id}_kp2d.json'
        if not kp_path.exists():continue
        kp_by={f['frame']:np.asarray(f['keypoints']) for f in json.loads(kp_path.read_text())['frames']}
        rest=beta_adjusted_rest_joints(tr.betas,model)
        w=compute_all_joint_worlds_batch(tr.thetas,tr.root_R,tr.root_t,rest)
        px=np.full((len(tr.frames),len(PAIRS),2),np.nan)
        for i,f in enumerate(tr.frames):
            c=cam_by.get(int(f))
            if c is None:continue
            R,t,K=np.asarray(c.R),np.asarray(c.t),np.asarray(c.K)
            pc=w[i,[s for _,s in PAIRS]]@R.T+t
            xy=pc[:,:2]/np.maximum(pc[:,2:],.1)
            px[i]=xy@K[:2,:2].T+K[:2,2]
        for k,(cj,_) in enumerate(PAIRS):
            for i in range(len(tr.frames)-1):
                f0,f1=int(tr.frames[i]),int(tr.frames[i+1])
                if f1-f0!=1:continue
                k0,k1p=kp_by.get(f0),kp_by.get(f1)
                if k0 is None or k1p is None or k0[cj,2]<.5 or k1p[cj,2]<.5:continue
                obs.append(np.linalg.norm(k1p[cj,:2]-k0[cj,:2]))
                pred.append(np.linalg.norm(px[i+1,k]-px[i,k]))
        # All temporal derivatives respect tracking gaps: compute within
        # contiguous frame runs only, never across missing spans.
        for a,b in frame_runs(tr.frames):
            for j in (20,21,7,8,10,11):
                x=w[a:b,j]
                if len(x)>15:
                    hf.append(np.sqrt(((x-savgol_filter(x,11,2,axis=0))**2).sum(1)))
            for name,joints in (('knees',(4,5)),('elbows',(18,19))):
                for j in joints:
                    if b-a<2:continue
                    r=Rotation.from_rotvec(tr.thetas[a:b,j])
                    ang[name].append(np.degrees(np.linalg.norm((r[:-1].inv()*r[1:]).as_rotvec(),axis=1)))
            if b-a>1:foot.append((np.linalg.norm(np.diff(w[a:b,[10,11]],axis=0),axis=2)*fps).ravel())
            if b-a>2:racc.append(np.linalg.norm(np.diff(tr.root_t[a:b],2,axis=0),axis=1)*fps*fps)
            if b-a>1:
                rr=Rotation.from_matrix(tr.root_R[a:b])
                rstep.append(np.degrees(np.linalg.norm((rr[:-1].inv()*rr[1:]).as_rotvec(),axis=1)))
    obs=np.array(obs);pred=np.array(pred)
    fast=obs>np.percentile(obs,90)
    hf=np.concatenate(hf);foot=np.concatenate(foot);racc=np.concatenate(racc);rstep=np.concatenate(rstep)
    return {
        'samples':int(len(obs)),
        'speed_ratio_overall':float(np.median(pred/np.maximum(obs,1e-6))),
        'speed_ratio_sharp':float(np.median(pred[fast]/np.maximum(obs[fast],1e-6))),
        'pred_speed_p99_px':float(np.percentile(pred,99)),
        'obs_speed_p99_px':float(np.percentile(obs,99)),
        'hf_residual_p95_cm':float(np.percentile(hf,95)*100),
        'knee_step_p99_deg':float(np.percentile(np.concatenate(ang['knees']),99)),
        'elbow_step_p99_deg':float(np.percentile(np.concatenate(ang['elbows']),99)),
        'foot_speed_p99_m_s':float(np.percentile(foot,99)),
        'root_acc_p95_m_s2':float(np.percentile(racc,95)),
        'root_steps_over_90':int((rstep>90).sum()),
        'largest_root_step_deg':float(rstep.max()),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--animation',type=Path,required=True)
    ap.add_argument('--reference',type=Path,default=REPO/'output')
    ap.add_argument('--shot',default='gberch')
    ap.add_argument('--json',type=Path,default=None)
    args=ap.parse_args()
    result=evaluate(args.animation.resolve(),args.reference.resolve(),args.shot)
    for k,v in result.items():print(f'{k}: {v:.3f}' if isinstance(v,float) else f'{k}: {v}')
    if args.json:args.json.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()

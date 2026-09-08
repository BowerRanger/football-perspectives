"""Motion diagnostics independent of the optimizer's accepted contact labels."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils.smpl_skeleton import compute_all_joint_worlds_batch
from src.utils.kinematic_refinement import anatomical_violation_count, COCO_BODY, SMPL_BODY


def _stats(values):
    v=np.asarray(values).reshape(-1)
    return {'count':int(len(v)), 'mean':float(v.mean()) if len(v) else None,
            'p95':float(np.percentile(v,95)) if len(v) else None,
            'max':float(v.max()) if len(v) else None}


def motion_metrics(*, frames,thetas,root_R,root_t,rest_joints,fps,
                   candidate_contacts=None,verified_contacts=None,observations=None):
    frames=np.asarray(frames)
    valid=np.diff(frames)==1
    w=compute_all_joint_worlds_batch(thetas,root_R,root_t,rest_joints)
    root=Rotation.from_matrix(root_R)
    angles=np.rad2deg((root[:-1].inv()*root[1:]).magnitude())
    local=Rotation.from_rotvec(np.asarray(thetas)[:,1:22].reshape(-1,3)).as_matrix().reshape(-1,21,3,3)
    rel=np.swapaxes(local[:-1],-1,-2)@local[1:]
    joint_angles=np.rad2deg(Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude()).reshape(-1,21)
    speed=np.linalg.norm(np.diff(w[:,[10,11]],axis=0),axis=-1)*fps
    acc=(np.asarray(root_t)[2:]-2*np.asarray(root_t)[1:-1]+np.asarray(root_t)[:-2])*fps**2
    acceleration=acc[valid[:-1]&valid[1:]]
    result={'samples':len(frames),'root_step_deg':_stats(angles[valid]),
            'root_steps_over_90_deg':int(((angles>90)&valid).sum()),
            'largest_root_step_frame':int(frames[1:][np.argmax(np.where(valid,angles,-1))]) if valid.any() else None,
            'joint_step_deg':_stats(joint_angles[valid]),
            'largest_joint_step_frame':int(frames[1:][np.argmax(np.where(valid,joint_angles.max(axis=1),-1))]) if valid.any() else None,
            'root_acc_m_s2':_stats(np.linalg.norm(acceleration,axis=-1)),
            'root_xy_acc_m_s2':_stats(np.linalg.norm(acceleration[:,:2],axis=-1)),
            'root_z_acc_m_s2':_stats(np.abs(acceleration[:,2])),
            'foot_speed_m_s':_stats(speed[valid]),
            'anatomical_violating_joint_frames':anatomical_violation_count(thetas),
            'ground_penetrating_frames':int((w[:,[10,11],2].min(axis=1)<.024).sum())}
    for name,mask in (('candidate',candidate_contacts),('verified',verified_contacts)):
        if mask is None:
            continue
        mask=np.asarray(mask,dtype=bool)
        within=mask[:-1]&mask[1:]&valid[:,None]
        boundary=(mask[:-1]!=mask[1:])&valid[:,None]
        result[name+'_contact']={
            'frame_coverage':float(mask.any(axis=1).mean()),
            'stance_speed_m_s':_stats(speed[within]),
            'transition_speed_m_s':_stats(speed[boundary]),
            'transition_joint_step_deg':_stats(joint_angles[np.any(boundary,axis=1)]),
        }
    if observations is not None:
        pc=np.einsum('fij,fkj->fki',observations['R'],w[:,SMPL_BODY])+observations['t'][:,None]
        xy=pc[:,:,:2]/np.maximum(pc[:,:,2:],.1)
        k1,k2=observations.get('distortion',(0.,0.))
        radius=(xy**2).sum(-1,keepdims=True)
        xy*=1+k1*radius+k2*radius**2
        pix=np.einsum('fij,fkj->fki',observations['K'],np.concatenate((xy,np.ones_like(xy[:,:,:1])),axis=-1))[:,:,:2]
        kp=observations['kp2d'][:,COCO_BODY]
        good=(kp[:,:,2]>=.5)&(pc[:,:,2]>.1)
        result['body_reprojection_px']=_stats(np.linalg.norm(pix-kp[:,:,:2],axis=-1)[good])
    return result

"""Joint optimization of human pose and trajectory with a fixed calibrated camera.

This is kinematic constrained fitting, not a dynamics simulator. Contact is
evidence, not an assumption that every low foot is planted. Frames without
contact evidence remain unconstrained in height apart from nonpenetration.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.pose_temporal import frame_runs
from src.utils.smpl_skeleton import SMPL_PARENTS, compute_all_joint_worlds_batch
from src.utils.foot_contact import FootContacts, ContactSpan
from src.utils.foot_lock import penetration_guard

# Conservative joint-coordinate bounds in SMPL's canonical joint frames.
# These constrain flexion and off-hinge rotation, rather than clamping the
# magnitude of an arbitrary axis-angle vector. Shoulders/hips remain free.
# Knees/ankles use xyz; elbows use yxz so their primary Y flexion can pass
# 90 degrees without hitting the middle Euler axis's gimbal singularity.
# Temporal residuals use matrices/FK, never these chart coordinates.
JOINT_BOUNDS_DEG = {
    4: ((-5, -20, -20), (160, 20, 20)),
    5: ((-5, -20, -20), (160, 20, 20)),
    18: ((-30, -160, -30), (30, 5, 30)),
    19: ((-30, -5, -30), (30, 160, 30)),
    7: ((-65, -40, -40), (65, 40, 40)),
    8: ((-65, -40, -40), (65, 40, 40)),
}
COCO_BODY = np.arange(5, 17)
SMPL_BODY = np.array([16,17,18,19,20,21,1,2,4,5,7,8])


def _rotation_exp(v):
    import torch
    x,y,z = v.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(v.shape[:-1]+(3,3))
    angle = torch.linalg.vector_norm(v,dim=-1)[...,None,None]
    identity = torch.eye(3,dtype=v.dtype,device=v.device)
    return identity + torch.sinc(angle/torch.pi)*skew + .5*torch.sinc(angle/(2*torch.pi))**2*(skew@skew)


def _euler_matrix(e):
    import torch
    x,y,z = e.unbind(-1)
    cx,cy,cz = x.cos(),y.cos(),z.cos()
    sx,sy,sz = x.sin(),y.sin(),z.sin()
    return torch.stack((cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz,
                        cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-sx*cz,
                        -sy, sx*cy, cx*cy),-1).reshape(e.shape[:-1]+(3,3))


def _joint_angles(thetas, *, degrees=False):
    """Return x/y/z components in each joint's nonsingular flexion chart."""
    th = np.asarray(thetas)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        angles = Rotation.from_rotvec(th.reshape(-1,3)).as_euler('xyz',degrees=degrees).reshape(th.shape)
        for j in (18,19):
            angles[:,j] = Rotation.from_rotvec(th[:,j]).as_euler('yxz',degrees=degrees)[:,[1,0,2]]
    return angles


def _local_matrices(e):
    import torch
    result = _euler_matrix(e)
    # Rz @ Rx @ Ry for elbows; e retains anatomical x/y/z component order.
    x,y,z = e[:,[18,19]].unbind(-1)
    cx,cy,cz = x.cos(),y.cos(),z.cos()
    sx,sy,sz = x.sin(),y.sin(),z.sin()
    elbow = torch.stack((cy*cz-sx*sy*sz, -cx*sz, sy*cz+sx*cy*sz,
                         cy*sz+sx*sy*cz, cx*cz, sy*sz-sx*cy*cz,
                         -cx*sy, sx, cx*cy),-1).reshape(-1,2,3,3)
    result[:,[18,19]] = elbow
    return result


def _fk(local, root_R, root_t, rest):
    positions, rotations = [root_t], [root_R]
    for j in range(1,24):
        parent = SMPL_PARENTS[j]
        positions.append(positions[parent] + (rotations[parent] @ (rest[j]-rest[parent])[...,None]).squeeze(-1))
        rotations.append(rotations[parent] @ local[:,j])
    import torch
    return torch.stack(positions,1)


def anatomical_violation_count(thetas):
    count = 0
    angles = _joint_angles(thetas, degrees=True)
    for j,(lo,hi) in JOINT_BOUNDS_DEG.items():
        e = angles[:,j]
        count += int(np.any((e < np.array(lo)-1e-3) | (e > np.array(hi)+1e-3),axis=1).sum())
    return count


def refine_motion(*, frames, thetas, root_R, root_t, betas, rest_joints,
                  contacts: FootContacts, fps, observations=None, cfg=None):
    """Fit each contiguous observed run jointly, including contact transitions.

    observations optionally contains frame-aligned K/R/t, COCO kp2d and radial
    distortion. Camera arrays are constants throughout. Contact targets share
    one height with the final clearance check. Accepted contacts are verified
    on FINAL FK, after clearance, and are only reporting labels: failed spans
    are still included in candidate-contact metrics.
    """
    import torch
    cfg = cfg or {}
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames = np.asarray(frames)
    theta_out = np.asarray(thetas,dtype=float).copy()
    rotation_out = np.asarray(root_R,dtype=float).copy()
    translation_out = np.asarray(root_t,dtype=float).copy()
    n = len(frames)
    clearance = float(cfg.get("sole_clearance_m",.025))
    fw0 = compute_all_joint_worlds_batch(thetas,root_R,root_t,rest_joints)
    pins = np.zeros((n,2,3))
    mask = np.zeros((n,2),dtype=bool)
    # Source-anchored, per-span targets, held fixed during optimization.
    span_pins = []
    for span in contacts.spans:
        xy = np.median(fw0[span.start:span.end,10+span.side,:2],axis=0)
        pin = np.r_[xy,clearance]
        pins[span.start:span.end,span.side] = pin
        mask[span.start:span.end,span.side] = True
        span_pins.append(pin)
    dtype = torch.float64
    tensor = lambda x: torch.as_tensor(np.asarray(x),dtype=dtype)
    histories = []
    for a,b in frame_runs(frames):
        if b-a < 3:
            continue
        th0 = np.asarray(thetas[a:b],dtype=float)
        canonical_angles = _joint_angles(th0)
        euler = canonical_angles.copy()
        # Unwrap only the optimizer's chart. Rotation residuals below remain
        # invariant to this choice of chart.
        euler = np.unwrap(euler,axis=0)
        bounded = np.zeros((24,3),dtype=bool)
        middle = np.zeros((24,3)); radius = np.ones((24,3))
        for j,(lo,hi) in JOINT_BOUNDS_DEG.items():
            lo,hi = np.deg2rad(lo),np.deg2rad(hi)
            middle[j],radius[j] = (lo+hi)/2,(hi-lo)/2
            bounded[j] = True
            canonical = canonical_angles[:,j]
            euler[:,j] = np.arctanh(np.clip((canonical-middle[j])/radius[j],-.995,.995))
        pose_var = torch.tensor(euler,dtype=dtype,requires_grad=True)
        root_delta = torch.zeros((b-a,3),dtype=dtype,requires_grad=True)
        translation_delta = torch.zeros((b-a,3),dtype=dtype,requires_grad=True)
        r0,t0 = tensor(root_R[a:b]),tensor(root_t[a:b])
        l0 = tensor(Rotation.from_rotvec(th0.reshape(-1,3)).as_matrix().reshape(-1,24,3,3))
        mid,rad,bnd = tensor(middle),tensor(radius),torch.as_tensor(bounded)
        rest = tensor(rest_joints)
        target,contact = tensor(pins[a:b]),torch.as_tensor(mask[a:b])
        obs = None
        if observations is not None:
            obs = {k:tensor(v[a:b]) for k,v in observations.items() if k in ('K','R','t','kp2d')}
            k1,k2 = observations.get('distortion',(0.,0.))
            kp = obs['kp2d'][:,COCO_BODY]
            conf = kp[:,:,2].clamp(0,1)**2
            # Never fit absent/occluded observations as zero-valued pixels.
            conf = conf * (kp[:,:,2] >= .3)
        optimizer = torch.optim.LBFGS([pose_var,root_delta,translation_delta],
            max_iter=int(cfg.get('iterations',160)), history_size=12,
            line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-9)
        losses=[]

        def state():
            angles = torch.where(bnd,mid+rad*torch.tanh(pose_var),pose_var)
            local = _local_matrices(angles)
            r = r0 @ _rotation_exp(root_delta)
            t = t0 + translation_delta
            return local,r,t,_fk(local,r,t,rest)

        def closure():
            optimizer.zero_grad()
            local,r,t,w = state()
            # Pose/data priors: permit evidence-supported changes without
            # fitting noisy 2D detections at any cost.
            loss = ((local-l0)**2).mean()/.25**2
            loss = loss + .25*((r-r0)**2).mean()/.2**2
            loss = loss + .3*(translation_delta/.2).square().mean()
            joint_acc = (w[2:]-2*w[1:-1]+w[:-2])*fps**2
            loss = loss + float(cfg.get('joint_acc_weight',2.0))*(joint_acc/25).square().mean()
            root_acc = (t[2:]-2*t[1:-1]+t[:-2])*fps**2
            loss = loss + float(cfg.get('root_acc_weight',2.0))*(root_acc/15).square().mean()
            # Changes of relative rotation approximate angular acceleration
            # in a local tangent frame and are safe across +/-pi charts.
            relative = local[:-1].transpose(-1,-2) @ local[1:]
            root_relative = r[:-1].transpose(-1,-2) @ r[1:]
            loss = loss + float(cfg.get('joint_angular_acc_weight',8.0))*((relative[1:]-relative[:-1])*fps**2/60).square().mean()
            loss = loss + .5*((root_relative[1:]-root_relative[:-1])*fps**2/40).square().mean()
            feet = w[:,[10,11]]
            loss = loss + 8*(torch.relu(clearance-feet[:,:,2])/.02).square().mean()
            if contact.any():
                loss = loss + float(cfg.get('contact_weight',4.0))*((feet[contact]-target[contact])/.025).square().mean()
            if obs is not None:
                pc = torch.einsum('fij,fkj->fki',obs['R'],w[:,SMPL_BODY])+obs['t'][:,None]
                xy = pc[:,:,:2]/pc[:,:,2:].clamp_min(.1)
                radial = 1+k1*(xy**2).sum(-1,keepdim=True)+k2*(xy**2).sum(-1,keepdim=True)**2
                homo = torch.cat((xy*radial,torch.ones_like(xy[:,:,:1])),dim=-1)
                pixels = torch.einsum('fij,fkj->fki',obs['K'],homo)[:,:,:2]
                residual = (pixels-kp[:,:,:2])/float(cfg.get('reprojection_sigma_px',8.0))
                robust = 2*(torch.sqrt(1+residual.square().sum(-1))-1)
                loss = loss + float(cfg.get('reprojection_weight',2.0))*(robust*conf).sum()/conf.sum().clamp_min(1)
                loss = loss + (torch.relu(.1-pc[:,:,2])*conf).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite kinematic objective')
            loss.backward()
            losses.append(float(loss.detach()))
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            local,r,t,_ = state()
            theta_out[a:b] = Rotation.from_matrix(local.numpy().reshape(-1,3,3)).as_rotvec().reshape(-1,24,3)
            theta_out[a:b,0] = 0  # world orientation is stored separately
            rotation_out[a:b],translation_out[a:b] = r.numpy(),t.numpy()
        histories.append({'start_frame':int(frames[a]),'end_frame':int(frames[b-1]),
                          'initial_loss':losses[0],'final_loss':losses[-1],'evaluations':len(losses)})

    # Minimal raise-only clearance; verify contacts AFTER this operation.
    # Run independently across occlusions so a guard never propagates a
    # correction across missing seconds.
    guard_stats={'frames_raised':0,'max_raise_cm':0.0}
    for a,b in frame_runs(frames):
        translation_out[a:b], gs = penetration_guard(thetas=theta_out[a:b],root_R=rotation_out[a:b],
            root_t=translation_out[a:b],betas=betas,rest_joints=rest_joints,sole_clearance_m=clearance)
        guard_stats['frames_raised'] += gs['frames_raised']
        guard_stats['max_raise_cm'] = max(guard_stats['max_raise_cm'],gs['max_raise_cm'])
    final = compute_all_joint_worlds_batch(theta_out,rotation_out,translation_out,rest_joints)
    resolved=[]; before_err=[]; after_err=[]
    for span,pin in zip(contacts.spans,span_pins):
        idx=10+span.side
        pre=np.linalg.norm(fw0[span.start:span.end,idx]-pin,axis=1)
        post=np.linalg.norm(final[span.start:span.end,idx]-pin,axis=1)
        steps=np.linalg.norm(np.diff(final[span.start:span.end,idx],axis=0),axis=1)
        before_err.extend(pre.tolist());after_err.extend(post.tolist())
        if (post.max(initial=0) <= float(cfg.get('resolved_pin_err_m',.06))
                and steps.max(initial=0) <= float(cfg.get('resolved_max_step_m',.02))):
            resolved.append(ContactSpan(span.side,span.start,span.end,pin))
    stats={
        'spans_locked':len(resolved),'spans_skipped':0,'spans_unresolved':len(contacts.spans)-len(resolved),
        'spans_unresolved_pass1':0,'spans_flipped_pass2':0,
        'mean_pin_err_m_before':float(np.mean(before_err)) if before_err else 0.,
        'mean_pin_err_m_after':float(np.mean(after_err)) if after_err else 0.,
        'max_root_corr_m':float(np.linalg.norm(translation_out-root_t,axis=1).max(initial=0)),
        **guard_stats, 'optimizer_runs':histories,
        'anatomical_violations_before':anatomical_violation_count(thetas),
        'anatomical_violations_after':anatomical_violation_count(theta_out),
        'resolved_spans':tuple(resolved),
    }
    return theta_out,rotation_out,translation_out,stats

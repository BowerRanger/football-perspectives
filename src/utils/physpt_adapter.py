"""Time/rotation adaptation for an isolated PhysPT experiment.

PhysPT uses interleaved matrix columns for 6D rotations (unlike PyTorch3D)
and a 20 Hz time step. Pitch coordinates already have its required Z-up.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def to_physpt_6d(matrices):
    return np.asarray(matrices)[..., :2].reshape(np.asarray(matrices).shape[:-2]+(6,))


def resample_motion(times, local, root_R, root_t, target_times):
    query=np.clip(target_times,times[0],times[-1])
    pose=np.asarray(local).copy()
    pose[:,0]=Rotation.from_matrix(root_R).as_rotvec()
    matrices=np.stack([Slerp(times,Rotation.from_rotvec(pose[:,j]))(query).as_matrix()
                       for j in range(24)],axis=1)
    translation=np.stack([np.interp(query,times,root_t[:,j]) for j in range(3)],axis=-1)
    return matrices,translation


def stitch_windows(rotations,translations,starts,n,initial_xy):
    """Select central predictions and integrate predicted XY increments.

    Absolute per-window XY origins are arbitrary. Stitching those positions
    produces seams; the author's demo integrates central XY increments.
    Use every frame including the final full window, with one starting XY
    anchor per contiguous run. Z remains the model's predicted height.
    """
    rot=np.zeros((n,24,3,3));delta=np.zeros((n,2));z=np.zeros(n)
    support=np.full(n,-1.);used=np.full(n,-1,dtype=int)
    for window,(rr,tt,start) in enumerate(zip(rotations,translations,starts)):
        m=len(tt)
        for j in range(m):
            i=int(start)+j
            score=min(j,m-1-j)
            if score<=support[i]:continue
            if i>0 and j==0:continue  # no within-window velocity evidence
            rot[i]=rr[j];z[i]=tt[j,2]
            if j:delta[i]=tt[j,:2]-tt[j-1,:2]
            support[i]=score;used[i]=window
    if np.any(used<0):raise ValueError('Uncovered PhysPT output frames')
    xy=np.asarray(initial_xy)+np.cumsum(delta,axis=0)
    return rot,np.c_[xy,z],used

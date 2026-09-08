import numpy as np
import torch
from scipy.spatial.transform import Rotation
from src.utils.kinematic_refinement import _rotation_exp, _euler_matrix, _fk, refine_motion
from src.utils.smpl_skeleton import SMPL_REST_JOINTS_YUP, compute_all_joint_worlds_batch
from src.utils.foot_contact import FootContacts, ContactSpan


def test_differentiable_fk_matches_exported_skeleton_and_has_finite_zero_gradient():
    rng=np.random.default_rng(21)
    e=rng.normal(0,.3,(4,24,3))
    th=Rotation.from_euler('xyz',e.reshape(-1,3)).as_rotvec().reshape(4,24,3)
    rr=Rotation.from_euler('x',[90]*4,degrees=True).as_matrix()
    rt=rng.normal(size=(4,3))
    local=_euler_matrix(torch.tensor(e))
    out=_fk(local,torch.tensor(rr),torch.tensor(rt),torch.tensor(SMPL_REST_JOINTS_YUP))
    np.testing.assert_allclose(out.numpy(),compute_all_joint_worlds_batch(th,rr,rt),atol=1e-12)
    delta=torch.zeros((4,3),dtype=torch.float64,requires_grad=True)
    (_rotation_exp(delta)*torch.arange(9).reshape(3,3)).sum().backward()
    assert torch.isfinite(delta.grad).all()


def test_joint_solve_reduces_jitter_enforces_knee_limits_and_preserves_lengths():
    n=24
    th=np.zeros((n,24,3)); th[:,4,0]=-.25  # hyperextended knee
    rr=Rotation.from_euler('x',[90]*n,degrees=True).as_matrix()
    rt=np.tile([0.,0.,.965],(n,1)); rt[10,0]=.15
    contacts=FootContacts(n,np.ones((n,2),bool),np.ones((n,2)),
        (ContactSpan(0,0,n,np.zeros(3)),ContactSpan(1,0,n,np.zeros(3))))
    out,rot,pos,stats=refine_motion(frames=np.arange(n),thetas=th,root_R=rr,root_t=rt,
        betas=np.zeros(10),rest_joints=SMPL_REST_JOINTS_YUP,contacts=contacts,fps=30,
        cfg={'iterations':35})
    assert stats['anatomical_violations_after']==0
    assert np.linalg.norm(np.diff(pos,n=2,axis=0),axis=1).max() < .1
    fw=compute_all_joint_worlds_batch(out,rot,pos)
    assert fw[:,[10,11],2].min() >= .025-1e-8
    np.testing.assert_allclose(np.linalg.norm(fw[:,7]-fw[:,4],axis=1),
        np.linalg.norm(SMPL_REST_JOINTS_YUP[7]-SMPL_REST_JOINTS_YUP[4]),atol=1e-9)
    assert stats['optimizer_runs'][0]['final_loss'] < stats['optimizer_runs'][0]['initial_loss']


def test_no_contact_does_not_pull_airborne_player_to_ground():
    n=8; th=np.zeros((n,24,3))
    rr=Rotation.from_euler('x',[90]*n,degrees=True).as_matrix()
    rt=np.tile([1.,2.,1.4],(n,1))
    c=FootContacts(n,np.zeros((n,2),bool),np.zeros((n,2)),())
    _,_,pos,_=refine_motion(frames=np.arange(n),thetas=th,root_R=rr,root_t=rt,
        betas=np.zeros(10),rest_joints=SMPL_REST_JOINTS_YUP,contacts=c,fps=30,cfg={'iterations':10})
    np.testing.assert_allclose(pos,rt,atol=.002)


def test_elbow_flexion_crosses_ninety_degrees_without_twist_or_snap():
    from src.utils.kinematic_refinement import _joint_angles, _local_matrices, anatomical_violation_count
    n=30
    th=np.zeros((n,24,3))
    th[:,19,1]=np.deg2rad(np.linspace(60,140,n))
    th[:,18,1]=-th[:,19,1]
    angles=_joint_angles(th)
    np.testing.assert_allclose(_local_matrices(torch.tensor(angles)).numpy(),
        Rotation.from_rotvec(th.reshape(-1,3)).as_matrix().reshape(n,24,3,3),atol=1e-12)
    assert anatomical_violation_count(th)==0
    rr=Rotation.from_euler('x',[90]*n,degrees=True).as_matrix()
    rt=np.tile([0.,0.,1.4],(n,1))
    contacts=FootContacts(n,np.zeros((n,2),bool),np.zeros((n,2)),())
    out,_,_,stats=refine_motion(frames=np.arange(n),thetas=th,root_R=rr,root_t=rt,
        betas=np.zeros(10),rest_joints=SMPL_REST_JOINTS_YUP,contacts=contacts,fps=30,
        cfg={'iterations':35})
    r=Rotation.from_rotvec(out[:,19])
    assert np.rad2deg((r[:-1].inv()*r[1:]).magnitude()).max()<5
    assert np.rad2deg(out[-1,19,1])>125
    assert stats['anatomical_violations_after']==0

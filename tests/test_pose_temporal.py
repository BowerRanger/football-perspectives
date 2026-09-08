import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.pose_temporal import frame_runs, smooth_pose, smooth_rotations, interpolate_pose


def test_rotation_6d_shim_roundtrip_matches_smpl_convention():
    import torch
    path = Path(__file__).resolve().parents[1] / "third_party/gvhmr_shims/pytorch3d/transforms.py"
    spec = importlib.util.spec_from_file_location("rotation_shim_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    matrices = Rotation.random(50, random_state=4).as_matrix()
    r = torch.tensor(matrices)
    actual = mod.rotation_6d_to_matrix(mod.matrix_to_rotation_6d(r))
    np.testing.assert_allclose(actual.numpy(), matrices, atol=1e-12)
    np.testing.assert_allclose(mod.axis_angle_to_matrix(mod.matrix_to_axis_angle(r)).numpy(), matrices, atol=1e-6)


def test_smoothing_removes_flip_and_preserves_turn():
    angles = np.arange(50)*3.0
    r = Rotation.from_euler("z", angles, degrees=True).as_matrix()
    damaged = r.copy()
    damaged[25] = Rotation.from_euler("z", angles[25]+170, degrees=True).as_matrix()
    out = smooth_rotations(damaged, robust=True, window=7)
    error = (Rotation.from_matrix(r).inv()*Rotation.from_matrix(out)).magnitude()
    assert np.rad2deg(error).max() < 4


def test_pose_smoothing_and_interpolation_cross_pi_without_unwinding():
    theta = np.zeros((11,24,3))
    theta[:,4,0] = np.deg2rad([179]*5+[-179]*6)
    out = smooth_pose(theta,window=9)
    assert np.min(np.linalg.norm(out[:,4],axis=1)) > np.deg2rad(177)
    interp = interpolate_pose(np.array([0,2]),theta[[0,-1]],np.array([0,1,2]))
    assert np.linalg.norm(interp[1,4]) > np.deg2rad(179.9)


def test_lean_threshold_is_continuous():
    from src.stages.refined_poses import _reduce_root_lean
    r = Rotation.from_euler("x",[60.1,59.9],degrees=True).as_matrix()
    out,_ = _reduce_root_lean(r,np.zeros((2,3)))
    jump = (Rotation.from_matrix(out[:1]).inv()*Rotation.from_matrix(out[1:])).magnitude()
    assert np.rad2deg(jump)[0] < .3


def test_runs_and_angular_speed_limit():
    assert frame_runs(np.array([1,2,3,40,41])) == [(0,3),(3,5)]
    r = Rotation.from_euler("z", [0]*10+[170]*10, degrees=True).as_matrix()
    out = smooth_rotations(r,window=5,fps=30,max_speed_deg_s=720)
    q = Rotation.from_matrix(out)
    assert np.rad2deg((q[:-1].inv()*q[1:]).magnitude()).max() <= 24.00001


def test_overlapping_inference_never_bridges_missing_video_time(tmp_path, monkeypatch):
    from src.utils import gvhmr_estimator as module
    frame_ids=np.r_[np.arange(12),np.arange(90,99)]
    seen=[]
    class Estimator:
        def estimate_sequence(self,frames,bboxes,**kwargs):
            ids=np.array([int(f[0,0,0]) for f in frames]); seen.append(ids)
            m=len(ids)
            return {'global_orient':np.zeros((m,3)), 'body_pose':np.zeros((m,63)),
                    'transl':np.column_stack((ids,np.zeros((m,2)))),
                    'kp2d':np.zeros((m,17,3)), 'betas':np.zeros(10)}
    monkeypatch.setattr(module,'_read_video_frames',lambda p,ids:[np.full((2,2,3),i,np.uint8) for i in ids])
    checkpoint=tmp_path/'model';checkpoint.touch()
    out=module.run_on_track([(int(i),(0,0,1,1)) for i in frame_ids],
        video_path=tmp_path/'video',checkpoint=checkpoint,device='cpu',batch_size=1,
        max_sequence_length=6,overlap_frames=2,estimator=Estimator())
    assert len(seen)>4  # overlap, not disjoint packing
    assert all(np.all(np.diff(s)==1) for s in seen)
    np.testing.assert_equal(out['root_t_cam'][:,0],frame_ids)
    np.testing.assert_equal(out['frame_indices'],frame_ids)


def test_render_shape_matches_refinement_rest_joints_without_mutating_asset():
    from src.utils.blender_scene_io import load_smpl_body_data,shape_smpl_body_data
    from src.utils.smpl_skeleton import beta_adjusted_rest_joints,load_smpl_neutral_model
    root=Path(__file__).resolve().parents[1]
    data,_=load_smpl_body_data(root,np)
    if data is None:
        import pytest
        pytest.skip('SMPL asset not installed')
    original=data['joint_positions'].copy()
    beta=np.linspace(-1,1,10)
    shaped,pelvis=shape_smpl_body_data(data,beta,np)
    expected=beta_adjusted_rest_joints(beta,load_smpl_neutral_model())
    np.testing.assert_allclose(shaped['joint_positions']-pelvis,expected,atol=2e-7)
    np.testing.assert_equal(data['joint_positions'],original)


def test_pipeline_fk_matches_upstream_smpl_fk_for_asymmetric_pose():
    import torch
    from src.utils.gvhmr_estimator import GVHMREstimator
    GVHMREstimator(checkpoint='unused',device='cpu')._ensure_imports()
    from hmr4d.utils.body_model.smplx_lite import batch_rigid_transform_v2
    from pytorch3d.transforms import axis_angle_to_matrix
    from src.utils.smpl_skeleton import SMPL_REST_JOINTS_YUP,SMPL_PARENTS,compute_all_joint_worlds_batch
    theta=np.zeros((2,24,3));theta[:,4]=[.8,.1,-.15];theta[:,18]=[.1,-1.2,.1]
    theta[:,1]=[-.3,.1,.05];theta[:,19]=[.2,.7,-.1]
    theta[:,0]=[.1,.7,-.3]
    root=Rotation.from_rotvec(theta[:,0]).as_matrix()
    joints,_=batch_rigid_transform_v2(axis_angle_to_matrix(torch.tensor(theta)),
        torch.tensor(SMPL_REST_JOINTS_YUP),torch.tensor(SMPL_PARENTS))
    actual=compute_all_joint_worlds_batch(theta,root,np.zeros((2,3)))
    np.testing.assert_allclose(actual,joints.numpy(),atol=2e-8)

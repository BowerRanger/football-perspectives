import importlib.util
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from src.utils.physpt_adapter import to_physpt_6d, resample_motion, stitch_windows


def test_physpt_column_packing_roundtrips_with_author_decoder():
    path=Path('third_party/PhysPT/assets/utils.py')
    if not path.exists():
        import pytest
        pytest.skip('Optional PhysPT checkout missing')
    spec=importlib.util.spec_from_file_location('physpt_author_utils_test',path)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    rotations=Rotation.random(30,random_state=17).as_matrix()
    actual=author.rot6d_to_rotmat(torch.tensor(to_physpt_6d(rotations))).numpy()
    np.testing.assert_allclose(actual,rotations,atol=1e-12)


def test_resampling_keeps_real_speed_and_rotation_branch():
    t=np.arange(31)/30.;target=np.arange(21)/20.
    root_t=np.c_[t*3,np.zeros(31),np.ones(31)]
    root=Rotation.from_euler('z',170+30*t,degrees=True).as_matrix()
    pose=np.zeros((31,24,3))
    matrices,translation=resample_motion(t,pose,root,root_t,target)
    np.testing.assert_allclose(np.diff(translation[:,0])*20,3,atol=1e-10)
    np.testing.assert_allclose(matrices[:,0],Rotation.from_euler('z',170+30*target,degrees=True).as_matrix(),atol=1e-10)


def test_stitching_uses_velocity_not_independent_window_origins_and_keeps_tail():
    n=25;starts=np.arange(n-16+1)
    r=np.tile(np.eye(3),(len(starts),16,24,1,1));t=np.zeros((len(starts),16,3))
    for k in range(len(starts)):
        t[k,:,0]=k*100+np.arange(16)*.1;t[k,:,2]=1.2
    rot,trans,used=stitch_windows(r,t,starts,n,[10,20])
    np.testing.assert_allclose(trans[:,0],10+np.arange(n)*.1,atol=1e-12)
    np.testing.assert_allclose(trans[:,2],1.2)
    assert len(rot)==n and used[-1]==len(starts)-1

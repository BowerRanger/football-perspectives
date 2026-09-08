import numpy as np
from scipy.spatial.transform import Rotation
from src.utils.animation_quality import motion_metrics
from src.utils.smpl_skeleton import SMPL_REST_JOINTS_YUP


def test_reports_touchdown_snap_even_when_stance_is_stationary():
    n=12; rt=np.zeros((n,3));rt[:,2]=1.;rt[:5,0]=.3
    mask=np.zeros((n,2),bool);mask[5:,0]=True
    m=motion_metrics(frames=np.arange(n),thetas=np.zeros((n,24,3)),
        root_R=Rotation.from_euler('x',[90]*n,degrees=True).as_matrix(),root_t=rt,
        rest_joints=SMPL_REST_JOINTS_YUP,fps=30,candidate_contacts=mask)
    assert m['candidate_contact']['stance_speed_m_s']['max']==0
    assert m['candidate_contact']['transition_speed_m_s']['max'] > 8.9


def test_motion_metrics_do_not_score_occlusion_as_one_frame():
    frames=np.array([0,1,2,90,91,92]);rt=np.zeros((6,3));rt[3:,0]=20
    m=motion_metrics(frames=frames,thetas=np.zeros((6,24,3)),root_R=np.tile(np.eye(3),(6,1,1)),
        root_t=rt,rest_joints=SMPL_REST_JOINTS_YUP,fps=30,candidate_contacts=np.ones((6,2),bool))
    assert m['foot_speed_m_s']['max']==0
    assert m['root_acc_m_s2']['max']==0

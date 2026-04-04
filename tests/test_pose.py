import numpy as np
import pytest
from src.utils.pose_estimator import FakePoseEstimator, PoseEstimator
from src.schemas.poses import COCO_KEYPOINT_NAMES


def test_fake_pose_estimator_returns_17_keypoints():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(100.0, 50.0))
    assert len(kps) == 17


def test_fake_pose_estimator_applies_offset():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(100.0, 50.0))
    # All keypoints should have x >= 100 (offset applied)
    assert all(kp.x >= 100.0 for kp in kps)


def test_fake_pose_estimator_keypoint_names():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(0.0, 0.0))
    names = [kp.name for kp in kps]
    assert names == COCO_KEYPOINT_NAMES


def test_pose_estimator_is_abstract():
    assert issubclass(FakePoseEstimator, PoseEstimator)

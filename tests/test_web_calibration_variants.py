"""Smoke tests for the calibration variant endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.web.server import create_app


def _write_stub_calibration(path: Path, shot_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    CalibrationResult(
        shot_id=shot_id,
        camera_type="static",
        frames=[
            CameraFrame(
                frame=0,
                intrinsic_matrix=[[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
                rotation_vector=[0.0, 0.0, 0.0],
                translation_vector=[0.0, 0.0, 10.0],
                reprojection_error=1.0,
                num_correspondences=12,
                confidence=0.9,
                tracked_landmark_types=[],
            ),
        ],
    ).save(path)


class TestVariantsEndpoint:
    def test_lists_primary_only_when_no_subdirs(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "calibration"
        _write_stub_calibration(
            cal_dir / "shot_001_calibration.json", "shot_001",
        )
        client = TestClient(create_app(tmp_path))
        r = client.get("/api/calibration/variants")
        assert r.status_code == 200
        variants = r.json()["variants"]
        assert [v["id"] for v in variants] == ["primary"]

    def test_lists_primary_plus_subdirs(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "calibration"
        _write_stub_calibration(cal_dir / "shot_001_calibration.json", "shot_001")
        _write_stub_calibration(
            cal_dir / "tvcalib" / "shot_001_calibration.json", "shot_001",
        )
        _write_stub_calibration(
            cal_dir / "pnlcalib" / "shot_001_calibration.json", "shot_001",
        )
        # debug/ and annotations/ siblings must be ignored.
        (cal_dir / "debug").mkdir()
        (cal_dir / "annotations").mkdir()
        # A subdir with no calibration files is skipped too.
        (cal_dir / "empty").mkdir()

        client = TestClient(create_app(tmp_path))
        variants = client.get("/api/calibration/variants").json()["variants"]
        ids = [v["id"] for v in variants]
        assert ids[0] == "primary"
        assert set(ids) == {"primary", "pnlcalib", "tvcalib"}

    def test_empty_when_no_calibration(self, tmp_path: Path) -> None:
        client = TestClient(create_app(tmp_path))
        r = client.get("/api/calibration/variants")
        assert r.status_code == 200
        assert r.json() == {"variants": []}


class TestInterpolatedVariantRouting:
    def test_rejects_invalid_variant_name(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "calibration"
        _write_stub_calibration(cal_dir / "shot_001_calibration.json", "shot_001")
        client = TestClient(create_app(tmp_path))
        r = client.get("/api/calibration/shot_001/interpolated?variant=../etc")
        assert r.status_code == 400

    def test_404_when_variant_subdir_missing(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "calibration"
        _write_stub_calibration(cal_dir / "shot_001_calibration.json", "shot_001")
        client = TestClient(create_app(tmp_path))
        r = client.get("/api/calibration/shot_001/interpolated?variant=tvcalib")
        assert r.status_code == 404

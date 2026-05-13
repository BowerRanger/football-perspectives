"""Round-trip tests for the cloud manifest schemas."""

from __future__ import annotations

import json

import pytest

from src.cloud.manifest import (
    SCHEMA_VERSION,
    ArrayManifest,
    JobManifest,
    JobStatus,
)


def _make_manifest(**overrides: object) -> JobManifest:
    base = dict(
        run_id="20260513T120000Z-abcd1234",
        shot_id="play_01",
        player_id="p17",
        video_uri="s3://bucket/runs/x/shots/play_01.mp4",
        camera_track_uri="s3://bucket/runs/x/camera/play_01_camera_track.json",
        track_frames=((100, (10, 20, 100, 200)), (101, (12, 22, 102, 202))),
        hmr_world_cfg={"device": "cuda:0", "max_sequence_length": 120},
        output_prefix="s3://bucket/runs/x/jobs/play_01__p17",
    )
    base.update(overrides)
    return JobManifest(**base)


def test_job_manifest_round_trip() -> None:
    original = _make_manifest()
    roundtripped = JobManifest.from_json(original.to_json())
    assert roundtripped == original


def test_job_manifest_track_frames_normalised() -> None:
    """JSON loses tuple-vs-list distinction; from_dict re-tuples."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "x",
        "shot_id": "s",
        "player_id": "p",
        "video_uri": "s3://b/v.mp4",
        "camera_track_uri": "s3://b/c.json",
        "track_frames": [[10, [1, 2, 3, 4]]],
        "hmr_world_cfg": {},
        "output_prefix": "s3://b/o",
    }
    manifest = JobManifest.from_dict(payload)
    assert isinstance(manifest.track_frames, tuple)
    assert isinstance(manifest.track_frames[0], tuple)
    assert isinstance(manifest.track_frames[0][1], tuple)
    assert manifest.track_frames[0] == (10, (1, 2, 3, 4))


def test_job_manifest_rejects_wrong_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        JobManifest.from_dict({
            "schema_version": 999,
            "run_id": "x",
            "shot_id": "s",
            "player_id": "p",
            "video_uri": "s3://b/v.mp4",
            "camera_track_uri": "s3://b/c.json",
            "track_frames": [],
            "hmr_world_cfg": {},
            "output_prefix": "s3://b/o",
        })


def test_array_manifest_round_trip() -> None:
    manifest = ArrayManifest(
        run_id="r1",
        entries=("s3://b/a/0.json", "s3://b/a/1.json", "s3://b/a/2.json"),
    )
    roundtripped = ArrayManifest.from_json(manifest.to_json())
    assert roundtripped == manifest


def test_job_status_serialisation() -> None:
    status = JobStatus(
        status="ok",
        duration_seconds=47.2,
        frames=348,
        git_sha="abc1234",
        metadata={"backend": "gvhmr"},
    )
    payload = json.loads(status.to_json())
    assert payload["status"] == "ok"
    assert payload["duration_seconds"] == 47.2
    assert payload["frames"] == 348
    assert payload["git_sha"] == "abc1234"
    assert payload["metadata"] == {"backend": "gvhmr"}


def test_job_status_error_payload() -> None:
    status = JobStatus(
        status="error",
        duration_seconds=3.1,
        frames=0,
        error_type="RuntimeError",
        error_message="boom",
        traceback="Traceback...",
    )
    payload = json.loads(status.to_json())
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_message"] == "boom"
    assert payload["traceback"] == "Traceback..."

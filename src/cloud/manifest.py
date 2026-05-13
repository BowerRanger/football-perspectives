"""Schemas for the per-job manifest and the array-job index.

The orchestrator writes one ``JobManifest`` per (shot_id, player_id) plus
one ``ArrayManifest`` listing them in array-index order. The container's
handler reads ``AWS_BATCH_JOB_ARRAY_INDEX`` and fetches the matching
``JobManifest`` via the index.

Both schemas are JSON-serialisable; ``track_frames`` survives the round
trip because list-of-lists is JSON-native (the dataclass uses tuples
internally for immutability and converts on the boundary).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


# ``track_frames`` element type: (frame_idx, (x1, y1, x2, y2)).
TrackFrame = tuple[int, tuple[int, int, int, int]]


@dataclass(frozen=True)
class JobManifest:
    """All inputs one Batch job needs to process one player track."""

    run_id: str
    shot_id: str
    player_id: str
    video_uri: str           # s3://... in batch mode, file://... locally
    camera_track_uri: str
    track_frames: tuple[TrackFrame, ...]
    hmr_world_cfg: dict[str, Any]
    output_prefix: str        # destination directory or s3 prefix
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "shot_id": self.shot_id,
            "player_id": self.player_id,
            "video_uri": self.video_uri,
            "camera_track_uri": self.camera_track_uri,
            "track_frames": [
                [int(fi), list(int(x) for x in bbox)]
                for fi, bbox in self.track_frames
            ],
            "hmr_world_cfg": dict(self.hmr_world_cfg),
            "output_prefix": self.output_prefix,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobManifest":
        v = int(data.get("schema_version", SCHEMA_VERSION))
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"JobManifest schema_version {v} != supported {SCHEMA_VERSION}"
            )
        frames = tuple(
            (int(fi), tuple(int(x) for x in bbox))
            for fi, bbox in data["track_frames"]
        )
        return cls(
            run_id=str(data["run_id"]),
            shot_id=str(data["shot_id"]),
            player_id=str(data["player_id"]),
            video_uri=str(data["video_uri"]),
            camera_track_uri=str(data["camera_track_uri"]),
            track_frames=frames,
            hmr_world_cfg=dict(data["hmr_world_cfg"]),
            output_prefix=str(data["output_prefix"]),
            schema_version=v,
        )

    @classmethod
    def from_json(cls, text: str) -> "JobManifest":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_path(cls, path: Path) -> "JobManifest":
        return cls.from_json(path.read_text())


@dataclass(frozen=True)
class ArrayManifest:
    """Maps Batch array-job indices to the manifest URI for each child job."""

    run_id: str
    entries: tuple[str, ...]    # entries[idx] = manifest URI for job idx
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "entries": list(self.entries),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "ArrayManifest":
        data = json.loads(text)
        return cls(
            run_id=str(data["run_id"]),
            entries=tuple(str(e) for e in data["entries"]),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class JobStatus:
    """Written to ``status.json`` next to the outputs."""

    status: str                 # "ok" | "error" | "too_short" | "cached"
    duration_seconds: float
    frames: int
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    git_sha: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "status": self.status,
                "duration_seconds": self.duration_seconds,
                "frames": self.frames,
                "error_type": self.error_type,
                "error_message": self.error_message,
                "traceback": self.traceback,
                "git_sha": self.git_sha,
                "metadata": dict(self.metadata),
            },
            indent=2,
        )

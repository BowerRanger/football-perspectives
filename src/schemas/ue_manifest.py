"""Versioned manifest contract between the pipeline and the UE5 editor utility.

Producer: ``src/stages/export.py`` writes ``output/export/ue_manifest.json``
after FBX export. Consumer: UE5 editor utility ``Content/Python/football_perspectives/
manifest.py`` reads the same file.

Bump ``SCHEMA_VERSION`` on any breaking change. The UE side rejects
unknown versions with a clear message.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 2


class UeManifestError(ValueError):
    """Raised when a manifest fails validation."""


@dataclass
class WorldBBox:
    min: tuple[float, float, float]
    max: tuple[float, float, float]


@dataclass
class PitchInfo:
    length_m: float
    width_m: float


@dataclass
class PlayerEntry:
    player_id: str
    fbx: str
    frame_range: tuple[int, int]
    world_bbox: WorldBBox
    display_name: str = ""
    # Canonical kit role (see src.utils.team_roles.KIT_ROLES): one of
    # home/away/home_gk/away_gk/referee/unknown. The UE side looks this
    # up in UeManifest.kits to colour the player's kit material.
    kit_role: str = "unknown"

    def __post_init__(self) -> None:
        # Default display_name to player_id when omitted, so the UE side
        # always has a non-empty asset-name source.
        if not self.display_name:
            self.display_name = self.player_id


@dataclass
class BallEntry:
    fbx: str
    frame_range: tuple[int, int]
    # Optional path to the per-frame translation source. The ball is a
    # static mesh actor on the UE side; reading the JSON directly into
    # a MovieScene3DTransformTrack avoids the SkeletalMesh + AnimSequence
    # round-trip needed when binding to an FBX. Empty string when the
    # legacy FBX path is the only source.
    track_json: str = ""


@dataclass
class CameraEntry:
    fbx: str
    image_size: tuple[int, int]
    frame_range: tuple[int, int]
    # Pipeline-relative path to the camera_track.json with per-frame
    # R/t/K data. Preferred over the FBX on the UE side — CineCameraActor
    # is a regular actor, not a skeletal asset, so a transform + focal
    # length track driven from JSON is the right semantic.
    track_json: str = ""


@dataclass
class NamedCameraEntry:
    name: str
    fbx: str
    image_size: tuple[int, int]
    frame_range: tuple[int, int]
    track_json: str = ""


@dataclass
class UeManifest:
    schema_version: int
    clip_name: str
    fps: float
    frame_range: tuple[int, int]
    pitch: PitchInfo
    players: list[PlayerEntry] = field(default_factory=list)
    ball: Optional[BallEntry] = None
    camera: Optional[CameraEntry] = None
    cameras: list[NamedCameraEntry] = field(default_factory=list)
    # Role -> hex-colour palette (e.g. {"home": "#3b82f6", ...}). The UE
    # side builds a kit material instance per role from this map and
    # assigns it to each player by PlayerEntry.kit_role. Empty when the
    # clip has no match metadata.
    kits: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise UeManifestError(
                f"schema_version {self.schema_version} != {SCHEMA_VERSION}"
            )
        if not self.clip_name:
            raise UeManifestError("clip_name must be non-empty")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise UeManifestError(f"fps must be finite and positive, got {self.fps}")
        if self.frame_range[0] > self.frame_range[1]:
            raise UeManifestError(
                f"frame_range start > end: {self.frame_range}"
            )
        if not self.players:
            raise UeManifestError("players must be non-empty")
        for p in self.players:
            if p.frame_range[0] > p.frame_range[1]:
                raise UeManifestError(
                    f"player {p.player_id} frame_range start > end"
                )

    def save(self, path: Path) -> None:
        self.validate()
        raw: dict = {
            "schema_version": self.schema_version,
            "clip_name": self.clip_name,
            "fps": self.fps,
            "frame_range": list(self.frame_range),
            "pitch": asdict(self.pitch),
            "players": [
                {
                    "player_id": p.player_id,
                    "display_name": p.display_name,
                    "fbx": p.fbx,
                    "frame_range": list(p.frame_range),
                    "world_bbox": {
                        "min": list(p.world_bbox.min),
                        "max": list(p.world_bbox.max),
                    },
                    "kit_role": p.kit_role,
                }
                for p in self.players
            ],
        }
        if self.kits:
            raw["kits"] = dict(self.kits)
        if self.ball is not None:
            raw["ball"] = {
                "fbx": self.ball.fbx,
                "frame_range": list(self.ball.frame_range),
            }
            if self.ball.track_json:
                raw["ball"]["track_json"] = self.ball.track_json
        if self.camera is not None:
            raw["camera"] = {
                "fbx": self.camera.fbx,
                "image_size": list(self.camera.image_size),
                "frame_range": list(self.camera.frame_range),
            }
            if self.camera.track_json:
                raw["camera"]["track_json"] = self.camera.track_json
        if self.cameras:
            raw["cameras"] = [
                {
                    "name": c.name,
                    "fbx": c.fbx,
                    "image_size": list(c.image_size),
                    "frame_range": list(c.frame_range),
                    **({"track_json": c.track_json} if c.track_json else {}),
                }
                for c in self.cameras
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, indent=2))

    @classmethod
    def load(cls, path: Path) -> "UeManifest":
        raw = json.loads(path.read_text())
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise UeManifestError(
                f"schema_version {raw.get('schema_version')} != {SCHEMA_VERSION}"
            )
        m = cls(
            schema_version=int(raw["schema_version"]),
            clip_name=str(raw["clip_name"]),
            fps=float(raw["fps"]),
            frame_range=tuple(raw["frame_range"]),
            pitch=PitchInfo(**raw["pitch"]),
            players=[
                PlayerEntry(
                    player_id=p["player_id"],
                    display_name=p.get("display_name", ""),
                    fbx=p["fbx"],
                    frame_range=tuple(p["frame_range"]),
                    world_bbox=WorldBBox(
                        min=tuple(p["world_bbox"]["min"]),
                        max=tuple(p["world_bbox"]["max"]),
                    ),
                    kit_role=p.get("kit_role", "unknown"),
                )
                for p in raw["players"]
            ],
            ball=(
                BallEntry(
                    fbx=raw["ball"]["fbx"],
                    frame_range=tuple(raw["ball"]["frame_range"]),
                    track_json=raw["ball"].get("track_json", ""),
                )
                if "ball" in raw
                else None
            ),
            camera=(
                CameraEntry(
                    fbx=raw["camera"]["fbx"],
                    image_size=tuple(raw["camera"]["image_size"]),
                    frame_range=tuple(raw["camera"]["frame_range"]),
                    track_json=raw["camera"].get("track_json", ""),
                )
                if "camera" in raw
                else None
            ),
            kits=raw.get("kits", {}) or {},
            cameras=[
                NamedCameraEntry(
                    name=c["name"],
                    fbx=c["fbx"],
                    image_size=tuple(c["image_size"]),
                    frame_range=tuple(c["frame_range"]),
                    track_json=c.get("track_json", ""),
                )
                for c in raw.get("cameras", [])
            ],
        )
        m.validate()
        return m

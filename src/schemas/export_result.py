from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class ExportResult:
    """Manifest of exported files from Stage 9."""

    players: list[str] = field(default_factory=list)  # player IDs
    gltf_file: str | None = None  # relative path to scene.glb
    metadata_file: str | None = None  # relative path to scene_metadata.json
    fbx_files: list[str] = field(default_factory=list)  # relative paths

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ExportResult":
        data = json.loads(path.read_text())
        return cls(**data)

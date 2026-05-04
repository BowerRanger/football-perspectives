"""Stage 7: Export to glTF and FBX (UE5).

TODO(Phase 4b): Reimplement to consume per-track ``hmr_world/`` outputs and
``ball/ball_track.json`` and produce ``export/gltf/scene.glb`` plus the
metadata sidecar.  See spec section 5.7.
"""

from src.pipeline.base import BaseStage


class ExportStage(BaseStage):
    name = "export"

    def is_complete(self) -> bool:
        return (self.output_dir / "export" / "gltf" / "scene.glb").exists()

    def run(self) -> None:
        raise NotImplementedError(
            "ExportStage is not implemented yet — see Phase 4b of "
            "docs/superpowers/plans/2026-05-04-broadcast-mono-pipeline.md"
        )

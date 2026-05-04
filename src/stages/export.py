"""Stage 9: Export to glTF and optionally FBX.

Converts SMPL parameter sequences into animation files for the 3D viewer
and Unreal Engine.
"""

import json
import logging
from pathlib import Path

from src.pipeline.base import BaseStage
from src.schemas.export_result import ExportResult
from src.schemas.player_matches import PlayerMatches
from src.schemas.smpl_result import SmplResult
from src.utils.gltf_builder import build_minimal_glb, build_scene_metadata


class ExportStage(BaseStage):
    name = "export"

    def _is_hmr_mode(self) -> bool:
        return self.config.get("pipeline", {}).get("mode") == "hmr"

    def _hmr_has_results(self) -> bool:
        hmr_dir = self.output_dir / "hmr"
        return hmr_dir.exists() and any(hmr_dir.glob("*_hmr.npz"))

    def is_complete(self) -> bool:
        if self._is_hmr_mode():
            # In HMR mode the viewer reads HmrResult directly — no GLB.
            return self._hmr_has_results()
        return (self.output_dir / "export" / "gltf" / "scene.glb").exists()

    def run(self) -> None:
        if self._is_hmr_mode():
            if self._hmr_has_results():
                print("  -> HMR mode: viewer loads HmrResult directly, "
                      "skipping glTF export")
            else:
                logging.warning("HMR mode but no HMR results found at "
                                "%s — nothing to export", self.output_dir / "hmr")
            return

        cfg = self.config.get("export", {})
        gltf_enabled = bool(cfg.get("gltf_enabled", True))
        fbx_enabled = bool(cfg.get("fbx_enabled", False))

        smpl_dir = self.output_dir / "smpl"
        if not smpl_dir.exists() or not any(smpl_dir.glob("*.npz")):
            logging.warning("No SMPL results found — skipping export")
            return

        # Load SMPL results
        smpl_results: list[SmplResult] = []
        for npz_path in sorted(smpl_dir.glob("*.npz")):
            smpl_results.append(SmplResult.load(npz_path))

        # Load player team info from matching
        player_teams: dict[str, str] = {}
        match_path = self.output_dir / "matching" / "player_matches.json"
        if match_path.exists():
            matches = PlayerMatches.load(match_path)
            for mp in matches.matched_players:
                player_teams[mp.player_id] = mp.team

        export_dir = self.output_dir / "export"
        player_ids = [r.player_id for r in smpl_results]
        result = ExportResult(players=player_ids)

        print(f"  -> exporting {len(smpl_results)} players")

        # glTF export
        if gltf_enabled:
            gltf_dir = export_dir / "gltf"
            gltf_dir.mkdir(parents=True, exist_ok=True)

            fps = smpl_results[0].fps if smpl_results else 25.0

            # Build and write GLB
            glb_bytes = build_minimal_glb(smpl_results, player_teams)
            glb_path = gltf_dir / "scene.glb"
            glb_path.write_bytes(glb_bytes)
            result.gltf_file = "export/gltf/scene.glb"
            print(f"     glTF: {glb_path} ({len(glb_bytes)} bytes)")

            # Build and write metadata
            metadata = build_scene_metadata(smpl_results, player_teams, fps)
            meta_path = gltf_dir / "scene_metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2))
            result.metadata_file = "export/gltf/scene_metadata.json"

        # FBX export (optional, requires Blender)
        if fbx_enabled:
            import shutil

            blender_path = str(cfg.get("blender_path", "blender"))
            if not shutil.which(blender_path):
                logging.warning(
                    "Blender not found at '%s' — skipping FBX export. "
                    "Install Blender and set export.blender_path in config.",
                    blender_path,
                )
            else:
                logging.info("FBX export via Blender is not yet implemented")

        # Save export manifest
        manifest_path = export_dir / "export_result.json"
        export_dir.mkdir(parents=True, exist_ok=True)
        result.save(manifest_path)
        print(f"  -> export manifest saved to {manifest_path}")

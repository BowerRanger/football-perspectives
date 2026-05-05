"""Stage 7: Export to glTF (web viewer) and FBX (UE5).

Inputs:
    - ``output/camera/camera_track.json``  (CameraTrack)
    - ``output/hmr_world/*_smpl_world.npz`` (SmplWorldTrack list)
    - ``output/ball/ball_track.json``       (BallTrack, optional)

Outputs:
    - ``output/export/gltf/scene.glb``
    - ``output/export/gltf/scene_metadata.json``
    - ``output/export/fbx/*.fbx``  (when Blender is available + ``fbx_enabled``)

The glTF emission uses :func:`src.utils.gltf_builder.build_glb` and produces a
v1 scene: pitch plane + line markings, one capsule per player at root_t/root_R,
an animated ball sphere, and an animated broadcast camera.  Full SMPL skinning
+ per-joint pose application from ``thetas`` is deferred (see notes in
``gltf_builder.py``).

The FBX emission shells out to Blender (``cfg.export.blender_path``) using the
``scripts/blender_export_fbx.py`` helper.  When Blender is missing the stage
logs a clear warning and continues — the glTF export remains the source of
truth for the web viewer.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from src.pipeline.base import BaseStage
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.utils.gltf_builder import SceneBundle, build_glb

logger = logging.getLogger(__name__)


class ExportStage(BaseStage):
    name = "export"

    def is_complete(self) -> bool:
        return (self.output_dir / "export" / "gltf" / "scene.glb").exists()

    # ------------------------------------------------------------------

    def run(self) -> None:
        export_cfg = self.config.get("export", {}) or {}
        pitch_cfg = self.config.get("pitch", {}) or {}
        ball_cfg = self.config.get("ball", {}) or {}

        gltf_enabled = bool(export_cfg.get("gltf_enabled", True))
        fbx_enabled = bool(export_cfg.get("fbx_enabled", True))

        if gltf_enabled:
            self._export_gltf(pitch_cfg, ball_cfg)
        else:
            logger.info("[export] gltf_enabled=false, skipping glTF emission")

        if fbx_enabled:
            self._export_fbx(export_cfg)
        else:
            logger.info("[export] fbx_enabled=false, skipping FBX emission")

    # ------------------------------------------------------------------
    # glTF
    # ------------------------------------------------------------------

    def _export_gltf(self, pitch_cfg: dict, ball_cfg: dict) -> None:
        camera_path = self.output_dir / "camera" / "camera_track.json"
        if not camera_path.exists():
            raise FileNotFoundError(
                f"camera_track.json not found at {camera_path}; run the camera "
                "stage first"
            )
        camera_track = CameraTrack.load(camera_path)

        hmr_dir = self.output_dir / "hmr_world"
        npz_files = sorted(hmr_dir.glob("*_smpl_world.npz")) if hmr_dir.exists() else []
        players: list[SmplWorldTrack] = [SmplWorldTrack.load(p) for p in npz_files]

        ball_path = self.output_dir / "ball" / "ball_track.json"
        ball_track = BallTrack.load(ball_path) if ball_path.exists() else None

        bundle = SceneBundle(
            camera_track=camera_track,
            players=tuple(players),
            ball_track=ball_track,
            pitch_length_m=float(pitch_cfg.get("length_m", 105.0)),
            pitch_width_m=float(pitch_cfg.get("width_m", 68.0)),
            ball_radius_m=float(ball_cfg.get("ball_radius_m", 0.11)),
        )
        glb_bytes, metadata = build_glb(bundle)

        gltf_dir = self.output_dir / "export" / "gltf"
        gltf_dir.mkdir(parents=True, exist_ok=True)
        glb_path = gltf_dir / "scene.glb"
        glb_path.write_bytes(glb_bytes)
        meta_path = gltf_dir / "scene_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.info(
            "[export] wrote glTF %s (%d bytes); %d player(s)",
            glb_path, len(glb_bytes), len(players),
        )

    # ------------------------------------------------------------------
    # FBX (optional, requires Blender)
    # ------------------------------------------------------------------

    def _export_fbx(self, export_cfg: dict) -> None:
        blender_path = export_cfg.get("blender_path", "blender")
        blender = shutil.which(blender_path) if not Path(blender_path).is_absolute() else (
            blender_path if Path(blender_path).exists() else None
        )
        if not blender:
            logger.warning(
                "[export] Blender not found on PATH (configured: %s); skipping "
                "FBX. See scripts/blender_export_fbx.py for manual usage.",
                blender_path,
            )
            return

        script_path = Path(__file__).resolve().parents[2] / "scripts" / "blender_export_fbx.py"
        if not script_path.exists():
            logger.warning(
                "[export] FBX helper script missing at %s; skipping FBX",
                script_path,
            )
            return

        fbx_dir = self.output_dir / "export" / "fbx"
        fbx_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            blender,
            "--background",
            "--python", str(script_path),
            "--",
            "--output-dir", str(self.output_dir),
        ]
        logger.info("[export] invoking Blender: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as exc:
            logger.warning("[export] Blender invocation failed: %s; skipping FBX", exc)
            return
        except subprocess.TimeoutExpired:
            logger.warning("[export] Blender FBX export timed out after 600s")
            return
        if result.returncode != 0:
            logger.warning(
                "[export] Blender FBX export returned %d; stderr=%s",
                result.returncode, result.stderr.strip()[-500:],
            )
            return
        logger.info("[export] FBX export complete")

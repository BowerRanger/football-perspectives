"""Render stage — headless Blender toon renders of each shot.

Thin orchestrator: resolves the Blender binary (render.blender_path,
falling back to export.blender_path), then shells to
scripts/blender_render_scene.py once per active shot. Degrades to a
warning when Blender is missing — same posture as FBX export.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from src.pipeline.base import BaseStage
from src.schemas.shots import ShotsManifest

logger = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "blender_render_scene.py"


class RenderStage(BaseStage):
    name = "render"

    def is_complete(self) -> bool:
        render_dir = self.output_dir / "render"
        return render_dir.exists() and any(render_dir.rglob("*.mp4"))

    def _render_cfg(self) -> dict:
        return self.config.get("render", {})

    def _resolve_blender(self) -> str | None:
        cfg = self._render_cfg()
        path = cfg.get("blender_path") or self.config.get("export", {}).get(
            "blender_path", "blender")
        if Path(path).is_absolute():
            return path if Path(path).exists() else None
        return shutil.which(path)

    def _blender_args(self, shot_id: str) -> list[str]:
        cfg = self._render_cfg()
        w, h = cfg.get("resolution", [1920, 1080])
        blender = self._resolve_blender() or "blender"
        args = [
            blender, "--background", "--python", str(_SCRIPT), "--",
            "--output-dir", str(self.output_dir),
            "--shot", shot_id,
            "--cameras", ",".join(cfg.get("cameras", ["broadcast"])),
            "--width", str(int(w)), "--height", str(int(h)),
            "--samples", str(int(cfg.get("samples", 16))),
            "--style-json", json.dumps(cfg.get("style", {})),
        ]
        if cfg.get("vertical_variant"):
            args.append("--vertical")
        if cfg.get("aov_passes"):
            args.append("--aov")
        if cfg.get("save_blend"):
            args.append("--save-blend")
        return args

    def _active_shot_ids(self) -> list[str]:
        # Mirrors ExportStage._export_shot_ids (src/stages/export.py:357):
        # active shots via ShotsManifest.active_shots() (excludes shots
        # with excluded=True), honouring an optional shot_filter. No
        # manifest on disk means the legacy single-shot layout — "" is
        # this stage's str-typed stand-in for export's `None`.
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return [""]
        manifest = ShotsManifest.load(manifest_path)
        shot_filter = getattr(self, "shot_filter", None)
        return [
            s.id for s in manifest.active_shots()
            if shot_filter is None or s.id == shot_filter
        ]

    def run(self) -> None:
        if not self._render_cfg().get("enabled", True):
            logger.info("render: disabled via config; skipping")
            return
        if self._resolve_blender() is None:
            logger.warning(
                "render: Blender not found (render.blender_path / "
                "export.blender_path); skipping renders. Install Blender "
                ">= 3.6 or set the config path.")
            return
        timings: dict[str, float] = {}
        for shot_id in self._active_shot_ids():
            args = self._blender_args(shot_id)
            logger.info("render: shot=%s -> %s", shot_id or "<legacy>", args)
            t0 = time.time()
            result = subprocess.run(args, capture_output=True, text=True)
            timings[shot_id or "clip"] = round(time.time() - t0, 1)
            if result.returncode != 0:
                logger.error("render: Blender failed for shot %s:\n%s",
                             shot_id, result.stderr[-4000:])
        out = self.output_dir / "render"
        out.mkdir(parents=True, exist_ok=True)
        (out / "render_timings.json").write_text(json.dumps(timings, indent=2))

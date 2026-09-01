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
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.schemas.shots import ShotsManifest
from src.schemas.smpl_world import SmplWorldTrack
from src.stages.export import _per_shot_smpl_tracks
from src.utils import virtual_cameras as vcam

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

    def _blender_args(self, shot_id: str, cameras: list[str] | None = None) -> list[str]:
        cfg = self._render_cfg()
        w, h = cfg.get("resolution", [1920, 1080])
        blender = self._resolve_blender() or "blender"
        cam_list = cameras if cameras is not None else cfg.get("cameras", ["broadcast"])
        args = [
            blender, "--background", "--python", str(_SCRIPT), "--",
            "--output-dir", str(self.output_dir),
            "--shot", shot_id,
            "--cameras", ",".join(cam_list),
            "--width", str(int(w)), "--height", str(int(h)),
            "--samples", str(int(cfg.get("samples", 16))),
            "--style-json", json.dumps(
                {**cfg.get("style", {}), "teams": cfg.get("teams", {})}),
        ]
        if cfg.get("vertical_variant"):
            args.append("--vertical")
        if cfg.get("aov_passes"):
            args.append("--aov")
        if cfg.get("save_blend"):
            args.append("--save-blend")
        return args

    def _virtual_camera_cfg(self) -> vcam.RigConfig:
        # Same construction as ExportStage._virtual_camera_cfg
        # (src/stages/export.py:233), extended with the drone_* fields
        # ExportStage never reads (it only ever builds pov/ots rigs) —
        # RenderStage is the first consumer of build_drone_track, and
        # config/default.yaml's export.virtual_cameras block already
        # carries drone_fov_deg/drone_height_m/drone_back_m/
        # drone_smooth_frames for it to read.
        raw = ((self.config.get("export", {}) or {}).get("virtual_cameras", {})) or {}
        return vcam.RigConfig(
            pov_fov_deg=float(raw.get("pov_fov_deg", 75.0)),
            ots_fov_deg=float(raw.get("ots_fov_deg", 60.0)),
            ots_back_m=float(raw.get("ots_back_m", 0.4)),
            ots_up_m=float(raw.get("ots_up_m", 0.3)),
            ots_right_m=float(raw.get("ots_right_m", 0.0)),
            ball_target_max_occlusion_frames=int(
                raw.get("ball_target_max_occlusion_frames", 10)
            ),
            drone_fov_deg=float(raw.get("drone_fov_deg", 55.0)),
            drone_height_m=float(raw.get("drone_height_m", 40.0)),
            drone_back_m=float(raw.get("drone_back_m", 25.0)),
            drone_smooth_frames=int(raw.get("drone_smooth_frames", 25)),
        )

    def _build_one_virtual_camera(
        self,
        cam_id: str,
        tracks_by_pid: dict[str, SmplWorldTrack],
        ball_track: BallTrack | None,
        cfg: vcam.RigConfig,
        image_size: tuple[int, int],
        fps: float,
        clip_id: str,
    ) -> CameraTrack | None:
        if cam_id == "drone":
            return vcam.build_drone_track(
                list(tracks_by_pid.values()), ball_track, cfg, image_size, fps, clip_id)
        rig, _, player_id = cam_id.partition(":")
        if not player_id:
            logger.warning("render: unknown virtual camera id %r; skipping", cam_id)
            return None
        track = tracks_by_pid.get(player_id)
        if track is None:
            logger.warning(
                "render: no player track for %s (camera %r); skipping",
                player_id, cam_id,
            )
            return None
        if rig == "pov":
            return vcam.build_pov_track(track, cfg, image_size, fps, clip_id)
        if rig == "ots":
            return vcam.build_ots_track(track, ball_track, cfg, image_size, fps, clip_id)
        logger.warning("render: unknown virtual camera rig %r in %r; skipping", rig, cam_id)
        return None

    def _write_virtual_camera_tracks(
        self, shot_id: str, camera_ids: list[str]
    ) -> list[str]:
        """Build + write a CameraTrack JSON for each requested virtual camera.

        Writes ``output/render/<shot|clip>/cameras/<safe_id>_camera_track.json``
        — ``safe_id`` is ``cam_id`` with ``:`` replaced by ``_``, matching
        the filename scripts/blender_render_scene.py resolves non-broadcast
        camera ids against. Serialises via ``CameraTrack.save`` — the exact
        writer the camera stage uses for ``camera/camera_track.json``
        (src/stages/camera.py) — so the script's generic reader works
        identically for real and virtual tracks.

        Returns the subset of ``camera_ids`` successfully written. Unknown
        player references (e.g. ``pov:P999`` with no matching track) are
        skipped with a warning rather than raising, so one bad camera id
        never fails the whole shot's render.
        """
        if not camera_ids:
            return []

        prefix = f"{shot_id}_" if shot_id else ""
        broadcast_path = self.output_dir / "camera" / f"{prefix}camera_track.json"
        if not broadcast_path.exists():
            logger.warning(
                "render: no broadcast camera for shot %s; skipping virtual "
                "camera tracks %s", shot_id or "<legacy>", camera_ids,
            )
            return []
        broadcast = CameraTrack.load(broadcast_path)
        fps = float(broadcast.fps)

        render_cfg = self._render_cfg()
        w, h = render_cfg.get("resolution", [1920, 1080])
        image_size = (int(w), int(h))
        cfg = self._virtual_camera_cfg()
        clip_id = shot_id or "clip"

        tracks_by_pid = {
            t.player_id: t
            for t in _per_shot_smpl_tracks(self.output_dir, shot_id=shot_id or None)
        }
        ball_path = self.output_dir / "ball" / f"{prefix}ball_track.json"
        ball_track = BallTrack.load(ball_path) if ball_path.exists() else None

        cams_dir = self.output_dir / "render" / clip_id / "cameras"
        written: list[str] = []
        for cam_id in camera_ids:
            cam = self._build_one_virtual_camera(
                cam_id, tracks_by_pid, ball_track, cfg, image_size, fps, clip_id)
            if cam is None or not cam.frames:
                if cam is not None:
                    logger.warning(
                        "render: empty camera track for %r; skipping", cam_id)
                continue
            safe_id = cam_id.replace(":", "_")
            cam.save(cams_dir / f"{safe_id}_camera_track.json")
            written.append(cam_id)
        return written

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
        requested = list(self._render_cfg().get("cameras", ["broadcast"]))
        broadcast_ids = [c for c in requested if c == "broadcast"]
        virtual_ids = [c for c in requested if c != "broadcast"]
        for shot_id in self._active_shot_ids():
            satisfied = set(self._write_virtual_camera_tracks(shot_id, virtual_ids))
            cameras = broadcast_ids + [c for c in virtual_ids if c in satisfied]
            args = self._blender_args(shot_id, cameras=cameras)
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

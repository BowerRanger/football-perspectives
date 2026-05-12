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

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.schemas.refined_pose import RefinedPose
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import SyncMap
from src.schemas.ue_manifest import (
    BallEntry,
    CameraEntry,
    PitchInfo,
    PlayerEntry,
    UeManifest,
    WorldBBox,
)
from src.utils.gltf_builder import SceneBundle, build_glb
from src.utils.player_names import display_name_for, load_player_names

logger = logging.getLogger(__name__)


def _derive_clip_name(output_dir: Path) -> str:
    """Read the prepared-shots manifest and return a folder-safe clip name.

    Falls back through ``shots_manifest.json`` (current schema with
    ``source_file``) → ``manifest.json`` (legacy with ``source``) →
    ``"clip"``.
    """
    candidates = [
        (output_dir / "shots" / "shots_manifest.json", "source_file"),
        (output_dir / "shots" / "manifest.json", "source"),
    ]
    for path, key in candidates:
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        # Prefer the first shot's clip_file stem if available — survives
        # source_file == "unknown".
        shots = raw.get("shots") or []
        if shots:
            first = shots[0]
            for shot_key in ("id", "clip_file"):
                if shot_key in first and first[shot_key] not in (None, ""):
                    name = Path(str(first[shot_key])).stem
                    break
            else:
                name = "clip"
        else:
            name = Path(str(raw.get(key, "clip"))).stem
        if name in ("", "unknown"):
            name = "clip"
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        return safe or "clip"
    return "clip"


def _resolve_primary_shot(output_dir: Path, clip_name: str) -> str | None:
    """Return ``clip_name`` if it names a real shot in shots_manifest.

    Used by ``write_ue_manifest`` to switch the FBX-path lookup between
    the multi-shot prefixed layout (``fbx/{shot_id}__{name}.fbx``) and
    the legacy single-shot layout (``fbx/{name}.fbx``). Returns ``None``
    when no shots manifest exists or when ``clip_name`` doesn't match
    any shot id, preserving legacy behaviour.
    """
    shots_path = output_dir / "shots" / "shots_manifest.json"
    if not shots_path.exists():
        return None
    raw = json.loads(shots_path.read_text())
    shot_ids = {shot.get("id") for shot in raw.get("shots", []) if shot.get("id")}
    return clip_name if clip_name in shot_ids else None


def _per_shot_smpl_tracks(
    output_dir: Path, *, shot_id: str | None
) -> list[SmplWorldTrack]:
    """Return per-shot SmplWorldTracks consumed by the export builder.

    Prefers ``output/refined_poses/*_refined.npz`` (one fused track per
    player on the reference timeline). For each refined track we slice
    the frames inside the shot's local window after applying the
    reverse sync offset (``f_local = f_ref + offset``) and emit a
    SmplWorldTrack-shaped view that the existing builder already knows
    how to consume.

    Falls back to ``output/hmr_world/*_smpl_world.npz`` when the
    refined directory is empty (e.g. user re-runs ``--stages export``
    without first running refined_poses).
    """
    refined_dir = output_dir / "refined_poses"
    refined_files = (
        sorted(refined_dir.glob("*_refined.npz")) if refined_dir.exists() else []
    )
    if refined_files:
        sync_path = output_dir / "shots" / "sync_map.json"
        sync_map = (
            SyncMap.load(sync_path)
            if sync_path.exists()
            else SyncMap(reference_shot="", alignments=[])
        )
        offset = sync_map.offset_for(shot_id) if shot_id else 0
        out: list[SmplWorldTrack] = []
        for p in refined_files:
            r = RefinedPose.load(p)
            ref_frames = np.asarray(r.frames, dtype=np.int64)
            local = ref_frames + offset
            out.append(
                SmplWorldTrack(
                    player_id=r.player_id,
                    frames=local,
                    betas=np.asarray(r.betas),
                    thetas=np.asarray(r.thetas),
                    root_R=np.asarray(r.root_R),
                    root_t=np.asarray(r.root_t),
                    confidence=np.asarray(r.confidence),
                    shot_id=shot_id or "",
                )
            )
        return out

    # Legacy fallback: per-shot SmplWorldTracks from hmr_world/.
    hmr_dir = output_dir / "hmr_world"
    if not hmr_dir.exists():
        return []
    tracks = [SmplWorldTrack.load(p) for p in sorted(hmr_dir.glob("*_smpl_world.npz"))]
    if shot_id is None:
        return tracks
    return [t for t in tracks if getattr(t, "shot_id", "") == shot_id]


class ExportStage(BaseStage):
    name = "export"

    def is_complete(self) -> bool:
        from src.schemas.shots import ShotsManifest
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return (self.output_dir / "export" / "gltf" / "scene.glb").exists()
        manifest = ShotsManifest.load(manifest_path)
        return all(
            (self.output_dir / "export" / "gltf" / f"{shot.id}_scene.glb").exists()
            for shot in manifest.shots
        )

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

        clip_name = _derive_clip_name(self.output_dir)
        self.write_ue_manifest(clip_name)

    # ------------------------------------------------------------------
    # glTF
    # ------------------------------------------------------------------

    def _export_gltf(self, pitch_cfg: dict, ball_cfg: dict) -> None:
        from src.schemas.shots import ShotsManifest

        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        gltf_dir = self.output_dir / "export" / "gltf"
        gltf_dir.mkdir(parents=True, exist_ok=True)

        # Legacy single-shot path: no manifest, but a singular camera_track
        # exists. Emit one scene.glb (no shot prefix) and bail.
        if not manifest_path.exists():
            cam_path = self.output_dir / "camera" / "camera_track.json"
            if not cam_path.exists():
                raise FileNotFoundError(
                    f"camera_track.json not found at {cam_path}; run the "
                    "camera stage first"
                )
            self._export_gltf_for_shot(
                shot_id=None,
                camera_path=cam_path,
                ball_path=self.output_dir / "ball" / "ball_track.json",
                gltf_dir=gltf_dir,
                pitch_cfg=pitch_cfg,
                ball_cfg=ball_cfg,
            )
            return

        manifest = ShotsManifest.load(manifest_path)
        shot_filter = getattr(self, "shot_filter", None)
        for shot in manifest.shots:
            if shot_filter is not None and shot.id != shot_filter:
                continue
            cam_path = self.output_dir / "camera" / f"{shot.id}_camera_track.json"
            if not cam_path.exists():
                logger.warning(
                    "[export] skipping shot %s — no camera_track at %s",
                    shot.id, cam_path,
                )
                continue
            self._export_gltf_for_shot(
                shot_id=shot.id,
                camera_path=cam_path,
                ball_path=self.output_dir / "ball" / f"{shot.id}_ball_track.json",
                gltf_dir=gltf_dir,
                pitch_cfg=pitch_cfg,
                ball_cfg=ball_cfg,
            )

    def _export_gltf_for_shot(
        self,
        shot_id: str | None,
        camera_path: Path,
        ball_path: Path,
        gltf_dir: Path,
        pitch_cfg: dict,
        ball_cfg: dict,
    ) -> None:
        camera_track = CameraTrack.load(camera_path)

        # Per-player SMPL tracks for this shot. Pulls from the fused
        # refined_poses output when present (default), with a fallback to
        # raw hmr_world tracks when refined_poses hasn't run.
        players = _per_shot_smpl_tracks(self.output_dir, shot_id=shot_id)

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

        prefix = "" if shot_id is None else f"{shot_id}_"
        glb_path = gltf_dir / f"{prefix}scene.glb"
        glb_path.write_bytes(glb_bytes)
        meta_path = gltf_dir / f"{prefix}scene_metadata.json"
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

    # ------------------------------------------------------------------
    # UE manifest
    # ------------------------------------------------------------------

    def write_ue_manifest(self, clip_name: str) -> None:
        """Write ``output/export/ue_manifest.json`` for UE5 ingest.

        Manifest is written iff at least one player FBX exists. Ball and
        camera entries are included only if their FBX files exist.
        """
        export_dir = self.output_dir / "export"
        fbx_dir = export_dir / "fbx"
        if not fbx_dir.exists():
            logger.info("[export] no fbx dir, skipping manifest")
            return

        # Multi-shot outputs name FBXs as ``{shot_id}__{display_name}.fbx``;
        # legacy single-shot outputs use ``{display_name}.fbx``. Pick the
        # right lookup mode by resolving the primary shot from
        # shots_manifest. Option A (single manifest, primary shot only):
        # we emit the manifest for whichever shot ``_derive_clip_name``
        # selected.
        primary_shot = _resolve_primary_shot(self.output_dir, clip_name)
        all_tracks = _per_shot_smpl_tracks(self.output_dir, shot_id=primary_shot)

        name_mapping = load_player_names(self.output_dir)
        players: list[PlayerEntry] = []
        for track in all_tracks:
            display_name = display_name_for(track.player_id, name_mapping)
            shot_id = getattr(track, "shot_id", "") or ""
            keyed = f"{shot_id}__{display_name}" if shot_id else display_name
            fbx_rel = f"fbx/{keyed}.fbx"
            if not (export_dir / fbx_rel).exists():
                # Try a few fallback names so older FBX files (from
                # pre-multi-shot export runs) still register: bare
                # display_name and bare player_id without the shot prefix.
                candidates = [
                    f"fbx/{display_name}.fbx",
                    f"fbx/{track.player_id}.fbx",
                ]
                resolved = next(
                    (c for c in candidates if (export_dir / c).exists()),
                    None,
                )
                if resolved is None:
                    continue
                fbx_rel = resolved
            n = int(track.frames.shape[0])
            if n == 0:
                continue
            mn = track.root_t.min(axis=0)
            mx = track.root_t.max(axis=0)
            players.append(
                PlayerEntry(
                    player_id=track.player_id,
                    display_name=display_name,
                    fbx=fbx_rel,
                    frame_range=(int(track.frames[0]), int(track.frames[-1])),
                    world_bbox=WorldBBox(
                        min=(float(mn[0]), float(mn[1]), float(mn[2])),
                        max=(float(mx[0]), float(mx[1]), float(mx[2])),
                    ),
                )
            )

        if not players:
            logger.info("[export] no player FBX present, skipping manifest")
            return

        # Multi-shot layout writes camera_track + ball_track + their FBXs
        # with the {shot_id}_ prefix; legacy single-shot uses unprefixed
        # paths. Pick whichever the primary shot resolved to and fall
        # back to legacy.
        if primary_shot:
            camera_path = self.output_dir / "camera" / f"{primary_shot}_camera_track.json"
            if not camera_path.exists():
                camera_path = self.output_dir / "camera" / "camera_track.json"
        else:
            camera_path = self.output_dir / "camera" / "camera_track.json"
        if not camera_path.exists():
            logger.warning("[export] camera_track.json missing; using defaults")
            fps = 30.0
            clip_range = (
                min(p.frame_range[0] for p in players),
                max(p.frame_range[1] for p in players),
            )
            cam_meta = {}
        else:
            cam_meta = json.loads(camera_path.read_text())
            fps = float(cam_meta.get("fps", 30.0)) or 30.0
            cam_frames = cam_meta.get("frames", [])
            if cam_frames:
                clip_range = (
                    int(cam_frames[0]["frame"]),
                    int(cam_frames[-1]["frame"]),
                )
            else:
                clip_range = (
                    min(p.frame_range[0] for p in players),
                    max(p.frame_range[1] for p in players),
                )

        ball_entry: BallEntry | None = None
        if primary_shot:
            ball_fbx = fbx_dir / f"{primary_shot}_ball.fbx"
            ball_track_path = self.output_dir / "ball" / f"{primary_shot}_ball_track.json"
            ball_fbx_rel = f"fbx/{primary_shot}_ball.fbx"
        else:
            ball_fbx = fbx_dir / "ball.fbx"
            ball_track_path = self.output_dir / "ball" / "ball_track.json"
            ball_fbx_rel = "fbx/ball.fbx"
        if ball_fbx.exists() and ball_track_path.exists():
            ball_meta = json.loads(ball_track_path.read_text())
            ball_frames = [
                f["frame"]
                for f in ball_meta.get("frames", [])
                if f.get("world_xyz")
            ]
            if ball_frames:
                ball_entry = BallEntry(
                    fbx=ball_fbx_rel,
                    frame_range=(int(ball_frames[0]), int(ball_frames[-1])),
                )

        camera_entry: CameraEntry | None = None
        if primary_shot:
            camera_fbx = fbx_dir / f"{primary_shot}_camera.fbx"
            camera_fbx_rel = f"fbx/{primary_shot}_camera.fbx"
        else:
            camera_fbx = fbx_dir / "camera.fbx"
            camera_fbx_rel = "fbx/camera.fbx"
        if camera_fbx.exists() and cam_meta:
            cam_frames = cam_meta.get("frames", [])
            image_size = tuple(cam_meta.get("image_size", [1920, 1080]))
            if cam_frames:
                camera_entry = CameraEntry(
                    fbx=camera_fbx_rel,
                    image_size=(int(image_size[0]), int(image_size[1])),
                    frame_range=(
                        int(cam_frames[0]["frame"]),
                        int(cam_frames[-1]["frame"]),
                    ),
                )

        pitch_cfg = self.config.get("pitch", {}) or {}
        manifest = UeManifest(
            schema_version=1,
            clip_name=clip_name,
            fps=fps,
            frame_range=clip_range,
            pitch=PitchInfo(
                length_m=float(pitch_cfg.get("length_m", 105.0)),
                width_m=float(pitch_cfg.get("width_m", 68.0)),
            ),
            players=players,
            ball=ball_entry,
            camera=camera_entry,
        )
        manifest.save(export_dir / "ue_manifest.json")
        logger.info(
            "[export] wrote ue_manifest.json with %d player(s), ball=%s, camera=%s",
            len(players),
            ball_entry is not None,
            camera_entry is not None,
        )

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
from src.schemas.camera_selection import CameraSelection
from src.schemas.camera_track import CameraTrack
from src.schemas.refined_pose import RefinedPose
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import SyncMap
from src.schemas.ue_manifest import (
    SCHEMA_VERSION,
    BallEntry,
    CameraEntry,
    NamedCameraEntry,
    PitchInfo,
    PlayerEntry,
    UeManifest,
    WorldBBox,
)
from src.utils import virtual_cameras as vcam
from src.utils.gltf_builder import SceneBundle, build_glb
from src.utils.player_names import (
    display_name_for,
    load_kit_roles,
    load_player_names,
)
from src.utils.team_roles import derive_kit_role, kits_map

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


def _player_team_class_map(output_dir: Path) -> dict[str, tuple[str, str]]:
    """Map ``player_id -> (team, class_name)`` from the tracking outputs.

    Reads every ``tracks/*_tracks.json`` (one ``TracksResult`` per shot)
    and keys by the global ``player_id``. A player can appear in several
    shots; we prefer the first entry whose team is classified (not
    ``"unknown"``) so a late unclassified shot doesn't clobber an
    earlier good label. Returns an empty map when no tracks exist.
    """
    from src.schemas.tracks import TracksResult

    tracks_dir = output_dir / "tracks"
    if not tracks_dir.exists():
        return {}
    out: dict[str, tuple[str, str]] = {}
    for path in sorted(tracks_dir.glob("*_tracks.json")):
        try:
            result = TracksResult.load(path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("[export] could not read tracks %s: %s", path, exc)
            continue
        for track in result.tracks:
            pid = getattr(track, "player_id", "") or ""
            if not pid:
                continue
            team = getattr(track, "team", "") or ""
            cls = getattr(track, "class_name", "") or ""
            existing = out.get(pid)
            # First write wins, but upgrade an "unknown" team if a later
            # shot classified the same player.
            if existing is None or (
                existing[0].strip().lower() == "unknown"
                and team.strip().lower() != "unknown"
            ):
                out[pid] = (team, cls)
    return out


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
    # Virtual cameras
    # ------------------------------------------------------------------

    def _virtual_camera_cfg(self) -> vcam.RigConfig:
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
        )

    def _generate_virtual_cameras(self, shot_id: str | None) -> list[NamedCameraEntry]:
        """Read the per-shot selection, write one CameraTrack JSON per rig,
        and return NamedCameraEntry rows for the manifest. No-op when no
        selection file exists."""
        if not hasattr(self, "_virtual_camera_entries"):
            self._virtual_camera_entries: dict[str | None, list[NamedCameraEntry]] = {}
        prefix = "" if shot_id is None else f"{shot_id}_"
        sel_path = self.output_dir / "export" / f"{prefix}camera_selection.json"
        if not sel_path.exists():
            self._virtual_camera_entries[shot_id] = []
            return []
        selection = CameraSelection.load(sel_path)

        bcast_path = self.output_dir / "camera" / f"{prefix}camera_track.json"
        if not bcast_path.exists():
            logger.warning(
                "[export] no broadcast camera for %s; skipping virtual cameras", shot_id
            )
            self._virtual_camera_entries[shot_id] = []
            return []
        bcast = CameraTrack.load(bcast_path)
        image_size = tuple(bcast.image_size)
        fps = float(bcast.fps)
        cfg = self._virtual_camera_cfg()

        players = {t.player_id: t for t in _per_shot_smpl_tracks(self.output_dir, shot_id=shot_id)}
        ball_path = self.output_dir / "ball" / f"{prefix}ball_track.json"
        ball_track = BallTrack.load(ball_path) if ball_path.exists() else None

        entries: list[NamedCameraEntry] = []
        for sel in selection.selections:
            track = players.get(sel.player_id)
            if track is None:
                logger.warning(
                    "[export] selection player %s not in shot %s; skipping",
                    sel.player_id, shot_id,
                )
                continue
            for rig in sel.rigs:
                clip_id = f"{prefix}{sel.player_id}_{rig}"
                if rig == "pov":
                    cam = vcam.build_pov_track(track, cfg, image_size, fps, clip_id)
                elif rig == "ots":
                    cam = vcam.build_ots_track(track, ball_track, cfg, image_size, fps, clip_id)
                else:
                    logger.warning("[export] unknown rig %r for %s; skipping", rig, sel.player_id)
                    continue
                if not cam.frames:
                    logger.warning(
                        "[export] empty camera track for %s/%s; skipping",
                        sel.player_id, rig,
                    )
                    continue
                cam_path = self.output_dir / "camera" / f"{clip_id}_camera_track.json"
                cam.save(cam_path)
                entries.append(NamedCameraEntry(
                    name=f"{sel.player_id}_{rig}",
                    fbx="",
                    image_size=(int(image_size[0]), int(image_size[1])),
                    frame_range=(int(cam.frames[0].frame), int(cam.frames[-1].frame)),
                    track_json=f"camera/{clip_id}_camera_track.json",
                ))
        self._virtual_camera_entries[shot_id] = entries
        logger.info(
            "[export] generated %d virtual camera(s) for shot %s", len(entries), shot_id
        )
        return entries

    # ------------------------------------------------------------------

    def run(self) -> None:
        export_cfg = self.config.get("export", {}) or {}
        pitch_cfg = self.config.get("pitch", {}) or {}
        ball_cfg = self.config.get("ball", {}) or {}

        gltf_enabled = bool(export_cfg.get("gltf_enabled", True))
        fbx_enabled = bool(export_cfg.get("fbx_enabled", True))

        # Generate virtual cameras up front so the rig CameraTrack JSONs exist
        # for both glTF node emission and FBX baking, and so the manifest gets
        # them even when glTF/FBX are disabled.
        self._virtual_camera_entries = {}
        for shot_id in self._export_shot_ids():
            self._generate_virtual_cameras(shot_id)

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

    def _export_shot_ids(self) -> list[str | None]:
        """Shot ids to export (mirrors _export_gltf's shot resolution).
        Returns [None] for the legacy single-shot layout."""
        from src.schemas.shots import ShotsManifest
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return [None]
        manifest = ShotsManifest.load(manifest_path)
        shot_filter = getattr(self, "shot_filter", None)
        return [s.id for s in manifest.shots if shot_filter is None or s.id == shot_filter]

    # ------------------------------------------------------------------
    # glTF
    # ------------------------------------------------------------------

    def _export_gltf(self, pitch_cfg: dict, ball_cfg: dict) -> None:
        from dataclasses import asdict as _asdict

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
                match=None,
            )
            return

        manifest = ShotsManifest.load(manifest_path)
        # Mirrored into every per-shot scene_metadata.json so the viewer
        # can pick up kit colours / score / venue without a second fetch.
        match_dict = _asdict(manifest.match) if manifest.match is not None else None
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
                match=match_dict,
            )

    def _export_gltf_for_shot(
        self,
        shot_id: str | None,
        camera_path: Path,
        ball_path: Path,
        gltf_dir: Path,
        pitch_cfg: dict,
        ball_cfg: dict,
        match: dict | None = None,
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
            match=match,
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

    def _load_match_dict(self) -> dict | None:
        """Return the ``asdict``-ed match block from shots_manifest, or None.

        Used to source kit colours for the UE manifest. Mirrors the load
        in ``_export_gltf`` but kept separate so manifest writing works
        even when the glTF path was skipped.
        """
        from dataclasses import asdict as _asdict

        from src.schemas.shots import ShotsManifest

        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return None
        manifest = ShotsManifest.load(manifest_path)
        return _asdict(manifest.match) if manifest.match is not None else None

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
        # Kit-role inputs: per-player team/class from the tracking output,
        # plus optional hand-authored overrides in players.json.
        team_class = _player_team_class_map(self.output_dir)
        role_overrides = load_kit_roles(self.output_dir)
        players: list[PlayerEntry] = []
        for track in all_tracks:
            display_name = display_name_for(track.player_id, name_mapping)
            kit_role = role_overrides.get(track.player_id)
            if kit_role is None:
                team, class_name = team_class.get(track.player_id, ("", ""))
                kit_role = derive_kit_role(team, class_name)
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
                    kit_role=kit_role,
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
            ball_track_rel = f"ball/{primary_shot}_ball_track.json"
        else:
            ball_fbx = fbx_dir / "ball.fbx"
            ball_track_path = self.output_dir / "ball" / "ball_track.json"
            ball_fbx_rel = "fbx/ball.fbx"
            ball_track_rel = "ball/ball_track.json"
        # The UE side prefers ``track_json`` for ball animation — reads
        # per-frame world_xyz directly into a MovieScene3DTransformTrack
        # on the BP_BallActor binding (static mesh, not a skeletal
        # asset). ``fbx`` is kept for backwards compatibility but unused
        # by the current EUW load path.
        if ball_track_path.exists():
            ball_meta = json.loads(ball_track_path.read_text())
            ball_frames = [
                f["frame"]
                for f in ball_meta.get("frames", [])
                if f.get("world_xyz")
            ]
            if ball_frames:
                ball_entry = BallEntry(
                    fbx=ball_fbx_rel if ball_fbx.exists() else "",
                    frame_range=(int(ball_frames[0]), int(ball_frames[-1])),
                    track_json=ball_track_rel,
                )

        camera_entry: CameraEntry | None = None
        if primary_shot:
            camera_fbx = fbx_dir / f"{primary_shot}_camera.fbx"
            camera_fbx_rel = f"fbx/{primary_shot}_camera.fbx"
            camera_track_rel = f"camera/{primary_shot}_camera_track.json"
        else:
            camera_fbx = fbx_dir / "camera.fbx"
            camera_fbx_rel = "fbx/camera.fbx"
            camera_track_rel = "camera/camera_track.json"
        # Same architectural choice as the ball: prefer ``track_json``
        # for the camera (per-frame R/t/K on a CineCameraActor binding)
        # over a skeletal-FBX round-trip. ``fbx`` is kept only when the
        # file is present, for backwards compatibility.
        if cam_meta:
            cam_frames = cam_meta.get("frames", [])
            image_size = tuple(cam_meta.get("image_size", [1920, 1080]))
            if cam_frames:
                camera_entry = CameraEntry(
                    fbx=camera_fbx_rel if camera_fbx.exists() else "",
                    image_size=(int(image_size[0]), int(image_size[1])),
                    frame_range=(
                        int(cam_frames[0]["frame"]),
                        int(cam_frames[-1]["frame"]),
                    ),
                    track_json=camera_track_rel,
                )

        # Named-cameras list: broadcast first, then per-rig virtual cameras
        # generated for the primary shot.
        named_cameras: list[NamedCameraEntry] = []
        if camera_entry is not None:
            named_cameras.append(NamedCameraEntry(
                name="broadcast",
                fbx=camera_entry.fbx,
                image_size=camera_entry.image_size,
                frame_range=camera_entry.frame_range,
                track_json=camera_entry.track_json,
            ))
        named_cameras.extend(
            getattr(self, "_virtual_camera_entries", {}).get(primary_shot, [])
        )

        pitch_cfg = self.config.get("pitch", {}) or {}
        manifest = UeManifest(
            schema_version=SCHEMA_VERSION,
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
            cameras=named_cameras,
            kits=kits_map(self._load_match_dict()),
        )
        manifest.save(export_dir / "ue_manifest.json")
        logger.info(
            "[export] wrote ue_manifest.json with %d player(s), ball=%s, camera=%s",
            len(players),
            ball_entry is not None,
            camera_entry is not None,
        )

"""Aggregate per-stage diagnostics into ``output/quality_report.json``.

Called at the end of :func:`src.pipeline.runner.run_pipeline` after all
stages have completed.  Each section (camera, hmr_world, ball) is
independent — missing inputs simply omit that section from the report.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.schemas.anchor import AnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.schemas.shots import ShotsManifest


def _prepare_shots_section(output_dir: Path, manifest: ShotsManifest) -> dict:
    """Ingestion diagnostics: exclusions, groups, alignment confidence."""
    from src.schemas.sync_map import SyncMap

    excluded: dict[str, int] = {}
    for s in manifest.shots:
        if s.excluded:
            reason = s.exclude_reason or "unknown"
            excluded[reason] = excluded.get(reason, 0) + 1

    sync_path = output_dir / "shots" / "sync_map.json"
    sync_map = None
    if sync_path.exists():
        try:
            sync_map = SyncMap.load(sync_path)
        except Exception:
            sync_map = None

    groups: list[dict] = []
    low_confidence: list[str] = []
    for g in manifest.groups:
        alignment: dict = {"min_confidence": None, "methods": []}
        gs = sync_map.group(g.id) if sync_map is not None else None
        if gs is not None and gs.alignments:
            confidences = [a.confidence for a in gs.alignments]
            alignment = {
                "min_confidence": round(min(confidences), 3),
                "methods": sorted({a.method for a in gs.alignments}),
            }
            if min(confidences) < 0.5:
                low_confidence.append(g.id)
        groups.append({
            "id": g.id,
            "label": g.label,
            "shots": len(g.shot_ids),
            "boundary_rule": g.boundary_rule,
            "boundary_confidence": g.boundary_confidence,
            "alignment": alignment,
        })

    return {
        "total_shots": len(manifest.shots),
        "active_shots": len(manifest.active_shots()),
        "excluded": excluded,
        "group_count": len(manifest.groups),
        "groups": groups,
        "low_confidence_groups": low_confidence,
    }


def _ball_shot_entry(track_path: Path, shot_id: str) -> dict:
    """Per-shot ball diagnostics: track stats + solver/anchoring sidecar."""
    ball = BallTrack.load(track_path)
    states = [f.state for f in ball.frames]
    residuals = [s.fit_residual_px for s in ball.flight_segments]
    spin_seg_ids = {
        s.id for s in ball.flight_segments
        if s.parabola.get("spin_omega_rad_s") is not None
    }
    flight_frames = [f for f in ball.frames if f.state == "flight"]
    spin_frames = [
        f for f in flight_frames if f.flight_segment_id in spin_seg_ids
    ]
    entry: dict = {
        "shot_id": shot_id,
        "grounded_frames": states.count("grounded"),
        "flight_frames": states.count("flight"),
        "missing_frames": states.count("missing"),
        "flight_segments": len(ball.flight_segments),
        "mean_flight_fit_residual_px": (
            float(np.mean(residuals)) if residuals else 0.0
        ),
        "max_flight_fit_residual_px": (
            float(np.max(residuals)) if residuals else 0.0
        ),
        "spin_coverage_pct": (
            100.0 * len(spin_frames) / len(flight_frames)
            if flight_frames else 0.0
        ),
    }
    diag_path = track_path.with_name(
        track_path.name.replace("ball_track", "ball_diag")
    )
    if diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text())
        except Exception:
            diag = {}
        events = diag.get("events", [])
        event_counts: dict[str, int] = {}
        for e in events:
            event_counts[e.get("kind", "?")] = (
                event_counts.get(e.get("kind", "?"), 0) + 1
            )
        contact_gaps = [g.get("gap_m", 0.0) for g in diag.get("contact_gaps", [])]
        entry.update({
            "anchors": diag.get("anchors", {}),
            "events": event_counts,
            "underconstrained_spans": diag.get("underconstrained_spans", []),
            "flagged_bounces": [
                b for b in diag.get("bounces", []) if b.get("flagged")
            ],
            "splits": diag.get("splits", 0),
            "max_contact_gap_m": (
                float(np.max(contact_gaps)) if contact_gaps else None
            ),
            "goal_impacts": [
                e for e in events if e.get("kind") == "goal_impact"
            ],
            "detection_coverage": diag.get("detection_coverage"),
            "cross_replay": diag.get("cross_replay"),
            "mode_search": diag.get("mode_search"),
            "out_of_view_spans": diag.get("out_of_view_spans", []),
        })
    return entry


def _ball_section(output_dir: Path, manifest: ShotsManifest | None) -> dict | None:
    """Ball diagnostics across active shots (legacy single-track fallback)."""
    shots: list[tuple[str, Path]] = []
    if manifest is not None:
        for shot in manifest.active_shots():
            p = output_dir / "ball" / f"{shot.id}_ball_track.json"
            if p.exists():
                shots.append((shot.id, p))
    if not shots:
        legacy = output_dir / "ball" / "ball_track.json"
        if legacy.exists():
            shots.append(("", legacy))
    if not shots:
        return None
    entries = [_ball_shot_entry(p, sid) for sid, p in shots]
    return {
        "shots": entries,
        "flight_segments": sum(e["flight_segments"] for e in entries),
        "missing_frames": sum(e["missing_frames"] for e in entries),
        "max_flight_fit_residual_px": max(
            e["max_flight_fit_residual_px"] for e in entries
        ),
        "underconstrained_span_count": sum(
            len(e.get("underconstrained_spans", [])) for e in entries
        ),
        "flagged_bounce_count": sum(
            len(e.get("flagged_bounces", [])) for e in entries
        ),
    }


def write_quality_report(output_dir: Path) -> None:
    """Aggregate diagnostics from camera/, hmr_world/, ball/ into a single JSON."""
    report: dict = {}

    # Match metadata (if set) is mirrored at the top of the report so
    # downstream tooling can answer "which match was this from?" without
    # re-reading the manifest. ``None`` when unset or no manifest yet.
    manifest_path = output_dir / "shots" / "shots_manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = ShotsManifest.load(manifest_path)
        except Exception:
            manifest = None
        if manifest is not None and manifest.match is not None:
            report["match"] = asdict(manifest.match)

    # Highlights-ingestion summary: drop counts, groups, and per-group
    # alignment confidence. Only emitted when the manifest carries
    # ingestion state (groups or exclusions) — flat copy-mode manifests
    # have nothing to review here.
    if manifest is not None and (manifest.groups
                                 or any(s.excluded for s in manifest.shots)):
        report["prepare_shots"] = _prepare_shots_section(output_dir, manifest)

    cam_path = output_dir / "camera" / "camera_track.json"
    anchors_path = output_dir / "camera" / "anchors.json"
    if cam_path.exists() and anchors_path.exists():
        cam = CameraTrack.load(cam_path)
        anchors = AnchorSet.load(anchors_path)
        confs = np.array([f.confidence for f in cam.frames])
        low_mask = confs < 0.6
        ranges: list[list[int]] = []
        i = 0
        while i < len(low_mask):
            if low_mask[i]:
                j = i
                while j < len(low_mask) and low_mask[j]:
                    j += 1
                ranges.append([cam.frames[i].frame, cam.frames[j - 1].frame])
                i = j
            else:
                i += 1

        # Reprojection residual per landmark for each anchor frame, averaged.
        # Uses per-frame t when available (post Phase 1) and applies the
        # clip-shared distortion (post Phase 2) so the metric reflects the
        # actual model the camera stage solved, not a linearised approximation.
        from src.utils.camera_projection import project_world_to_image

        anchor_residuals: list[float] = []
        camera_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        t_world_fallback = np.array(cam.t_world)
        for f in cam.frames:
            t_f = np.array(f.t) if f.t is not None else t_world_fallback
            camera_by_frame[f.frame] = (np.array(f.K), np.array(f.R), t_f)
        distortion = cam.distortion
        for anchor in anchors.anchors:
            if anchor.frame not in camera_by_frame:
                continue
            K, R, t_f = camera_by_frame[anchor.frame]
            world = np.array(
                [lm.world_xyz for lm in anchor.landmarks], dtype=np.float64,
            )
            if len(world) == 0:
                continue
            obs = np.array(
                [lm.image_xy for lm in anchor.landmarks], dtype=np.float64,
            )
            try:
                proj = project_world_to_image(K, R, t_f, distortion, world)
            except Exception:
                continue
            for i in range(len(world)):
                anchor_residuals.append(
                    float(np.linalg.norm(obs[i] - proj[i]))
                )

        # Static-camera contract: with cam.camera_centre set, every frame's
        # -R^T @ t must equal that centre. Surface the worst drift so a
        # regression in the inter-anchor interpolation can't slip past us.
        if cam.camera_centre is not None:
            C = np.asarray(cam.camera_centre, dtype=float)
            drifts: list[float] = []
            for f in cam.frames:
                if f.t is None:
                    continue
                R = np.asarray(f.R, dtype=float)
                t = np.asarray(f.t, dtype=float)
                drifts.append(float(np.linalg.norm(-R.T @ t - C)))
            body_drift_max_m: float | None = max(drifts) if drifts else 0.0
        else:
            body_drift_max_m = None

        report["camera"] = {
            "anchor_count": len(anchors.anchors),
            "low_confidence_frame_count": int(low_mask.sum()),
            "low_confidence_frame_ranges": ranges,
            "mean_anchor_residual_px": (
                float(np.mean(anchor_residuals)) if anchor_residuals else 0.0
            ),
            "body_drift_max_m": body_drift_max_m,
            "distortion": list(cam.distortion),
        }

    hmr_dir = output_dir / "hmr_world"
    if hmr_dir.exists():
        npz_files = sorted(hmr_dir.glob("*_smpl_world.npz"))
        per_player_conf: list[float] = []
        low_players: list[str] = []
        for p in npz_files:
            z = np.load(p)
            mc = float(z["confidence"].mean()) if z["confidence"].size else 0.0
            per_player_conf.append(mc)
            if mc < 0.5:
                low_players.append(str(z["player_id"]))
        report["hmr_world"] = {
            "tracked_players": len(npz_files),
            "mean_per_player_confidence": (
                float(np.mean(per_player_conf)) if per_player_conf else 0.0
            ),
            "low_confidence_players": low_players,
        }

    ball_section = _ball_section(output_dir, manifest)
    if ball_section is not None:
        report["ball"] = ball_section

    refined_summary_path = output_dir / "refined_poses" / "refined_poses_summary.json"
    if refined_summary_path.exists():
        refined_summary = json.loads(refined_summary_path.read_text())
        report["refined_poses"] = refined_summary
        # Lift the cross-player jitter-correction stats to the top
        # level so a dashboard or CI gate can read them without
        # descending into the nested refined-poses payload. Older
        # summaries (pre-jitter) simply lack the inner block and skip.
        jitter = refined_summary.get("jitter")
        if jitter is not None:
            total = int(jitter.get("total_frames_evaluated", 0))
            corrected = int(jitter.get("corrected_frames", 0))
            report["jitter_correction"] = {
                "enabled": bool(jitter.get("enabled", True)),
                "corrected_frames": corrected,
                "total_frames_evaluated": total,
                "fraction_corrected": (corrected / total) if total > 0 else 0.0,
                "max_offset_m": float(jitter.get("max_offset_m", 0.0)),
                "mean_offset_m": float(jitter.get("mean_offset_m", 0.0)),
            }

    out = output_dir / "quality_report.json"
    out.write_text(json.dumps(report, indent=2))

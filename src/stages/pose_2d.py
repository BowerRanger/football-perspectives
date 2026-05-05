"""Stage 4: 2D pose estimation (ViTPose / COCO-17).

Used by ``hmr_world`` for foot anchoring (left/right ankle keypoints) and
by the viewer for 2D overlays.

This module currently provides a *placeholder* implementation: it walks the
shots manifest + tracks and writes one ``pose_2d/{player_id}_pose.json`` per
tracked player with an empty ``frames`` list and a clear runtime warning.
The real ViTPose runner (MMPose) is a future enhancement (D13).

The placeholder keeps downstream stages and tests unblocked: ``hmr_world``
gracefully treats missing pose data as "unanchored" (low-confidence,
zero translation), and the file shape conforms to the schema downstream
expects (``{"player_id": str, "frames": [{"frame": int, "keypoints":
[[x, y, conf], ...]}]}``).

See spec section 5.4 and decisions log D13.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.pipeline.base import BaseStage
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import TracksResult

logger = logging.getLogger(__name__)


class Pose2DStage(BaseStage):
    name = "pose_2d"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        **_: object,
    ) -> None:
        super().__init__(config, output_dir)

    def is_complete(self) -> bool:
        pose_dir = self.output_dir / "pose_2d"
        return pose_dir.exists() and any(pose_dir.glob("*_pose.json"))

    def run(self) -> None:
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"pose_2d requires shots manifest at {manifest_path}"
            )
        manifest = ShotsManifest.load(manifest_path)

        tracks_dir = self.output_dir / "tracks"
        if not tracks_dir.exists():
            raise FileNotFoundError(
                f"pose_2d requires tracks at {tracks_dir} — run tracking first"
            )

        pose_dir = self.output_dir / "pose_2d"
        pose_dir.mkdir(parents=True, exist_ok=True)

        logger.warning(
            "pose_2d stage is using a PLACEHOLDER implementation: empty "
            "keypoints. Real ViTPose / MMPose integration is pending — see "
            "docs/superpowers/decisions/2026-05-05-implementation-decisions.md "
            "(D13) and spec section 5.4."
        )

        written = 0
        for shot in manifest.shots:
            tracks_path = tracks_dir / f"{shot.id}_tracks.json"
            if not tracks_path.exists():
                logger.info("  [SKIP] %s: no tracks file", shot.id)
                continue
            tracks = TracksResult.load(tracks_path)
            for track in tracks.tracks:
                if track.class_name not in ("player", "goalkeeper"):
                    continue
                player_id = track.player_id or track.track_id
                payload = {
                    "player_id": player_id,
                    "shot_id": shot.id,
                    "frames": [],
                }
                out_path = pose_dir / f"{player_id}_pose.json"
                out_path.write_text(json.dumps(payload, indent=2))
                written += 1
        logger.info("pose_2d: wrote %d placeholder pose files", written)

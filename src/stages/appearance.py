"""Appearance stage — post-hoc kit / team evidence from the footage.

Runs AFTER ``ball`` (it needs ``hmr_world`` keypoints, ``refined_poses``
pitch positions and the camera for white balance) and writes SUGGESTIONS
only under ``<out>/appearance/``:

* ``kits.json``              — per-role KitSpecs with provenance
* ``players_suggested.json`` — ``{pid: {"kit_role": ...}}``

It never reads or writes ``players.json`` and never touches tracks
(re-tracking would renumber PIDs); operator labels win downstream via
``player_names.load_kit_roles`` and ``kit_resolution.effective_team_kits``.
The heavy lifting lives in ``src/utils/{appearance_evidence,
team_clustering,kit_palette,appearance_solver}.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas import appearance as schema
from src.schemas.shots import ShotsManifest
from src.utils import kit_palette as kp
from src.utils.appearance_evidence import collect_shot_evidence, player_pitch_x
from src.utils.appearance_solver import solve_appearance
from src.utils.kit_library import load_library, resolve_kit
from src.utils.team_clustering import aggregate_samples, cluster_teams

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "enabled": True,
    "sample_frames_per_shot": 16,
    "min_keypoint_conf": 0.4,
    "white_balance": {"enabled": True, "target": kp.WHITE_TARGET_HEX, "min_line_pixels": 40,
                      "gain_clip": [0.8, 1.25], "exposure": False},
    "clustering": {"keeper_third_m": 35.0, "outlier_min_de": 18.0, "outlier_factor": 3.0},
    "library": {"snap_de": kp.SNAP_DELTA_E, "lightness_weight": kp.LIGHTNESS_WEIGHT},
}


def _merged(cfg: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _match_from_config(raw: dict | None):
    """``appearance.match: {home_team, away_team, date}`` stands in when the
    manifest carries no MatchInfo (clip configs / eval harness)."""
    if not raw:
        return None
    from types import SimpleNamespace

    return SimpleNamespace(home_team=raw.get("home_team"), away_team=raw.get("away_team"),
                           date=raw.get("date"))


class AppearanceStage(BaseStage):
    name = "appearance"

    def __init__(self, config: dict, output_dir: Path, **kwargs) -> None:
        super().__init__(config, output_dir, **kwargs)

    def _out(self) -> Path:
        return schema.appearance_dir(self.output_dir)

    def is_complete(self) -> bool:
        out = self._out()
        return (out / schema.KITS_FILE).exists() and (out / schema.PLAYERS_SUGGESTED_FILE).exists()

    def _clip_kits(self, acfg: dict) -> dict[str, dict]:
        kits: dict[str, dict] = {}
        for role, value in (acfg.get("kits") or {}).items():
            try:
                kits[role] = resolve_kit(value)
            except Exception as exc:  # noqa: BLE001 - bad config entry: warn, ignore
                logger.warning("[appearance] appearance.kits.%s invalid: %s", role, exc)
        return kits

    def run(self) -> None:
        acfg = _merged(self.config.get("appearance") or {})
        if not acfg["enabled"]:
            logger.info("[appearance] disabled")
            return
        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        shots = [s for s in manifest.active_shots()
                 if self.shot_filter in (None, s.id)]
        samples: dict[str, list[dict[str, np.ndarray]]] = {}
        line_px: list[np.ndarray] = []
        for shot in shots:
            ev = collect_shot_evidence(
                self.output_dir, shot,
                frames_per_shot=int(acfg["sample_frames_per_shot"]),
                min_conf=float(acfg["min_keypoint_conf"]),
            )
            logger.info("[appearance] %s: %d frames, %d players, %d line px",
                        shot.id, ev.n_frames, len(ev.samples), len(ev.line_pixels))
            for pid, lst in ev.samples.items():
                samples.setdefault(pid, []).extend(lst)
            line_px.append(ev.line_pixels)
        if not samples:
            logger.warning("[appearance] no keypoint evidence found; nothing written")
            return

        wbc = acfg["white_balance"]
        if wbc.get("enabled", True) and line_px:
            wb = kp.white_balance_gains(
                np.concatenate(line_px), target_hex=wbc["target"],
                min_pixels=int(wbc["min_line_pixels"]), gain_clip=tuple(wbc["gain_clip"]),
                exposure=bool(wbc["exposure"]))
        else:
            wb = kp.IDENTITY_WB

        player_parts = {pid: aggregate_samples(lst) for pid, lst in samples.items()}
        ccfg = acfg["clustering"]
        clustering = cluster_teams(
            player_parts, player_pitch_x(self.output_dir, player_parts),
            pitch_length=float(self.config.get("pitch", {}).get("length_m", 105.0)),
            keeper_third_m=float(ccfg["keeper_third_m"]),
            outlier_min_de=float(ccfg["outlier_min_de"]),
            outlier_factor=float(ccfg["outlier_factor"]),
        )
        result = solve_appearance(
            clustering, player_parts, wb,
            library=load_library(), clip_kits=self._clip_kits(acfg),
            match=manifest.match or _match_from_config(acfg.get("match")),
            snap_de=float(acfg["library"]["snap_de"]),
            lightness_weight=float(acfg["library"]["lightness_weight"]),
        )
        payload = schema.build_kits_payload(
            result.kits, teams=result.teams, clustering=clustering.to_dict(),
            white_balance=wb.to_dict(), needs_confirmation=result.needs_confirmation + clustering.notes,
            shots=[s.id for s in shots])
        out = self._out()
        schema.write_json(out / schema.KITS_FILE, payload)
        schema.write_json(out / schema.PLAYERS_SUGGESTED_FILE,
                          schema.build_players_suggested(result.player_roles))
        logger.info("[appearance] wrote %d kits, %d player roles (%s)", len(result.kits),
                    len(result.player_roles), ", ".join(payload["needs_confirmation"]) or "no flags")

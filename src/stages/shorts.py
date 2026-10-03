"""Shorts stage -- vertical YouTube Shorts from a shot with a goal.

Thin orchestrator: derive moments -> resolve each template (framing-checked on
real virtual-camera tracks) -> render each pass (cached by fingerprint under
``shorts/passes/<fp>/``) -> soundtrack -> compose ``shorts/<shot>_<template>.mp4``
and write ``shorts/<shot>_shorts.json`` (operator block preserved, operator pins
win). Algorithms live in ``src/utils/shorts_*.py`` / ``short_compositor.py``.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from src.pipeline.base import BaseStage
from src.schemas.shorts import (
    effective_moments, empty_sidecar, load_sidecar, merge_regenerated, save_sidecar,
    sidecar_path, template_entry)
from src.schemas.shots import ShotsManifest
from src.utils import render_pass_runner as rpr
from src.utils import shorts_audio
from src.utils.short_compositor import compose
from src.utils.shorts_moments import derive_moments
from src.utils.shorts_pipeline import (
    audio_plan, inputs_digest, make_framing_check, pass_fingerprint, resolve_captions)
from src.utils.shorts_templates import fill_sources, load_template, resolve_template

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES = ("matchday", "keeper", "comic")


class ShortsStage(BaseStage):
    name = "shorts"

    # --- config / selection -------------------------------------------------
    def _cfg(self) -> dict:
        return self.config.get("shorts", {}) or {}

    def _shot_ids(self) -> list[str]:
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if manifest_path.exists():
            ids = [s.id for s in ShotsManifest.load(manifest_path).active_shots()]
        else:
            ids = sorted(p.name.removesuffix("_ball_track.json")
                         for p in (self.output_dir / "ball").glob("*_ball_track.json"))
        if self.shot_filter:
            ids = [i for i in ids if i == self.shot_filter]
        return ids

    def _target_shots(self) -> list[str]:
        want = self._cfg().get("shot", "auto")
        ids = self._shot_ids()
        if want != "auto":
            return [i for i in ids if i == want]
        out = []
        for sid in ids:
            m = derive_moments(self.output_dir, sid)
            if m.get("impact") is not None and m.get("strike") is not None:
                out.append(sid)
        return out

    def _templates(self, operator: dict | None) -> list[str]:
        pinned = (operator or {}).get("templates")
        return list(pinned or self._cfg().get("templates") or DEFAULT_TEMPLATES)

    def _out_mp4(self, shot: str, template: str) -> Path:
        return self.output_dir / "shorts" / f"{shot}_{template}.mp4"

    def is_complete(self) -> bool:
        if self._cfg().get("enabled", True) is False:
            return True
        for shot in self._target_shots():
            side = load_sidecar(sidecar_path(self.output_dir, shot))
            if side is None:
                return False
            if not all(self._out_mp4(shot, t).exists()
                       for t in self._templates(side.get("operator"))):
                return False
        return True

    # --- run ------------------------------------------------------------------
    def run(self) -> None:
        if self._cfg().get("enabled", True) is False:
            logger.info("[shorts] disabled")
            return
        shots = self._target_shots()
        if not shots:
            logger.warning("[shorts] no shot with a goal event (strike + impact) found")
        for shot in shots:
            self._run_shot(shot)

    def _run_shot(self, shot: str) -> None:
        side_path = sidecar_path(self.output_dir, shot)
        existing = load_sidecar(side_path)
        operator = (existing or {}).get("operator")
        moments = effective_moments(derive_moments(self.output_dir, shot), operator)
        quality = rpr.QUALITY_PRESETS[self._cfg().get("quality", "clean")]
        framing_check = make_framing_check(self.output_dir, shot, self.config, quality)
        digest = inputs_digest(self.output_dir, shot)
        generated = {**(existing or empty_sidecar(shot)), "moments": moments,
                     "templates": dict((existing or {}).get("templates") or {})}
        for name in self._templates(operator):
            generated["templates"][name] = self._build_template(
                shot, name, moments, operator, framing_check, quality, digest)
            save_sidecar(side_path, merge_regenerated(existing, generated))

    def _build_template(self, shot, name, moments, operator, framing_check, quality, digest):
        tpl = load_template(name)
        op = operator or {}
        caps = resolve_captions(tpl, (self._cfg().get("captions") or {}).get(name),
                                (op.get("captions") or {}).get(name))
        resolved = resolve_template(
            tpl, moments, framing_check=framing_check, captions_override=caps,
            candidate_pins=(op.get("candidates") or {}).get(name))
        if not resolved["ok"]:
            bad = [s["id"] for s in resolved["slots"] if s["status"] == "unresolved"]
            logger.error("[shorts] %s/%s: unresolved slot(s) %s -- not rendered", shot, name, bad)
            return template_entry(resolved, {"error": f"unresolved slots {bad}"})
        paths, cache = {}, {}
        for spec in resolved["passes"]:
            paths[spec["id"]], cache[spec["id"]] = self._render_cached(shot, spec, quality, digest)
        edl = fill_sources(resolved["edl"], lambda pid: paths[pid])
        resolved = {**resolved, "edl": edl}
        out_mp4 = self._out_mp4(shot, name)
        audio_wav = self._audio(shot, edl, moments, out_mp4)
        info = compose(edl, out_mp4, audio=audio_wav)
        return template_entry(resolved, {
            "mp4": str(out_mp4.relative_to(self.output_dir)),
            "duration_s": info["duration_s"], "audio": audio_wav is not None,
            "pass_cache": cache})

    def _render_cached(self, shot: str, spec: dict, quality: dict, digest: str):
        style = rpr.resolve_style_payload(self.output_dir, shot, self.config, spec.get("style"),
                                          write_sidecar=False)
        fp = pass_fingerprint(spec, shot=shot, quality=quality, digest=digest,
                              style_payload=style)
        root = self.output_dir / "shorts" / "passes" / fp
        marker = root / "pass.json"
        if marker.exists():
            hit = json.loads(marker.read_text()).get("mp4")
            if hit and (root / hit).exists():
                logger.info("[shorts] pass %s cached (%s)", spec["id"], fp)
                return root / hit, {"fingerprint": fp, "cached": True}
        shutil.rmtree(root, ignore_errors=True)
        mp4 = Path(rpr.render_pass(self.output_dir, shot, spec, self.config, quality,
                                   out_dir=root / shot, vertical_only=True))
        marker.write_text(json.dumps({"spec": spec, "mp4": str(mp4.relative_to(root))},
                                     indent=2, default=str))
        return mp4, {"fingerprint": fp, "cached": False}

    def _audio(self, shot, edl, moments, out_mp4: Path) -> Path | None:
        acfg = dict(self._cfg().get("audio") or {})
        if not acfg.pop("enabled", True):
            return None
        src = self.output_dir / "shots" / f"{shot}.mp4"
        try:
            fps = float(rpr._load_broadcast_camera(self.output_dir, shot).fps)
            windows, events, dur = audio_plan(edl, fps, moments)
            return shorts_audio.build_audio(src, windows, events, dur, acfg,
                                            out_path=out_mp4.with_suffix(".wav"))
        except (ValueError, RuntimeError, OSError) as exc:
            logger.warning("[shorts] %s: no soundtrack (%s) -- silent AAC", shot, exc)
            return None

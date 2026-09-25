"""Run the CURRENT, unmodified ball stage (``src.stages.ball.BallStage``)
against both a synthetic evidence world and the real detector, and convert
its dense per-frame output into the PoC's shared ``Track`` contract
(``method="current"``) — see ``CONTRACT.md`` and task brief SPIKE A2.

Two independent modes, both built on the overlay machinery already proven
by ``scripts/eval_ball_accuracy.py`` (imported, not copied):

* ``synthetic`` — :func:`run_synthetic` feeds the stage a
  :class:`~prototypes.ball_hybrid_poc.types.SynthRun` (A1's synthetic
  detector stream) through a :class:`SyntheticDetector` keyed on frame
  CONTENT hash (``CachingBallDetector._key``), never call order, so a
  seek-then-read pass (second_pass/foot_guided/strike_window all seek)
  still gets the right synthetic evidence for whatever frame it actually
  decoded. Crop-based re-detection passes (second-pass zoom, foot-guided
  zoom) are config'd OFF so the synthetic world is never asked to explain
  a real-pixel crop it has no evidence for; the stage's overall shape
  (detect loop -> second pass -> strike window -> solve) still runs
  unmodified.

* ``real`` — :func:`run_real` runs the shipped default config (real WASB
  detector) through the *same* overlay + optional 2-fold holdout split
  used by the sub-20cm accuracy campaign, wrapped in a content-hash
  detection cache seeded from a read-only COPY of the main repo's
  committed cache (never mutates the main-repo file).

Outputs land under ``$M/output-ball-poc/<clip>/`` only (never any other
``$M/output*`` dir — the overlay this module builds symlinks every other
stage's input read-only and writes only its own scratch ``ball/`` dir).

CLI:
    .venv311/bin/python -m prototypes.ball_hybrid_poc.run_current synthetic \\
        --clips gberch --scenarios standin
    .venv311/bin/python -m prototypes.ball_hybrid_poc.run_current real \\
        --clips gberch --folds full,0,1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parents[1]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from scripts.eval_ball_accuracy import _run_fold, build_overlay  # noqa: E402
from src.schemas.ball_anchor import BallAnchor, BallAnchorSet  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.stages.ball import BallStage  # noqa: E402
from src.utils import ball_eval as BE  # noqa: E402
from src.utils.ball_detection_cache import CachingBallDetector  # noqa: E402
from src.utils.ball_detector import BallDetector  # noqa: E402

from .ctx import M, ClipContext, load_clip  # noqa: E402
from .types import (  # noqa: E402
    Observation,
    SynthRun,
    Track,
    TrackFrame,
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = _WORKTREE_ROOT / "config" / "default.yaml"
POC_OUT = Path(M) / "output-ball-poc"
DET_CACHE_SRC_DIR = (
    Path(M) / "docs" / "superpowers" / "notes" / "ball-accuracy" / "det_cache"
)


def _clip_output_dir(clip_id: str) -> Path:
    d = POC_OUT / clip_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _track_to_contract(clip_id: str, track: BallTrack) -> Track:
    """``BallTrack`` (stage output) -> contract ``Track`` (``method="current"``).

    ``mode`` is the stage's own per-frame ``state`` string
    (``"grounded"|"flight"|"occluded"|"missing"``); ``xyz``/``conf`` map
    straight across.
    """
    return Track(
        clip_id=clip_id,
        method="current",
        frames=tuple(
            TrackFrame(frame=f.frame, xyz=f.world_xyz, mode=f.state,
                       conf=f.confidence)
            for f in track.frames
        ),
    )


# ---------------------------------------------------------------------------
# Mode 1: synthetic
# ---------------------------------------------------------------------------

def _synth_config_overrides(config: dict) -> dict:
    """Config overrides for a synthetic run (mutates and returns ``config``).

    Every key verified against ``config/default.yaml`` and
    ``src/stages/ball.py`` (line numbers as of this spike's HEAD):

    - ``ball.second_pass.zoom_min_ball_px: 0`` — the zoom-retry gate at
      ``ball.py`` ~1257 is ``size < zoom_min_ball_px``; with the floor at
      0 an apparent-ball-px estimate (always >= 0) never trips it, so
      ``_zoom_detect`` (which crops real pixels, ~1273-1309) never runs.
    - ``ball.foot_guided.enabled: false`` — ``ball.py`` ~1572-1574 gates
      the whole foot-guided pass (which also crops via ``_zoom_detect``,
      ~1311-1339) on this flag.
    - ``ball.appearance_bridge.enabled: false`` — the bridge
      (``ball.py`` ~1086-1162) reads real frame pixels to template-match
      through a miss; in a synthetic world that would leak the REAL
      ball's appearance into evidence the detector never actually
      reported, so it must be off.

    Left ALONE (both already default-on and neither crops or reads real
    pixels): ``ball.second_pass.enabled`` (full-frame corridor pass,
    ~1207-1271) and ``ball.second_pass.strike_window.enabled`` (full-frame
    low-threshold candidate pass, ~1341-1396 — confirmed by reading
    ``_strike_window_candidates_loop``: it always calls
    ``detector.detect_candidates(frame, ...)`` on the frame ``cap.read()``
    handed it, never a crop). ``ball.context_prior`` is also left as
    shipped.
    """
    config["ball"]["second_pass"]["zoom_min_ball_px"] = 0
    config["ball"]["foot_guided"]["enabled"] = False
    config["ball"]["appearance_bridge"]["enabled"] = False
    return config


SYNTH_CONFIG_OVERRIDES_APPLIED = (
    "ball.second_pass.zoom_min_ball_px=0",
    "ball.foot_guided.enabled=false",
    "ball.appearance_bridge.enabled=false",
)


def _normalize_frame_keyed(d: dict | list | None) -> dict[int, list]:
    """Frame-keyed candidates -> ``{int: [(u, v, score), ...]}``.

    Accepts ``{frame(str|int): [[u,v,score], ...]}`` (JSON round-trip
    yields str keys; an in-memory stand-in may use int) or the flat list
    ``[{frame, uv, score}, ...]`` that ``synth_detector`` writes."""
    out: dict[int, list] = {}
    if isinstance(d, list):
        for c in d:
            u, v = c["uv"]
            out.setdefault(int(c["frame"]), []).append(
                (float(u), float(v), float(c["score"])))
        return out
    for k, v in (d or {}).items():
        out[int(k)] = [tuple(c) for c in v]
    return out


def _anchor_from_dict(d: dict) -> BallAnchor:
    """A loose ``SynthRun.anchors`` dict (BallAnchor-like) -> ``BallAnchor``."""
    raw_xy = d.get("image_xy")
    raw_end = d.get("end_frame")
    return BallAnchor(
        frame=int(d["frame"]),
        image_xy=(tuple(float(x) for x in raw_xy) if raw_xy is not None else None),
        state=d["state"],
        player_id=d.get("player_id"),
        bone=d.get("bone"),
        goal_element=d.get("goal_element"),
        touch_type=d.get("touch_type"),
        spin=d.get("spin"),
        confidence=float(d.get("confidence", 1.0)),
        end_frame=(int(raw_end) if raw_end is not None else None),
        landmark=d.get("landmark"),
    )


def synth_anchor_set(clip_id: str, image_size: tuple[int, int],
                      anchor_dicts: tuple[dict, ...]) -> BallAnchorSet:
    """SynthRun anchors (loose dicts) -> a real ``BallAnchorSet`` for the
    overlay's ``ball/<shot>_ball_anchors.json``. Always uses the clip's
    camera-track ``image_size`` (``ClipContext.image_size``), never
    whatever (possibly bogus — origi01's real file has ``[0, 0]``) value
    might otherwise be floating around."""
    return BallAnchorSet(
        clip_id=clip_id,
        image_size=image_size,
        anchors=tuple(_anchor_from_dict(d) for d in anchor_dicts),
    )


def build_standin_synth_run(ctx: ClipContext, scenario: str = "standin") -> SynthRun:
    """A1's ``synth_obs_<scenario>.json`` may not exist yet. Build a
    stand-in directly from real operator data, ONLY for smoke-testing
    :func:`run_synthetic`'s plumbing (frame-content keying, config
    overrides, contract conversion) before A1 lands: observations = the
    clip's real detector observations (``ClipContext.observations``),
    anchors = the clip's real manual anchors, converted to the loose-dict
    shape ``SynthRun.anchors`` expects. This is NOT a synthetic-truth
    scenario (it still uses real detector evidence) — it exists solely to
    exercise this module's own code paths end to end.
    """
    anchor_dicts = tuple(dataclasses.asdict(a) for a in ctx.anchors.anchors)
    return SynthRun(
        clip_id=ctx.clip_id, scenario=scenario,
        observations=ctx.observations, anchors=anchor_dicts, noise_model={},
    )


class SyntheticDetector(BallDetector):
    """Serves a :class:`SynthRun`'s observations keyed by frame CONTENT,
    not call order.

    The stage's detect loop reads frames sequentially, but second_pass /
    foot_guided / strike_window all ``cap.set(CAP_PROP_POS_FRAMES, ...)``
    seek before reading — so this detector cannot assume the Nth call it
    receives corresponds to the Nth frame of the clip. Instead it decodes
    ``video_path`` once up front, hashes every frame with
    ``CachingBallDetector._key`` (md5 of ``frame[::4,::4]`` + shape — the
    exact key the real detection cache uses), and resolves every
    ``detect``/``detect_candidates`` call back to a frame index by that
    hash.

    Two invariants the caller (``run_synthetic``) enforces after the run:
    ``hash_misses == 0`` (every call's content hash was seen during
    indexing) and ``crop_calls == 0`` (every call passed a full frame,
    never a smaller crop — a crop call would mean the config overrides
    failed to suppress a real-pixel-cropping pass).
    """

    SUPPORTS_REDETECT = True

    def __init__(self, run: SynthRun, video_path: Path | None = None,
                 frames: list[np.ndarray] | None = None) -> None:
        """``video_path`` decodes the real clip (the production path);
        ``frames`` (an explicit, in-order list of frame arrays) is the
        test-only path — see ``tests/test_run_current.py`` — so the
        content-hash keying logic can be unit-tested against tiny fake
        frames without writing a real video file. Exactly one must be
        given."""
        if (video_path is None) == (frames is None):
            raise ValueError("SyntheticDetector: pass exactly one of "
                              "video_path or frames")
        self.hash_misses = 0
        self.crop_calls = 0
        self.positional_fallbacks = 0
        self._expected_next_idx = 0
        self._frame_shape: tuple[int, int] | None = None

        obs_by_frame: dict[int, list[Observation]] = {}
        for o in run.observations:
            obs_by_frame.setdefault(o.frame, []).append(o)
        self._obs_by_frame = obs_by_frame

        noise = run.noise_model or {}
        self._fp_by_frame = _normalize_frame_keyed(noise.get("false_positives"))
        self._weak_by_frame = _normalize_frame_keyed(noise.get("weak_candidates"))

        self._hash_to_frame: dict[str, int] = {}
        self._n_frames = 0
        if video_path is not None:
            self._index_video(video_path)
        else:
            self._index_frames(frames)

    def _index_one(self, idx: int, frame: np.ndarray) -> None:
        if self._frame_shape is None:
            self._frame_shape = frame.shape[:2]
        key = CachingBallDetector._key(frame)
        # First writer wins for an exact pixel-identical repeat frame
        # (e.g. a frozen/still opening) — fine here since SynthRun
        # observations are truth-derived, not appearance dependent, so a
        # duplicate frame legitimately shares the same synthetic evidence.
        self._hash_to_frame.setdefault(key, idx)

    def _index_video(self, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"SyntheticDetector: cannot open {video_path}")
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self._index_one(idx, frame)
                idx += 1
        finally:
            cap.release()
        self._n_frames = idx

    def _index_frames(self, frames: list[np.ndarray]) -> None:
        for idx, frame in enumerate(frames):
            self._index_one(idx, frame)
        self._n_frames = len(frames)

    def _resolve_frame(self, frame_img: np.ndarray) -> int:
        key = CachingBallDetector._key(frame_img)
        idx = self._hash_to_frame.get(key)
        if idx is None:
            # Seek-decode produced content whose hash matches NO frame
            # from our sequential index pass. Every real caller only
            # seeks-then-reads strictly forward within a bounded run, so
            # the nearest-in-the-decoded-GOP recovery is simply "the next
            # frame after whatever we last resolved" — no pixel-similarity
            # metric on the hash itself is needed or possible.
            self.hash_misses += 1
            self.positional_fallbacks += 1
            idx = min(self._expected_next_idx, max(self._n_frames - 1, 0))
            logger.warning(
                "SyntheticDetector: hash miss (frame content unseen during "
                "indexing) — falling back to positional frame %d", idx,
            )
        self._expected_next_idx = idx + 1
        return idx

    def _is_full_frame(self, frame_img: np.ndarray) -> bool:
        shape = frame_img.shape[:2]
        if self._frame_shape is not None and shape != self._frame_shape:
            self.crop_calls += 1
            return False
        return True

    def detect(self, frame: np.ndarray):
        if not self._is_full_frame(frame):
            return None
        idx = self._resolve_frame(frame)
        obs_list = self._obs_by_frame.get(idx)
        if not obs_list:
            return None
        best = max(obs_list, key=lambda o: o.conf)
        return (float(best.uv[0]), float(best.uv[1]), float(best.conf))

    def detect_candidates(self, frame: np.ndarray, min_score: float,
                           top_k: int = 5):
        if not self._is_full_frame(frame):
            return []
        idx = self._resolve_frame(frame)
        cands: list[tuple[float, float, float]] = []
        for o in self._obs_by_frame.get(idx, []):
            cands.append((float(o.uv[0]), float(o.uv[1]), float(o.conf)))
        cands.extend(
            (float(c[0]), float(c[1]), float(c[2]))
            for c in self._fp_by_frame.get(idx, [])
        )
        cands.extend(
            (float(c[0]), float(c[1]), float(c[2]))
            for c in self._weak_by_frame.get(idx, [])
        )
        cands = [c for c in cands if c[2] >= min_score]
        cands.sort(key=lambda c: -c[2])
        return cands[:top_k]

    def reset(self) -> None:
        return None


def run_synthetic(clip_id: str, scenario: str) -> Track:
    """Run the current ball stage against ``SynthRun`` ``scenario`` for
    ``clip_id``; writes and returns the contract ``Track``.

    ``scenario="standin"`` builds its own stand-in run (see
    :func:`build_standin_synth_run`) when A1's file isn't there yet;
    every other scenario name requires
    ``$M/output-ball-poc/<clip>/synth_obs_<scenario>.json`` to exist.
    """
    ctx = load_clip(clip_id)
    out_dir = _clip_output_dir(clip_id)
    synth_path = out_dir / f"synth_obs_{scenario}.json"

    if synth_path.exists():
        run = SynthRun.from_json(load_json(synth_path))
    elif scenario == "standin":
        run = build_standin_synth_run(ctx, scenario=scenario)
        save_json(synth_path, run)
    else:
        raise FileNotFoundError(
            f"no SynthRun at {synth_path} (A1 hasn't produced scenario "
            f"{scenario!r} for {clip_id!r} yet); use scenario='standin' "
            "for a self-built smoke-test stand-in"
        )

    config = _synth_config_overrides(_load_config())
    anchors = synth_anchor_set(clip_id, ctx.image_size, run.anchors)
    detector = SyntheticDetector(run, ctx.video_path)

    with tempfile.TemporaryDirectory(prefix=f"ball_poc_synth_{clip_id}_") as tmp:
        overlay = build_overlay(ctx.output_dir, Path(tmp), ctx.shot_id, anchors)
        clip_path = overlay / "shots" / f"{ctx.shot_id}.mp4"
        cam_path = overlay / "camera" / f"{ctx.shot_id}_camera_track.json"
        track_out = overlay / "ball" / f"{ctx.shot_id}_ball_track.json"
        stage = BallStage(config=config, output_dir=overlay, ball_detector=detector)
        t0 = time.time()
        stage._run_shot(ctx.shot_id, clip_path, cam_path, track_out,
                         config["ball"], detector)
        elapsed = time.time() - t0
        track = BallTrack.load(track_out)
        _persist_auto_anchors(overlay, ctx.shot_id,
                              out_dir / f"auto_anchors_current_{scenario}.json")

    logger.info(
        "run_synthetic(%s, %s): %d frames in %.1fs, hash_misses=%d "
        "crop_calls=%d positional_fallbacks=%d, overrides=%s",
        clip_id, scenario, len(track.frames), elapsed, detector.hash_misses,
        detector.crop_calls, detector.positional_fallbacks,
        ",".join(SYNTH_CONFIG_OVERRIDES_APPLIED),
    )
    if detector.hash_misses > 0 or detector.crop_calls > 0:
        raise RuntimeError(
            f"run_synthetic({clip_id!r}, {scenario!r}): synthetic-world "
            f"invariant violated — hash_misses={detector.hash_misses} "
            f"crop_calls={detector.crop_calls} "
            f"(positional_fallbacks={detector.positional_fallbacks}); "
            "the stage asked the synthetic detector to explain a real "
            "pixel crop or decoded content it never indexed"
        )

    out = _track_to_contract(clip_id, track)
    save_json(out_dir / f"track_current_{scenario}.json", out)
    return out


def _persist_auto_anchors(overlay: Path, shot_id: str, dst: Path) -> None:
    """Keep the stage's auto-event sidecar before the temp overlay is
    deleted: the hybrid's ``+events`` variant consumes the current stage's
    auto events (method evidence, never truth) as extra knots."""
    src = Path(overlay) / "ball" / f"{shot_id}_ball_anchors_auto.json"
    if src.exists():
        shutil.copyfile(src, dst)
    else:
        logger.warning("no auto-anchor sidecar at %s", src)


# ---------------------------------------------------------------------------
# Mode 2: real
# ---------------------------------------------------------------------------

def _det_cache_copy(clip_id: str) -> Path:
    """Copy-on-first-use of the main repo's committed det cache into this
    clip's PoC output dir; the main-repo file is never opened for
    writing. Reused (and grown) across a clip's full/fold0/fold1 runs."""
    dst = _clip_output_dir(clip_id) / "det_cache.json"
    if not dst.exists():
        src = DET_CACHE_SRC_DIR / f"{clip_id}.json"
        if src.exists():
            shutil.copyfile(src, dst)
        else:
            dst.write_text(json.dumps({"detect": {}, "candidates": {}}))
    return dst


def dry_run_cache_hit_rate(clip_id: str) -> dict:
    """Cheap pre-flight (no detector invocation): fraction of the clip's
    frames whose content hash already has a ``detect`` cache entry in the
    copied cache. Only estimates the ``detect`` cache (the ``candidates``
    cache's keys are suffixed by whichever ``min_score``/``top_k`` a given
    pass used, so a call-independent estimate isn't cheap to compute)."""
    ctx = load_clip(clip_id)
    cache_path = _det_cache_copy(clip_id)
    cache = json.loads(cache_path.read_text())
    known = set(cache.get("detect", {}).keys())
    cap = cv2.VideoCapture(str(ctx.video_path))
    total = 0
    hits = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total += 1
            if CachingBallDetector._key(frame) in known:
                hits += 1
    finally:
        cap.release()
    rate = (hits / total) if total else 0.0
    return {
        "clip_id": clip_id,
        "total_frames": total,
        "detect_cache_hits": hits,
        "detect_cache_hit_rate": rate,
        "cache_detect_entries": len(known),
    }


def run_real(clip_id: str, fold: Optional[int]) -> Track:
    """Run the current ball stage with the real (default-config, WASB)
    detector for ``clip_id``. ``fold=None`` uses every manual anchor
    ("operational"); ``fold in (0, 1)`` uses a 2-fold holdout split
    (``src.utils.ball_eval.split_anchors``, identical semantics/call to
    ``scripts/eval_ball_accuracy.py``) and records the split to
    ``real_split_fold{fold}.json`` so the hybrid extractor can be run on
    the identical held-out set."""
    ctx = load_clip(clip_id)
    out_dir = _clip_output_dir(clip_id)
    config = _load_config()  # unmodified — the real stage exactly as shipped
    det_cache = _det_cache_copy(clip_id)

    kept_set: BallAnchorSet | None = None
    label = "full"
    if fold is not None:
        kept, held = BE.split_anchors(ctx.anchors.anchors, fold=fold, n_folds=2)
        kept_set = dataclasses.replace(ctx.anchors, anchors=tuple(kept))
        label = f"fold{fold}"
        save_json(out_dir / f"real_split_fold{fold}.json", {
            "clip_id": clip_id, "fold": fold, "n_folds": 2,
            "kept_frames": sorted(a.frame for a in kept),
            "heldout_frames": sorted(a.frame for a in held),
        })

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"ball_poc_real_{clip_id}_") as tmp:
        overlay, track_out = _run_fold(
            ctx.output_dir, ctx.shot_id, config, "wasb", kept_set, Path(tmp),
            det_cache=det_cache,
        )
        track = BallTrack.load(track_out)
        _persist_auto_anchors(overlay, ctx.shot_id,
                              out_dir / f"auto_anchors_current_real_{label}.json")
    elapsed = time.time() - t0

    logger.info("run_real(%s, %s): %d frames in %.1fs", clip_id, label,
                len(track.frames), elapsed)
    out = _track_to_contract(clip_id, track)
    save_json(out_dir / f"track_current_real_{label}.json", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    sp_syn = sub.add_parser("synthetic")
    sp_syn.add_argument("--clips", required=True,
                         help="comma-separated clip ids")
    sp_syn.add_argument("--scenarios", required=True,
                         help="comma-separated scenario names ('standin' "
                              "self-builds a smoke-test stand-in)")

    sp_real = sub.add_parser("real")
    sp_real.add_argument("--clips", required=True)
    sp_real.add_argument("--folds", default="full,0,1",
                          help="comma-separated: 'full' and/or '0'/'1' "
                               "(2-fold ids)")
    sp_real.add_argument("--dry-run-cache", action="store_true",
                          help="only print detect-cache hit rate; no stage run")

    args = ap.parse_args(argv)
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]

    if args.mode == "synthetic":
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        for clip_id in clips:
            for scenario in scenarios:
                run_synthetic(clip_id, scenario)
        return

    if args.dry_run_cache:
        for clip_id in clips:
            print(json.dumps(dry_run_cache_hit_rate(clip_id)))
        return

    folds_raw = [f.strip() for f in args.folds.split(",") if f.strip()]
    folds: list[Optional[int]] = [None if f == "full" else int(f) for f in folds_raw]
    for clip_id in clips:
        for fold in folds:
            run_real(clip_id, fold)


if __name__ == "__main__":
    main()

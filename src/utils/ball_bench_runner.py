"""Run the real, unmodified ball stage (``src.stages.ball.BallStage``)
against synthetic evidence or the real WASB detector, and convert its
dense per-frame output into the bench's shared ``Track`` contract.

Promoted from the ``ball-hybrid-integration`` spike
(``prototypes/ball_hybrid_poc/run_current.py`` + the overlay/fold helpers
of ``scripts/eval_ball_accuracy.py``, reproduced here rather than imported
so ``src/utils`` never depends on the top-level ``scripts/`` package).

Two independent evidence sources, both built on an OVERLAY of the clip's
real output dir (every stage input symlinked; only ``ball/`` replaced with
a real dir holding whatever anchors this run uses) so the real outputs are
never touched:

* ``run_synthetic`` — feeds the stage a
  :class:`~src.utils.ball_bench_types.SynthRun` (``ball_bench_synth.py``'s
  synthetic detector stream) through :class:`SyntheticDetector`, keyed on
  frame CONTENT hash (``CachingBallDetector._key``), never call order, so
  a seek-then-read pass (second_pass/foot_guided/strike_window all seek)
  still gets the right synthetic evidence for whatever frame it actually
  decoded. Crop-based re-detection passes (second-pass zoom, foot-guided
  zoom, appearance bridge) are config'd OFF (see
  ``SYNTH_CONFIG_OVERRIDES``) so the synthetic world is never asked to
  explain a real-pixel crop it has no evidence for; the stage's overall
  shape (detect loop -> second pass -> strike window -> solve) still runs
  unmodified.

* ``run_real`` — runs the shipped ball stage with the real WASB detector,
  wrapped in a content-hash detection cache seeded from a read-only COPY
  of ``$M/output-ball-poc/<clip>/det_cache.json`` (falling back to the
  sub-20cm campaign's committed cache at
  ``$M/docs/superpowers/notes/ball-accuracy/det_cache/<clip>.json`` on
  first use) — the main repo's ``docs/`` copy is never opened for writing.

Neither entry point edits ``src/stages/ball.py`` or ``config/default.yaml``
— every behaviour change (trajectory mode, synth-world overrides, ad hoc
``--set`` overrides) is an in-memory mutation of the config dict loaded
from ``config/default.yaml``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.stages.ball import BallStage, _build_detector
from src.utils import ball_eval as BE
from src.utils.ball_detection_cache import CachingBallDetector
from src.utils.ball_detector import BallDetector

from .ball_bench_clip import M, ClipContext
from .ball_bench_types import Observation, SynthRun, Track, TrackFrame

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
BENCH_OUT_ROOT = Path(M) / "output-ball-poc"
DET_CACHE_FALLBACK_DIR = (
    Path(M) / "docs" / "superpowers" / "notes" / "ball-accuracy" / "det_cache"
)

DEFAULT_TRAJECTORY = "reference"
N_FOLDS = 2

# Overlay build: top-level entries never linked in (ball outputs are
# replaced by the overlay's own ball/; render/export/logs are heavyweight
# or must not leak stage logging into the source dir).
_SKIP_PREFIXES = ("ball",)
_SKIP_NAMES = ("renders", "logs", "export")

# Config overrides applied for every synthetic-evidence run. Every key
# verified against config/default.yaml and src/stages/ball.py:
#
# - ball.second_pass.zoom_min_ball_px: 0 — the zoom-retry gate is
#   `size < zoom_min_ball_px`; with the floor at 0 an apparent-ball-px
#   estimate (always >= 0) never trips it, so `_zoom_detect` (which crops
#   real pixels) never runs.
# - ball.foot_guided.enabled: false — gates the whole foot-guided pass
#   (which also crops via `_zoom_detect`) on this flag.
# - ball.appearance_bridge.enabled: false — the bridge reads real frame
#   pixels to template-match through a miss; in a synthetic world that
#   would leak the REAL ball's appearance into evidence the detector
#   never actually reported, so it must be off.
#
# Left ALONE (both already default-on and neither crops nor reads real
# pixels): ball.second_pass.enabled (full-frame corridor pass) and
# ball.second_pass.strike_window.enabled (full-frame low-threshold
# candidate pass — always calls detect_candidates on the whole decoded
# frame, never a crop). ball.context_prior is also left as shipped.
SYNTH_CONFIG_OVERRIDES = (
    ("ball.second_pass.zoom_min_ball_px", 0),
    ("ball.foot_guided.enabled", False),
    ("ball.appearance_bridge.enabled", False),
)


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------

def load_base_config() -> dict:
    """Fresh copy of ``config/default.yaml`` — never mutated in place by
    the module-level constant, so every caller gets an independent dict."""
    return yaml.safe_load(CONFIG_PATH.read_text())


def _set_dotted(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = config
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(
                f"cannot set {dotted_key!r}: {p!r} is not a dict in config")
    node[parts[-1]] = value


def apply_synth_overrides(config: dict) -> dict:
    """Mutate and return ``config`` with :data:`SYNTH_CONFIG_OVERRIDES`
    applied — required whenever the evidence source is synthetic."""
    for key, value in SYNTH_CONFIG_OVERRIDES:
        _set_dotted(config, key, value)
    return config


def apply_trajectory(config: dict, trajectory: str) -> dict:
    """Mutate and return ``config`` with ``ball.trajectory`` set.

    Forward-compatible with IC-A's ``ball.trajectory: reference|hybrid``
    config switch (not yet landed as of this module's authorship — until
    it lands, ``BallStage`` simply ignores the unknown key and every
    trajectory produces the same reference behaviour). Never edits
    ``config/default.yaml`` itself.
    """
    _set_dotted(config, "ball.trajectory", trajectory)
    return config


def apply_set_overrides(config: dict, overrides: list[str]) -> dict:
    """Mutate and return ``config`` with ``--set key.path=value``
    overrides applied. Values are parsed with ``yaml.safe_load`` so
    ``true``/``1.5``/``"a string"`` all come through as the right Python
    type; a bare unquoted word becomes a string."""
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"--set override must be key=value, got {raw!r}")
        key, _, value_str = raw.partition("=")
        value = yaml.safe_load(value_str)
        _set_dotted(config, key.strip(), value)
    return config


def build_config(*, trajectory: str, synthetic: bool,
                  overrides: Optional[list[str]] = None) -> dict:
    """Assemble one run's config: base -> (synth overrides if
    ``synthetic``) -> trajectory -> ``--set`` overrides (applied last so
    an operator override always wins)."""
    config = load_base_config()
    if synthetic:
        apply_synth_overrides(config)
    apply_trajectory(config, trajectory)
    if overrides:
        apply_set_overrides(config, overrides)
    return config


# ---------------------------------------------------------------------------
# overlay + BallTrack -> Track conversion
# ---------------------------------------------------------------------------

def build_overlay(src_output: Path, tmp_root: Path, shot_id: str,
                   kept: BallAnchorSet | None) -> Path:
    """Create the overlay dir: symlinked inputs + a real ``ball/``
    (holding ``kept`` if given, else a copy of the clip's real anchors)."""
    src_output = Path(src_output)
    ov = Path(tmp_root) / "overlay"
    ov.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src_output.iterdir()):
        name = entry.name
        if name in _SKIP_NAMES or any(
                name == p or name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        link = ov / name
        if not link.exists():
            link.symlink_to(entry.resolve())
    (ov / "logs").mkdir(exist_ok=True)
    (ov / "ball").mkdir(exist_ok=True)
    anchors_src = src_output / "ball" / f"{shot_id}_ball_anchors.json"
    dst = ov / "ball" / f"{shot_id}_ball_anchors.json"
    if kept is not None:
        kept.save(dst)
    elif anchors_src.exists():
        dst.write_text(anchors_src.read_text())
    return ov


def _track_to_contract(clip_id: str, method: str, track: BallTrack) -> Track:
    """``BallTrack`` (stage output) -> bench ``Track``.

    ``mode`` is the stage's own per-frame ``state`` string
    (``"grounded"|"flight"|"occluded"|"missing"``); ``xyz``/``conf`` map
    straight across.
    """
    return Track(
        clip_id=clip_id,
        method=method,
        frames=tuple(
            TrackFrame(frame=f.frame, xyz=f.world_xyz, mode=f.state,
                       conf=f.confidence)
            for f in track.frames
        ),
    )


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
    """``SynthRun`` anchors (loose dicts) -> a real ``BallAnchorSet`` for
    the overlay's ``ball/<shot>_ball_anchors.json``. Always uses the
    clip's camera-track ``image_size``, never whatever value might
    otherwise be floating around in a stale anchors file."""
    return BallAnchorSet(
        clip_id=clip_id,
        image_size=image_size,
        anchors=tuple(_anchor_from_dict(d) for d in anchor_dicts),
    )


# ---------------------------------------------------------------------------
# SyntheticDetector: content-hash-keyed BallDetector over a SynthRun
# ---------------------------------------------------------------------------

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
        test-only path, so the content-hash keying logic can be
        unit-tested against tiny fake frames without writing a real video
        file. Exactly one must be given."""
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


def _normalize_frame_keyed(d: dict | list | None) -> dict[int, list]:
    """Frame-keyed candidates -> ``{int: [(u, v, score), ...]}``.

    Accepts ``{frame(str|int): [[u,v,score], ...]}`` (JSON round-trip
    yields str keys; an in-memory stand-in may use int) or the flat list
    ``[{frame, uv, score}, ...]`` that ``ball_bench_synth`` writes."""
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


class SyntheticWorldViolation(RuntimeError):
    """Raised when the stage asked the synthetic detector to explain a
    real pixel crop or decoded content it never indexed."""


# ---------------------------------------------------------------------------
# run_synthetic
# ---------------------------------------------------------------------------

def run_synthetic(clip_ctx: ClipContext, synth: SynthRun, scenario: str,
                   trajectory: str = DEFAULT_TRAJECTORY,
                   overrides: Optional[list[str]] = None) -> Track:
    """Run the real ball stage against ``synth`` for ``clip_ctx``'s clip
    and convert its output into a bench ``Track`` named ``trajectory``."""
    config = build_config(trajectory=trajectory, synthetic=True,
                           overrides=overrides)
    anchors = synth_anchor_set(clip_ctx.clip_id, clip_ctx.image_size,
                                synth.anchors)
    detector = SyntheticDetector(synth, clip_ctx.video_path)

    with tempfile.TemporaryDirectory(
            prefix=f"ball_bench_synth_{clip_ctx.clip_id}_") as tmp:
        overlay = build_overlay(clip_ctx.output_dir, Path(tmp),
                                 clip_ctx.shot_id, anchors)
        clip_path = overlay / "shots" / f"{clip_ctx.shot_id}.mp4"
        cam_path = overlay / "camera" / f"{clip_ctx.shot_id}_camera_track.json"
        track_out = overlay / "ball" / f"{clip_ctx.shot_id}_ball_track.json"
        stage = BallStage(config=config, output_dir=overlay,
                           ball_detector=detector)
        t0 = time.time()
        stage._run_shot(clip_ctx.shot_id, clip_path, cam_path, track_out,
                         config["ball"], detector)
        elapsed = time.time() - t0
        track = BallTrack.load(track_out)

    logger.info(
        "run_synthetic(%s, %s, trajectory=%s): %d frames in %.1fs, "
        "hash_misses=%d crop_calls=%d positional_fallbacks=%d",
        clip_ctx.clip_id, scenario, trajectory, len(track.frames), elapsed,
        detector.hash_misses, detector.crop_calls, detector.positional_fallbacks,
    )
    if detector.hash_misses > 0 or detector.crop_calls > 0:
        raise SyntheticWorldViolation(
            f"run_synthetic({clip_ctx.clip_id!r}, {scenario!r}): synthetic-"
            f"world invariant violated — hash_misses={detector.hash_misses} "
            f"crop_calls={detector.crop_calls} "
            f"(positional_fallbacks={detector.positional_fallbacks})")

    return _track_to_contract(clip_ctx.clip_id, trajectory, track)


# ---------------------------------------------------------------------------
# real detector + 2-fold held-out evaluation
# ---------------------------------------------------------------------------

def det_cache_path(clip_id: str, bench_root: Optional[Path] = None) -> Path:
    """Copy-on-first-use of a det cache into
    ``<bench_root>/<clip_id>/det_cache.json``: the main repo's committed
    ``docs/.../det_cache/<clip_id>.json`` is only ever a READ source; this
    function never opens it for writing. Reused (and grown) across a
    clip's real runs — every caller sharing ``bench_root`` shares the
    cache."""
    root = Path(bench_root) if bench_root is not None else BENCH_OUT_ROOT
    clip_dir = root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    dst = clip_dir / "det_cache.json"
    if not dst.exists():
        src = DET_CACHE_FALLBACK_DIR / f"{clip_id}.json"
        if src.exists():
            shutil.copyfile(src, dst)
            logger.info("det_cache_path(%s): seeded from %s", clip_id, src)
        else:
            dst.write_text(json.dumps({"detect": {}, "candidates": {}}))
            logger.info("det_cache_path(%s): no fallback cache at %s; "
                        "starting empty", clip_id, src)
    return dst


def dry_run_cache_hit_rate(clip_ctx: ClipContext, cache_path: Path) -> dict:
    """Cheap pre-flight (no detector invocation): fraction of the clip's
    frames whose content hash already has a ``detect`` cache entry.
    Logged by the CLI/capture script so a slow real run's cache-hit rate
    is visible before the expensive stage run starts."""
    cache = json.loads(cache_path.read_text())
    known = set(cache.get("detect", {}).keys())
    cap = cv2.VideoCapture(str(clip_ctx.video_path))
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
        "clip_id": clip_ctx.clip_id,
        "total_frames": total,
        "detect_cache_hits": hits,
        "detect_cache_hit_rate": rate,
        "cache_detect_entries": len(known),
    }


def get_split(clip_ctx: ClipContext, fold: int, bench_root: Optional[Path] = None,
              n_folds: int = N_FOLDS) -> tuple[tuple, tuple]:
    """Returns ``(kept_anchors, held_anchors)`` for ``fold``, persisting
    the split once to ``<bench_root>/<clip_id>/real_split_fold{fold}.json``
    so every caller against the same clip agrees on the same frames."""
    root = Path(bench_root) if bench_root is not None else BENCH_OUT_ROOT
    clip_dir = root / clip_ctx.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    path = clip_dir / f"real_split_fold{fold}.json"
    all_anchors = clip_ctx.anchors.anchors
    if path.exists():
        data = json.loads(path.read_text())
        kept_frames = set(int(f) for f in data.get("kept_frames", []))
        held_frames = set(int(f) for f in data.get("heldout_frames", []))
        kept = tuple(a for a in all_anchors if a.frame in kept_frames)
        held = tuple(a for a in all_anchors if a.frame in held_frames)
        return kept, held
    kept, held = BE.split_anchors(all_anchors, fold=fold, n_folds=n_folds)
    path.write_text(json.dumps({
        "kept_frames": sorted(a.frame for a in kept),
        "heldout_frames": sorted(a.frame for a in held),
    }, indent=1))
    return kept, held


def run_real(clip_ctx: ClipContext, trajectory: str = DEFAULT_TRAJECTORY,
             fold: Optional[int] = None, overrides: Optional[list[str]] = None,
             det_cache: Optional[Path] = None,
             bench_root: Optional[Path] = None) -> Track:
    """Run the real ball stage with the real (WASB) detector.

    ``fold=None`` uses every manual anchor ("operational"); ``fold in (0,
    1)`` uses the 2-fold holdout split from :func:`get_split`.
    """
    config = build_config(trajectory=trajectory, synthetic=False,
                           overrides=overrides)
    cache_path = det_cache or det_cache_path(clip_ctx.clip_id, bench_root)

    kept_set: BallAnchorSet | None = None
    if fold is not None:
        kept, _held = get_split(clip_ctx, fold, bench_root)
        kept_set = dataclasses.replace(clip_ctx.anchors, anchors=tuple(kept))

    det = _build_detector(config["ball"])
    det = CachingBallDetector(det, cache_path)

    t0 = time.time()
    with tempfile.TemporaryDirectory(
            prefix=f"ball_bench_real_{clip_ctx.clip_id}_") as tmp:
        overlay = build_overlay(clip_ctx.output_dir, Path(tmp),
                                 clip_ctx.shot_id, kept_set)
        clip = overlay / "shots" / f"{clip_ctx.shot_id}.mp4"
        cam_path = overlay / "camera" / f"{clip_ctx.shot_id}_camera_track.json"
        track_out = overlay / "ball" / f"{clip_ctx.shot_id}_ball_track.json"
        stage = BallStage(config=config, output_dir=overlay, ball_detector=det)
        stage._run_shot(clip_ctx.shot_id, clip, cam_path, track_out,
                         config["ball"], det)
        det.save()
        track = BallTrack.load(track_out)
    elapsed = time.time() - t0

    logger.info("run_real(%s, trajectory=%s, fold=%s): %d frames in %.1fs",
                clip_ctx.clip_id, trajectory, fold, len(track.frames), elapsed)
    return _track_to_contract(clip_ctx.clip_id, trajectory, track)


def anchor_heldout_error(clip_ctx: ClipContext, track: Track, fold: int,
                          bench_root: Optional[Path] = None) -> dict:
    """3-D error at the fold's held-out anchor frames, mirroring
    ``scripts/eval_ball_accuracy.py``'s anchor grading (3-D error where
    ground truth is known, else the ray-lateral distance as a
    necessarily-optimistic lower bound)."""
    _kept, held = get_split(clip_ctx, fold, bench_root)
    held_frames = frozenset(a.frame for a in held)
    cams = {f: (clip_ctx.per_frame_K[f], clip_ctx.per_frame_R[f],
                clip_ctx.per_frame_t[f]) for f in clip_ctx.frames}
    world = {tf.frame: tf.xyz for tf in track.frames if tf.xyz is not None}
    ball_radius = 0.11

    def _joint_world_fn(frame, player_id, bone):
        try:
            return clip_ctx.player_context().joint_world(frame, player_id, bone)
        except Exception:  # noqa: BLE001
            return None

    rows = BE.eval_rows_at_anchors(
        world, clip_ctx.anchors.anchors, cams, ball_radius=ball_radius,
        distortion=clip_ctx.distortion, joint_world_fn=_joint_world_fn,
        held_out_frames=held_frames,
        evidence_frames=frozenset(o.frame for o in clip_ctx.observations))
    held_rows = [r for r in rows if r.held_out]
    errs = [(r.err_3d_m if r.err_3d_m is not None else r.lateral_m)
            for r in held_rows]
    errs = [e for e in errs if e is not None]
    return {
        "n": len(held_rows),
        "p50": float(np.percentile(errs, 50)) if errs else None,
        "p95": float(np.percentile(errs, 95)) if errs else None,
        "errs": errs,
    }


__all__ = [
    "REPO_ROOT", "CONFIG_PATH", "BENCH_OUT_ROOT", "DET_CACHE_FALLBACK_DIR",
    "DEFAULT_TRAJECTORY", "N_FOLDS", "SYNTH_CONFIG_OVERRIDES",
    "load_base_config", "apply_synth_overrides", "apply_trajectory",
    "apply_set_overrides", "build_config", "build_overlay",
    "synth_anchor_set", "SyntheticDetector", "SyntheticWorldViolation",
    "run_synthetic", "det_cache_path", "dry_run_cache_hit_rate",
    "get_split", "run_real", "anchor_heldout_error",
]

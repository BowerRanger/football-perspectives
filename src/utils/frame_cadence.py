"""Frame cadence of a shot video: repeated frames and content time.

Broadcast clips delivered at 30 fps are often 25 fps content with a
pulldown: one display frame in six repeats its predecessor. Frame ``f``
then does not show the instant ``f / fps`` — the true content instant runs
ahead of it by up to most of a frame and snaps back at each repeat. For a
ball moving at 20 m/s that sawtooth is worth tens of centimetres, so any
physics fit over display frames needs the content time.

``content_time_shift`` returns, per display frame, ``content_time -
display_time`` in seconds, detrended to be locally zero-mean (the sync map
aligned shots at the display level, so only the sawtooth is corrected).
With no repeats the shift is exactly zero and every consumer reduces to the
uniform ``f / fps`` model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Mean absolute grey-level difference (0-255, on a <=480 px wide proxy)
# below which a frame counts as a repeat of its predecessor. Real motion,
# even on a static wide shot, measures several units; codec noise on a true
# repeat measures ~0.
REPEAT_DIFF_MAX = 0.6
_PROXY_WIDTH = 480


@dataclass(frozen=True)
class Cadence:
    n_frames: int
    repeats: tuple[int, ...]


def detect_repeats(video_path: Path, diff_max: float = REPEAT_DIFF_MAX) -> list[int]:
    """Display frames whose image repeats the previous frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video_path}")
    repeats: list[int] = []
    prev: np.ndarray | None = None
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            if w > _PROXY_WIDTH:
                img = cv2.resize(img, (_PROXY_WIDTH, max(1, round(h * _PROXY_WIDTH / w))),
                                 interpolation=cv2.INTER_AREA)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
            if prev is not None and float(np.abs(grey - prev).mean()) < diff_max:
                repeats.append(i)
            prev = grey
            i += 1
    finally:
        cap.release()
    return repeats


def _count_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def load_or_detect(video_path: Path, cache_dir: Path) -> Cadence:
    """Cached ``Cadence`` for a video (keyed on its size + mtime)."""
    st = video_path.stat()
    key = {"size": st.st_size, "mtime": st.st_mtime}
    cache = cache_dir / f"{video_path.stem}.json"
    if cache.exists():
        try:
            raw = json.loads(cache.read_text())
            if raw.get("key") == key:
                return Cadence(int(raw["n_frames"]), tuple(int(f) for f in raw["repeats"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("frame cadence: ignoring bad cache %s (%s)", cache, exc)
    repeats = detect_repeats(video_path)
    cad = Cadence(_count_frames(video_path), tuple(repeats))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"key": key, "n_frames": cad.n_frames,
                               "repeats": list(cad.repeats)}))
    tmp.replace(cache)
    return cad


def content_time_shift(n_frames: int, repeats: list[int] | tuple[int, ...], fps: float,
                       window_s: float = 1.0) -> np.ndarray:
    """Per display frame: content time minus ``f / fps`` (seconds).

    Fresh frames advance the content clock by one source frame
    (``1 / source_fps``, source_fps = fps x fresh/total); repeats hold it.
    The result is detrended by a centred moving mean over ``window_s`` so it
    carries only the local sawtooth, never a drift.
    """
    n = int(n_frames)
    if n <= 0:
        return np.zeros(0)
    fresh = np.ones(n, dtype=bool)
    reps = [int(f) for f in repeats if 0 < int(f) < n]
    if not reps:
        return np.zeros(n)
    fresh[reps] = False
    src_index = np.cumsum(fresh) - 1
    src_fps = fps * fresh.sum() / n
    raw = src_index / src_fps - np.arange(n) / fps
    win = max(1, int(round(window_s * fps)))
    if win >= n:
        return raw - raw.mean()
    full = np.convolve(raw, np.ones(win) / win, mode="valid")  # len n - win + 1
    lead = (win - 1) // 2
    trend = np.empty(n)
    trend[lead:lead + len(full)] = full
    trend[:lead] = full[0]
    trend[lead + len(full):] = full[-1]
    return raw - trend

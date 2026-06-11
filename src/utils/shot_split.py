"""Shot-boundary detection for highlights reels (prepare_shots split mode).

Thin wrapper around PySceneDetect plus span hygiene. Resurrected from
the legacy segmentation stage (deleted in 262d08a) with the same
detector defaults; ``AdaptiveDetector`` is the default because plain
content thresholds misfire on fast broadcast pans/zooms.

``AdaptiveDetector`` normalises each frame's content change by the mean
of a tiny (2-frame) neighbourhood — inside continuous fast action
(celebration sequences, busy build-up) that denominator inflates and
real cuts vanish below the ratio threshold. Measured on the Liverpool
4-0 Barcelona reel: 28/36 spike-confirmed cuts found at the default
threshold. *Spike rescue* closes the gap with a second, statistically
robust pass: a cut is an isolated outlier of the frame-diff curve
against a wide (25-frame) median/MAD window, which fast action cannot
inflate the way a 2-frame mean can. Detector ∪ rescue recovers 36/36 on
that reel while adding no false cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video


@dataclass(frozen=True)
class ShotSpan:
    """One detected shot, in source-video frames/seconds."""

    start_frame: int
    end_frame: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def detect_spans(
    video_path: Path,
    *,
    detector: str = "adaptive",
    threshold: float = 27.0,
    adaptive_threshold: float = 3.0,
    min_scene_len_frames: int = 13,
    adaptive_min_content_val: float = 15.0,
    min_shot_duration_s: float = 0.0,
    spike_rescue: bool = True,
    spike_z_min: float = 4.0,
    spike_abs_min: float = 18.0,
    spike_window_frames: int = 25,
    dissolve_split: bool = False,
    dissolve_uniformity_min: float = 10.0,
    dissolve_flow_max: float = 1.25,
    dissolve_min_run_frames: int = 5,
) -> list[ShotSpan]:
    """Detect hard cuts in ``video_path`` and return shot spans.

    ``spike_rescue`` unions the detector's cuts with frame-diff outlier
    cuts (see module docstring); ``dissolve_split`` additionally unions
    cross-dissolve cuts (:func:`dissolve_cuts` — fades between clips
    that neither the detector nor spike rescue can see). Cuts from
    either pass closer than ``min_scene_len_frames`` to an existing cut
    are discarded so the detector's placement wins. Spans shorter than
    ``min_shot_duration_s`` are dropped (sub-second flashes are useless
    for reconstruction). If no cuts are found at all, the whole video is
    returned as a single span so the caller always has something to work
    with.
    """
    video = open_video(str(video_path))
    manager = SceneManager()
    if detector == "adaptive":
        manager.add_detector(AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len_frames,
            min_content_val=adaptive_min_content_val,
        ))
    elif detector == "content":
        manager.add_detector(ContentDetector(
            threshold=threshold, min_scene_len=min_scene_len_frames,
        ))
    else:
        raise ValueError(
            "prepare_shots.split.detector must be 'adaptive' or 'content'"
        )
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()

    fps = float(video.frame_rate) or 25.0
    total_frames = (
        scenes[-1][1].get_frames()
        if scenes else _whole_video_span(video_path).end_frame + 1
    )
    cut_frames = sorted(s[0].get_frames() for s in scenes[1:])

    extra_cuts: list[int] = []
    if spike_rescue:
        extra_cuts.extend(diff_spike_cuts(
            video_path,
            z_min=spike_z_min,
            abs_min=spike_abs_min,
            window_frames=spike_window_frames,
        ))
    if dissolve_split:
        extra_cuts.extend(dissolve_cuts(
            video_path,
            uniformity_min=dissolve_uniformity_min,
            flow_max=dissolve_flow_max,
            min_run_frames=dissolve_min_run_frames,
            min_gap_frames=max(25, min_scene_len_frames),
        ))
    if extra_cuts:
        existing = [0, total_frames] + cut_frames
        for cut in sorted(extra_cuts):
            if all(abs(cut - c) >= min_scene_len_frames for c in existing):
                cut_frames.append(cut)
                existing.append(cut)
        cut_frames.sort()

    spans = _spans_from_cuts(cut_frames, total_frames, fps)
    if not spans:
        spans = [_whole_video_span(video_path)]
    if min_shot_duration_s > 0:
        spans = [s for s in spans if s.duration_s >= min_shot_duration_s]
    return spans


def _spans_from_cuts(
    cut_frames: list[int],
    total_frames: int,
    fps: float,
) -> list[ShotSpan]:
    """Contiguous spans from sorted cut positions (cut = first frame of
    the new shot)."""
    if total_frames <= 0:
        return []
    starts = [0] + [c for c in cut_frames if 0 < c < total_frames]
    spans = []
    for i, start in enumerate(starts):
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else total_frames - 1
        if end < start:
            continue
        spans.append(ShotSpan(
            start_frame=start,
            end_frame=end,
            start_s=start / fps,
            end_s=(end + 1) / fps,
        ))
    return spans


def _block_median_diff(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    grid: tuple[int, int] = (4, 8),
) -> float:
    """Median per-block mean |diff| — how *uniformly* the frame changed.

    A cross-dissolve changes every block by a similar amount (the whole
    image fades), so the median block diff stays high. Localised change
    (players moving in front of a static camera) leaves most blocks
    untouched, so the median stays low even when the frame-mean diff is
    large. Measured on the Liverpool reel: dissolve frames ≥ 9.5 (p10),
    static-camera action ≤ 7.3 (p50), pans ≈ 8.
    """
    d = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
    gh, gw = grid
    h, w = d.shape
    blocks = [
        float(d[i * h // gh:(i + 1) * h // gh,
                j * w // gw:(j + 1) * w // gw].mean())
        for i in range(gh) for j in range(gw)
    ]
    return float(np.median(blocks))


def dissolve_cuts(
    video_path: Path,
    *,
    width_px: int = 320,
    uniformity_min: float = 10.0,
    flow_max: float = 1.25,
    min_run_frames: int = 5,
    min_gap_frames: int = 25,
) -> list[int]:
    """Cut candidates from cross-dissolves (fades between clips).

    A dissolve changes the *whole frame* steadily without anything
    moving. Two gates together isolate that: spatial uniformity of the
    change (block-median diff ≥ ``uniformity_min`` — rejects
    static-camera action, where change is localised to the players) and
    near-zero optical flow (median LK flow ≤ ``flow_max`` — rejects
    pans, which change every block but carry real motion). Within each
    dissolve run, cuts land on local maxima spaced ``min_gap_frames``
    apart (default ≈1 s: one fade = one cut, while a continuously-
    dissolving montage still yields one cut per chained fade). Returns
    first-frame-of-new-shot indices.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    uniformity: list[float] = []
    flows: list[float] = []
    prev: np.ndarray | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = width_px / max(1, w)
            small = cv2.resize(
                frame, (width_px, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                uniformity.append(_block_median_diff(prev, gray))
                # Median LK flow over tracked corners. With very few
                # corners (flat content — fade endpoints, blank frames)
                # LK emits garbage displacements, and motion can't be
                # claimed from a handful of points anyway: treat as no
                # flow. Pans always track hundreds of corners.
                flow = 0.0
                corners = cv2.goodFeaturesToTrack(
                    prev, maxCorners=120, qualityLevel=0.01, minDistance=7,
                )
                if corners is not None and len(corners):
                    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev, gray, corners, None,
                    )
                    tracked = status.reshape(-1) == 1
                    if tracked.sum() >= 15:
                        flow = float(np.median(np.linalg.norm(
                            (nxt - corners).reshape(-1, 2)[tracked], axis=1,
                        )))
                flows.append(flow)
            prev = gray
    finally:
        cap.release()

    diff_arr = np.asarray(uniformity)
    flow_arr = np.asarray(flows)
    mask = (diff_arr >= uniformity_min) & (flow_arr <= flow_max)

    cuts: list[int] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_run_frames:
            # Local diff maxima within the run, strongest first, spaced
            # at least min_gap_frames apart.
            order = sorted(range(i, j), key=lambda k: -diff_arr[k])
            accepted: list[int] = []
            for k in order:
                if all(abs(k - a) >= min_gap_frames for a in accepted):
                    accepted.append(k)
            cuts.extend(k + 1 for k in accepted)  # new shot starts at k+1
        i = j
    return sorted(cuts)


def diff_spike_cuts(
    video_path: Path,
    *,
    width_px: int = 160,
    z_min: float = 4.0,
    abs_min: float = 18.0,
    window_frames: int = 25,
) -> list[int]:
    """Cut candidates from frame-diff outliers (robust local statistics).

    Computes the mean |gray frame-diff| curve at reduced resolution and
    flags frames whose diff is a ``z_min``-sigma outlier of the
    surrounding ``window_frames`` median/MAD *and* above ``abs_min``
    absolute change. Adjacent flags (dissolves spread over 2-3 frames)
    collapse to the strongest. Returns first-frame-of-new-shot indices.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    diffs: list[float] = []
    prev = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = width_px / max(1, w)
            small = cv2.resize(
                frame, (width_px, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(gray - prev))))
            prev = gray
    finally:
        cap.release()

    curve = np.asarray(diffs)
    n = len(curve)
    flagged: list[tuple[int, float]] = []
    for i in range(n):
        if curve[i] < abs_min:
            continue
        lo, hi = max(0, i - window_frames), min(n, i + window_frames + 1)
        neighbourhood = np.concatenate([curve[lo:i], curve[i + 1:hi]])
        if len(neighbourhood) < 5:
            continue
        med = float(np.median(neighbourhood))
        mad = float(np.median(np.abs(neighbourhood - med))) + 1e-6
        z = (curve[i] - med) / (1.4826 * mad)
        if z >= z_min:
            # diff index i = change between frames i and i+1, so the new
            # shot starts at frame i+1.
            flagged.append((i + 1, float(curve[i])))

    collapsed: list[tuple[int, float]] = []
    for frame, strength in flagged:
        if collapsed and frame - collapsed[-1][0] <= 3:
            if strength > collapsed[-1][1]:
                collapsed[-1] = (frame, strength)
        else:
            collapsed.append((frame, strength))
    return [frame for frame, _ in collapsed]


def _whole_video_span(video_path: Path) -> ShotSpan:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    end_frame = max(0, frames - 1)
    return ShotSpan(
        start_frame=0,
        end_frame=end_frame,
        start_s=0.0,
        end_s=frames / fps,
    )


def merge_short_spans(
    spans: list[ShotSpan],
    *,
    max_short_duration_s: float,
    max_gap_s: float,
) -> list[ShotSpan]:
    """Merge likely false cuts: contiguous spans where either side is
    very short (camera whip-pans read as cuts to content detectors)."""
    if not spans:
        return []
    merged: list[ShotSpan] = [spans[0]]
    for current in spans[1:]:
        prev = merged[-1]
        gap_s = current.start_s - prev.end_s
        should_merge = gap_s <= max_gap_s and (
            prev.duration_s <= max_short_duration_s
            or current.duration_s <= max_short_duration_s
        )
        if should_merge:
            merged[-1] = ShotSpan(
                start_frame=prev.start_frame,
                end_frame=current.end_frame,
                start_s=prev.start_s,
                end_s=current.end_s,
            )
        else:
            merged.append(current)
    return merged

"""Kit-colour evidence extraction for automatic match suggestion (W4).

Samples player-torso pixel colours from a handful of frames per shot,
clusters them in CIELAB space into the two dominant kit colours (plus a
smallest cluster dropped as referee/noise), and scores that evidence
against a candidate fixture's known kit hexes.

This module is deliberately CPU-only (cv2 + numpy + sklearn) and never
raises: it is used as best-effort dashboard evidence, and a missing or
malformed pipeline artifact should degrade to "no evidence" (``None``)
rather than blow up a request handler.

The track ``team`` field ("A" | "B" | "referee" | "unknown") is IGNORED
here on purpose -- the default tracking config wires up
``FakeTeamClassifier``, which always reports "unknown". Kit colour is
recovered independently by clustering actual pixel samples.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.schemas.shots import ShotsManifest
from src.schemas.tracks import TracksResult
from src.utils.video_reader import read_frames

logger = logging.getLogger(__name__)

# Pitch-green HSV band. Matches the convention already used for shot
# classification in src/utils/shot_features.py (OpenCV hue is 0-179).
_PITCH_GREEN_HSV_LOW = (35, 40, 40)
_PITCH_GREEN_HSV_HIGH = (95, 255, 255)

# Torso crop window inside a player bbox, as a fraction of (height, width):
# rows 20-45% down the box, columns 25-75% across -- avoids head/shorts and
# the bbox's left/right edges, which are more likely to include background.
_TORSO_ROW_RANGE = (0.20, 0.45)
_TORSO_COL_RANGE = (0.25, 0.75)

# Track classes treated as "a kit is visible here". Excludes "ball".
_KIT_CLASSES = frozenset({"player", "goalkeeper", "referee"})

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# Normalisation constant for kit_match_score's CIE76 (Euclidean-in-Lab)
# distance. Derived once, offline, as the largest pairwise CIE76 distance
# among the 8 sRGB-cube corners (black, white, the 3 primaries, the 3
# secondaries) run through cv2's BGR2LAB conversion -- the largest gap
# found was pure green vs. pure blue, ~258.47. Real kit-colour pairs sit
# well inside this bound, so clamping against it keeps the [0, 1] score
# from saturating at 0 for merely "different", reserving that for genuine
# gamut-extreme opposites.
_MAX_LAB_DISTANCE = 258.5


@dataclass(frozen=True)
class KitEvidence:
    team_hexes: tuple[str, str]  # "#rrggbb" dominant torso colour per team cluster
    cluster_sizes: tuple[int, int]  # player-samples backing each cluster
    n_frames_sampled: int


def _bgr_pixel_to_lab(bgr: np.ndarray) -> tuple[float, float, float]:
    """Convert one BGR uint8 pixel (shape (3,)) to true CIELAB.

    OpenCV's 8-bit LAB output rescales L to [0, 255] and offsets a/b by
    +128 for byte storage; this undoes that so downstream math works in
    real CIELAB units (L in [0, 100], a/b roughly [-128, 127]).
    """
    img = bgr.reshape(1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)[0, 0]
    return (lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0)


def _lab_to_hex(lab: np.ndarray) -> str:
    """Inverse of ``_bgr_pixel_to_lab``: CIELAB -> "#rrggbb"."""
    l_byte = np.clip(lab[0] * 255.0 / 100.0, 0, 255)
    a_byte = np.clip(lab[1] + 128.0, 0, 255)
    b_byte = np.clip(lab[2] + 128.0, 0, 255)
    lab_img = np.array([[[l_byte, a_byte, b_byte]]], dtype=np.uint8)
    bgr = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)[0, 0]
    b, g, r = (int(c) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_lab(hex_str: str) -> np.ndarray | None:
    """Parse "#rrggbb" -> CIELAB ndarray, or None if malformed/empty."""
    if not isinstance(hex_str, str):
        return None
    candidate = hex_str.strip()
    if not _HEX_RE.fullmatch(candidate):
        return None
    r = int(candidate[1:3], 16)
    g = int(candidate[3:5], 16)
    b = int(candidate[5:7], 16)
    return np.array(_bgr_pixel_to_lab(np.array([b, g, r], dtype=np.uint8)))


def _torso_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Slice the torso sub-region out of a player bbox, clamped to the
    frame bounds. Returns None for a degenerate (empty/inverted) result."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return None
    frame_h, frame_w = frame.shape[:2]
    row0 = max(0, int(round(y1 + _TORSO_ROW_RANGE[0] * h)))
    row1 = min(frame_h, int(round(y1 + _TORSO_ROW_RANGE[1] * h)))
    col0 = max(0, int(round(x1 + _TORSO_COL_RANGE[0] * w)))
    col1 = min(frame_w, int(round(x1 + _TORSO_COL_RANGE[1] * w)))
    if row1 <= row0 or col1 <= col0:
        return None
    return frame[row0:row1, col0:col1]


def _kit_colour_sample(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Median BGR colour of ``crop_bgr`` with pitch-green pixels masked
    out. Returns None when no non-green pixel remains."""
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, _PITCH_GREEN_HSV_LOW, _PITCH_GREEN_HSV_HIGH)
    keep = green_mask == 0
    pixels = crop_bgr[keep]
    if pixels.shape[0] == 0:
        return None
    median = np.median(pixels.astype(np.float64), axis=0)
    return np.clip(median, 0, 255).astype(np.uint8)


def _sample_indices(total_frames: int, max_frames: int) -> list[int]:
    n = min(max_frames, total_frames)
    if n <= 0:
        return []
    return sorted({int(i) for i in np.linspace(0, total_frames - 1, num=n)})


def _collect_shot_samples(
    output_dir: Path, shot, max_frames: int
) -> tuple[list[np.ndarray], int]:
    """Return (lab_samples, n_frames_read) for one shot. Never raises --
    any missing/unreadable artifact for this shot yields ([], 0), letting
    the caller fall through to other shots."""
    tracks_path = output_dir / "tracks" / f"{shot.id}_tracks.json"
    try:
        tracks_result = TracksResult.load(tracks_path)
    except Exception:
        logger.warning("kit_evidence: could not load tracks for shot %s", shot.id)
        return [], 0

    kit_tracks = [t for t in tracks_result.tracks if t.class_name in _KIT_CLASSES]
    if not kit_tracks:
        return [], 0

    clip_path = output_dir / shot.clip_file
    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            logger.warning("kit_evidence: could not open clip for shot %s", shot.id)
            return [], 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if total_frames <= 0:
        return [], 0

    frame_indices = _sample_indices(total_frames, max_frames)
    if not frame_indices:
        return [], 0

    # Sequential cv2 reads via video_reader.read_frames -- never per-frame
    # seeks (project convention; see src/utils/video_reader.py).
    frames_by_index = read_frames(clip_path, frame_indices)
    if not frames_by_index:
        return [], 0

    bbox_by_frame_per_track = [
        {tf.frame: tf.bbox for tf in t.frames} for t in kit_tracks
    ]

    samples: list[np.ndarray] = []
    for frame_idx, frame in frames_by_index.items():
        for bbox_by_frame in bbox_by_frame_per_track:
            bbox = bbox_by_frame.get(frame_idx)
            if bbox is None:
                continue
            crop = _torso_crop(frame, bbox)
            if crop is None:
                continue
            sample_bgr = _kit_colour_sample(crop)
            if sample_bgr is None:
                continue
            samples.append(np.array(_bgr_pixel_to_lab(sample_bgr)))

    return samples, len(frames_by_index)


def extract_kit_evidence(output_dir: Path, *, max_frames: int = 12) -> KitEvidence | None:
    """Extract dominant kit-colour evidence for an output directory.

    Iterates ``manifest.active_shots()``, reads each shot's tracks
    sidecar and clip, samples up to ``max_frames`` evenly-spaced frames
    per clip via sequential cv2 reads, takes one torso-colour sample per
    player bbox present on a sampled frame, and clusters all samples
    (Lab space, k-means k=3) to recover the two dominant kit colours,
    dropping the smallest cluster as referee/noise.

    Returns None (never raises) when the manifest, tracks, or clips are
    missing/unreadable, or when fewer than 2 usable clusters emerge.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "shots" / "shots_manifest.json"
    try:
        manifest = ShotsManifest.load(manifest_path)
    except Exception:
        logger.warning("kit_evidence: could not load manifest at %s", manifest_path)
        return None

    try:
        all_samples: list[np.ndarray] = []
        n_frames_sampled = 0
        for shot in manifest.active_shots():
            samples, n_read = _collect_shot_samples(output_dir, shot, max_frames)
            all_samples.extend(samples)
            n_frames_sampled += n_read

        if len(all_samples) < 3:
            return None

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("kit_evidence: sklearn not available")
            return None

        data = np.stack(all_samples, axis=0)
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
        with warnings.catch_warnings():
            # Two teams wearing near-uniform kits legitimately produce
            # duplicate/near-duplicate Lab points -- sklearn's warning
            # about fewer distinct clusters than k=3 is expected here,
            # not a real problem (the empty third cluster is exactly
            # the referee/noise slot we intend to drop below).
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            labels = kmeans.fit_predict(data)

        sizes = np.bincount(labels, minlength=3)
        order = np.argsort(sizes)[::-1]  # descending by cluster size
        top_two = order[:2]
        if sizes[top_two[0]] == 0 or sizes[top_two[1]] == 0:
            # Fewer than 2 usable (non-empty) clusters emerged.
            return None

        team_hexes = tuple(_lab_to_hex(kmeans.cluster_centers_[i]) for i in top_two)
        cluster_sizes = tuple(int(sizes[i]) for i in top_two)

        return KitEvidence(
            team_hexes=team_hexes,  # type: ignore[arg-type]
            cluster_sizes=cluster_sizes,  # type: ignore[arg-type]
            n_frames_sampled=n_frames_sampled,
        )
    except Exception:
        logger.exception("kit_evidence: unexpected failure extracting evidence from %s", output_dir)
        return None


def _similarity(lab_a: np.ndarray | None, lab_b: np.ndarray | None) -> float | None:
    """CIE76 similarity in [0, 1], or None if either side is unusable."""
    if lab_a is None or lab_b is None:
        return None
    distance = float(np.linalg.norm(lab_a - lab_b))
    similarity = 1.0 - distance / _MAX_LAB_DISTANCE
    return max(0.0, min(1.0, similarity))


def kit_match_score(evidence: KitEvidence, home_hex: str, away_hex: str) -> float:
    """Score ``evidence`` against candidate ``home_hex``/``away_hex`` kits.

    Converts every hex to CIELAB and scores both possible assignments of
    the evidence's two clusters to (home, away), returning the better
    one -- the evidence carries no inherent home/away labelling. Distance
    is normalised via ``_MAX_LAB_DISTANCE`` (see module docstring) and
    clamped to [0, 1] per side, then averaged over whichever side(s) are
    usable.

    A malformed or empty hex on either side is dropped from the
    comparison (that side is "ignored"); if both home_hex and away_hex
    are unusable, the result is 0.0.
    """
    home_lab = _hex_to_lab(home_hex)
    away_lab = _hex_to_lab(away_hex)
    if home_lab is None and away_lab is None:
        return 0.0

    ev0 = _hex_to_lab(evidence.team_hexes[0])
    ev1 = _hex_to_lab(evidence.team_hexes[1])

    def _assignment_score(ev_home: np.ndarray | None, ev_away: np.ndarray | None) -> float:
        similarities = [
            s
            for s in (_similarity(ev_home, home_lab), _similarity(ev_away, away_lab))
            if s is not None
        ]
        if not similarities:
            return 0.0
        return sum(similarities) / len(similarities)

    score_direct = _assignment_score(ev0, ev1)
    score_swapped = _assignment_score(ev1, ev0)
    return max(score_direct, score_swapped)

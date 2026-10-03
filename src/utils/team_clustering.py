"""Team clustering from kit colour evidence.

Two halves:

* **Sampling** (``sample_player_regions`` / ``collect_player_features``):
  per-player body-region colours (torso, upper arms, shorts, socks) read
  from COCO-17 keypoint polygons on a handful of frames, grass-masked,
  as a per-player CIELAB median.
* **Clustering** (``cluster_teams``): k=2 teams from the outfielders,
  with keepers and officials peeled off as colour outliers. A keeper is
  an outlier whose pitch x stays in a defensive third (at most one per
  goal); every other outlier is an official.

Pure numpy / sklearn / cv2; the IO half takes already-loaded frames so
tests run on synthetic images.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from src.utils.kit_palette import srgb_to_lab

logger = logging.getLogger(__name__)

# COCO-17 indices
L_SHO, R_SHO, L_ELB, R_ELB = 5, 6, 7, 8
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

REGIONS = ("torso", "sleeves", "shorts", "socks")
_GREEN_HSV_LOW = (35, 40, 40)
_GREEN_HSV_HIGH = (95, 255, 255)
_MIN_REGION_PIXELS = 6


# --- sampling ----------------------------------------------------------------

def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def _shrink(poly: np.ndarray, f: float) -> np.ndarray:
    c = poly.mean(axis=0)
    return c + (poly - c) * (1.0 - f)


def region_shapes(kp: np.ndarray, min_conf: float) -> dict[str, list[tuple[str, np.ndarray, int]]]:
    """Region geometry from one player's ``(17, 3)`` keypoints.

    Returns ``{region: [(kind, pts, thickness), ...]}`` where ``kind`` is
    ``"poly"`` (filled polygon) or ``"line"`` (thick segment). Regions
    whose keypoints are below ``min_conf`` are absent.
    """
    xy, conf = kp[:, :2].astype(np.float64), kp[:, 2]

    def ok(*idx: int) -> bool:
        return bool(all(conf[i] >= min_conf for i in idx))

    out: dict[str, list] = {}
    sw = float(np.linalg.norm(xy[L_SHO] - xy[R_SHO])) if ok(L_SHO, R_SHO) else 0.0
    if ok(L_SHO, R_SHO, L_HIP, R_HIP):
        quad = np.array([xy[L_SHO], xy[R_SHO], xy[R_HIP], xy[L_HIP]])
        out["torso"] = [("poly", _shrink(quad, 0.25), 0)]
        sw = max(sw, 1.0)
    thick = max(2, int(round(0.28 * sw))) if sw else 2
    sleeves = []
    for sho, elb in ((L_SHO, L_ELB), (R_SHO, R_ELB)):
        if ok(sho, elb):
            sleeves.append(("line", np.array([_lerp(xy[sho], xy[elb], 0.2), _lerp(xy[sho], xy[elb], 0.75)]), thick))
    if sleeves:
        out["sleeves"] = sleeves
    if ok(L_HIP, R_HIP, L_KNE, R_KNE):
        lower = np.array([xy[L_HIP], xy[R_HIP], _lerp(xy[R_HIP], xy[R_KNE], 0.7), _lerp(xy[L_HIP], xy[L_KNE], 0.7)])
        out["shorts"] = [("poly", _shrink(lower, 0.2), 0)]
    socks = []
    for kne, ank in ((L_KNE, L_ANK), (R_KNE, R_ANK)):
        if ok(kne, ank):
            socks.append(("line", np.array([_lerp(xy[kne], xy[ank], 0.45), _lerp(xy[kne], xy[ank], 0.92)]),
                          max(2, int(round(0.2 * sw)) if sw else 2)))
    if socks:
        out["socks"] = socks
    return out


def _region_lab_median(frame_bgr: np.ndarray, shapes: Sequence) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    allpts = np.concatenate([s[1] for s in shapes])
    pad = max(s[2] for s in shapes) + 2
    x0, y0 = np.floor(allpts.min(axis=0) - pad).astype(int)
    x1, y1 = np.ceil(allpts.max(axis=0) + pad).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    for kind, pts, thick in shapes:
        local = np.round(pts - [x0, y0]).astype(np.int32)
        if kind == "poly":
            cv2.fillPoly(mask, [local], 255)
        else:
            cv2.line(mask, tuple(local[0]), tuple(local[1]), 255, thick)
    green = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), _GREEN_HSV_LOW, _GREEN_HSV_HIGH)
    keep = (mask > 0) & (green == 0)
    if int(keep.sum()) < _MIN_REGION_PIXELS:
        return None
    rgb = crop[keep][:, ::-1].astype(np.float64)
    return srgb_to_lab(np.median(rgb, axis=0))


def sample_player_regions(frame_bgr: np.ndarray, kp: np.ndarray, *, min_conf: float = 0.4) -> dict[str, np.ndarray]:
    """``{region: Lab median}`` for the regions visible on this frame."""
    out: dict[str, np.ndarray] = {}
    for region, shapes in region_shapes(np.asarray(kp), min_conf).items():
        lab = _region_lab_median(frame_bgr, shapes)
        if lab is not None:
            out[region] = lab
    return out


def aggregate_samples(per_frame: Iterable[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Median Lab per region across frames."""
    buckets: dict[str, list[np.ndarray]] = {}
    for sample in per_frame:
        for region, lab in sample.items():
            buckets.setdefault(region, []).append(lab)
    return {r: np.median(np.stack(v), axis=0) for r, v in buckets.items()}


# --- clustering --------------------------------------------------------------

@dataclass
class TeamClustering:
    team_of: dict[str, int] = field(default_factory=dict)        # outfield pid -> 0/1
    keepers: dict[str, int] = field(default_factory=dict)        # pid -> goal side (0: x=0, 1: x=L)
    referees: list[str] = field(default_factory=list)
    team_features: list[dict[str, np.ndarray]] = field(default_factory=list)
    team_mean_x: list[float | None] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "team_of": self.team_of,
            "keepers": {p: {"goal_side": s} for p, s in self.keepers.items()},
            "referees": self.referees,
            "team_mean_x": [None if x is None else round(float(x), 2) for x in self.team_mean_x],
            "notes": self.notes,
        }


def _feature_vector(parts: Mapping[str, np.ndarray]) -> np.ndarray | None:
    torso = parts.get("torso")
    if torso is None:
        return None
    shorts = parts.get("shorts")
    return np.concatenate([torso, 0.5 * (shorts if shorts is not None else torso)])


def _kmeans2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(x)
    return km.labels_, km.cluster_centers_


def cluster_teams(
    players: Mapping[str, Mapping[str, np.ndarray]],
    pitch_x: Mapping[str, float],
    *,
    pitch_length: float = 105.0,
    keeper_third_m: float = 35.0,
    outlier_min_de: float = 18.0,
    outlier_factor: float = 3.0,
    max_outlier_rounds: int = 3,
) -> TeamClustering:
    """k=2 team clustering with keeper / official outlier peeling.

    ``players``: pid -> region Lab medians (needs ``torso``);
    ``pitch_x``: pid -> median pitch x (m) from refined poses (optional).
    """
    result = TeamClustering()
    feats = {pid: v for pid, p in players.items() if (v := _feature_vector(p)) is not None}
    pids = sorted(feats)
    if len(pids) < 4:
        result.notes.append("too_few_players_for_clustering")
        for pid in pids:
            result.team_of[pid] = 0
        return result

    inliers = list(pids)
    outliers: list[str] = []
    for _ in range(max_outlier_rounds):
        x = np.stack([feats[p] for p in inliers])
        labels, centers = _kmeans2(x)
        d = np.linalg.norm(x - centers[labels], axis=1)
        thresh = max(outlier_min_de, outlier_factor * float(np.median(d)))
        bad = [p for p, di in zip(inliers, d) if di > thresh]
        # never peel so many that the teams collapse
        if not bad or len(inliers) - len(bad) < 4 or len(bad) > 5:
            break
        outliers.extend(bad)
        inliers = [p for p in inliers if p not in bad]
    x = np.stack([feats[p] for p in inliers])
    labels, centers = _kmeans2(x)
    # deterministic team ids: team 0 = cluster holding the smallest pid
    first = labels[0]
    order = {int(first): 0, int(1 - first): 1}
    for pid, lab in zip(inliers, labels):
        result.team_of[pid] = order[int(lab)]

    # outliers -> keepers (defensive third, <=1 per goal) or officials
    sides: dict[int, list[tuple[float, str]]] = {0: [], 1: []}
    for pid in outliers:
        px = pitch_x.get(pid)
        if px is None:
            result.referees.append(pid)
            result.notes.append(f"{pid}:outlier_without_pitch_x_treated_as_referee")
        elif px <= keeper_third_m:
            sides[0].append((px, pid))
        elif px >= pitch_length - keeper_third_m:
            sides[1].append((pitch_length - px, pid))
        else:
            result.referees.append(pid)
    for side, cand in sides.items():
        if not cand:
            continue
        cand.sort()
        result.keepers[cand[0][1]] = side          # closest to its own goal line
        result.referees.extend(pid for _, pid in cand[1:])
    result.referees = sorted(result.referees)

    for team in (0, 1):
        members = [p for p, t in result.team_of.items() if t == team]
        result.team_features.append(aggregate_samples([players[p] for p in members]))
        xs = [pitch_x[p] for p in members if p in pitch_x]
        result.team_mean_x.append(float(np.mean(xs)) if xs else None)
    return result


def keeper_team(side: int, clustering: TeamClustering, *, min_margin_m: float = 3.0) -> tuple[int, bool]:
    """Which team does the keeper at goal ``side`` play for?

    The defending team is the one whose outfielders sit closer to that
    goal on average. Returns ``(team, confident)``; confident is False
    when the mean-x gap is under ``min_margin_m`` (or unknown).
    """
    xs = clustering.team_mean_x
    if len(xs) < 2 or xs[0] is None or xs[1] is None:
        return 0 if side == 0 else 1, False
    gap = xs[0] - xs[1]
    # side 0 = goal at x=0: defenders have the smaller mean x
    team = (0 if gap < 0 else 1) if side == 0 else (0 if gap > 0 else 1)
    return team, abs(gap) >= min_margin_m

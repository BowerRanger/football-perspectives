"""Post-triangulation player de-duplication.

The matching stage occasionally splits a single physical player into
two ``MatchedPlayer`` records (one per camera angle, when cross-view
re-id fails to associate them).  After triangulation those records
collapse onto the same world-space trajectory but keep separate
``player_id``s, which shows up in the bird's-eye preview as a player
"duplicating across the pitch."

This module identifies and merges duplicates *after* triangulation,
using the recovered 3D hip midpoint as the matching cue.  The hip
midpoint is the most stable joint pair (compared to wrists/feet which
swing) and lives at a near-constant world height, so a (x, y) median
distance comparison is enough to flag co-located players.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.schemas.triangulated import TriangulatedPlayer

_LEFT_HIP = 11
_RIGHT_HIP = 12


def deduplicate_players(
    players: list[TriangulatedPlayer],
    *,
    distance_m: float = 1.5,
    min_overlap_frames: int = 30,
) -> list[TriangulatedPlayer]:
    """Merge spatially co-located triangulated players.

    Two players are considered duplicates if their hip-midpoint (x, y)
    median distance over overlapping valid frames is below
    ``distance_m`` *and* they overlap on at least ``min_overlap_frames``
    frames.  Union-find is used so chains (a≈b, b≈c) collapse into a
    single merged player even when ``a`` and ``c`` don't directly
    overlap.

    The kept player is the one with the most valid hip-midpoint frames;
    other members of the cluster fill its NaN frames where they have
    data.  Player identity (id, name, team) is taken from the kept
    player.
    """
    n = len(players)
    if n < 2:
        return list(players)

    parents = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            median_d, n_overlap = _pair_median_distance(
                players[i].positions, players[j].positions,
            )
            if n_overlap >= min_overlap_frames and median_d < distance_m:
                _union(parents, i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(parents, i)
        groups.setdefault(root, []).append(i)

    merged: list[TriangulatedPlayer] = []
    for member_indices in groups.values():
        if len(member_indices) == 1:
            merged.append(players[member_indices[0]])
        else:
            merged.append(_merge_players([players[i] for i in member_indices]))
    return merged


def _hip_midpoints(positions: np.ndarray) -> np.ndarray:
    """Return ``(N, 2)`` hip-midpoint xy per frame.

    NaN propagates: if either hip is NaN at a frame, the midpoint is
    NaN at that frame.
    """
    lh = positions[:, _LEFT_HIP, :2]
    rh = positions[:, _RIGHT_HIP, :2]
    return (lh + rh) / 2.0


def _pair_median_distance(
    a_positions: np.ndarray,
    b_positions: np.ndarray,
) -> tuple[float, int]:
    """Median hip-midpoint (x, y) distance over overlapping valid frames.

    Returns ``(median_distance_m, n_overlap_frames)``.  Returns
    ``(inf, 0)`` when there is no overlap.
    """
    n = min(len(a_positions), len(b_positions))
    if n == 0:
        return float("inf"), 0
    a_mid = _hip_midpoints(a_positions[:n])
    b_mid = _hip_midpoints(b_positions[:n])
    valid = ~(np.isnan(a_mid[:, 0]) | np.isnan(b_mid[:, 0]))
    if not np.any(valid):
        return float("inf"), 0
    diffs = a_mid[valid] - b_mid[valid]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.median(dists)), int(valid.sum())


def _merge_players(players: list[TriangulatedPlayer]) -> TriangulatedPlayer:
    """Merge a cluster of duplicate players into one record.

    Keeps the player with the most valid hip-midpoint frames as the
    base; fills its NaN joint frames from the other members in order
    of valid-frame count.  Identity (id, name, team) is taken from the
    base player.
    """
    def _hip_valid_count(p: TriangulatedPlayer) -> int:
        mid = _hip_midpoints(p.positions)
        return int(np.sum(~np.isnan(mid[:, 0])))

    sorted_players = sorted(players, key=_hip_valid_count, reverse=True)
    base = sorted_players[0]
    others = sorted_players[1:]

    n_frames = len(base.positions)
    positions = base.positions.copy()
    confidences = base.confidences.copy()
    reproj_errors = base.reprojection_errors.copy()
    num_views = base.num_views.copy()

    for other in others:
        if len(other.positions) != n_frames:
            # Different frame ranges shouldn't happen in the current
            # pipeline (every player is built from the same
            # frame_range), but guard against it rather than crash.
            continue
        base_nan = np.isnan(positions[..., 0])      # (n_frames, 17)
        other_valid = ~np.isnan(other.positions[..., 0])
        fill = base_nan & other_valid
        positions[fill] = other.positions[fill]
        confidences[fill] = other.confidences[fill]
        reproj_errors[fill] = other.reprojection_errors[fill]
        num_views[fill] = other.num_views[fill]

    return replace(
        base,
        positions=positions,
        confidences=confidences,
        reprojection_errors=reproj_errors,
        num_views=num_views,
    )


def _find(parents: list[int], i: int) -> int:
    while parents[i] != i:
        parents[i] = parents[parents[i]]  # path compression
        i = parents[i]
    return i


def _union(parents: list[int], i: int, j: int) -> None:
    ri = _find(parents, i)
    rj = _find(parents, j)
    if ri != rj:
        parents[ri] = rj

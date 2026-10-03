"""Turn a team clustering + colour evidence into kits and per-player roles.

Pure logic (no IO): the ``appearance`` stage feeds it region colours,
the white balance and the config, and writes what it returns.

Candidate pools, tried in order, for each kit (first pool that snaps wins):

1. the clip config's ``appearance.kits`` (role-labelled, operator intent);
2. library kits of the MatchInfo clubs (season-filtered);
3. the whole library.

A snap needs a full-vector ΔE under ``snap_de`` (see ``kit_palette``);
otherwise the white-balance-corrected sample is kept (``source: sampled``).
Roles come from the winning pool: pool 1 names them; pool 2 maps a kit's
club to MatchInfo home/away; otherwise team 0 -> home, team 1 -> away with
``needs_confirmation``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.utils import kit_palette as kp
from src.utils.kit_library import KitLibrary, season_for_date
from src.utils.team_clustering import TeamClustering, keeper_team

TEAM_ROLES = ("home", "away")


@dataclass
class AppearanceResult:
    kits: dict[str, dict] = field(default_factory=dict)
    player_roles: dict[str, str] = field(default_factory=dict)
    needs_confirmation: list[str] = field(default_factory=list)
    teams: list[dict] = field(default_factory=list)


def _parts_hex(parts: Mapping[str, np.ndarray], wb: kp.WhiteBalance) -> dict[str, str]:
    """Region Lab medians -> white-balanced hex sample for snapping."""
    hexes = kp.sample_to_hex_parts({k: v for k, v in parts.items() if k != "sleeves"}, wb)
    sl = parts.get("sleeves")
    if sl is not None and "torso" in parts:
        sleeve_hex = kp.sample_to_hex_parts({"s": sl}, wb)["s"]
        torso_hex = hexes.get("torso")
        # skin-toned arms read as "sleeves"; only report a sleeve colour when it
        # clearly differs from the torso AND is not skin-like
        if torso_hex and kp.delta_e_hex(sleeve_hex, torso_hex) > 18 and not _skin_like(sl):
            hexes["sleeve_color"] = sleeve_hex
    socks = parts.get("socks")
    if socks is not None and _skin_like(socks):
        hexes.pop("socks", None)      # bare shin / rolled socks read as skin: not evidence
    if "torso" in hexes:
        hexes["shirt"] = hexes.pop("torso")
    return hexes


def _skin_like(lab: np.ndarray) -> bool:
    L, a, b = (float(v) for v in lab)
    hue = np.degrees(np.arctan2(b, a)) % 360
    chroma = float(np.hypot(a, b))
    return 15 <= hue <= 80 and 8 <= chroma <= 45 and 20 <= L <= 80


def _best_distinct(dists: list[dict[str, float]]) -> list[str | None]:
    """Pick one ref per team minimising total distance with refs distinct."""
    if len(dists) == 1:
        return [min(dists[0], key=dists[0].get) if dists[0] else None]
    refs0, refs1 = list(dists[0]), list(dists[1])
    if not refs0 or not refs1:
        return [min(d, key=d.get) if d else None for d in dists]
    best, best_cost = None, float("inf")
    for r0, r1 in itertools.product(refs0, refs1):
        if r0 == r1:
            continue
        cost = dists[0][r0] + dists[1][r1]
        if cost < best_cost:
            best, best_cost = [r0, r1], cost
    return best or [min(d, key=d.get) for d in dists]


def _finalise(spec: Mapping, pool_name: str, clip_kit: Mapping | None) -> dict:
    """Provenance tidy-up: clip-pool snaps carry the operator's kit, not a ``clip:`` ref."""
    out = dict(spec)
    if pool_name == "clip" and clip_kit is not None:
        out = dict(clip_kit)
        out["source"] = "clip"
        out["delta_e"] = spec.get("delta_e")
    out["pool"] = pool_name
    return out


def _decisive(dists: list[dict[str, float]], picks: list[str | None], ratio: float = 1.4) -> bool:
    """Is the chosen team->kit pairing clearly better than swapping the two?"""
    if len(picks) != 2:
        return False
    best = dists[0][picks[0]] + dists[1][picks[1]]
    swapped = dists[0].get(picks[1], float("inf")) + dists[1].get(picks[0], float("inf"))
    return swapped >= ratio * best


def _limit(pool_name: str, snap_de: float) -> float:
    """Accept radius per pool. Pools constrained by operator config or MatchInfo
    are trusted at 2x ``snap_de`` (the broadcast grade darkens/dulls saturated
    kits more than a library-wide search can tolerate); loose snaps are flagged."""
    return 2 * snap_de if pool_name in ("clip", "match") else snap_de


def _slug(ref: str) -> str:
    return ref.split("/", 1)[0]


def _team_sample_hex(clustering: TeamClustering, wb: kp.WhiteBalance) -> list[dict[str, str]]:
    return [_parts_hex(parts, wb) for parts in clustering.team_features]


def solve_appearance(
    clustering: TeamClustering,
    player_parts: Mapping[str, Mapping[str, np.ndarray]],
    wb: kp.WhiteBalance,
    *,
    library: KitLibrary,
    clip_kits: Mapping[str, Mapping] | None = None,
    match=None,
    snap_de: float = kp.SNAP_DELTA_E,
    lightness_weight: float = kp.LIGHTNESS_WEIGHT,
) -> AppearanceResult:
    clip_kits = dict(clip_kits or {})
    res = AppearanceResult()
    season = season_for_date(getattr(match, "date", None))
    home_slug = library.club_slug(getattr(match, "home_team", None))
    away_slug = library.club_slug(getattr(match, "away_team", None))
    match_slugs = [s for s in (home_slug, away_slug) if s]

    def snap_args() -> dict:
        return {"snap_de": snap_de, "lightness_weight": lightness_weight}

    # --- outfield teams -------------------------------------------------
    samples = _team_sample_hex(clustering, wb)
    pools: list[tuple[str, dict[str, Mapping]]] = []
    p1 = {f"clip:{r}": clip_kits[r] for r in TEAM_ROLES if r in clip_kits}
    if p1:
        pools.append(("clip", p1))
    if match_slugs:
        pools.append(("match", library.select(slugs=match_slugs, season=season) or
                      library.select(slugs=match_slugs)))
    pools.append(("library", library.select()))

    team_choice: list[tuple[str | None, kp.SnapResult, str]] = []
    chosen = None
    for pool_name, pool in pools:
        if not pool or not all(s.get("shirt") for s in samples):
            continue
        dists = [{ref: kp.kit_distance(s, kit, lightness_weight=lightness_weight)
                  for ref, kit in pool.items()} for s in samples]
        picks = _best_distinct(dists)
        limit = _limit(pool_name, snap_de)
        if any(p is None for p in picks):
            continue
        if all(dists[i][p] < limit for i, p in enumerate(picks)):
            chosen = (pool_name, pool, picks, limit)
            break
        if pool_name in ("clip", "match") and _decisive(dists, picks) \
                and all(dists[i][p] < 4 * snap_de for i, p in enumerate(picks)):
            chosen = (pool_name, pool, picks, 4 * snap_de)   # role-naming is clear; kit fit is loose
            break
    if chosen:
        pool_name, pool, picks, limit = chosen
        for i, ref in enumerate(picks):
            result = kp.snap_kit(samples[i], {ref: pool[ref]}, snap_de=limit,
                                 lightness_weight=lightness_weight)
            team_choice.append((ref, result, pool_name))
    else:
        for i, s in enumerate(samples):
            result = kp.snap_kit(s, {}, **snap_args()) if s.get("shirt") else None
            team_choice.append((None, result, "sampled"))

    team_role: dict[int, str] = {}
    for i, (ref, result, pool_name) in enumerate(team_choice):
        role = None
        if ref is not None and pool_name == "clip":
            role = ref.split(":", 1)[1]
        elif ref is not None and pool_name == "match":
            slug = _slug(ref)
            role = "home" if slug == home_slug else "away" if slug == away_slug else None
        team_role[i] = role or ""
    if sorted(team_role.values()) != sorted(TEAM_ROLES):
        # pools could not name both roles (library-wide / sampled / same club twice)
        for i in team_role:
            team_role[i] = TEAM_ROLES[i]
        res.needs_confirmation.append("home_away_unassigned")
    for i, (ref, result, pool_name) in enumerate(team_choice):
        role = team_role[i]
        if result is None:
            res.needs_confirmation.append(f"team_{i}_no_shirt_evidence")
            continue
        spec = _finalise(result.spec, pool_name, clip_kits.get(role))
        res.kits[role] = spec
        res.teams.append({"team": i, "role": role, "sample": samples[i],
                          "ref": spec.get("ref") if result.snapped else None,
                          "delta_e": round(result.delta_e, 2) if np.isfinite(result.delta_e) else None,
                          "snapped": result.snapped, "pool": pool_name})
        if not result.snapped:
            res.needs_confirmation.append(f"{role}_kit_not_in_library")
        elif pool_name != "clip" and result.delta_e >= snap_de:
            res.needs_confirmation.append(f"{role}_snap_loose")
    for pid, team in clustering.team_of.items():
        res.player_roles[pid] = team_role[team]

    # --- keepers ---------------------------------------------------------
    role_slug = {"home": home_slug, "away": away_slug}
    gk_pools: list[tuple[str, dict]] = []
    p1gk = {f"clip:{r}": clip_kits[r] for r in ("home_gk", "away_gk") if r in clip_kits}
    if p1gk:
        gk_pools.append(("clip", p1gk))
    if match_slugs:
        gk_pools.append(("match", library.select(slugs=match_slugs, season=season, keepers=True) or
                         library.select(slugs=match_slugs, keepers=True)))
    gk_pools.append(("library", library.select(keepers=True)))
    used_gk_roles: set[str] = set()
    for pid, side in sorted(clustering.keepers.items(), key=lambda kv: kv[1]):
        sample = _parts_hex(player_parts[pid], wb)
        team, confident = keeper_team(side, clustering)
        role = f"{team_role[team]}_gk"
        if not confident:
            res.needs_confirmation.append(f"{pid}:keeper_side_low_margin")
        snap_res, pool_name = None, "sampled"
        for name, pool in gk_pools:
            if not pool:
                continue
            r = kp.snap_kit(sample, pool, snap_de=_limit(name, snap_de),
                            lightness_weight=lightness_weight)
            if r.snapped:
                snap_res, pool_name = r, name
                if name == "clip":
                    role = r.ref.split(":", 1)[1]
                elif name == "match" and _slug(r.ref) in (home_slug, away_slug):
                    role = "home_gk" if _slug(r.ref) == home_slug else "away_gk"
                break
        if snap_res is None and p1gk and f"clip:{role}" in p1gk:
            # Operator named this side's keeper kit and the keeper is a colour
            # outlier by construction: a distant keeper's few pixels are
            # contaminated (grass/boards), so trust the named kit at 4x radius.
            r = kp.snap_kit(sample, {f"clip:{role}": p1gk[f"clip:{role}"]}, snap_de=4 * snap_de,
                            lightness_weight=lightness_weight)
            if r.snapped:
                snap_res, pool_name = r, "clip"
                res.needs_confirmation.append(f"{role}_snap_loose")
        if role in used_gk_roles:                      # two keepers, one role: use the other side's
            other = "away_gk" if role == "home_gk" else "home_gk"
            if other not in used_gk_roles:
                role = other
                res.needs_confirmation.append(f"{pid}:keeper_role_swapped_to_stay_distinct")
        used_gk_roles.add(role)
        spec = _finalise((snap_res or kp.snap_kit(sample, {}, **snap_args())).spec, pool_name,
                         clip_kits.get(role))
        res.kits[role] = spec
        res.player_roles[pid] = role
        if snap_res is None:
            res.needs_confirmation.append(f"{role}_kit_not_in_library")

    # --- officials -------------------------------------------------------
    ref_pool = library.select(referees=True)
    for pid in clustering.referees:
        sample = _parts_hex(player_parts[pid], wb)
        r = kp.snap_kit(sample, ref_pool, **snap_args())
        res.kits.setdefault("referee", _finalise(r.spec, "library" if r.snapped else "sampled", None))
        res.player_roles[pid] = "referee"
    return res

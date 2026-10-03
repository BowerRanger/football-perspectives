"""Key moments of a goal sequence, derived from existing pipeline outputs.

``derive_moments(output_dir, shot)`` is the single entry point the shorts
stage and templates use; it reads the ball track / diag / operator anchors,
``refined_poses`` root tracks and ``players.json`` roles. The maths lives in
``derive_from_data`` (pure, no I/O) so it is unit-testable on synthetic
inputs.

Moments (all scene-frame ints or ``None``):

* ``impact``      - the net/post/bar contact: operator ``goal_impact``
                    anchor (net contacts preferred over woodwork), else the
                    diag ``goal_impact`` events, else ``goal_check.goal_frame``.
* ``line_cross``  - T1's ``goal_check.line_cross.frame``, else the first
                    frame the dense track crosses a goal line inside the mouth.
* ``strike``      - operator shot/volley touch anchor, else the last diag
                    touch before the finish at which the ball's velocity
                    actually changed (shoulder grazes on a ball already in
                    flight don't), else the last touch.
* ``keeper_dive`` - peak lateral root velocity of the defending keeper in
                    ``[strike-10, impact]``.
* ``buildup_start`` - first touch of the scorer's team's possession chain
                    (only if at least ``MIN_BUILDUP_LEAD`` before the strike),
                    else ``strike - 75``.

Roles: ``scorer_pid`` is the strike's player; ``keeper_pid`` the ``*_gk``
player who defends the goal that was scored in; ``goal_end`` is ``left``
(x = 0) or ``right`` (x = 105), matching the ``goal:<side>`` camera ids.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_HALF_WIDTH = 3.66
CROSSBAR_M = 2.44
MOUTH_MARGIN_M = 0.6          # slack on the mouth for the track-based crossing
NET_ELEMENTS = frozenset({"back_net", "side_net"})
MOMENT_KEYS = ("strike", "line_cross", "impact", "keeper_dive", "buildup_start")
STRIKE_MIN_DV_M_S = 4.0       # |dv| across a touch for it to count as a strike
KEEPER_DIVE_PRE_FRAMES = 10
MAX_CHAIN_GAP_FRAMES = 40     # touches further apart than this break a chain
MIN_BUILDUP_LEAD = 45
DEFAULT_BUILDUP_LEAD = 75
EPS = 1e-6                    # a track clamped onto the line counts as on it
SHOT_TOUCH_TYPES = frozenset({"shot", "volley"})


def _team(role: str | None) -> str | None:
    if not role or role in ("referee", "unknown"):
        return None
    return role.removesuffix("_gk")


def _goal_end_for_x(x: float) -> str:
    return "left" if x < PITCH_LENGTH / 2 else "right"


def _ball_arrays(frames: Mapping[int, Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    ks = np.array(sorted(frames), dtype=int)
    xyz = np.array([frames[int(k)] for k in ks], dtype=float).reshape(-1, 3)
    return ks, xyz


def find_line_crossing(
    frames: Mapping[int, Sequence[float]], after: int | None = None,
    before: int | None = None,
) -> tuple[int, str, tuple[float, float, float]] | None:
    """First frame in ``(after, before]`` at which the track is on/past a
    goal line (x <= 0 or x >= 105) having been inside the pitch the frame
    before, with the crossing point inside the mouth (+margin)."""
    if len(frames) < 2:
        return None
    ks, xyz = _ball_arrays(frames)
    for i in range(1, len(ks)):
        f = int(ks[i])
        if after is not None and f <= after:
            continue
        if before is not None and f > before:
            break
        for line_x, end in ((0.0, "left"), (PITCH_LENGTH, "right")):
            a, b = xyz[i - 1], xyz[i]
            inside_prev = a[0] > line_x + EPS if end == "left" else a[0] < line_x - EPS
            past = b[0] <= line_x + EPS if end == "left" else b[0] >= line_x - EPS
            if not (inside_prev and past) or ks[i] - ks[i - 1] != 1:
                continue
            s = (line_x - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 1.0
            p = a + s * (b - a)
            if (abs(p[1] - PITCH_WIDTH / 2) <= GOAL_HALF_WIDTH + MOUTH_MARGIN_M
                    and p[2] <= CROSSBAR_M + MOUTH_MARGIN_M):
                return f, end, (float(p[0]), float(p[1]), float(p[2]))
    return None


def _pick_impact(operator_anchors: Sequence[Mapping], events: Sequence[Mapping],
                 goal_check: Mapping | None) -> tuple[int | None, str | None]:
    cands: list[tuple[int, int, str]] = []  # (priority, frame, source)
    for a in operator_anchors:
        if a.get("state") == "goal_impact" or a.get("goal_element"):
            el = a.get("goal_element")
            if a.get("state") == "goal_impact" or el in NET_ELEMENTS:
                cands.append((0 if el in NET_ELEMENTS else 1, int(a["frame"]), "operator_anchor"))
    if cands:
        cands.sort(key=lambda c: (c[0], -c[1]))
        return cands[0][1], cands[0][2]
    ev = [e for e in events if e.get("kind") == "goal_impact"]
    if ev:
        ev.sort(key=lambda e: (0 if e.get("goal_element") in NET_ELEMENTS else 1, -int(e["frame"])))
        return int(ev[0]["frame"]), "diag_event"
    if goal_check and goal_check.get("goal_frame") is not None:
        return int(goal_check["goal_frame"]), "goal_check"
    return None, None


def _velocity_change(frames: Mapping[int, Sequence[float]], f: int, fps: float,
                     win: int = 3) -> float | None:
    def mean_v(a: int, b: int):
        pts = [(k, np.asarray(frames[k], float)) for k in range(a, b + 1) if k in frames]
        if len(pts) < 2:
            return None
        return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]) * fps
    before, after = mean_v(f - win, f), mean_v(f, f + win)
    if before is None or after is None:
        return None
    return float(np.linalg.norm(after - before))


def _can_score(t: Mapping, keeper_pid: str | None, defending_team: str | None,
               roles: Mapping[str, str]) -> bool:
    """The defending keeper, or anyone on the defending team, never scores
    (origi01: ter Stegen's parry at 425 / a later auto touch at 447 must not
    beat Origi's tap-in at 440)."""
    pid = t.get("player_id")
    if pid is None or pid == keeper_pid:
        return False
    team = _team(roles.get(pid))
    return not (defending_team and team == defending_team)


def _pick_strike(ref: int | None, touches: Sequence[Mapping],
                 operator_anchors: Sequence[Mapping],
                 frames: Mapping[int, Sequence[float]], fps: float,
                 keeper_pid: str | None = None, defending_team: str | None = None,
                 roles: Mapping[str, str] | None = None,
                 ) -> tuple[Mapping | None, str | None]:
    horizon = ref if ref is not None else 10 ** 9
    roles = roles or {}
    ok = lambda t: _can_score(t, keeper_pid, defending_team, roles)  # noqa: E731
    shots = [a for a in operator_anchors
             if a.get("state") == "player_touch" and a.get("touch_type") in SHOT_TOUCH_TYPES
             and int(a["frame"]) <= horizon and ok(a)]
    if shots:
        a = max(shots, key=lambda a: int(a["frame"]))
        return a, "operator_shot_anchor"
    # Operator input wins over inferred events: the latest attacking operator
    # touch before the goal is the strike when it really changes the ball.
    op_touches = sorted((a for a in operator_anchors
                         if a.get("state") == "player_touch" and a.get("player_id")
                         and int(a["frame"]) <= horizon and ok(a)),
                        key=lambda a: int(a["frame"]))
    if op_touches:
        last = op_touches[-1]
        later_attacking = [t for t in touches
                           if int(last["frame"]) < int(t["frame"]) <= horizon and ok(t)]
        if not later_attacking:
            return last, "operator_touch"
    prior = sorted((t for t in touches if int(t["frame"]) <= horizon and ok(t)),
                   key=lambda t: (int(t["frame"]), float(t.get("score") or 0.0)))
    if not prior:
        return None, None
    for t in reversed(prior):
        dv = _velocity_change(frames, int(t["frame"]), fps) if frames else None
        if dv is not None and dv >= STRIKE_MIN_DV_M_S:
            return t, "velocity_change"
    return prior[-1], "last_touch"


def _dedupe_touches(touches: Sequence[Mapping]) -> list[Mapping]:
    best: dict[tuple[int, str | None], Mapping] = {}
    for t in touches:
        key = (int(t["frame"]), t.get("player_id"))
        if key not in best or float(t.get("score") or 0) > float(best[key].get("score") or 0):
            best[key] = t
    return sorted(best.values(), key=lambda t: int(t["frame"]))


def _buildup_start(strike: int, scorer_pid: str | None, touches: Sequence[Mapping],
                   roles: Mapping[str, str]) -> tuple[int, str]:
    my_team = _team(roles.get(scorer_pid)) if scorer_pid else None
    chain_start, prev = strike, strike
    for t in sorted((t for t in touches if int(t["frame"]) < strike),
                    key=lambda t: -int(t["frame"])):
        f = int(t["frame"])
        team = _team(roles.get(t.get("player_id")))
        if prev - f > MAX_CHAIN_GAP_FRAMES:
            break
        if my_team and team and team != my_team:
            break
        chain_start, prev = f, f
    if strike - chain_start >= MIN_BUILDUP_LEAD:
        return chain_start, "possession_chain"
    return strike - DEFAULT_BUILDUP_LEAD, "strike_minus_default"


def _pick_keeper(roles: Mapping[str, str], goal_end: str | None,
                 root_xy: Mapping[str, tuple[np.ndarray, np.ndarray]],
                 near_frame: int | None) -> str | None:
    gks = sorted(p for p, r in roles.items() if r.endswith("_gk"))
    if len(gks) <= 1 or goal_end is None or near_frame is None:
        return gks[0] if gks else None
    goal_x = 0.0 if goal_end == "left" else PITCH_LENGTH

    def dist(pid: str) -> float:
        if pid not in root_xy:
            return 1e9
        fr, xy = root_xy[pid]
        if len(fr) == 0:
            return 1e9
        i = int(np.argmin(np.abs(fr - near_frame)))
        return abs(float(xy[i, 0]) - goal_x)
    return min(gks, key=dist)


def _keeper_dive_frame(pid: str | None, root_xy: Mapping[str, tuple[np.ndarray, np.ndarray]],
                       lo: int, hi: int, goal_end: str | None, fps: float) -> int | None:
    if pid is None or pid not in root_xy or hi < lo:
        return None
    fr, xy = root_xy[pid]
    if len(fr) < 5:
        return None
    lateral = xy[:, 1]  # goals sit on x = 0/105, so lateral == pitch y
    k = np.ones(3) / 3.0
    sm = np.convolve(np.pad(lateral, 1, mode="edge"), k, mode="valid")
    v = np.abs(np.gradient(sm, fr.astype(float))) * fps
    mask = (fr >= lo) & (fr <= hi)
    if not mask.any():
        return None
    idx = np.where(mask)[0]
    return int(fr[idx[int(np.argmax(v[idx]))]])


def derive_from_data(
    *,
    ball_frames: Mapping[int, Sequence[float]],
    fps: float,
    events: Sequence[Mapping],
    operator_anchors: Sequence[Mapping],
    roles: Mapping[str, str],
    root_xy: Mapping[str, tuple[np.ndarray, np.ndarray]],
    goal_check: Mapping | None = None,
) -> dict:
    """Pure moment derivation. ``root_xy[pid] = (frames, xy[N,2])``."""
    sources: dict[str, str] = {}
    impact, src = _pick_impact(operator_anchors, events, goal_check)
    if src:
        sources["impact"] = src

    line_cross, goal_end = None, None
    lc = (goal_check or {}).get("line_cross")
    if lc and lc.get("frame") is not None:
        line_cross = int(round(float(lc["frame"])))  # e.g. 393.98 -> 394
        sources["line_cross"] = "goal_check"
        if lc.get("xyz"):
            goal_end = _goal_end_for_x(float(lc["xyz"][0]))
        elif goal_check.get("goal_end_x") is not None:
            goal_end = _goal_end_for_x(float(goal_check["goal_end_x"]))
    if line_cross is None:
        hi = impact + 3 if impact is not None else None
        found = None
        # latest crossing at/before the impact (a goal), else the first one
        cur = None
        while True:
            c = find_line_crossing(ball_frames, after=cur, before=hi)
            if c is None:
                break
            found, cur = c, c[0]
            if impact is None:
                break
        if found:
            line_cross, goal_end = found[0], found[1]
            sources["line_cross"] = "track_crossing"
    if goal_end is None and goal_check and goal_check.get("goal_end_x") is not None:
        goal_end = _goal_end_for_x(float(goal_check["goal_end_x"]))
    ref = impact if impact is not None else line_cross

    touches = _dedupe_touches(
        [e for e in events if e.get("kind") == "touch" and e.get("player_id")]
        + [a for a in operator_anchors if a.get("state") == "player_touch" and a.get("player_id")])
    keeper = _pick_keeper(roles, goal_end, root_xy, impact if impact is not None else line_cross)
    defending = _team(roles.get(keeper)) if keeper else None
    touch_row, src = _pick_strike(ref, touches, operator_anchors, ball_frames, fps,
                                  keeper_pid=keeper, defending_team=defending, roles=roles)
    strike = int(touch_row["frame"]) if touch_row else None
    scorer = touch_row.get("player_id") if touch_row else None
    if src:
        sources["strike"] = src

    dive = None
    if strike is not None and ref is not None:
        dive = _keeper_dive_frame(keeper, root_xy, strike - KEEPER_DIVE_PRE_FRAMES, ref,
                                  goal_end, fps)
    buildup = None
    if strike is not None:
        buildup, sources["buildup_start"] = _buildup_start(strike, scorer, touches, roles)
    return {
        "strike": strike, "line_cross": line_cross, "impact": impact,
        "keeper_dive": dive, "buildup_start": buildup,
        "scorer_pid": scorer, "keeper_pid": keeper, "goal_end": goal_end,
        "sources": sources,
    }


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[shorts_moments] cannot read %s: %s", path, exc)
        return None


def _load_root_xy(output_dir: Path, pids: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid in pids:
        p = output_dir / "refined_poses" / f"{pid}_refined.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            out[pid] = (np.asarray(z["frames"], dtype=int),
                        np.asarray(z["root_t"], dtype=float)[:, :2])
    return out


def derive_moments(output_dir: Path, shot: str) -> dict:
    """Moments for ``shot`` from ``output_dir``'s ball / refined_poses /
    players.json outputs. Missing inputs yield ``None`` moments rather than
    raising (the framing/template layer reports unresolved moments)."""
    from src.utils.player_names import load_kit_roles

    output_dir = Path(output_dir)
    ball_dir = output_dir / "ball"
    track = _read_json(ball_dir / f"{shot}_ball_track.json") or {}
    fps = float(track.get("fps") or 30.0)
    frames = {int(f["frame"]): f["world_xyz"] for f in track.get("frames", [])
              if f.get("world_xyz") is not None}
    diag = _read_json(ball_dir / f"{shot}_ball_diag.json") or {}
    anchors = (_read_json(ball_dir / f"{shot}_ball_anchors.json") or {}).get("anchors", [])
    roles = load_kit_roles(output_dir)
    root_xy = _load_root_xy(output_dir, sorted(roles))
    return derive_from_data(
        ball_frames=frames, fps=fps, events=diag.get("events", []),
        operator_anchors=anchors, roles=roles, root_xy=root_xy,
        goal_check=diag.get("goal_check"))

"""Persisted ball anchor data — user-supplied per-frame ball positions
and state tags. Read by ``BallStage`` as a Layer 5 input before the
WASB detection loop runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


BallAnchorState = Literal[
    "grounded",
    "airborne_low",
    "airborne_mid",
    "airborne_high",
    "kick",
    "catch",
    "bounce",
    "header",
    "volley",
    "chest",
    "player_touch",
    "goal_impact",
    "off_screen_flight",
]

_VALID_STATES: frozenset[str] = frozenset({
    "grounded", "airborne_low", "airborne_mid", "airborne_high",
    "kick", "catch", "bounce", "header", "volley", "chest",
    "player_touch", "goal_impact", "off_screen_flight",
})


@dataclass(frozen=True)
class BallAnchor:
    """One per-frame anchor.

    ``player_id`` and ``bone`` are required when ``state == "player_touch"``;
    they identify which player's body part the ball is contacting at this
    frame so the ball stage can drive the trajectory through that bone's
    actual world position (via SMPL forward kinematics on the player's
    hmr_world track). Both are ``None`` for all other states.

    ``goal_element`` is required when ``state == "goal_impact"``; it names
    which part of the goal frame or net the ball struck (one of ``"post"``,
    ``"crossbar"``, ``"back_net"``, ``"side_net"``) so the ball stage can
    pin the trajectory through the geometrically-known contact point.

    ``touch_type`` is an optional sub-tag accepted only on
    ``"player_touch"`` anchors: ``"shot"`` for a deliberate strike at
    goal, ``"volley"`` for a mid-flight strike, or ``None`` for a plain
    contact (pass, control, dribble). Selecting ``"shot"`` or
    ``"volley"`` enables the ``spin`` sub-tag below.

    ``spin`` is an optional hint accepted on ``"player_touch"`` anchors
    whose ``touch_type`` is ``"shot"`` or ``"volley"``: a categorical
    preset (e.g. ``"instep_curl_right"``, ``"topspin"``) that the ball
    stage maps to an angular-velocity seed for the Magnus fitter and
    that relaxes the Magnus-acceptance threshold for the flight
    segment containing the anchor.

    ``confidence`` is the detector's score for an auto-generated anchor
    (carried over from ``BallEvent.score``), clamped to ``[0, 1]``.
    Manual anchors are user-supplied ground truth and default to ``1.0``.
    The web editor uses it to render auto suggestions distinctly from
    confirmed/manual entries.

    ``end_frame`` is the inclusive end of a *span* event (e.g. a
    ``carry``/possession dribble); ``None`` for point events. When set it
    must be greater than ``frame``.

    ``landmark`` is a pitch-feature coincidence for a grounded anchor: a LANDMARK_CATALOGUE
    name (point) or "line:<LINE_CATALOGUE name>". The ball stage snaps the
    anchor's world x,y to the feature (exact hard knot); the clicked pixel
    remains authoring provenance. None for ordinary anchors.
    """
    frame: int
    # None only when state == "off_screen_flight".
    image_xy: tuple[float, float] | None
    state: BallAnchorState
    player_id: str | None = None
    bone: str | None = None
    goal_element: str | None = None
    touch_type: str | None = None
    spin: str | None = None
    confidence: float = 1.0
    end_frame: int | None = None
    landmark: str | None = None


@dataclass(frozen=True)
class DismissedAuto:
    """One operator-dismissed auto suggestion. Identity is the full
    (frame, state, player_id, bone) tuple: auto sets regenerate every run,
    so a dismissal only suppresses an auto anchor that still matches
    exactly; otherwise it is inert."""
    frame: int
    state: str
    player_id: str | None = None
    bone: str | None = None


@dataclass(frozen=True)
class BallAnchorSet:
    clip_id: str
    image_size: tuple[int, int]
    anchors: tuple[BallAnchor, ...]
    # Operator-authored shot chains: each entry is >= 2 strictly-ascending
    # member anchor frames (strike -> [deflections...] -> terminal impact).
    # Grouping only — members are ordinary anchors; the ball stage validates
    # each chain and reports warnings in the diag sidecar.
    shot_chains: tuple[tuple[int, ...], ...] = ()
    # Operator-dismissed auto suggestions (exhaustive-annotation workflow):
    # matching auto anchors are excluded from the merge; the recall report
    # counts them as reviewed false positives.
    dismissed_auto: tuple[DismissedAuto, ...] = ()

    @classmethod
    def load(cls, path: Path) -> "BallAnchorSet":
        # Lazy import to avoid a cycle: ball_anchor_heights imports nothing
        # from this module, but loaders for tests sometimes import both.
        from src.utils.ball_anchor_heights import VALID_BONES, VALID_GOAL_ELEMENTS
        from src.utils.ball_spin_presets import (
            SPIN_ENABLED_STATES,
            SPIN_ENABLED_TOUCH_TYPES,
            VALID_SPIN_PRESETS,
            VALID_TOUCH_TYPES,
        )

        with path.open() as fh:
            data = json.load(fh)
        anchors = []
        for a in data.get("anchors", []):
            state = str(a["state"])
            if state not in _VALID_STATES:
                raise ValueError(f"unknown ball anchor state: {state!r}")
            raw_xy = a.get("image_xy")
            if raw_xy is None:
                if state != "off_screen_flight":
                    raise ValueError(
                        f"image_xy is required for state {state!r} "
                        f"(only off_screen_flight may omit it)"
                    )
                image_xy = None
            else:
                image_xy = (float(raw_xy[0]), float(raw_xy[1]))
            player_id = a.get("player_id")
            bone = a.get("bone")
            goal_element = a.get("goal_element")
            if state == "player_touch":
                if not player_id:
                    raise ValueError(
                        "player_id is required for state 'player_touch'"
                    )
                if not bone:
                    raise ValueError(
                        "bone is required for state 'player_touch'"
                    )
                if bone not in VALID_BONES:
                    raise ValueError(
                        f"unknown bone {bone!r}; valid: {sorted(VALID_BONES)}"
                    )
            if state == "goal_impact":
                if not goal_element:
                    raise ValueError(
                        "goal_element is required for state 'goal_impact'"
                    )
                if goal_element not in VALID_GOAL_ELEMENTS:
                    raise ValueError(
                        f"unknown goal_element {goal_element!r}; "
                        f"valid: {sorted(VALID_GOAL_ELEMENTS)}"
                    )
            touch_type = a.get("touch_type")
            if touch_type:
                if state != "player_touch":
                    raise ValueError(
                        f"touch_type is only valid on state 'player_touch'; "
                        f"got state {state!r}"
                    )
                if touch_type not in VALID_TOUCH_TYPES:
                    raise ValueError(
                        f"unknown touch_type {touch_type!r}; "
                        f"valid: {sorted(VALID_TOUCH_TYPES)}"
                    )
            spin = a.get("spin")
            if spin:
                if state not in SPIN_ENABLED_STATES:
                    raise ValueError(
                        f"spin is only valid on states {sorted(SPIN_ENABLED_STATES)}; "
                        f"got state {state!r}"
                    )
                if touch_type not in SPIN_ENABLED_TOUCH_TYPES:
                    raise ValueError(
                        "spin requires touch_type in "
                        f"{sorted(SPIN_ENABLED_TOUCH_TYPES)}; "
                        f"got touch_type {touch_type!r}"
                    )
                if spin not in VALID_SPIN_PRESETS:
                    raise ValueError(
                        f"unknown spin preset {spin!r}; "
                        f"valid: {sorted(VALID_SPIN_PRESETS)}"
                    )
            landmark = a.get("landmark")
            if landmark:
                if state != "grounded":
                    raise ValueError(
                        f"landmark is only valid on state 'grounded'; "
                        f"got state {state!r}"
                    )
                from src.utils.pitch_landmarks import LANDMARK_CATALOGUE
                from src.utils.pitch_lines_catalogue import LINE_CATALOGUE
                if landmark.startswith("line:"):
                    if landmark[5:] not in LINE_CATALOGUE:
                        raise ValueError(
                            f"unknown landmark line {landmark!r}"
                        )
                elif landmark not in LANDMARK_CATALOGUE:
                    raise ValueError(f"unknown landmark {landmark!r}")
            confidence = float(a.get("confidence", 1.0))
            confidence = min(1.0, max(0.0, confidence))
            raw_end = a.get("end_frame")
            end_frame = int(raw_end) if raw_end is not None else None
            if end_frame is not None and end_frame <= int(a["frame"]):
                raise ValueError(
                    f"end_frame ({end_frame}) must be greater than frame "
                    f"({a['frame']})"
                )
            anchors.append(BallAnchor(
                frame=int(a["frame"]),
                image_xy=image_xy,
                state=state,  # type: ignore[arg-type]
                player_id=str(player_id) if player_id else None,
                bone=str(bone) if bone else None,
                goal_element=str(goal_element) if goal_element else None,
                touch_type=str(touch_type) if touch_type else None,
                spin=str(spin) if spin else None,
                confidence=confidence,
                end_frame=end_frame,
                landmark=str(landmark) if landmark else None,
            ))
        shot_chains: list[tuple[int, ...]] = []
        for chain in data.get("shot_chains", []):
            frames = tuple(int(f) for f in chain)
            if len(frames) < 2:
                raise ValueError(
                    f"shot_chain needs >= 2 member frames; got {frames}"
                )
            if any(b <= a_ for a_, b in zip(frames, frames[1:])):
                raise ValueError(
                    f"shot_chain frames must be strictly ascending; got {frames}"
                )
            shot_chains.append(frames)
        dismissed: list[DismissedAuto] = []
        for d in data.get("dismissed_auto", []):
            dismissed.append(DismissedAuto(
                frame=int(d["frame"]),
                state=str(d["state"]),
                player_id=(str(d["player_id"]) if d.get("player_id") else None),
                bone=(str(d["bone"]) if d.get("bone") else None),
            ))
        return cls(
            clip_id=str(data["clip_id"]),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            anchors=tuple(anchors),
            shot_chains=tuple(shot_chains),
            dismissed_auto=tuple(dismissed),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(
                asdict(self), fh, indent=2,
                default=lambda v: list(v) if isinstance(v, tuple) else v,
            )

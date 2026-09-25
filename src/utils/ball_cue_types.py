"""Shared evidence type for the single-camera ball event-cue modules
(``ball_cue_audio``, ``ball_cue_net``, ``ball_cue_blur``) and their
fusion layer (``ball_cue_fusion``).

IC-A is landing ``CueEvidence`` in ``src/utils/ball_hybrid_types.py`` as
part of the hybrid foundation. This module defines the identical shape
(``CueEvidence(frame:int, kind:str, cue:str, conf:float, xyz:tuple|None,
uv:tuple|None)``) so the cue modules aren't blocked on that commit
landing first. Once ``ball_hybrid_types.CueEvidence`` exists, downstream
code should import from there instead -- this module can then either
re-export it or be deleted; see the IC-D wiring note for the exact
switch-over instructions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CueEvidence:
    """One piece of candidate evidence for a ball event at ``frame``.

    ``kind`` is a loose label for what kind of event this evidence points
    at (e.g. ``"contact"`` for a generic touch/bounce a single-camera cue
    cannot disambiguate further, or ``"goal_impact"`` when the cue is
    unambiguous about the event type, as with the net-energy cue).
    ``cue`` names the specific cue module/signal that produced this
    evidence (e.g. ``"audio_onset"``, ``"net_energy"``,
    ``"blur_direction_change"``). ``conf`` is in ``[0, 1]``. ``xyz`` is a
    3D world-frame position when the cue can resolve one (e.g. net-energy
    back-projection); ``uv`` is the source pixel location when available.
    """

    frame: int
    kind: str
    cue: str
    conf: float
    xyz: tuple[float, float, float] | None = None
    uv: tuple[float, float] | None = None


__all__ = ["CueEvidence"]

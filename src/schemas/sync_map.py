"""SyncMap — manual alignment of multi-shot clips on a shared timeline.

Each clip in a multi-shot run starts its own internal frame counter at
zero. To compare or fuse animations across shots (e.g. a future
convergence stage that reconciles a player's pose between two shots
covering the same moment), every shot needs a known offset onto a
shared timeline. We pick one shot as the *reference* (offset = 0) and
record each other shot's ``frame_offset`` relative to it.

Sign convention (matches the dashboard's UX):

    frame_offset = matched_frame_in_this_shot - matched_frame_in_reference

i.e. a positive offset means *this shot is N frames ahead of the
reference at the same wall-clock instant*. So shot ``X``'s local frame
``f`` corresponds to reference frame ``f - X.frame_offset``.

The SyncMap is operator-edited from the dashboard's Prepare Shots
panel and persisted to ``output/shots/sync_map.json``. It is *not*
required for any current pipeline stage; the convergence stage (TBD)
will read it later. Storing it separately from ``shots_manifest.json``
keeps the manifest immutable across re-runs of prepare_shots.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ``method`` values record how an alignment was derived. Today only
# ``"manual"`` is produced; the literals here document the future
# automatic-solver placeholders so a downstream consumer can branch on
# trustworthiness without us renaming the field later.
_METHOD_MANUAL = "manual"
_VALID_METHODS = frozenset({
    _METHOD_MANUAL,
    "ball_trajectory",
    "player_formation",
    "hybrid",
    "low_confidence",
})


@dataclass
class Alignment:
    """One shot's offset onto the reference shot's timeline."""

    shot_id: str
    frame_offset: int
    method: str = _METHOD_MANUAL
    confidence: float = 1.0


@dataclass
class SyncMap:
    """Reference shot + per-shot offsets keyed by ``shot_id``."""

    reference_shot: str
    alignments: list[Alignment] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "SyncMap":
        data = json.loads(path.read_text())
        alignments = [Alignment(**a) for a in data.pop("alignments", [])]
        return cls(alignments=alignments, **data)

    def offset_for(self, shot_id: str) -> int:
        """Return the saved frame_offset for ``shot_id`` (0 if absent)."""
        for a in self.alignments:
            if a.shot_id == shot_id:
                return a.frame_offset
        return 0

    def with_alignment(self, alignment: Alignment) -> "SyncMap":
        """Return a new SyncMap with ``alignment`` upserted by shot_id."""
        kept = [a for a in self.alignments if a.shot_id != alignment.shot_id]
        kept.append(alignment)
        kept.sort(key=lambda a: a.shot_id)
        return SyncMap(reference_shot=self.reference_shot, alignments=kept)


def default_sync_map(reference_shot: str, shot_ids: list[str]) -> SyncMap:
    """Build a SyncMap with every shot at offset 0 (identity baseline).

    Used as the starting state when no ``sync_map.json`` is on disk so
    the dashboard can render the editor with one row per shot from the
    moment the operator opens it.
    """
    return SyncMap(
        reference_shot=reference_shot,
        alignments=[
            Alignment(shot_id=sid, frame_offset=0, method=_METHOD_MANUAL)
            for sid in sorted(shot_ids)
        ],
    )


def validate_method(method: str) -> str:
    """Return ``method`` if recognised, else raise ``ValueError``."""
    if method not in _VALID_METHODS:
        raise ValueError(
            f"unknown sync method {method!r}; expected one of "
            f"{sorted(_VALID_METHODS)}"
        )
    return method

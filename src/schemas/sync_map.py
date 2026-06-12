"""SyncMap — alignment of shots onto a shared timeline, per highlight group.

Shots inside one highlight group cover the same real-world moment (live
action + replays from other angles), so offsets are only meaningful
*within* a group. Each ``GroupSync`` picks one member as the *reference*
(offset = 0) and records every other member's ``frame_offset`` relative
to it.

Sign convention (matches the dashboard's UX):

    frame_offset = matched_frame_in_this_shot - matched_frame_in_reference

i.e. a positive offset means *this shot is N frames ahead of the
reference at the same wall-clock instant*. So shot ``X``'s local frame
``f`` corresponds to reference frame ``f - X.frame_offset``.

History: v1 stored a single flat ``reference_shot`` + ``alignments``
(every shot against one global reference). ``load()`` migrates v1 files
into a single group with ``group_id = ""`` — the "ungrouped" bucket the
dashboard renders for manually-added clips.

The SyncMap is written by prepare_shots' auto-aligner
(``method="motion_profile"`` / ``"low_confidence"``) and edited from the
dashboard's Prepare Shots panel (``method="manual"``). Persisted to
``output/shots/sync_map.json``, separate from ``shots_manifest.json`` so
manifest re-runs never clobber operator-tuned offsets.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ``method`` values record how an alignment was derived. ``manual`` =
# operator-edited; ``motion_profile`` = prepare_shots' motion-energy NCC
# aligner; ``low_confidence`` = the aligner's align-ends fallback. The
# remaining literals are placeholders for future tracking-based solvers
# so downstream consumers can branch on trustworthiness without a
# rename later.
_METHOD_MANUAL = "manual"
_VALID_METHODS = frozenset({
    _METHOD_MANUAL,
    "motion_profile",
    "ball_trajectory",
    "player_formation",
    "hybrid",
    "low_confidence",
})


@dataclass
class Alignment:
    """One shot's offset onto its group reference's timeline."""

    shot_id: str
    frame_offset: int
    method: str = _METHOD_MANUAL
    confidence: float = 1.0


@dataclass
class GroupSync:
    """Per-group timeline: reference shot + offsets of member shots."""

    group_id: str
    reference_shot: str
    alignments: list[Alignment] = field(default_factory=list)

    def offset_for(self, shot_id: str) -> int:
        for a in self.alignments:
            if a.shot_id == shot_id:
                return a.frame_offset
        return 0

    def with_alignment(self, alignment: Alignment) -> "GroupSync":
        """Return a new GroupSync with ``alignment`` upserted by shot_id."""
        kept = [a for a in self.alignments if a.shot_id != alignment.shot_id]
        kept.append(alignment)
        kept.sort(key=lambda a: a.shot_id)
        return GroupSync(
            group_id=self.group_id,
            reference_shot=self.reference_shot,
            alignments=kept,
        )


@dataclass
class SyncMap:
    """Group-scoped sync state (v2)."""

    version: int = 2
    groups: list[GroupSync] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write — readers must never observe a torn sync map.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SyncMap":
        data = json.loads(path.read_text())
        if "groups" not in data:
            # v1 flat file → single ungrouped bucket.
            alignments = [
                Alignment(**a) for a in data.get("alignments", [])
            ]
            return cls(groups=[GroupSync(
                group_id="",
                reference_shot=data.get("reference_shot", ""),
                alignments=alignments,
            )])
        groups = [
            GroupSync(
                group_id=g["group_id"],
                reference_shot=g.get("reference_shot", ""),
                alignments=[
                    Alignment(**a) for a in g.get("alignments", [])
                ],
            )
            for g in data.get("groups", [])
        ]
        return cls(version=2, groups=groups)

    def group(self, group_id: str) -> GroupSync | None:
        for g in self.groups:
            if g.group_id == group_id:
                return g
        return None

    def offset_for(self, group_id: str, shot_id: str) -> int:
        """Saved frame_offset for ``shot_id`` in ``group_id`` (0 if absent)."""
        g = self.group(group_id)
        return g.offset_for(shot_id) if g is not None else 0

    def offset_for_shot(self, shot_id: str) -> int:
        """Saved frame_offset for ``shot_id`` in whichever group holds it
        (0 if no group does). Convenience for consumers that key by shot
        alone, e.g. the refined-poses preview endpoint."""
        for g in self.groups:
            for a in g.alignments:
                if a.shot_id == shot_id:
                    return a.frame_offset
        return 0

    def with_group(self, group_sync: GroupSync) -> "SyncMap":
        """Return a new SyncMap with ``group_sync`` upserted by group_id."""
        kept = [g for g in self.groups if g.group_id != group_sync.group_id]
        kept.append(group_sync)
        kept.sort(key=lambda g: g.group_id)
        return SyncMap(version=2, groups=kept)

    def with_group_alignment(
        self,
        group_id: str,
        reference_shot: str,
        alignment: Alignment,
    ) -> "SyncMap":
        """Return a new SyncMap with ``alignment`` upserted into the
        named group (the group is created if absent)."""
        existing = self.group(group_id)
        base = existing if existing is not None else GroupSync(
            group_id=group_id, reference_shot=reference_shot, alignments=[],
        )
        return self.with_group(base.with_alignment(alignment))


def default_group_sync(
    group_id: str,
    reference_shot: str,
    shot_ids: list[str],
) -> GroupSync:
    """Build a GroupSync with every shot at offset 0 (identity baseline).

    Used as the starting state when a manifest group has no saved sync
    yet, so the dashboard renders one row per member immediately.
    """
    return GroupSync(
        group_id=group_id,
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

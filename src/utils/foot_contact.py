"""Per-foot ground-contact spans — the shared currency between the
foot-contact-aware locomotion components (see
docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md).

This module currently holds only the frozen dataclasses and their JSON
serialization (Task 1 of the implementation plan). Detection algorithms
(``detect_contacts`` ray-casting ViTPose ankles, and the FK-only
``derive_contacts_from_fk`` fallback) land in Task 3 — deliberately not
implemented here so Tasks 3 and 4 can build against a stable interface
in parallel.

``FootContacts`` is meant to travel as a JSON sidecar
(``{shot}__{pid}_foot_contacts.json``, see ``src/schemas/foot_contacts.py``
once Task 5 adds it) alongside the ``hmr_world`` / ``refined_poses`` npz
tracks it was computed from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactSpan:
    """One contiguous run of a single foot being planted on the ground.

    ``start``/``end`` are frame-array positions (half-open: frames
    ``[start, end)``), aligned with whatever per-frame array
    (``FootContacts.in_contact`` today, an ``SmplWorldTrack``/
    ``RefinedPose``'s ``frames`` later) the enclosing ``FootContacts``
    was computed from. ``pin`` is the robust (median) world-frame pitch
    position (metres, z-up) the foot should be held at for the whole
    span — ``pin[2]`` is the pitch-plane ankle/foot-plane height
    constant used by the detector, not a per-span measurement.
    """

    side: int          # 0 = left, 1 = right
    start: int
    end: int
    pin: np.ndarray     # (3,) pitch-world metres

    def to_json(self) -> dict:
        return {
            "side": int(self.side),
            "start": int(self.start),
            "end": int(self.end),
            "pin": [float(x) for x in np.asarray(self.pin, dtype=float)],
        }

    @classmethod
    def from_json(cls, d: dict) -> "ContactSpan":
        return cls(
            side=int(d["side"]),
            start=int(d["start"]),
            end=int(d["end"]),
            pin=np.array(d["pin"], dtype=float),
        )


@dataclass(frozen=True)
class FootContacts:
    """Per-foot contact state for one player track.

    ``in_contact``/``quality`` are dense per-frame-position arrays
    (shape ``(n_frames, 2)``, column order ``[L, R]``) aligned 1:1 with
    the track's row index — NOT resampled by :meth:`shifted`, see there.
    ``spans`` is the derived contiguous-run view of ``in_contact`` with
    a robust pin position attached to each run; downstream consumers
    that only need "is this frame's foot planted, and where" can use
    ``in_contact`` directly, while consumers that need a stable pin
    target (root-solve, foot-lock IK) use ``spans``.
    """

    n_frames: int
    in_contact: np.ndarray          # (F, 2) bool  [L, R]
    quality: np.ndarray             # (F, 2) float in [0, 1]
    spans: tuple[ContactSpan, ...]

    def to_json(self) -> dict:
        return {
            "n_frames": int(self.n_frames),
            "in_contact": np.asarray(self.in_contact, dtype=bool).tolist(),
            "quality": np.asarray(self.quality, dtype=float).tolist(),
            "spans": [s.to_json() for s in self.spans],
        }

    @classmethod
    def from_json(cls, d: dict) -> "FootContacts":
        return cls(
            n_frames=int(d["n_frames"]),
            in_contact=np.array(d["in_contact"], dtype=bool),
            quality=np.array(d["quality"], dtype=float),
            spans=tuple(ContactSpan.from_json(s) for s in d["spans"]),
        )

    def shifted(self, offset: int) -> "FootContacts":
        """Re-base span frame indices by ``offset`` (e.g. a sync_map
        shot->reference-timeline offset, or ``-i_first`` when a caller
        trims its own per-frame arrays and needs the spans to stay
        aligned with the trimmed row indices).

        Only the span ``start``/``end`` (frame-position labels) move.
        ``in_contact``/``quality``/``n_frames`` are copied unchanged —
        they are dense arrays already indexed 0..n_frames-1 by array
        position, and a pure relabelling of what "position 0" means
        doesn't itself resample them. A caller that also needs to crop
        the dense arrays to a sub-range (e.g. after slicing its own
        track to an anchored span) does that slicing itself and then
        calls ``shifted`` for the span bookkeeping.
        """
        return FootContacts(
            n_frames=self.n_frames,
            in_contact=np.array(self.in_contact, copy=True),
            quality=np.array(self.quality, copy=True),
            spans=tuple(
                ContactSpan(
                    side=s.side,
                    start=s.start + offset,
                    end=s.end + offset,
                    pin=np.array(s.pin, copy=True),
                )
                for s in self.spans
            ),
        )

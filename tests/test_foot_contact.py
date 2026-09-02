"""Tests for the FootContacts / ContactSpan dataclasses.

Task 1 scope only: dataclass shape + JSON round-trip + ``shifted()``.
Detection algorithms (``detect_contacts`` / ``derive_contacts_from_fk``)
land in Task 3 — not tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.foot_contact import ContactSpan, FootContacts


def _sample_contacts() -> FootContacts:
    n = 10
    in_contact = np.zeros((n, 2), dtype=bool)
    in_contact[2:6, 0] = True
    in_contact[5:9, 1] = True
    quality = np.zeros((n, 2), dtype=float)
    quality[2:6, 0] = 0.9
    quality[5:9, 1] = 0.8
    spans = (
        ContactSpan(side=0, start=2, end=6, pin=np.array([1.0, 2.0, 0.05])),
        ContactSpan(side=1, start=5, end=9, pin=np.array([3.0, -1.0, 0.05])),
    )
    return FootContacts(n_frames=n, in_contact=in_contact, quality=quality, spans=spans)


def test_contact_span_to_json_from_json_round_trip() -> None:
    span = ContactSpan(side=1, start=5, end=9, pin=np.array([3.0, -1.0, 0.05]))
    d = span.to_json()
    restored = ContactSpan.from_json(d)
    assert restored.side == span.side
    assert restored.start == span.start
    assert restored.end == span.end
    np.testing.assert_allclose(restored.pin, span.pin)


def test_contact_span_to_json_is_plain_json_types() -> None:
    span = ContactSpan(side=0, start=2, end=6, pin=np.array([1.0, 2.0, 0.05]))
    d = span.to_json()
    assert isinstance(d["side"], int)
    assert isinstance(d["start"], int)
    assert isinstance(d["end"], int)
    assert isinstance(d["pin"], list)
    assert all(isinstance(x, float) for x in d["pin"])


def test_foot_contacts_to_json_from_json_round_trip() -> None:
    fc = _sample_contacts()
    d = fc.to_json()
    restored = FootContacts.from_json(d)
    assert restored.n_frames == fc.n_frames
    np.testing.assert_array_equal(restored.in_contact, fc.in_contact)
    np.testing.assert_allclose(restored.quality, fc.quality)
    assert len(restored.spans) == len(fc.spans)
    for a, b in zip(restored.spans, fc.spans):
        assert a.side == b.side
        assert a.start == b.start
        assert a.end == b.end
        np.testing.assert_allclose(a.pin, b.pin)


def test_foot_contacts_to_json_round_trips_through_real_json_module() -> None:
    import json

    fc = _sample_contacts()
    text = json.dumps(fc.to_json())
    restored = FootContacts.from_json(json.loads(text))
    np.testing.assert_array_equal(restored.in_contact, fc.in_contact)


def test_foot_contacts_empty_spans_round_trip() -> None:
    fc = FootContacts(
        n_frames=5,
        in_contact=np.zeros((5, 2), dtype=bool),
        quality=np.zeros((5, 2), dtype=float),
        spans=(),
    )
    restored = FootContacts.from_json(fc.to_json())
    assert restored.spans == ()
    assert restored.n_frames == 5


def test_foot_contacts_shifted_offsets_span_frame_indices() -> None:
    fc = _sample_contacts()
    shifted = fc.shifted(-2)
    assert len(shifted.spans) == len(fc.spans)
    for orig, new in zip(fc.spans, shifted.spans):
        assert new.start == orig.start - 2
        assert new.end == orig.end - 2
        assert new.side == orig.side
        np.testing.assert_allclose(new.pin, orig.pin)


def test_foot_contacts_shifted_preserves_dense_arrays_and_n_frames() -> None:
    """shifted() re-labels span frame numbers (e.g. for sync_map offset
    application); it does not resample the dense per-position arrays."""
    fc = _sample_contacts()
    shifted = fc.shifted(7)
    assert shifted.n_frames == fc.n_frames
    np.testing.assert_array_equal(shifted.in_contact, fc.in_contact)
    np.testing.assert_allclose(shifted.quality, fc.quality)


def test_foot_contacts_shifted_zero_is_a_no_op_copy() -> None:
    fc = _sample_contacts()
    shifted = fc.shifted(0)
    assert shifted is not fc
    for orig, new in zip(fc.spans, shifted.spans):
        assert new.start == orig.start
        assert new.end == orig.end


def test_foot_contacts_is_frozen() -> None:
    fc = _sample_contacts()
    with pytest.raises(Exception):
        fc.n_frames = 99  # type: ignore[misc]


def test_contact_span_is_frozen() -> None:
    span = ContactSpan(side=0, start=0, end=1, pin=np.zeros(3))
    with pytest.raises(Exception):
        span.start = 5  # type: ignore[misc]

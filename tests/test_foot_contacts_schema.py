"""Round-trip tests for the foot-contact sidecar schema."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.schemas.foot_contacts import load_foot_contacts, save_foot_contacts
from src.utils.foot_contact import ContactSpan, FootContacts


def _sample_contacts() -> FootContacts:
    in_contact = np.zeros((10, 2), dtype=bool)
    in_contact[2:7, 0] = True
    quality = np.zeros((10, 2), dtype=float)
    quality[2:7, 0] = 0.8
    spans = (
        ContactSpan(side=0, start=2, end=7, pin=np.array([10.0, 20.0, 0.05])),
    )
    return FootContacts(
        n_frames=10, in_contact=in_contact, quality=quality, spans=spans,
    )


@pytest.mark.unit
def test_sidecar_round_trip(tmp_path):
    fc = _sample_contacts()
    path = tmp_path / "gberch__P001_foot_contacts.json"
    save_foot_contacts(
        path, fc, shot_id="gberch", player_id="P001", anchor_mode="contact",
    )
    loaded, meta = load_foot_contacts(path)
    assert meta["shot_id"] == "gberch"
    assert meta["player_id"] == "P001"
    assert meta["anchor_mode"] == "contact"
    assert meta["version"] == 1
    assert loaded.n_frames == fc.n_frames
    np.testing.assert_array_equal(loaded.in_contact, fc.in_contact)
    np.testing.assert_allclose(loaded.quality, fc.quality)
    assert len(loaded.spans) == 1
    assert (loaded.spans[0].side, loaded.spans[0].start, loaded.spans[0].end) == (0, 2, 7)
    np.testing.assert_allclose(loaded.spans[0].pin, [10.0, 20.0, 0.05])


@pytest.mark.unit
def test_load_rejects_foreign_json(tmp_path):
    path = tmp_path / "not_a_sidecar.json"
    path.write_text(json.dumps({"schema": "something_else"}))
    with pytest.raises(ValueError, match="not a foot_contacts sidecar"):
        load_foot_contacts(path)

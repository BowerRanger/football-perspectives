"""Tests for the shot_id sanitiser used by prepare_shots and the dashboard."""

from __future__ import annotations

import pytest

from src.schemas.shots import _sanitise_shot_id


@pytest.mark.unit
def test_passes_through_safe_id():
    assert _sanitise_shot_id("match_first_half") == "match_first_half"


@pytest.mark.unit
def test_strips_spaces():
    assert _sanitise_shot_id("my clip") == "myclip"


@pytest.mark.unit
def test_strips_brackets_and_dots():
    assert _sanitise_shot_id("my.clip[v2]") == "myclipv2"


@pytest.mark.unit
def test_keeps_underscore_and_hyphen():
    assert _sanitise_shot_id("clip_a-b") == "clip_a-b"


@pytest.mark.unit
def test_truncates_to_64_chars():
    long = "x" * 100
    assert _sanitise_shot_id(long) == "x" * 64


@pytest.mark.unit
def test_empty_input_raises():
    with pytest.raises(ValueError):
        _sanitise_shot_id("")


@pytest.mark.unit
def test_all_invalid_chars_raises():
    with pytest.raises(ValueError):
        _sanitise_shot_id("...   ")

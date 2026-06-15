"""Export manual ball anchors to WASB/CVAT training-annotation format
(the gold-label half of the ball-detector fine-tuning scaffold)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_finetune_export import anchors_to_cvat_xml


def _points(xml: str):
    root = ET.fromstring(xml)
    out = {}
    for track in root.iter("track"):
        for p in track.iter("points"):
            fid = int(p.attrib["frame"])
            uig = next((c.text for c in p if c.attrib.get("name") == "used_in_game"), None)
            out[fid] = {
                "outside": p.attrib["outside"],
                "occluded": p.attrib["occluded"],
                "points": p.attrib["points"],
                "used_in_game": uig,
            }
    return out


def test_labelled_anchor_becomes_visible_used_in_game_point():
    anchors = [BallAnchor(frame=5, image_xy=(100.0, 200.0), state="grounded")]
    pts = _points(anchors_to_cvat_xml("gberch", anchors))
    assert 5 in pts
    assert pts[5]["outside"] == "0"
    assert pts[5]["occluded"] == "0"
    assert pts[5]["used_in_game"] == "1"
    assert pts[5]["points"] == "100.00,200.00"


def test_off_screen_anchor_is_marked_outside():
    anchors = [BallAnchor(frame=8, image_xy=None, state="off_screen_flight")]
    pts = _points(anchors_to_cvat_xml("gberch", anchors))
    assert pts[8]["outside"] == "1"


def test_player_touch_anchor_is_a_visible_ball_label():
    # a touch anchor's image_xy is the operator's ball pixel — a gold label
    anchors = [BallAnchor(frame=49, image_xy=(1612.0, 525.0), state="player_touch",
                          player_id="P014", bone="r_foot")]
    pts = _points(anchors_to_cvat_xml("gberch", anchors))
    assert pts[49]["points"] == "1612.00,525.00" and pts[49]["used_in_game"] == "1"


def test_xml_is_well_formed_and_has_one_track():
    anchors = [BallAnchor(frame=i, image_xy=(float(i), 2.0 * i), state="grounded")
               for i in range(3)]
    root = ET.fromstring(anchors_to_cvat_xml("clip", anchors))
    assert len(list(root.iter("track"))) == 1
    assert len(list(root.iter("points"))) == 3

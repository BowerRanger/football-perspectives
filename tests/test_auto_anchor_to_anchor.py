from src.schemas.anchor import Anchor
from src.utils.auto_anchor import keypoints_to_anchor


def test_keypoints_to_anchor_maps_world_coords():
    pixels = {
        1: (100.0, 900.0),     # far-left corner -> (0, 68, 0)
        4: (200.0, 880.0),     # -> (0, 54.16, 0)
        16: (950.0, 220.0),    # goalpost top -> (0, 30.34, 2.44) (non-coplanar)
        5: (500.0, 700.0),
        7: (1700.0, 880.0),
        9: (600.0, 690.0),
    }
    anchor = keypoints_to_anchor(pixels, frame=0, min_points=4)
    assert isinstance(anchor, Anchor)
    assert anchor.frame == 0
    assert len(anchor.landmarks) == 6
    lm16 = next(l for l in anchor.landmarks if l.image_xy == (950.0, 220.0))
    assert lm16.world_xyz[2] == 2.44


def test_keypoints_to_anchor_returns_none_below_min_points():
    anchor = keypoints_to_anchor({1: (1.0, 2.0)}, frame=0, min_points=4)
    assert anchor is None


def test_keypoints_to_anchor_skips_unknown_ids():
    anchor = keypoints_to_anchor(
        {1: (1.0, 2.0), 999: (3.0, 4.0), 4: (5.0, 6.0),
         5: (7.0, 8.0), 7: (9.0, 10.0)},
        frame=3, min_points=4,
    )
    assert len(anchor.landmarks) == 4  # 999 dropped

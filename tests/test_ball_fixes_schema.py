"""BallFixSet sidecar round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.schemas.ball_fixes import BallFix, BallFixSet


@pytest.mark.unit
def test_round_trip(tmp_path: Path):
    fs = BallFixSet(
        clip_id="origi01",
        group_id="",
        cross_replay={
            "partner_shots": ["origi02"],
            "saved_offset": -142.0,
            "refined_offset": -144.0,
            "offset_disagreement_frames": 2.0,
            "n_pairs": 21,
            "n_inlier_fixes": 11,
            "median_ray_miss_m": 0.31,
            "median_parallax_deg": 26.4,
        },
        fixes=(
            BallFix(frame=328, xyz=(12.7, 50.1, 8.28), ray_miss_m=0.15,
                    parallax_deg=23.1, partner_shot="origi02",
                    partner_frame=184),
        ),
    )
    p = tmp_path / "origi01_ball_fixes.json"
    fs.save(p)
    loaded = BallFixSet.load(p)
    assert loaded == fs
    assert loaded.fixes[0].xyz == pytest.approx((12.7, 50.1, 8.28))

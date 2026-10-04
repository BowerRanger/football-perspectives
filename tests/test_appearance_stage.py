"""AppearanceStage end-to-end on a synthetic output dir (no ML, no camera)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.shots import Shot, ShotsManifest
from src.stages.appearance import AppearanceStage
from src.utils.kit_palette import hex_to_rgb
from src.utils.player_names import load_kit_roles

W, H, N_FRAMES = 640, 360, 12
KITS = {  # name -> (shirt, shorts, socks) hex
    "red": ("#c8102e", "#c8102e", "#c8102e"),
    "blue": ("#034694", "#034694", "#f2f2f2"),
    "gk": ("#33373b", "#2a2d30", "#2a2d30"),
    "ref": ("#e8df3a", "#151515", "#151515"),
}


def _bgr(hex_str: str) -> tuple[int, int, int]:
    r, g, b = (int(c) for c in hex_to_rgb(hex_str))
    return (b, g, r)


def _kp(cx: int, top: int) -> list[list[float]]:
    kp = np.zeros((17, 3))
    pts = {5: (cx - 14, top + 22), 6: (cx + 14, top + 22), 7: (cx - 22, top + 40), 8: (cx + 22, top + 40),
           11: (cx - 10, top + 62), 12: (cx + 10, top + 62), 13: (cx - 10, top + 85), 14: (cx + 10, top + 85),
           15: (cx - 10, top + 110), 16: (cx + 10, top + 110)}
    for i, (x, y) in pts.items():
        kp[i] = (x, y, 0.9)
    return kp.tolist()


def _build(tmp: Path, roster: list[tuple[str, str, float, int]]) -> None:
    """roster: (pid, kit, pitch_x, column_slot)."""
    (tmp / "shots").mkdir()
    (tmp / "hmr_world").mkdir()
    (tmp / "refined_poses").mkdir()
    writer = cv2.VideoWriter(str(tmp / "shots" / "s1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for _ in range(N_FRAMES):
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (40, 150, 40)
        for pid, kit, _, slot in roster:
            cx, top = 40 + 60 * (slot % 10), 20 + 150 * (slot // 10)
            shirt, shorts, socks = KITS[kit]
            img[top + 22:top + 62, cx - 14:cx + 15] = _bgr(shirt)
            img[top + 62:top + 85, cx - 10:cx + 11] = _bgr(shorts)
            img[top + 85:top + 112, cx - 10:cx + 11] = _bgr(socks)
        writer.write(img)
    writer.release()
    for pid, _, px, slot in roster:
        cx, top = 40 + 60 * (slot % 10), 20 + 150 * (slot // 10)
        frames = [{"frame": f, "keypoints": _kp(cx, top)} for f in range(N_FRAMES)]
        (tmp / "hmr_world" / f"s1__{pid}_kp2d.json").write_text(
            json.dumps({"player_id": pid, "shot_id": "s1", "frames": frames}))
        root_t = np.tile([px, 30.0, 0.9], (N_FRAMES, 1))
        np.savez(tmp / "refined_poses" / f"{pid}_refined.npz", frames=np.arange(N_FRAMES), root_t=root_t)
    ShotsManifest(
        source_file="x.mp4", fps=25.0, total_frames=N_FRAMES,
        shots=[Shot(id="s1", start_frame=0, end_frame=N_FRAMES - 1, start_time=0.0, end_time=1.0,
                    clip_file="shots/s1.mp4")],
    ).save(tmp / "shots" / "shots_manifest.json")


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    roster = [(f"P{i:03d}", "red", 20.0 + 3 * i, i) for i in range(5)]
    roster += [(f"P{i:03d}", "blue", 12.0 + 3 * (i - 5), i) for i in range(5, 10)]
    roster += [("P010", "gk", 3.0, 10), ("P011", "ref", 55.0, 11)]
    _build(tmp_path, roster)
    return tmp_path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CLIP_CFG = {"appearance": {"kits": {"home": "liverpool/2025-26/home", "away": "chelsea/2025-26/home",
                                    "away_gk": "chelsea/2025-26/gk", "referee": "referees/yellow"}}}


def test_stage_writes_suggestions_only(out_dir: Path) -> None:
    (out_dir / "players.json").write_text(json.dumps({"P000": {"name": "Op", "kit_role": "away"}}))
    before = _sha(out_dir / "players.json")
    stage = AppearanceStage(CLIP_CFG, out_dir)
    assert not stage.is_complete()
    stage.run()
    assert stage.is_complete()
    assert _sha(out_dir / "players.json") == before            # operator file never touched
    kits = json.loads((out_dir / "appearance" / "kits.json").read_text())
    assert kits["schema"] == "appearance_kits"
    assert set(kits["kits"]) == {"home", "away", "away_gk", "referee"}
    assert kits["kits"]["home"]["shirt"] == "#c8102e" and kits["kits"]["home"]["source"] == "clip"
    sugg = json.loads((out_dir / "appearance" / "players_suggested.json").read_text())
    assert sugg["P000"]["kit_role"] == "home" and sugg["P007"]["kit_role"] == "away"
    assert sugg["P010"]["kit_role"] == "away_gk" and sugg["P011"]["kit_role"] == "referee"
    assert len(sugg) == 12
    # the suggestion pass never read players.json (P000 suggestion ignores the operator's "away")
    # ... but the merge honours the operator:
    assert load_kit_roles(out_dir)["P000"] == "away"
    assert load_kit_roles(out_dir)["P001"] == "home"


def test_stage_library_snap_from_match_config(out_dir: Path) -> None:
    cfg = {"appearance": {"match": {"home_team": "Liverpool", "away_team": "Chelsea", "date": "2025-09-14"}}}
    AppearanceStage(cfg, out_dir).run()
    kits = json.loads((out_dir / "appearance" / "kits.json").read_text())["kits"]
    assert kits["home"]["ref"] == "liverpool/2025-26/home"
    assert kits["away"]["ref"] == "chelsea/2025-26/home"
    assert kits["away_gk"]["ref"] == "chelsea/2025-26/gk"


def test_stage_disabled_and_no_evidence(tmp_path: Path) -> None:
    (tmp_path / "shots").mkdir()
    ShotsManifest(source_file="x.mp4", fps=25.0, total_frames=1, shots=[
        Shot(id="s1", start_frame=0, end_frame=0, start_time=0, end_time=0, clip_file="shots/s1.mp4")
    ]).save(tmp_path / "shots" / "shots_manifest.json")
    AppearanceStage({"appearance": {"enabled": False}}, tmp_path).run()
    AppearanceStage({}, tmp_path).run()                           # no kp2d: warns, writes nothing
    assert not (tmp_path / "appearance").exists()


def test_stage_respects_shot_filter(out_dir: Path) -> None:
    stage = AppearanceStage(CLIP_CFG, out_dir)
    stage.shot_filter = "other"
    stage.run()
    assert not (out_dir / "appearance").exists()

# Ball Phase 3 — Context Prior, Attribution Refinement, Event List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status (2026-07-02): EXECUTED — historical record.** The shipped semantics diverged from this plan during empirical verification: the context prior became a factor VETO (no confidence scaling) and touch_attribution shipped default-off. The spec's §4.2/§4.4 "Measured outcome" paragraphs are authoritative; see .superpowers/sdd/task-8-report.md for the three-wave measurement history.

**Goal:** Implement Phase 3 of `docs/superpowers/specs/2026-07-02-ball-stage-improvement-design.md` (as amended 2026-07-02): (§4.2) a context-prior score modifier that lets existing gates drop scoreboard/crowd false detections; (§4.4) a touch bone-attribution refinement post-pass (validated headroom: gberch strict recall 2/8 vs 4/8 loose ceiling — the gap is wrong-bone labels); (§5.2) the event-list panel with confirm/dismiss, persisted `dismissed_auto`, span `end_frame` editing, and an FP breakdown in the recall report — the exhaustive touch-annotation workflow.

**Architecture:** Two new pure modules (`ball_context_prior.py`, `ball_touch_attribution.py`) follow the established pattern: torch-free logic, frozen `Cfg` dataclass, config block in `default.yaml`, builder in `ball.py`, swallow-with-warning wiring. The prior multiplies each pass-1 detection's confidence (and each second-pass candidate's score) — detections whose *penalized* score falls below existing thresholds are dropped by the gates that already exist, so it is a modifier, never a new gate. Attribution refinement relabels `(player_id, bone)` on touch events after `merge_touch_events` by minimal 3-D bone↔ball-ray gap (reusing `point_to_pixel_ray_distance` / `PlayerContext.joints_at`); it never adds, removes, or re-frames events. Dismissals ride the manual `BallAnchorSet` sidecar and suppress matching auto anchors in `merge_anchors`; the editor's two anchor sections merge into one chronological event list with promote/dismiss.

**Tech Stack:** Python 3.11, pytest, FastAPI + TestClient, numpy, vanilla JS in the single-file editor.

## Scope notes — conscious v1 simplifications (do not "fix" these)

- **Span editing is button-based, not drag**: manual `player_touch` rows get "⇥ end" (set `end_frame` to the current video frame) and clear controls. The spec's "drag handles on the timeline" is deferred as UX polish.
- **Player-proximity prior uses the tracking stage's 2-D boxes** (`tracks/{shot}_tracks.json`), not FK joints — boxes are available before the detect pass at zero extra compute; `class_name == "ball"` tracks are excluded (self-confirmation).
- **Prior weights are deliberately gentle**: any single signal leaves a confident detection above `drop_below`; only combined signals (e.g. static-in-image + no player nearby) drop it. A lone long ball far from every player must survive.
- **Attribution refinement relabels only** — the event frame is untouched (recall matching already tolerates ±2 frames).

## Global Constraints

- Type annotations on all new signatures; frozen dataclasses; never mutate inputs — return new objects (`dataclasses.replace` for relabelled events).
- New utility modules torch-free and import-light.
- Enrichment never kills the stage: prior/attribution/dismissal failures degrade with a `logger.warning` (`# noqa: BLE001` pattern used by sibling blocks).
- Schema changes backward compatible: old JSON without `dismissed_auto` loads unchanged; old readers ignore the new key.
- Config keys verbatim: `ball.context_prior.{enabled,drop_below,pitch_margin_m,pitch_penalty,player_max_dist_px,player_penalty,static_window,static_max_px,static_min_cam_deg,static_penalty,min_factor}`; `ball.touch_attribution.{enabled,window,max_gap_m,margin_m,min_fk_conf}`.
- Acceptance (spec §7 row 3): gberch strict union recall ≥ 4/8 via relabelling; frame-top FPs suppressed with no `detection_coverage` regression on origi/kroupi; event list usable for a full review pass.
- Commit format `<type>: <description>`, no attribution trailers.
- Tests via the repo venv from the repo root: `.venv/bin/python -m pytest`.
- Paths relative to `/Users/joebower/workplace/football-perspectives`.

---

### Task 1: Pure context-prior module (`src/utils/ball_context_prior.py`) + config block

**Files:**
- Create: `src/utils/ball_context_prior.py`
- Modify: `config/default.yaml` (add `context_prior:` block inside the `ball:` mapping, immediately after the `shot_chain:` block, 2-space indentation)
- Test: `tests/test_ball_context_prior.py`

**Interfaces:**
- Consumes: `ankle_ray_to_pitch(uv, *, K, R, t, plane_z, distortion) -> np.ndarray` from `src/utils/foot_anchor.py` (raises `ValueError` on near-parallel rays); `TracksResult.load(path)` from `src/schemas/tracks.py` (`tracks: list[Track]`, each `Track.class_name` in `{"player","goalkeeper","referee","ball"}` with `frames: list[TrackFrame]`, `TrackFrame.bbox = [x1,y1,x2,y2]`).
- Produces (later tasks rely on exactly these):
  - `ContextPriorCfg(enabled: bool = True, drop_below: float = 0.35, pitch_margin_m: float = 5.0, pitch_penalty: float = 0.5, player_max_dist_px: float = 180.0, player_penalty: float = 0.6, static_window: int = 45, static_max_px: float = 3.0, static_min_cam_deg: float = 2.0, static_penalty: float = 0.45, min_factor: float = 0.1)` — frozen dataclass.
  - `class ContextPrior` with `__init__(self, cfg: ContextPriorCfg, *, per_frame_K: dict[int, np.ndarray], per_frame_R: dict[int, np.ndarray], per_frame_t: dict[int, np.ndarray], distortion: tuple[float, float], pitch_length_m: float, pitch_width_m: float, player_boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] | None, ball_radius_m: float = 0.11)` and `factor(self, frame: int, uv: tuple[float, float]) -> float` — returns the multiplicative factor in `[cfg.min_factor, 1.0]` (`1.0` when disabled) AND records `(frame, uv)` in its internal history (call it exactly once per raw detection, in frame order).
  - `load_player_boxes(tracks_path: Path) -> dict[int, list[tuple[float, float, float, float]]] | None` — per-frame person boxes (all classes except `"ball"`); `None` when the file is absent/unreadable.
  - `bbox_distance_px(uv: tuple[float, float], bbox: tuple[float, float, float, float]) -> float` — 0.0 inside the box, else Euclidean distance to the nearest edge point.
  - `rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_context_prior.py`:

```python
"""Context prior: pitch / player-proximity / static-in-image penalties,
gentle single-signal semantics, factor floor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.ball_context_prior import (
    ContextPrior,
    ContextPriorCfg,
    bbox_distance_px,
    load_player_boxes,
    rotation_angle_deg,
)

CFG = ContextPriorCfg()


def _camera_pose(yaw_deg: float = 0.0):
    """Broadcast-ish pose; optional yaw about world z to simulate panning."""
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    a = np.deg2rad(yaw_deg)
    yaw = np.array([[np.cos(a), -np.sin(a), 0.0],
                    [np.sin(a), np.cos(a), 0.0],
                    [0.0, 0.0, 1.0]])
    R = R @ yaw
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _prior(n_frames: int = 90, yaw_per_frame: float = 0.0,
           boxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
           cfg: ContextPriorCfg = CFG) -> tuple[ContextPrior, dict]:
    Ks, Rs, ts = {}, {}, {}
    for i in range(n_frames):
        K, R, t = _camera_pose(yaw_deg=i * yaw_per_frame)
        Ks[i], Rs[i], ts[i] = K, R, t
    prior = ContextPrior(
        cfg, per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), pitch_length_m=105.0, pitch_width_m=68.0,
        player_boxes_by_frame=boxes,
    )
    return prior, {"K": Ks, "R": Rs, "t": ts}


def test_on_pitch_detection_with_player_nearby_is_unpenalized():
    K, R, t = _camera_pose()
    uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    boxes = {0: [(uv[0] - 30, uv[1] - 80, uv[0] + 30, uv[1] + 10)]}
    prior, _ = _prior(boxes=boxes)
    assert prior.factor(0, uv) == pytest.approx(1.0)


def test_single_signal_never_drops_a_confident_detection():
    # No player box within reach (boxes exist that frame) — player penalty
    # alone must keep 0.8 * factor >= drop_below.
    K, R, t = _camera_pose()
    uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    boxes = {0: [(uv[0] + 500, uv[1] + 500, uv[0] + 560, uv[1] + 620)]}
    prior, _ = _prior(boxes=boxes)
    f = prior.factor(0, uv)
    assert f == pytest.approx(CFG.player_penalty)
    assert 0.8 * f >= CFG.drop_below


def test_off_pitch_ground_intersection_penalized():
    K, R, t = _camera_pose()
    # A point 30 m beyond the far touchline at ground level.
    uv = _project(np.array([52.5, 68.0 + 30.0, 0.11]), K, R, t)
    prior, _ = _prior()
    assert prior.factor(0, uv) == pytest.approx(CFG.pitch_penalty)


def test_unresolvable_ground_ray_is_not_penalized():
    # A pixel above the horizon intersects the ground plane BEHIND the
    # camera (negative depth) — the pitch signal must abstain (airborne
    # balls legitimately do this). Verified empirically: ankle_ray_to_pitch
    # returns a behind-camera point rather than raising for such pixels.
    prior, _ = _prior()
    f = prior.factor(0, (640.0, -2000.0))
    # No boxes provided and pitch abstains -> only static could fire, and
    # there's no history yet.
    assert f == pytest.approx(1.0)


def test_static_in_image_under_pan_penalized_combined_with_player():
    # Same pixel for 60 frames while the camera pans 0.2 deg/frame, and no
    # player anywhere near: static * player must drop a 0.8-conf blob.
    uv = (640.0, 40.0)
    boxes = {i: [(100.0, 600.0, 160.0, 700.0)] for i in range(90)}
    prior, _ = _prior(yaw_per_frame=0.2, boxes=boxes)
    fs = [prior.factor(i, uv) for i in range(60)]
    f_late = fs[-1]
    assert f_late <= CFG.static_penalty * CFG.player_penalty + 1e-9
    assert 0.8 * f_late < CFG.drop_below


def test_static_not_triggered_when_camera_still():
    # Camera fixed: a world-static resting ball is image-static too — the
    # static signal must NOT fire (cam rotation below static_min_cam_deg).
    K, R, t = _camera_pose()
    uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    prior, _ = _prior(yaw_per_frame=0.0)
    fs = [prior.factor(i, uv) for i in range(60)]
    assert fs[-1] == pytest.approx(1.0)


def test_factor_floor():
    cfg = ContextPriorCfg(pitch_penalty=0.1, player_penalty=0.1,
                          static_penalty=0.1, min_factor=0.1)
    K, R, t = _camera_pose()
    uv = _project(np.array([52.5, 120.0, 0.11]), K, R, t)
    boxes = {0: [(0.0, 0.0, 10.0, 10.0)]}
    prior, _ = _prior(boxes=boxes, cfg=cfg)
    assert prior.factor(0, uv) >= cfg.min_factor


def test_disabled_returns_one():
    prior, _ = _prior(cfg=ContextPriorCfg(enabled=False))
    assert prior.factor(0, (9999.0, -9999.0)) == 1.0


def test_bbox_distance():
    assert bbox_distance_px((5.0, 5.0), (0.0, 0.0, 10.0, 10.0)) == 0.0
    assert bbox_distance_px((13.0, 14.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(5.0)


def test_rotation_angle():
    K, R0, _ = _camera_pose(0.0)
    _, R1, _ = _camera_pose(3.0)
    assert rotation_angle_deg(R0, R0) == pytest.approx(0.0, abs=1e-6)
    assert rotation_angle_deg(R0, R1) == pytest.approx(3.0, abs=1e-3)


def test_load_player_boxes_excludes_ball_and_missing_file(tmp_path: Path):
    payload = {
        "shot_id": "play",
        "tracks": [
            {"track_id": "1", "class_name": "player", "team": "A",
             "player_id": "P001", "player_name": "",
             "frames": [{"frame": 3, "bbox": [1.0, 2.0, 3.0, 4.0],
                         "confidence": 0.9, "pitch_position": None,
                         "interpolated": False}]},
            {"track_id": "2", "class_name": "ball", "team": "unknown",
             "player_id": "", "player_name": "",
             "frames": [{"frame": 3, "bbox": [9.0, 9.0, 10.0, 10.0],
                         "confidence": 0.9, "pitch_position": None,
                         "interpolated": False}]},
        ],
    }
    p = tmp_path / "play_tracks.json"
    p.write_text(json.dumps(payload))
    boxes = load_player_boxes(p)
    assert boxes == {3: [(1.0, 2.0, 3.0, 4.0)]}
    assert load_player_boxes(tmp_path / "missing.json") is None


def test_config_block_keys():
    import yaml
    cfg = yaml.safe_load(
        Path("config/default.yaml").read_text())["ball"]["context_prior"]
    assert cfg["enabled"] is True
    assert cfg["drop_below"] == 0.35
    assert cfg["min_factor"] == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_context_prior.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.ball_context_prior'`

- [ ] **Step 3: Implement the module**

Create `src/utils/ball_context_prior.py`:

```python
"""Context prior for ball detections (spec §4.2).

WASB confidently misdetects scoreboard digits / crowd blobs; cleaning the
track afterwards removes wrong points but cannot add right ones, so the
lever is at detect time. This module computes a multiplicative factor in
[min_factor, 1] per raw detection from three cheap signals available
before the solve pass:

- pitch: the detection's ground-ray intersection lies far off the pitch
  (crowd/stand). Abstains when the ray misses the ground plane — high
  balls legitimately do that.
- player proximity: no tracked person box anywhere near the pixel (only
  when boxes exist for that frame).
- static-in-image: the pixel position is near-constant over a trailing
  window while the camera visibly pans — glued to the IMAGE, not the
  world (overlays, scoreboards).

The factor multiplies the detector confidence; the EXISTING acceptance
thresholds then do any dropping. Signals are deliberately gentle: no
single signal drops a confident detection (see default penalties).
Pure and torch-free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.foot_anchor import ankle_ray_to_pitch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextPriorCfg:
    enabled: bool = True
    # Penalized confidence below this is treated as "no detection" by the
    # wiring in ball.py (the drop happens via existing gate semantics).
    drop_below: float = 0.35
    pitch_margin_m: float = 5.0
    pitch_penalty: float = 0.5
    player_max_dist_px: float = 180.0
    player_penalty: float = 0.6
    static_window: int = 45
    static_max_px: float = 3.0
    static_min_cam_deg: float = 2.0
    static_penalty: float = 0.45
    min_factor: float = 0.1


def bbox_distance_px(
    uv: tuple[float, float], bbox: tuple[float, float, float, float],
) -> float:
    """Distance from a pixel to a box: 0 inside, else nearest-edge distance."""
    u, v = float(uv[0]), float(uv[1])
    x1, y1, x2, y2 = (float(b) for b in bbox)
    dx = max(x1 - u, 0.0, u - x2)
    dy = max(y1 - v, 0.0, v - y2)
    return float(np.hypot(dx, dy))


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    cos = (float(np.trace(R1.T @ R2)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def load_player_boxes(
    tracks_path: Path,
) -> dict[int, list[tuple[float, float, float, float]]] | None:
    """Per-frame person boxes from the tracking sidecar; None when absent.

    Every class except "ball" counts as a person (the ball's own track
    would make the proximity prior self-confirming).
    """
    if not tracks_path.exists():
        return None
    try:
        data = json.loads(tracks_path.read_text())
        out: dict[int, list[tuple[float, float, float, float]]] = {}
        for track in data.get("tracks", []):
            if track.get("class_name") == "ball":
                continue
            for fr in track.get("frames", []):
                b = fr["bbox"]
                out.setdefault(int(fr["frame"]), []).append(
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                )
        return out
    except Exception as exc:  # noqa: BLE001 — prior input is enrichment
        logger.warning("context prior: unreadable tracks at %s: %s",
                       tracks_path, exc)
        return None


class ContextPrior:
    """Stateful per-shot prior; call ``factor`` once per raw detection in
    frame order (it records the detection for the static-window check)."""

    def __init__(
        self,
        cfg: ContextPriorCfg,
        *,
        per_frame_K: dict[int, np.ndarray],
        per_frame_R: dict[int, np.ndarray],
        per_frame_t: dict[int, np.ndarray],
        distortion: tuple[float, float],
        pitch_length_m: float,
        pitch_width_m: float,
        player_boxes_by_frame: dict[
            int, list[tuple[float, float, float, float]]] | None,
        ball_radius_m: float = 0.11,
    ) -> None:
        self._cfg = cfg
        self._K = per_frame_K
        self._R = per_frame_R
        self._t = per_frame_t
        self._distortion = distortion
        self._length = float(pitch_length_m)
        self._width = float(pitch_width_m)
        self._boxes = player_boxes_by_frame
        self._ball_radius = float(ball_radius_m)
        # frame -> uv of the raw detections seen so far (static check).
        self._history: dict[int, tuple[float, float]] = {}

    def factor(self, frame: int, uv: tuple[float, float]) -> float:
        cfg = self._cfg
        u, v = float(uv[0]), float(uv[1])
        if not cfg.enabled:
            return 1.0
        f = 1.0

        # -- pitch signal ---------------------------------------------------
        # Abstains when the ray misses the ground AHEAD of the camera: a
        # near-parallel ray raises, and an above-horizon pixel intersects
        # the plane BEHIND the camera (negative depth) — both are the
        # signature of a legitimately high ball, not a crowd blob.
        K, R, t = self._K.get(frame), self._R.get(frame), self._t.get(frame)
        if K is not None and R is not None and t is not None:
            try:
                world = ankle_ray_to_pitch(
                    (u, v), K=K, R=R, t=t,
                    plane_z=self._ball_radius, distortion=self._distortion,
                )
                depth = float((R @ np.asarray(world) + t)[2])
                if depth > 0.0:
                    m = cfg.pitch_margin_m
                    if not (-m <= world[0] <= self._length + m
                            and -m <= world[1] <= self._width + m):
                        f *= cfg.pitch_penalty
            except ValueError:
                pass  # near-parallel ray: abstain

        # -- player-proximity signal (abstains without boxes that frame) --
        if self._boxes is not None:
            boxes = self._boxes.get(frame)
            if boxes:
                nearest = min(bbox_distance_px((u, v), b) for b in boxes)
                if nearest > cfg.player_max_dist_px:
                    f *= cfg.player_penalty

        # -- static-in-image signal ---------------------------------------
        past_frame = frame - cfg.static_window
        past_uv = self._history.get(past_frame)
        if past_uv is not None:
            moved = float(np.hypot(u - past_uv[0], v - past_uv[1]))
            R_now = self._R.get(frame)
            R_then = self._R.get(past_frame)
            if (moved <= cfg.static_max_px
                    and R_now is not None and R_then is not None
                    and rotation_angle_deg(R_then, R_now)
                    >= cfg.static_min_cam_deg):
                f *= cfg.static_penalty

        self._history[frame] = (u, v)
        return max(cfg.min_factor, f)
```

In `config/default.yaml`, immediately after the `shot_chain:` block (inside `ball:`, 2-space indentation), add:

```yaml
  context_prior:
    enabled: true
    drop_below: 0.35         # penalized conf below this => treated as no detection
    pitch_margin_m: 5.0
    pitch_penalty: 0.5       # ground intersection far off-pitch (crowd/stand)
    player_max_dist_px: 180.0
    player_penalty: 0.6      # no tracked person box anywhere near the pixel
    static_window: 45
    static_max_px: 3.0
    static_min_cam_deg: 2.0
    static_penalty: 0.45     # pixel-static while the camera pans (overlay)
    min_factor: 0.1          # floor on the combined factor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_context_prior.py -v`
Expected: 12 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_context_prior.py config/default.yaml tests/test_ball_context_prior.py
git commit -m "feat: context-prior module (pitch/player/static signals) + config"
```

---

### Task 2: Wire the context prior into the detection pass

**Files:**
- Modify: `src/stages/ball.py` — the pass-1 detection loop (`_detect_loop`, ~:822-918: detector call ~:870, image-bounds gate ~:871-874, `raw_confidences[frame_idx] = float(det[2])` ~:895) and its caller `_detect_shot` (camera matrices loaded ~:1077-1083); the second-pass candidate call site (where `best_gated_candidate` receives candidates); config builder next to `_shot_chain_cfg`.
- Test: `tests/test_ball_stage_context_prior.py`

**Interfaces:**
- Consumes: `ContextPrior`, `ContextPriorCfg`, `load_player_boxes` from Task 1 (exact signatures there). Tracks sidecar path: `<output_dir>/tracks/{shot_id}_tracks.json` (legacy single-shot: `tracks/tracks.json` — mirror `_load_ball_anchors`'s two-path pattern).
- Produces: pass-1 semantics — for each raw detection `det=(u,v,conf)` that survives the image-bounds gate, compute `adjusted = conf * prior.factor(frame_idx, (u,v))`; if `adjusted < prior_cfg.drop_below` treat the frame as no-detection (`det = None`, so the IMM never consumes the position); else store `raw_confidences[frame_idx] = adjusted`. Second-pass semantics — candidate scores are multiplied by `prior.factor(...)`-equivalent BEFORE `best_gated_candidate` (build the modified candidate list at the call site; `accept_min` compare is untouched). A `_context_prior_cfg(cfg_dict) -> ContextPriorCfg` builder mirrors `_kinematic_touch_cfg`.

- [ ] **Step 1: Write the failing stage-level tests**

Create `tests/test_ball_stage_context_prior.py`:

```python
"""Context prior wiring: a static overlay blob far from players is dropped
by the prior; a genuine moving ball is untouched; disabled flag restores
old behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector

N_FRAMES = 90
FPS = 30.0


def _camera_pose(yaw_deg: float = 0.0):
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    a = np.deg2rad(yaw_deg)
    yaw = np.array([[np.cos(a), -np.sin(a), 0.0],
                    [np.sin(a), np.cos(a), 0.0],
                    [0.0, 0.0, 1.0]])
    R = R @ yaw
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _build_scene(tmp_path: Path, *, panning: bool):
    """Scene with a PANNING camera (0.2 deg/frame yaw) so the static
    signal can fire, plus a tracks sidecar with one player box far from
    the frame top."""
    out = tmp_path / "out"
    clip = out / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for _ in range(N_FRAMES):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()

    frames = []
    for i in range(N_FRAMES):
        K, R, t = _camera_pose(yaw_deg=(i * 0.2 if panning else 0.0))
        frames.append(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                  confidence=1.0, is_anchor=(i == 0)))
    K0, R0, t0 = _camera_pose(0.0)
    CameraTrack(clip_id="play", fps=FPS, image_size=(1280, 720),
                t_world=t0.tolist(), frames=tuple(frames),
                ).save(out / "camera" / "play_camera_track.json")

    ShotsManifest(
        source_file="fake.mp4", fps=FPS, total_frames=N_FRAMES,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=N_FRAMES - 1,
                    start_time=0.0, end_time=(N_FRAMES - 1) / FPS)],
    ).save(out / "shots" / "shots_manifest.json")

    tracks = {
        "shot_id": "play",
        "tracks": [{
            "track_id": "1", "class_name": "player", "team": "A",
            "player_id": "P001", "player_name": "",
            "frames": [
                {"frame": i, "bbox": [200.0, 500.0, 260.0, 640.0],
                 "confidence": 0.9, "pitch_position": None,
                 "interpolated": False}
                for i in range(N_FRAMES)
            ],
        }],
    }
    tracks_path = out / "tracks" / "play_tracks.json"
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_path.write_text(json.dumps(tracks))
    return out


def _cfg(prior_enabled: bool) -> dict:
    return {
        "ball": {
            "detector": "fake",
            "appearance_bridge": {"enabled": False},
            "second_pass": {"enabled": False},
            "context_prior": {"enabled": prior_enabled},
        },
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


def _static_blob_detections() -> list:
    # Confident blob glued to the image near the frame top, every frame —
    # the scoreboard signature (static under pan + no player near).
    return [(640.0, 30.0, 0.8)] * N_FRAMES


@pytest.mark.integration
def test_prior_drops_static_overlay_blob(tmp_path: Path):
    out = _build_scene(tmp_path, panning=True)
    BallStage(config=_cfg(prior_enabled=True), output_dir=out,
              ball_detector=FakeBallDetector(_static_blob_detections())).run()
    obs = json.loads(
        (out / "ball" / "play_ball_observations.json").read_text())
    accepted_late = [
        f for f in obs["frames"]
        if f["frame"] >= 50 and f["source"] == "detector"
        and f["confidence"] > 0.0
    ]
    # After the static window fills, the combined static+player penalties
    # push 0.8 below drop_below and the blob stops being accepted.
    assert accepted_late == [], (
        f"expected the static blob to be dropped after frame 50; "
        f"got {len(accepted_late)} accepted detector frames"
    )


@pytest.mark.integration
def test_prior_disabled_keeps_old_behaviour(tmp_path: Path):
    out = _build_scene(tmp_path, panning=True)
    BallStage(config=_cfg(prior_enabled=False), output_dir=out,
              ball_detector=FakeBallDetector(_static_blob_detections())).run()
    obs = json.loads(
        (out / "ball" / "play_ball_observations.json").read_text())
    accepted_late = [
        f for f in obs["frames"]
        if f["frame"] >= 50 and f["source"] == "detector"
        and f["confidence"] > 0.0
    ]
    assert accepted_late, "disabled prior must not drop anything"


@pytest.mark.integration
def test_genuine_moving_ball_untouched_by_prior(tmp_path: Path):
    out = _build_scene(tmp_path, panning=True)
    K0, R0, t0 = _camera_pose(0.0)
    detections = []
    for i in range(N_FRAMES):
        # Roll across the pitch, near the tracked player's box.
        p = np.array([30.0 + 0.15 * i, 34.0, 0.11])
        Ki, Ri, ti = _camera_pose(yaw_deg=i * 0.2)
        cam = Ri @ p + ti
        pix = Ki @ cam
        detections.append((float(pix[0] / pix[2]), float(pix[1] / pix[2]), 0.9))
    # Put the player box under the rolling ball so proximity never fires.
    tracks_path = out / "tracks" / "play_tracks.json"
    tracks = json.loads(tracks_path.read_text())
    for i, fr in enumerate(tracks["tracks"][0]["frames"]):
        u, v, _ = detections[i]
        fr["bbox"] = [u - 40.0, v - 120.0, u + 40.0, v + 10.0]
    tracks_path.write_text(json.dumps(tracks))

    BallStage(config=_cfg(prior_enabled=True), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()
    obs = json.loads(
        (out / "ball" / "play_ball_observations.json").read_text())
    accepted = [f for f in obs["frames"]
                if f["source"] == "detector" and f["confidence"] > 0.0]
    assert len(accepted) >= int(0.9 * N_FRAMES), (
        f"prior must not eat a genuine moving ball; accepted {len(accepted)}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_stage_context_prior.py -v`
Expected: `test_prior_drops_static_overlay_blob` FAILS (blob accepted — no prior exists); the other two PASS vacuously or fail on config keys. Confirm at least the first fails for the right reason.

- [ ] **Step 3: Wire the prior**

In `src/stages/ball.py`:

1. Import at the top with the sibling ball utils:

```python
from src.utils.ball_context_prior import (
    ContextPrior,
    ContextPriorCfg,
    load_player_boxes,
)
```

2. Builder next to `_shot_chain_cfg`:

```python
def _context_prior_cfg(cfg_dict: dict) -> ContextPriorCfg:
    """Build a ContextPriorCfg from the ``ball.context_prior`` sub-tree."""
    base = ContextPriorCfg()
    return ContextPriorCfg(
        enabled=bool(cfg_dict.get("enabled", base.enabled)),
        drop_below=float(cfg_dict.get("drop_below", base.drop_below)),
        pitch_margin_m=float(cfg_dict.get(
            "pitch_margin_m", base.pitch_margin_m)),
        pitch_penalty=float(cfg_dict.get(
            "pitch_penalty", base.pitch_penalty)),
        player_max_dist_px=float(cfg_dict.get(
            "player_max_dist_px", base.player_max_dist_px)),
        player_penalty=float(cfg_dict.get(
            "player_penalty", base.player_penalty)),
        static_window=int(cfg_dict.get("static_window", base.static_window)),
        static_max_px=float(cfg_dict.get("static_max_px", base.static_max_px)),
        static_min_cam_deg=float(cfg_dict.get(
            "static_min_cam_deg", base.static_min_cam_deg)),
        static_penalty=float(cfg_dict.get(
            "static_penalty", base.static_penalty)),
        min_factor=float(cfg_dict.get("min_factor", base.min_factor)),
    )
```

3. In `_detect_shot` (after the camera matrices `per_frame_K/R/t` and `distortion` are built, ~:1077-1083), construct the prior and pass it to the detect loop. Read the actual code first — the construction is:

```python
        prior_cfg = _context_prior_cfg(cfg.get("context_prior", {}))
        prior: ContextPrior | None = None
        if prior_cfg.enabled:
            try:
                tracks_path = (
                    self.output_dir / "tracks" / f"{shot_id}_tracks.json"
                    if shot_id else self.output_dir / "tracks" / "tracks.json"
                )
                pitch_cfg = self.config.get("pitch", {})
                prior = ContextPrior(
                    prior_cfg,
                    per_frame_K=per_frame_K, per_frame_R=per_frame_R,
                    per_frame_t=per_frame_t, distortion=distortion,
                    pitch_length_m=float(pitch_cfg.get("length_m", 105.0)),
                    pitch_width_m=float(pitch_cfg.get("width_m", 68.0)),
                    player_boxes_by_frame=load_player_boxes(tracks_path),
                    ball_radius_m=float(cfg.get("ball_radius_m", 0.11)),
                )
            except Exception as exc:  # noqa: BLE001 — prior is enrichment
                logger.warning(
                    "ball: context prior unavailable (%s) — detections "
                    "unmodified", exc)
                prior = None
```

Thread `prior` and `prior_cfg.drop_below` into `_detect_loop` (extend its signature with `prior: ContextPrior | None = None, prior_drop_below: float = 0.0`; update the call site).

4. In `_detect_loop`, immediately after the image-bounds gate (~:871-874) and before `raw_confidences[frame_idx] = float(det[2])` (~:895), apply:

```python
            if det is not None and prior is not None:
                adj = float(det[2]) * prior.factor(
                    frame_idx, (float(det[0]), float(det[1])))
                if adj < prior_drop_below:
                    det = None  # penalized below the drop threshold
                else:
                    det = (det[0], det[1], adj)
```

(Read the loop first: the exact placement is between the bounds gate and where `uv`/`raw_confidences` are derived from `det`; if the loop's structure differs slightly, preserve the invariant "prior applies to every bounds-surviving raw detection exactly once, before the tracker or confidences see it".)

5. Second pass: find where second-pass candidates are gathered and passed to `best_gated_candidate` (search `best_gated_candidate(` in `src/stages/ball.py`). Before that call, when `prior is not None`, rebuild the candidate list with modified scores:

```python
                    if prior is not None:
                        candidates = [
                            (u, v, s * prior.factor(fr, (u, v)))
                            for (u, v, s) in candidates
                        ]
```

(Adapt `fr`/local names to the call site's actual frame variable. If the call site iterates frames, apply per frame. `best_gated_candidate`'s own `accept_min` compare stays untouched.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_stage_context_prior.py tests/test_ball_stage.py tests/test_ball_stage_second_pass.py tests/test_ball_stage_kinematic_wiring.py tests/test_ball_stage_shot_chains.py -v`
Expected: all PASS (existing stage suites confirm no regression — their scenes have no tracks sidecar and static cameras, so the prior abstains).

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py tests/test_ball_stage_context_prior.py
git commit -m "feat: wire context prior into pass-1 and second-pass detection scoring"
```

---

### Task 3: Pure attribution-refinement module (`src/utils/ball_touch_attribution.py`) + config block

**Files:**
- Create: `src/utils/ball_touch_attribution.py`
- Modify: `config/default.yaml` (add `touch_attribution:` block inside `ball:`, immediately after the `context_prior:` block)
- Test: `tests/test_ball_touch_attribution.py`

**Interfaces:**
- Consumes: `BallEvent` (frozen dataclass: frame/kind/score/player_id/bone/goal_element/end_frame); `point_to_pixel_ray_distance(world, ball_uv, K, R, t, distortion) -> float` from `src/utils/ball_kinematic_touch.py`; the PlayerContext duck-type: `player_ctx.joints_at(frame)` yields objects with `.player_id`, `.bone`, `.world_xyz`, `.uv`, `.confidence` (exactly what `ray_gap_series` consumes at `src/utils/ball_kinematic_touch.py:94-123`).
- Produces (Task 4 relies on exactly these):
  - `TouchAttributionCfg(enabled: bool = True, window: int = 2, max_gap_m: float = 0.45, margin_m: float = 0.05, min_fk_conf: float = 0.3)` — frozen dataclass.
  - `refine_touch_attribution(events: Sequence[BallEvent], *, player_ctx, ball_uvs: dict[int, np.ndarray], per_frame_K: dict[int, np.ndarray], per_frame_R: dict[int, np.ndarray], per_frame_t: dict[int, np.ndarray], distortion: tuple[float, float], cfg: TouchAttributionCfg) -> tuple[BallEvent, ...]` — same events, same order, same frames/kinds/scores; only `player_id`/`bone` of `kind == "touch"` events may change (via `dataclasses.replace`). A touch is relabelled to the joint with the smallest bone↔ball-ray gap over frames `[f-window, f+window]` (fk_conf ≥ min_fk_conf, gap ≤ max_gap_m) ONLY when that best gap beats the current attribution's own best gap in the window by more than `margin_m` (ambiguity keeps the original). Touches with no ball uv in the window, or whose current attribution can't be measured, are left unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_touch_attribution.py`:

```python
"""Touch bone-attribution refinement: relabel to the ray-closest joint,
keep originals on ambiguity, never add/remove/re-frame events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.utils.ball_auto_events import BallEvent
from src.utils.ball_touch_attribution import (
    TouchAttributionCfg,
    refine_touch_attribution,
)

CFG = TouchAttributionCfg()


def _camera():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


@dataclass(frozen=True)
class _Joint:
    player_id: str
    bone: str
    world_xyz: tuple[float, float, float]
    uv: tuple[float, float] | None
    confidence: float


class _Ctx:
    """PlayerContext stub: fixed joints at every frame."""

    def __init__(self, joints):
        self._joints = joints

    def joints_at(self, frame):
        return list(self._joints)


def _setup(ball_world=(40.0, 34.0, 0.11)):
    K, R, t = _camera()
    ball_uv = _project(np.array(ball_world), K, R, t)
    joints = [
        # l_foot right AT the ball; r_foot 0.8 m away.
        _Joint("P001", "l_foot", (40.0, 34.0, 0.11),
               _project(np.array([40.0, 34.0, 0.11]), K, R, t), 0.9),
        _Joint("P001", "r_foot", (40.8, 34.0, 0.11),
               _project(np.array([40.8, 34.0, 0.11]), K, R, t), 0.9),
    ]
    frames = range(8, 13)
    return (
        _Ctx(joints),
        {f: np.asarray(ball_uv) for f in frames},
        {f: K for f in frames}, {f: R for f in frames}, {f: t for f in frames},
    )


def _refine(events, ctx, uvs, Ks, Rs, ts, cfg=CFG):
    return refine_touch_attribution(
        events, player_ctx=ctx, ball_uvs=uvs,
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), cfg=cfg,
    )


def test_wrong_bone_relabelled_to_ray_closest_joint():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert len(out) == 1
    assert out[0].bone == "l_foot"
    assert out[0].player_id == "P001"
    assert out[0].frame == 10 and out[0].kind == "touch"
    assert out[0].score == pytest.approx(0.7)


def test_ambiguous_margin_keeps_original():
    # Both feet equidistant-ish: margin gate keeps the original label.
    K, R, t = _camera()
    ball_uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    joints = [
        _Joint("P001", "l_foot", (40.02, 34.0, 0.11),
               _project(np.array([40.02, 34.0, 0.11]), K, R, t), 0.9),
        _Joint("P001", "r_foot", (40.05, 34.0, 0.11),
               _project(np.array([40.05, 34.0, 0.11]), K, R, t), 0.9),
    ]
    ctx = _Ctx(joints)
    uvs = {10: np.asarray(ball_uv)}
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = refine_touch_attribution(
        events, player_ctx=ctx, ball_uvs=uvs,
        per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
        distortion=(0.0, 0.0), cfg=CFG,
    )
    assert out[0].bone == "r_foot"


def test_far_ball_no_candidate_keeps_original():
    ctx, uvs, Ks, Rs, ts = _setup(ball_world=(60.0, 10.0, 0.11))
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert out[0].bone == "r_foot"  # best gap exceeds max_gap_m -> unchanged


def test_non_touch_events_and_order_preserved():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (
        BallEvent(frame=5, kind="bounce", score=0.6),
        BallEvent(frame=10, kind="touch", score=0.7,
                  player_id="P001", bone="r_foot"),
        BallEvent(frame=20, kind="goal_impact", score=0.9,
                  goal_element="post"),
    )
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert [e.kind for e in out] == ["bounce", "touch", "goal_impact"]
    assert out[0] == events[0] and out[2] == events[2]


def test_no_ball_uv_in_window_keeps_original():
    ctx, _uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, {}, Ks, Rs, ts)
    assert out[0].bone == "r_foot"


def test_disabled_is_identity():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts,
                  cfg=TouchAttributionCfg(enabled=False))
    assert out == events


def test_config_block_keys():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        Path("config/default.yaml").read_text())["ball"]["touch_attribution"]
    assert cfg["enabled"] is True
    assert cfg["window"] == 2
    assert cfg["max_gap_m"] == 0.45
    assert cfg["margin_m"] == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_touch_attribution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.ball_touch_attribution'`

- [ ] **Step 3: Implement the module**

Create `src/utils/ball_touch_attribution.py`:

```python
"""Touch bone-attribution refinement (spec §4.4).

Validated on gberch: with the same ±2-frame tolerance, ignoring the bone
claim lifts touch recall from 0.25 to 0.50 — half the touch moments are
found but pinned to the wrong body part. The original attribution happens
at the (noisy) break/proposal moment; this post-pass re-picks each touch
event's (player, bone) as the joint with the smallest 3-D bone↔ball-ray
gap over a small window around the event frame, keeping the original when
the improvement is within an ambiguity margin. It relabels ONLY — never
adds, removes, re-frames, or re-scores events. Pure and torch-free.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from src.utils.ball_kinematic_touch import point_to_pixel_ray_distance

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.utils.ball_auto_events import BallEvent


@dataclass(frozen=True)
class TouchAttributionCfg:
    enabled: bool = True
    window: int = 2          # +/- frames considered around the event frame
    max_gap_m: float = 0.45  # candidate joints beyond this never relabel
    margin_m: float = 0.05   # new joint must beat the current one by this
    min_fk_conf: float = 0.3


def _best_gaps_in_window(
    frame: int,
    *,
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    cfg: TouchAttributionCfg,
) -> dict[tuple[str, str], float]:
    """Per-(player, bone) minimal ray gap over the window around ``frame``."""
    best: dict[tuple[str, str], float] = {}
    for f in range(frame - cfg.window, frame + cfg.window + 1):
        ball_uv = ball_uvs.get(f)
        K, R, t = per_frame_K.get(f), per_frame_R.get(f), per_frame_t.get(f)
        if ball_uv is None or K is None or R is None or t is None:
            continue
        for s in player_ctx.joints_at(f):
            if s.confidence < cfg.min_fk_conf or s.world_xyz is None:
                continue
            gap = float(point_to_pixel_ray_distance(
                np.asarray(s.world_xyz, dtype=float), ball_uv,
                K, R, t, distortion,
            ))
            key = (s.player_id, s.bone)
            if gap < best.get(key, float("inf")):
                best[key] = gap
    return best


def refine_touch_attribution(
    events: "Sequence[BallEvent]",
    *,
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    cfg: TouchAttributionCfg,
) -> "tuple[BallEvent, ...]":
    """Relabel touch events to the ray-closest joint; everything else
    passes through untouched (same order, same length)."""
    if not cfg.enabled:
        return tuple(events)
    out: "list[BallEvent]" = []
    for e in events:
        if e.kind != "touch" or not e.player_id or not e.bone:
            out.append(e)
            continue
        gaps = _best_gaps_in_window(
            e.frame, player_ctx=player_ctx, ball_uvs=ball_uvs,
            per_frame_K=per_frame_K, per_frame_R=per_frame_R,
            per_frame_t=per_frame_t, distortion=distortion, cfg=cfg,
        )
        if not gaps:
            out.append(e)
            continue
        (best_pid, best_bone), best_gap = min(
            gaps.items(), key=lambda kv: (kv[1], kv[0]))
        current_gap = gaps.get((e.player_id, e.bone))
        relabel = (
            best_gap <= cfg.max_gap_m
            and (best_pid, best_bone) != (e.player_id, e.bone)
            and (current_gap is None or best_gap + cfg.margin_m < current_gap)
        )
        if relabel:
            out.append(dataclasses.replace(
                e, player_id=best_pid, bone=best_bone))
        else:
            out.append(e)
    return tuple(out)
```

In `config/default.yaml`, immediately after the `context_prior:` block (inside `ball:`), add:

```yaml
  touch_attribution:
    enabled: true            # relabel touch events to the ray-closest joint
    window: 2                # +/- frames around the event frame
    max_gap_m: 0.45          # candidates beyond this never relabel
    margin_m: 0.05           # required improvement over current attribution
    min_fk_conf: 0.3
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_touch_attribution.py -v`
Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_touch_attribution.py config/default.yaml tests/test_ball_touch_attribution.py
git commit -m "feat: touch bone-attribution refinement module + config"
```

---

### Task 4: Wire attribution refinement into `BallStage` + wiring guard

**Files:**
- Modify: `src/stages/ball.py` — post-`merge_touch_events` region (~:1546-1558: the kinematic try/except ends, then `chain_cfg = _shot_chain_cfg(...)`); config builder next to `_shot_chain_cfg`.
- Test: `tests/test_ball_stage_attribution_wiring.py`

**Interfaces:**
- Consumes: `TouchAttributionCfg`, `refine_touch_attribution` from Task 3 (exact signatures there); existing locals in `_solve_shot`: `events`, `player_ctx`, `per_frame_K/R/t`, `distortion`, `steps` (ball uvs come from `{s.frame: np.asarray(s.uv) for s in steps if s.uv is not None}` — the same construction the kinematic block uses at ~:1527).
- Produces: after the kinematic merge and BEFORE `chain_cfg = _shot_chain_cfg(...)` (chains must pair against final labels), `events` is replaced by the refined tuple when `ball.touch_attribution.enabled` and `player_ctx.player_ids` is non-empty; failures degrade with `logger.warning("ball stage: touch attribution refinement failed ...")`.

- [ ] **Step 1: Write the failing wiring tests**

Create `tests/test_ball_stage_attribution_wiring.py` (reuses the Task-1-pattern scene; the monkeypatch style mirrors `tests/test_ball_stage_kinematic_wiring.py`):

```python
"""BallStage wiring guard for touch-attribution refinement: it runs on the
merged events, its output feeds the diag, the flag disables it, and a
crash degrades with a warning."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.smpl_world import SmplWorldTrack
from src.stages.ball import BallStage
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_detector import FakeBallDetector

N_FRAMES = 60
FPS = 30.0


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _build_scene(tmp_path: Path):
    out = tmp_path / "out"
    K, R, t = _camera_pose()
    clip = out / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for _ in range(N_FRAMES):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()
    CameraTrack(
        clip_id="play", fps=FPS, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                 confidence=1.0, is_anchor=(i == 0))
                     for i in range(N_FRAMES)),
    ).save(out / "camera" / "play_camera_track.json")
    ShotsManifest(
        source_file="fake.mp4", fps=FPS, total_frames=N_FRAMES,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=N_FRAMES - 1,
                    start_time=0.0, end_time=(N_FRAMES - 1) / FPS)],
    ).save(out / "shots" / "shots_manifest.json")

    base_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    thetas0 = np.zeros((24, 3), dtype=np.float32)
    frames = np.arange(N_FRAMES, dtype=np.int64)
    SmplWorldTrack(
        player_id="P001", frames=frames,
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.stack([thetas0] * N_FRAMES),
        root_R=np.stack([base_R.astype(np.float32)] * N_FRAMES),
        root_t=np.stack([np.array([40.0, 34.0, 1.0], dtype=np.float32)] * N_FRAMES),
        confidence=np.full(N_FRAMES, 0.8, dtype=np.float32),
        shot_id="play",
    ).save(out / "hmr_world" / "play__P001_smpl_world.npz")

    detections = []
    for i in range(N_FRAMES):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))
    return out, detections


def _cfg(**ball_extra) -> dict:
    cfg = {"ball": {"detector": "fake",
                    "appearance_bridge": {"enabled": False}},
           "pitch": {"length_m": 105.0, "width_m": 68.0}}
    cfg["ball"].update(ball_extra)
    return cfg


@pytest.mark.integration
def test_refinement_runs_and_relabel_reaches_diag(tmp_path, monkeypatch):
    out, detections = _build_scene(tmp_path)
    calls: dict = {}

    def fake_refine(events, **kwargs):
        calls["n"] = len(events)
        import dataclasses as dc
        return tuple(
            dc.replace(e, bone="head")
            if e.kind == "touch" and e.bone else e
            for e in events
        )

    synthetic = (BallEvent(frame=20, kind="touch", score=0.8,
                           player_id="P001", bone="r_foot"),)
    monkeypatch.setattr("src.stages.ball.detect_events",
                        lambda **kwargs: synthetic)
    monkeypatch.setattr("src.stages.ball.refine_touch_attribution",
                        fake_refine)
    BallStage(config=_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()
    assert "n" in calls, "refine_touch_attribution was never invoked"
    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    assert any(e["kind"] == "touch" and e["bone"] == "head"
               for e in diag["events"])


@pytest.mark.integration
def test_refinement_disabled_by_flag(tmp_path, monkeypatch):
    out, detections = _build_scene(tmp_path)
    calls: dict = {}
    monkeypatch.setattr(
        "src.stages.ball.refine_touch_attribution",
        lambda events, **kwargs: calls.setdefault("hit", True) and tuple(events),
    )
    BallStage(config=_cfg(touch_attribution={"enabled": False}),
              output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()
    assert "hit" not in calls


@pytest.mark.integration
def test_refinement_crash_degrades_with_warning(tmp_path, monkeypatch, caplog):
    out, detections = _build_scene(tmp_path)

    def broken(events, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.stages.ball.refine_touch_attribution", broken)
    with caplog.at_level("WARNING"):
        BallStage(config=_cfg(), output_dir=out,
                  ball_detector=FakeBallDetector(detections)).run()
    assert (out / "ball" / "play_ball_track.json").exists()
    assert any("touch attribution refinement failed" in r.message
               for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_stage_attribution_wiring.py -v`
Expected: FAIL — `AttributeError: <module 'src.stages.ball'> has no attribute 'refine_touch_attribution'`.

- [ ] **Step 3: Wire it**

In `src/stages/ball.py`:

1. Import at the top:

```python
from src.utils.ball_touch_attribution import (
    TouchAttributionCfg,
    refine_touch_attribution,
)
```

2. Builder next to `_shot_chain_cfg`:

```python
def _touch_attribution_cfg(cfg_dict: dict) -> TouchAttributionCfg:
    """Build a TouchAttributionCfg from ``ball.touch_attribution``."""
    base = TouchAttributionCfg()
    return TouchAttributionCfg(
        enabled=bool(cfg_dict.get("enabled", base.enabled)),
        window=int(cfg_dict.get("window", base.window)),
        max_gap_m=float(cfg_dict.get("max_gap_m", base.max_gap_m)),
        margin_m=float(cfg_dict.get("margin_m", base.margin_m)),
        min_fk_conf=float(cfg_dict.get("min_fk_conf", base.min_fk_conf)),
    )
```

3. In `_solve_shot`, after the kinematic-touch try/except block closes (~:1551, right before `chain_cfg = _shot_chain_cfg(...)`), insert:

```python
        # Bone-attribution refinement: half the strict-recall gap on real
        # clips is wrong-bone labels at the (noisy) break moment — re-pick
        # each touch's (player, bone) by minimal bone<->ball-ray gap over a
        # small window. Relabels only; runs before chain proposal so chains
        # pair against final labels.
        attr_cfg = _touch_attribution_cfg(cfg.get("touch_attribution", {}))
        if attr_cfg.enabled and player_ctx.player_ids:
            try:
                attr_ball_uvs = {
                    s.frame: np.asarray(s.uv, dtype=float)
                    for s in steps if s.uv is not None
                }
                events = refine_touch_attribution(
                    events, player_ctx=player_ctx, ball_uvs=attr_ball_uvs,
                    per_frame_K=per_frame_K, per_frame_R=per_frame_R,
                    per_frame_t=per_frame_t, distortion=distortion,
                    cfg=attr_cfg,
                )
            except Exception as exc:  # noqa: BLE001 — never kill the stage
                logger.warning(
                    "ball stage: touch attribution refinement failed (%s) — "
                    "keeping original attributions", exc,
                )
```

(Read the region first — `steps` may be named via `artifacts.steps` there; the kinematic block at ~:1527 shows the exact local names to reuse.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_stage_attribution_wiring.py tests/test_ball_stage_kinematic_wiring.py tests/test_ball_stage_shot_chains.py tests/test_ball_stage.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py tests/test_ball_stage_attribution_wiring.py
git commit -m "feat: wire touch-attribution refinement into BallStage before chain proposal"
```

---

### Task 5: `dismissed_auto` — schema, merge suppression, single-loader refactor

**Files:**
- Modify: `src/schemas/ball_anchor.py` (add `DismissedAuto` + `BallAnchorSet.dismissed_auto`; mirror the `shot_chains` parsing pattern at :228-244)
- Modify: `src/utils/ball_auto_anchor.py` (`merge_anchors`, :391-402)
- Modify: `src/stages/ball.py` (replace `_load_ball_anchors` + `_load_manual_shot_chains` with one `_load_manual_anchor_set`; update the `merge_anchors` call at ~:1582 and the chain-diag block)
- Test: `tests/test_ball_dismissed_auto.py`

**Interfaces:**
- Produces (Tasks 6-7 rely on exactly these):
  - `DismissedAuto(frame: int, state: str, player_id: str | None = None, bone: str | None = None)` — frozen dataclass in `src/schemas/ball_anchor.py`; JSON form `{"frame", "state", "player_id", "bone"}`.
  - `BallAnchorSet.dismissed_auto: tuple[DismissedAuto, ...] = ()` — loaded from `data.get("dismissed_auto", [])` (frame→int, state→str, missing player/bone→None; no other validation — dismissals reference auto output, not operator claims).
  - `merge_anchors(manual, auto, suppress_radius_frames, dismissed: Collection[DismissedAuto] = ())` — an auto anchor is additionally dropped when a dismissal matches ALL of (frame, state, player_id, bone) exactly. A dismissal that matches nothing is inert.
  - `_load_manual_anchor_set(output_dir: Path, shot_id: str) -> BallAnchorSet | None` in `ball.py` — single loader (same two-path naming as before, warning-and-None on failure); `manual_by_frame` / manual `shot_chains` / `dismissed_auto` all derive from it. `_load_manual_shot_chains` is DELETED (it was flagged as a double-load in the Phase-2 review); `_load_ball_anchors` is deleted or reduced to a thin wrapper — grep for other callers first (the preview endpoint constructs its own set; tests may import).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_dismissed_auto.py`:

```python
"""dismissed_auto: schema round-trip, merge suppression, stage plumbing."""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet, DismissedAuto
from src.utils.ball_auto_anchor import merge_anchors


def test_schema_roundtrip(tmp_path: Path):
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 5, "image_xy": [1.0, 2.0], "state": "grounded"}],
        "dismissed_auto": [
            {"frame": 20, "state": "player_touch",
             "player_id": "P003", "bone": "l_foot"},
            {"frame": 33, "state": "bounce"},
        ],
    }
    p = tmp_path / "a.json"
    p.write_text(json.dumps(payload))
    aset = BallAnchorSet.load(p)
    assert aset.dismissed_auto == (
        DismissedAuto(frame=20, state="player_touch",
                      player_id="P003", bone="l_foot"),
        DismissedAuto(frame=33, state="bounce"),
    )
    out = tmp_path / "b.json"
    aset.save(out)
    assert json.loads(out.read_text())["dismissed_auto"][0]["frame"] == 20


def test_legacy_payload_defaults_empty(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({
        "clip_id": "play", "image_size": [1280, 720], "anchors": []}))
    assert BallAnchorSet.load(p).dismissed_auto == ()


def _auto(frame: int, state: str = "player_touch",
          player_id: str | None = "P003",
          bone: str | None = "l_foot") -> BallAnchor:
    return BallAnchor(frame=frame, image_xy=(10.0, 10.0), state=state,
                      player_id=player_id, bone=bone, confidence=0.5)


def test_merge_drops_exactly_matching_dismissal():
    auto = {20: _auto(20), 30: _auto(30)}
    merged = merge_anchors(
        {}, auto, 3,
        dismissed=(DismissedAuto(frame=20, state="player_touch",
                                 player_id="P003", bone="l_foot"),),
    )
    assert set(merged) == {30}


def test_merge_partial_match_is_inert():
    auto = {20: _auto(20)}
    merged = merge_anchors(
        {}, auto, 3,
        dismissed=(
            DismissedAuto(frame=20, state="player_touch",
                          player_id="P003", bone="r_foot"),  # wrong bone
            DismissedAuto(frame=21, state="player_touch",
                          player_id="P003", bone="l_foot"),  # wrong frame
        ),
    )
    assert set(merged) == {20}


def test_merge_default_no_dismissals_unchanged():
    manual = {10: _auto(10, state="grounded", player_id=None, bone=None)}
    auto = {11: _auto(11), 30: _auto(30)}
    merged = merge_anchors(manual, auto, 3)
    # 11 suppressed by radius as before, 30 kept.
    assert set(merged) == {10, 30}
```

Also extend the existing stage test file `tests/test_ball_stage_shot_chains.py` with one plumbing test (append at the end):

```python
@pytest.mark.integration
def test_dismissed_auto_suppresses_auto_anchor(tmp_path: Path, monkeypatch):
    out, detections = _build_scene(tmp_path)
    synthetic = (
        BallEvent(frame=15, kind="touch", score=0.8,
                  player_id="P001", bone="r_foot"),
    )
    monkeypatch.setattr(
        "src.stages.ball.detect_events", lambda **kwargs: synthetic)

    # First run: the touch becomes an auto anchor.
    BallStage(config=_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()
    auto = json.loads(
        (out / "ball" / "play_ball_anchors_auto.json").read_text())
    touch_autos = [a for a in auto["anchors"]
                   if a["state"] == "player_touch"]
    assert touch_autos, "precondition: auto touch anchor exists"
    ta = touch_autos[0]

    # Operator dismisses it in the manual sidecar; second run must not
    # merge it (diag anchor count drops).
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720), anchors=(),
        dismissed_auto=(DismissedAuto(
            frame=int(ta["frame"]), state="player_touch",
            player_id=ta.get("player_id"), bone=ta.get("bone")),),
    ).save(out / "ball" / "play_ball_anchors.json")
    BallStage(config=_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()
    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    assert diag["anchors"]["merged"] < diag["anchors"]["auto_generated"], (
        "dismissed auto anchor must be excluded from the merge"
    )
```

with the imports `from src.schemas.ball_anchor import BallAnchor, BallAnchorSet, DismissedAuto` added to that file's import block (BallAnchor/BallAnchorSet may already be imported there — check).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_dismissed_auto.py -v`
Expected: FAIL — `ImportError: cannot import name 'DismissedAuto'`.

- [ ] **Step 3: Implement**

1. `src/schemas/ball_anchor.py` — add above `BallAnchorSet`:

```python
@dataclass(frozen=True)
class DismissedAuto:
    """One operator-dismissed auto suggestion. Identity is the full
    (frame, state, player_id, bone) tuple: auto sets regenerate every run,
    so a dismissal only suppresses an auto anchor that still matches
    exactly; otherwise it is inert."""
    frame: int
    state: str
    player_id: str | None = None
    bone: str | None = None
```

Add the field to `BallAnchorSet` (after `shot_chains`):

```python
    # Operator-dismissed auto suggestions (exhaustive-annotation workflow):
    # matching auto anchors are excluded from the merge; the recall report
    # counts them as reviewed false positives.
    dismissed_auto: tuple[DismissedAuto, ...] = ()
```

In `load`, after the `shot_chains` parsing block and before `return cls(...)`:

```python
        dismissed: list[DismissedAuto] = []
        for d in data.get("dismissed_auto", []):
            dismissed.append(DismissedAuto(
                frame=int(d["frame"]),
                state=str(d["state"]),
                player_id=(str(d["player_id"]) if d.get("player_id") else None),
                bone=(str(d["bone"]) if d.get("bone") else None),
            ))
```

and pass `dismissed_auto=tuple(dismissed),` in `return cls(...)`. (`save` needs no change — `asdict` serializes nested dataclasses.)

2. `src/utils/ball_auto_anchor.py` — extend `merge_anchors`:

```python
def merge_anchors(
    manual: Mapping[int, BallAnchor],
    auto: Mapping[int, BallAnchor],
    suppress_radius_frames: int,
    dismissed: Collection["DismissedAuto"] = (),
) -> dict[int, BallAnchor]:
    """Manual anchors win; auto anchors near a manual frame are dropped;
    auto anchors exactly matching an operator dismissal are dropped."""
    dismissed_keys = {
        (d.frame, d.state, d.player_id, d.bone) for d in dismissed
    }
    merged: dict[int, BallAnchor] = dict(manual)
    for f, anchor in auto.items():
        if any(abs(f - mf) <= suppress_radius_frames for mf in manual):
            continue
        if (anchor.frame, anchor.state, anchor.player_id,
                anchor.bone) in dismissed_keys:
            continue
        merged[f] = anchor
    return merged
```

(Import `Collection` from `collections.abc` and `DismissedAuto` under `TYPE_CHECKING` or directly — the module already imports `BallAnchor` from the schema, so a direct import is consistent.)

3. `src/stages/ball.py` — single-loader refactor:

```python
def _load_manual_anchor_set(
    output_dir: Path, shot_id: str
) -> BallAnchorSet | None:
    """The operator's manual anchor sidecar for a shot; None when absent
    or invalid. Single load point for anchors, shot chains and dismissals."""
    if shot_id:
        path = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
    else:
        path = output_dir / "ball" / "ball_anchors.json"
    if not path.exists():
        return None
    try:
        return BallAnchorSet.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ball stage: failed to load anchors at %s: %s",
                       path, exc)
        return None
```

Then: replace the body of `_load_ball_anchors` with a delegation (`aset = _load_manual_anchor_set(...); return {a.frame: a for a in aset.anchors} if aset else {}`) — grep for `_load_ball_anchors` callers first and keep its signature; DELETE `_load_manual_shot_chains` and, in the chain-diag block, obtain chains and dismissals from one `manual_set = _load_manual_anchor_set(self.output_dir, shot_id)` call (`manual_chains = manual_set.shot_chains if manual_set else ()`), and pass dismissals into the merge:

```python
        manual_set = _load_manual_anchor_set(self.output_dir, shot_id)
        anchor_by_frame = merge_anchors(
            manual_by_frame, auto_by_frame, anchor_cfg.suppress_radius_frames,
            dismissed=(manual_set.dismissed_auto if manual_set else ()),
        )
```

(Read the `_solve_shot` region: `manual_by_frame` comes from `artifacts.manual_by_frame` built in the detect pass — keep that; the `manual_set` load here serves dismissals + chains. One extra file read per shot is acceptable and still fewer than the previous double-parse.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_dismissed_auto.py tests/test_ball_stage_shot_chains.py tests/test_ball_auto_anchor.py tests/test_ball_anchor_schema_phase2.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/schemas/ball_anchor.py src/utils/ball_auto_anchor.py src/stages/ball.py tests/test_ball_dismissed_auto.py tests/test_ball_stage_shot_chains.py
git commit -m "feat: dismissed_auto schema + merge suppression + single manual-set loader"
```

---

### Task 6: Server passthrough + recall-report FP breakdown

**Files:**
- Modify: `src/web/server.py` — `BallAnchorPayload` (:1905-1909) gains `dismissed_auto`; the POST handler's `BallAnchorSet(...)` construction (~:2154-2158) passes it through (and the preview handler's equivalent construction).
- Modify: `src/utils/ball_touch_recall.py` — add `fp_breakdown`.
- Modify: `scripts/run_touch_recall_validation.py` and `scripts/ball_touch_recall_report.py` — print the breakdown.
- Test: `tests/test_web_ball_dismissed_api.py`, extend `tests/test_ball_touch_recall.py`

**Interfaces:**
- Consumes: `DismissedAuto` / `BallAnchorSet.dismissed_auto` from Task 5.
- Produces:
  - Pydantic: `class DismissedAutoEntry(BaseModel): frame: int; state: str; player_id: str | None = None; bone: str | None = None` and `BallAnchorPayload.dismissed_auto: list[DismissedAutoEntry] = []`; POST and preview persist it (`dismissed_auto=tuple(DismissedAuto(frame=d.frame, state=d.state, player_id=d.player_id, bone=d.bone) for d in payload.dismissed_auto)`).
  - `fp_breakdown(auto: list[Touch], manual: list[Touch], dismissed: list[Touch], *, frame_tol: int = 2) -> dict` in `ball_touch_recall.py` returning `{"fp_total": int, "fp_dismissed": int, "fp_unreviewed": int}` — FPs are the auto touches left unmatched by the same greedy strict matching as `match_touches`; an FP counts as dismissed when it equals a dismissed touch triple exactly.
  - `dismissed_touches_from_anchor_set(path: str | Path) -> list[Touch]` — the `player_touch` triples from the sidecar's `dismissed_auto` list.
  - Both report scripts print a `dismissed/unreviewed` FP column per config when the manual sidecar carries dismissals.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_ball_dismissed_api.py`:

```python
"""dismissed_auto round-trips through the save endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_post_persists_dismissed_auto(tmp_path: Path):
    client = _client(tmp_path)
    payload = {
        "clip_id": "play", "image_size": [1280, 720],
        "anchors": [{"frame": 5, "image_xy": [1.0, 2.0], "state": "grounded"}],
        "dismissed_auto": [
            {"frame": 20, "state": "player_touch",
             "player_id": "P003", "bone": "l_foot"},
        ],
    }
    r = client.post("/ball-anchors/play", json=payload)
    assert r.status_code == 200, r.text
    got = client.get("/ball-anchors/play").json()
    assert got["dismissed_auto"] == [
        {"frame": 20, "state": "player_touch",
         "player_id": "P003", "bone": "l_foot"},
    ]


def test_post_without_dismissals_unchanged(tmp_path: Path):
    client = _client(tmp_path)
    payload = {"clip_id": "play", "image_size": [1280, 720],
               "anchors": []}
    assert client.post("/ball-anchors/play", json=payload).status_code == 200
    assert client.get("/ball-anchors/play").json().get(
        "dismissed_auto", []) == []
```

Append to `tests/test_ball_touch_recall.py`:

```python
def test_fp_breakdown_partitions_dismissed_and_unreviewed():
    from src.utils.ball_touch_recall import fp_breakdown

    manual = [(10, "P1", "r_foot")]
    auto = [(10, "P1", "r_foot"),   # TP
            (30, "P2", "l_foot"),   # FP, dismissed
            (50, "P3", "head")]     # FP, unreviewed
    dismissed = [(30, "P2", "l_foot")]
    out = fp_breakdown(auto, manual, dismissed, frame_tol=2)
    assert out == {"fp_total": 2, "fp_dismissed": 1, "fp_unreviewed": 1}


def test_dismissed_touches_loader(tmp_path):
    import json
    from src.utils.ball_touch_recall import dismissed_touches_from_anchor_set

    p = tmp_path / "a.json"
    p.write_text(json.dumps({
        "clip_id": "x", "image_size": [1280, 720], "anchors": [],
        "dismissed_auto": [
            {"frame": 30, "state": "player_touch",
             "player_id": "P2", "bone": "l_foot"},
            {"frame": 40, "state": "bounce"},
        ],
    }))
    assert dismissed_touches_from_anchor_set(p) == [(30, "P2", "l_foot")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_web_ball_dismissed_api.py tests/test_ball_touch_recall.py -v`
Expected: the new tests FAIL (`dismissed_auto` dropped by the payload model / `ImportError: fp_breakdown`).

- [ ] **Step 3: Implement**

1. `src/web/server.py`: add next to `BallAnchorEntry`:

```python
    class DismissedAutoEntry(BaseModel):
        frame: int
        state: str
        player_id: str | None = None
        bone: str | None = None
```

add `dismissed_auto: list[DismissedAutoEntry] = []` to `BallAnchorPayload`, and in BOTH `BallAnchorSet(...)` constructions (save handler ~:2154 and the preview handler's temp-sidecar build) add:

```python
                dismissed_auto=tuple(
                    DismissedAuto(frame=int(d.frame), state=str(d.state),
                                  player_id=d.player_id, bone=d.bone)
                    for d in payload.dismissed_auto
                ),
```

with `DismissedAuto` added to the existing `from src.schemas.ball_anchor import ...` line.

2. `src/utils/ball_touch_recall.py`: append:

```python
def dismissed_touches_from_anchor_set(path: str | Path) -> list[Touch]:
    """The dismissed ``player_touch`` triples from a manual sidecar."""
    data = json.loads(Path(path).read_text())
    out: list[Touch] = []
    for d in data.get("dismissed_auto", []):
        if d.get("state") == "player_touch":
            out.append((int(d["frame"]), str(d.get("player_id") or ""),
                        str(d.get("bone") or "")))
    return sorted(out, key=lambda t: t[0])


def fp_breakdown(
    auto: list[Touch],
    manual: list[Touch],
    dismissed: list[Touch],
    *,
    frame_tol: int = 2,
) -> dict:
    """Partition strict-match false positives into operator-dismissed
    (reviewed, confirmed wrong) vs unreviewed (unknown — possibly real
    touches the manual set never annotated)."""
    manual_sorted = sorted(manual, key=lambda t: t[0])
    claimed = [False] * len(manual_sorted)
    fps: list[Touch] = []
    for af, ap, ab in sorted(auto, key=lambda t: t[0]):
        best_j = -1
        best_d = frame_tol + 1
        for j, (mf, _mp, mb) in enumerate(manual_sorted):
            if claimed[j]:
                continue
            d = abs(af - mf)
            if d > frame_tol or ab != mb:
                continue
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            claimed[best_j] = True
        else:
            fps.append((af, ap, ab))
    dismissed_set = set(dismissed)
    n_dismissed = sum(1 for fp in fps if fp in dismissed_set)
    return {
        "fp_total": len(fps),
        "fp_dismissed": n_dismissed,
        "fp_unreviewed": len(fps) - n_dismissed,
    }
```

3. `scripts/run_touch_recall_validation.py` (and mirror in `scripts/ball_touch_recall_report.py`'s `__main__`): after printing the table, load dismissals from the manual sidecar and print the breakdown per config:

```python
    from src.utils.ball_touch_recall import (
        dismissed_touches_from_anchor_set,
        fp_breakdown,
    )
    dismissed = dismissed_touches_from_anchor_set(manual_path)
    if dismissed:
        print(f"\nFP breakdown ({len(dismissed)} dismissed touches on record):")
        for name, auto_set in (("break_only", break_only), ("union", union)):
            b = fp_breakdown(auto_set, manual, dismissed)
            print(f"  {name:<12} fp={b['fp_total']}  "
                  f"dismissed={b['fp_dismissed']}  "
                  f"unreviewed={b['fp_unreviewed']}")
```

(In `ball_touch_recall_report.py` the manual path is `sys.argv[1]`; adapt names.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_web_ball_dismissed_api.py tests/test_ball_touch_recall.py tests/test_ball_kinematic_recall.py tests/test_run_touch_recall_validation.py tests/test_web_ball_phase2_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/web/server.py src/utils/ball_touch_recall.py scripts/run_touch_recall_validation.py scripts/ball_touch_recall_report.py tests/test_web_ball_dismissed_api.py tests/test_ball_touch_recall.py
git commit -m "feat: persist dismissed_auto via API + FP breakdown in recall reports"
```

---

### Task 7: Editor — merged chronological event list with confirm/dismiss + end_frame controls

**Files:**
- Modify: `src/web/static/ball_anchor_editor.html` — `renderAnchors()` (:274-331), `saveBtn.onclick` (:712-725), `loadShot()` (:798-827)
- Test: `tests/test_web_ball_editor_eventlist.py`

**Interfaces:**
- Consumes: save/load of `dismissed_auto` (Task 6 shapes); existing globals `anchors`, `autoAnchors`, `shotChains`, `setDirty`, `seekToFrame`, `currentFrame`, `drawOverlay`, TAGS.
- Produces: JS global `dismissedAuto: {frame,state,player_id,bone}[]`; a single chronological event list replacing the two-section list (manual rows solid, auto rows dashed with ＋ promote / ✕ dismiss; dismissed rows faded with ↩ undo); "⇥ end" / "clear end" controls on manual `player_touch` rows; payloads include `dismissed_auto`.

- [ ] **Step 1: Write the failing markup test**

Create `tests/test_web_ball_editor_eventlist.py`:

```python
"""The ball anchor editor ships the merged event list with dismiss/undo,
persisted dismissed_auto, and end-frame span controls."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_editor_served_with_event_list(tmp_path: Path):
    html = _client(tmp_path).get("/ball-anchor-editor").text
    assert "dismissedAuto" in html          # JS state
    assert "dismissed_auto" in html         # payload key round-trip
    assert 'title="Dismiss this suggestion' in html
    assert 'title="Undo dismissal' in html
    assert 'title="Set end frame' in html
    assert 'title="Clear end frame' in html
    # merged chronological list marker
    assert "Events (manual + auto" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_web_ball_editor_eventlist.py -v`
Expected: FAIL on the first assertion.

- [ ] **Step 3: Implement the JS/markup**

In `src/web/static/ball_anchor_editor.html`:

1. State (with the other `let` declarations, ~:194-200): `let dismissedAuto = [];`

2. Helpers (place above `renderAnchors`):

```javascript
function dismissKey(a) {
  return `${a.frame}|${a.state}|${a.player_id || ""}|${a.bone || ""}`;
}
function isDismissed(a) {
  return dismissedAuto.some(d => dismissKey(d) === dismissKey(a));
}
```

3. Replace `renderAnchors()` wholesale with the merged list (preserving the existing promote behaviour verbatim inside the auto branch):

```javascript
function renderAnchors() {
  anchorList.innerHTML = "";
  const head = document.createElement("div");
  head.style.cssText = "color:#64748b;font-size:10px;padding:2px 0;text-transform:uppercase;";
  head.textContent = `Events (manual + auto, ${anchors.length}+${autoAnchors.length})`;
  anchorList.appendChild(head);

  const rows = [
    ...anchors.map(a => ({ a, src: "manual" })),
    ...autoAnchors.map(a => ({ a, src: "auto" })),
  ].sort((x, y) => x.a.frame - y.a.frame || (x.src === "manual" ? -1 : 1));

  for (const { a, src } of rows) {
    const t = TAGS.find(x => x.id === a.state) || { color: "#94a3b8" };
    const row = document.createElement("div");
    row.className = "anchor-row";
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.gap = "6px";
    row.onclick = () => seekToFrame(a.frame);

    const detail = a.player_id
      ? ` (${a.player_id}/${a.bone || "?"}${a.touch_type ? " " + a.touch_type : ""})`
      : a.goal_element ? ` (${a.goal_element})`
      : a.landmark ? ` (⚑ ${a.landmark})` : "";
    const span = a.end_frame != null ? ` →${a.end_frame}` : "";

    if (src === "manual") {
      row.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${t.color};flex-shrink:0;"></span><span style="flex:1;min-width:0;">Frame ${a.frame}${span} — ${a.state}${detail}</span>`;
      if (a.state === "player_touch") {
        const setEnd = document.createElement("button");
        setEnd.textContent = "⇥";
        setEnd.title = "Set end frame to the current video frame (span event, e.g. carry)";
        setEnd.style.cssText = "background:#334155;color:#e2e8f0;border:1px solid #475569;border-radius:3px;font-size:10px;padding:0 5px;cursor:pointer;";
        setEnd.onclick = (ev) => {
          ev.stopPropagation();
          const fi = currentFrame();
          if (fi <= a.frame) { setStatus("end frame must be after the anchor frame"); return; }
          a.end_frame = fi;
          setDirty(true); renderAnchors();
        };
        row.appendChild(setEnd);
        if (a.end_frame != null) {
          const clr = document.createElement("button");
          clr.textContent = "⇤";
          clr.title = "Clear end frame (back to a point event)";
          clr.style.cssText = setEnd.style.cssText;
          clr.onclick = (ev) => {
            ev.stopPropagation();
            a.end_frame = null;
            setDirty(true); renderAnchors();
          };
          row.appendChild(clr);
        }
      }
      anchorList.appendChild(row);
      continue;
    }

    // Auto suggestion row.
    const dismissed = isDismissed(a);
    const suppressed = anchors.some(m => Math.abs(m.frame - a.frame) <= 3);
    if (dismissed || suppressed) row.style.opacity = "0.45";
    const conf = (a.confidence != null) ? ` · ${Math.round(a.confidence * 100)}%` : "";
    const flag = dismissed ? " · dismissed" : suppressed ? " · suppressed" : "";
    row.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;border:1.5px dashed ${t.color};flex-shrink:0;"></span><span style="flex:1;min-width:0;${dismissed ? "text-decoration:line-through;" : ""}">Frame ${a.frame} — ${a.state}${detail}${conf}${flag}</span>`;

    if (dismissed) {
      const undo = document.createElement("button");
      undo.textContent = "↩";
      undo.title = "Undo dismissal";
      undo.style.cssText = "background:#334155;color:#e2e8f0;border:1px solid #475569;border-radius:3px;font-size:11px;padding:0 6px;cursor:pointer;";
      undo.onclick = (ev) => {
        ev.stopPropagation();
        dismissedAuto = dismissedAuto.filter(d => dismissKey(d) !== dismissKey(a));
        setDirty(true); renderAnchors();
      };
      row.appendChild(undo);
    } else if (!suppressed) {
      const add = document.createElement("button");
      add.textContent = "＋";
      add.title = "Promote this suggestion to an editable anchor";
      add.style.cssText = "background:#334155;color:#e2e8f0;border:1px solid #475569;border-radius:3px;font-size:11px;padding:0 6px;cursor:pointer;";
      add.onclick = (ev) => {
        ev.stopPropagation();
        anchors = anchors.filter(m => m.frame !== a.frame);
        anchors.push({
          frame: a.frame, image_xy: a.image_xy ?? null, state: a.state,
          player_id: a.player_id || null, bone: a.bone || null,
          goal_element: a.goal_element || null,
          touch_type: a.touch_type || null, confidence: 1.0,
        });
        setDirty(true); renderAnchors(); drawOverlay();
      };
      row.appendChild(add);
      const dis = document.createElement("button");
      dis.textContent = "✕";
      dis.title = "Dismiss this suggestion (persisted; won't return on re-runs)";
      dis.style.cssText = "background:#334155;color:#f87171;border:1px solid #475569;border-radius:3px;font-size:11px;padding:0 6px;cursor:pointer;";
      dis.onclick = (ev) => {
        ev.stopPropagation();
        dismissedAuto.push({
          frame: a.frame, state: a.state,
          player_id: a.player_id || null, bone: a.bone || null,
        });
        setDirty(true); renderAnchors();
      };
      row.appendChild(dis);
    }
    anchorList.appendChild(row);
  }
}
```

4. `saveBtn.onclick` payload: `const payload = { clip_id: shotId || "", image_size: imageSize, anchors, shot_chains: shotChains, dismissed_auto: dismissedAuto };` (and mirror `dismissed_auto: dismissedAuto` into the preview payload construction).

5. `loadShot()`: after the `shotChains = ...` line add:

```javascript
  dismissedAuto = (ar && ar.dismissed_auto) ? ar.dismissed_auto.map(d => ({
    frame: d.frame, state: d.state,
    player_id: d.player_id ?? null, bone: d.bone ?? null,
  })) : [];
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_web_ball_editor_eventlist.py tests/test_web_ball_editor_phase2.py tests/test_web_ball_editor_touch_panel.py tests/test_web_ball_quality_timeline.py -v`
Expected: all PASS (existing editor markup tests confirm the restructure kept their markers; if one asserts on the old "Auto anchors (" section header, update THAT assertion to the new merged-list header in the same commit and say so in the report).

- [ ] **Step 5: Commit**

```bash
git add src/web/static/ball_anchor_editor.html tests/test_web_ball_editor_eventlist.py
git commit -m "feat: merged event list with dismiss/undo + end-frame span controls"
```

---

### Task 8: Full-suite verification + real-clip acceptance measurement

- [ ] **Step 1: Full suite + lint**

Run: `.venv/bin/python -m pytest -q`
Expected: everything passes except the known env-dependent Blender test.

Run: `.venv/bin/python -m ruff check src/utils/ball_context_prior.py src/utils/ball_touch_attribution.py src/stages/ball.py src/utils/ball_auto_anchor.py src/utils/ball_touch_recall.py src/schemas/ball_anchor.py src/web/server.py scripts/run_touch_recall_validation.py scripts/ball_touch_recall_report.py tests/test_ball_context_prior.py tests/test_ball_stage_context_prior.py tests/test_ball_touch_attribution.py tests/test_ball_stage_attribution_wiring.py tests/test_ball_dismissed_auto.py tests/test_web_ball_dismissed_api.py tests/test_web_ball_editor_eventlist.py`
Expected: clean. Commit fixups as `chore: phase-3 lint/test fixups` only if needed.

- [ ] **Step 2: Re-run the gberch recall validation (local, MPS — ~15 min)**

```bash
.venv/bin/python scripts/run_touch_recall_validation.py --output output --shot gberch
```

Baseline (2026-07-02, pre-Phase-3): break_only 0.125 recall / 1 tp / 11 fp; union 0.250 / 2 tp / 16 fp. Acceptance movement expected from this branch: strict union recall ≥ 3/8 (target 4/8 — the loose ceiling) via attribution relabelling; auto FP count not increased. Record the table in the task report and paste it into the commit message body of a final `docs:` commit updating the plan/ledger. If recall does NOT move: dump per-touch best-gap diagnostics (add a temporary script call, not committed) and report BLOCKED-style findings rather than tuning blindly.

- [ ] **Step 3: Coverage regression check on a second clip (origi/kroupi)**

```bash
.venv/bin/python recon.py run --output output-origi --stages ball 2>&1 | tail -5
.venv/bin/python -c "
import json
q = json.load(open('output-origi/quality_report.json'))
print(json.dumps(q.get('ball', {}).get('origi01', {}).get('detection_coverage', q), indent=2)[:400])
"
```

Expected: `detection_coverage.total` within 0.02 of its pre-branch value (the prior must not eat real-ball detections; its stage tests cover the mechanism, this covers the real clip). Record the numbers. (Skip gracefully with a note if `output-origi` lacks upstream artifacts on this machine.)

- [ ] **Step 4: Final docs commit**

Update `CLAUDE.md`'s ball config bullet list with one line for `ball.context_prior.*` + `ball.touch_attribution.*` (mirror the existing `ball.shot_chain.*` bullet style), then:

```bash
git add CLAUDE.md && git commit -m "docs: phase-3 config bullets + recall movement record"
```

# Refined Poses Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new pipeline stage `refined_poses` (between `ball` and `export`) that fuses each player's per-shot SMPL reconstructions into a single per-player track on the shared reference timeline, with robust per-frame outlier rejection, post-fusion smoothing, and per-player diagnostics surfaced in the quality report.

**Architecture:** New stage reads `output/hmr_world/{shot_id}__{player_id}_smpl_world.npz` files, groups by `player_id`, shifts each contributing shot's frames onto the reference timeline using `output/shots/sync_map.json`, and per reference frame fuses contributing views with confidence weighting + MAD-based outlier rejection. Rotations are fused via the chordal SO(3) mean inheriting the position-derived kept_mask. Smoothing reuses `src/utils/temporal_smoothing.py` (`savgol_axis`, `slerp_window`). Output: `output/refined_poses/{player_id}_refined.npz` + `_diagnostics.json` per player + `refined_poses_summary.json`. Export is updated to consume fused tracks per shot via reverse-offset; falls back to legacy `SmplWorldTrack` reads when the refined dir is empty.

**Tech Stack:** Python 3.11, NumPy, SciPy (existing), pytest. Reuses `src/schemas/sync_map.py`, `src/schemas/smpl_world.py`, `src/utils/temporal_smoothing.py`, `src/pipeline/base.py`.

**Spec:** `docs/superpowers/specs/2026-05-09-refined-poses-design.md`

---

## File structure

**Create:**
- `src/schemas/refined_pose.py` — `RefinedPose` (NPZ-persisted), `FrameDiagnostic`, `RefinedPoseDiagnostics` (JSON-persisted)
- `src/utils/pose_fusion.py` — `so3_chordal_mean`, `so3_geodesic_distance`, `robust_translation_fuse`
- `src/stages/refined_poses.py` — `RefinedPosesStage` (`BaseStage` subclass)
- `tests/test_refined_pose_schema.py` — schema round-trip tests
- `tests/test_pose_fusion.py` — math-utility unit tests
- `tests/test_refined_poses_stage.py` — stage integration tests

**Modify:**
- `src/pipeline/runner.py` — register `refined_poses` between `ball` and `export`
- `src/stages/export.py` — load `RefinedPose` per player, project frames back to shot-local timeline using `sync_map`; fall back to `SmplWorldTrack` when refined dir empty
- `src/pipeline/quality_report.py` — append `refined_poses` section from `refined_poses_summary.json`
- `config/default.yaml` — add `refined_poses:` section
- `tests/test_runner.py` — assert new stage in ordering
- `tests/test_quality_report.py` — assert new section
- `tests/test_export_stage.py` — assert refined-track consumption + fallback
- `src/web/static/index.html` — small Multi-shot status panel addition (refined-poses status column)

---

## Task 1: RefinedPose + diagnostics schema

**Files:**
- Create: `src/schemas/refined_pose.py`
- Test: `tests/test_refined_pose_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_refined_pose_schema.py
"""Round-trip tests for RefinedPose and RefinedPoseDiagnostics."""

from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import (
    FrameDiagnostic,
    RefinedPose,
    RefinedPoseDiagnostics,
)


@pytest.mark.unit
def test_refined_pose_round_trip(tmp_path: Path) -> None:
    pose = RefinedPose(
        player_id="P002",
        frames=np.array([0, 1, 2, 3], dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((4, 24, 3)),
        root_R=np.tile(np.eye(3), (4, 1, 1)),
        root_t=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        confidence=np.ones(4),
        view_count=np.array([1, 2, 2, 1], dtype=np.int32),
        contributing_shots=("origi01", "origi02"),
    )
    p = tmp_path / "P002_refined.npz"
    pose.save(p)
    loaded = RefinedPose.load(p)
    assert loaded.player_id == "P002"
    np.testing.assert_array_equal(loaded.frames, pose.frames)
    np.testing.assert_array_equal(loaded.view_count, pose.view_count)
    np.testing.assert_allclose(loaded.root_t, pose.root_t)
    assert loaded.contributing_shots == ("origi01", "origi02")


@pytest.mark.unit
def test_refined_pose_diagnostics_round_trip(tmp_path: Path) -> None:
    diag = RefinedPoseDiagnostics(
        player_id="P002",
        frames=(
            FrameDiagnostic(
                frame=0,
                contributing_shots=("origi01",),
                dropped_shots=(),
                pos_disagreement_m=0.0,
                rot_disagreement_rad=0.0,
                low_coverage=True,
                high_disagreement=False,
            ),
            FrameDiagnostic(
                frame=1,
                contributing_shots=("origi01", "origi02"),
                dropped_shots=("origi03",),
                pos_disagreement_m=0.05,
                rot_disagreement_rad=0.02,
                low_coverage=False,
                high_disagreement=False,
            ),
        ),
        contributing_shots=("origi01", "origi02", "origi03"),
        summary={
            "total_frames": 2,
            "single_view_frames": 1,
            "high_disagreement_frames": 0,
        },
    )
    p = tmp_path / "P002_diagnostics.json"
    diag.save(p)
    loaded = RefinedPoseDiagnostics.load(p)
    assert loaded.player_id == "P002"
    assert len(loaded.frames) == 2
    assert loaded.frames[0].low_coverage is True
    assert loaded.frames[1].contributing_shots == ("origi01", "origi02")
    assert loaded.frames[1].dropped_shots == ("origi03",)
    assert loaded.summary["total_frames"] == 2
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_refined_pose_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.schemas.refined_pose'`

- [ ] **Step 3: Implement the schema**

```python
# src/schemas/refined_pose.py
"""Per-player fused SMPL track and per-frame fusion diagnostics.

Output of the ``refined_poses`` stage. RefinedPose mirrors
SmplWorldTrack but is keyed by player_id only and indexed onto the
shared reference timeline (see src/schemas/sync_map.py for the
sign convention). Diagnostics record which shots contributed to each
fused frame and how much they disagreed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RefinedPose:
    """Per-player SMPL track fused across all shots that saw the player.

    Pitch-world frame matches SmplWorldTrack: z-up, x along nearside
    touchline, y toward far side, root_t in pitch metres.
    """

    player_id: str
    frames: np.ndarray              # (N,) reference-timeline frame indices
    betas: np.ndarray               # (10,) shared across the whole track
    thetas: np.ndarray              # (N, 24, 3) axis-angle
    root_R: np.ndarray              # (N, 3, 3)
    root_t: np.ndarray              # (N, 3) pitch metres
    confidence: np.ndarray          # (N,) fused confidence (sum of contributing weights)
    view_count: np.ndarray          # (N,) int — how many shots contributed at this frame
    contributing_shots: tuple[str, ...]  # union across all frames

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            player_id=self.player_id,
            frames=self.frames,
            betas=self.betas,
            thetas=self.thetas,
            root_R=self.root_R,
            root_t=self.root_t,
            confidence=self.confidence,
            view_count=self.view_count,
            contributing_shots=np.array(list(self.contributing_shots)),
        )

    @classmethod
    def load(cls, path: Path) -> "RefinedPose":
        z = np.load(path, allow_pickle=False)
        return cls(
            player_id=str(z["player_id"]),
            frames=z["frames"],
            betas=z["betas"],
            thetas=z["thetas"],
            root_R=z["root_R"],
            root_t=z["root_t"],
            confidence=z["confidence"],
            view_count=z["view_count"],
            contributing_shots=tuple(str(s) for s in z["contributing_shots"]),
        )


@dataclass(frozen=True)
class FrameDiagnostic:
    """One reference frame's fusion bookkeeping."""

    frame: int
    contributing_shots: tuple[str, ...]
    dropped_shots: tuple[str, ...]
    pos_disagreement_m: float
    rot_disagreement_rad: float
    low_coverage: bool
    high_disagreement: bool


@dataclass(frozen=True)
class RefinedPoseDiagnostics:
    """Per-player diagnostics; companion to a RefinedPose NPZ."""

    player_id: str
    frames: tuple[FrameDiagnostic, ...]
    contributing_shots: tuple[str, ...]
    summary: dict

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "player_id": self.player_id,
            "contributing_shots": list(self.contributing_shots),
            "frames": [
                {
                    "frame": f.frame,
                    "contributing_shots": list(f.contributing_shots),
                    "dropped_shots": list(f.dropped_shots),
                    "pos_disagreement_m": f.pos_disagreement_m,
                    "rot_disagreement_rad": f.rot_disagreement_rad,
                    "low_coverage": f.low_coverage,
                    "high_disagreement": f.high_disagreement,
                }
                for f in self.frames
            ],
            "summary": self.summary,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RefinedPoseDiagnostics":
        data = json.loads(path.read_text())
        return cls(
            player_id=data["player_id"],
            contributing_shots=tuple(data["contributing_shots"]),
            frames=tuple(
                FrameDiagnostic(
                    frame=int(f["frame"]),
                    contributing_shots=tuple(f["contributing_shots"]),
                    dropped_shots=tuple(f["dropped_shots"]),
                    pos_disagreement_m=float(f["pos_disagreement_m"]),
                    rot_disagreement_rad=float(f["rot_disagreement_rad"]),
                    low_coverage=bool(f["low_coverage"]),
                    high_disagreement=bool(f["high_disagreement"]),
                )
                for f in data["frames"]
            ),
            summary=data["summary"],
        )
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_refined_pose_schema.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/schemas/refined_pose.py tests/test_refined_pose_schema.py
git commit -m "feat(refined_poses): add RefinedPose and diagnostics schemas"
```

---

## Task 2: SO(3) primitives — chordal mean and geodesic distance

**Files:**
- Create: `src/utils/pose_fusion.py`
- Test: `tests/test_pose_fusion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pose_fusion.py
"""Unit tests for src.utils.pose_fusion (math primitives only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.pose_fusion import so3_chordal_mean, so3_geodesic_distance


def _rotation_matrix_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.unit
def test_so3_chordal_mean_returns_input_for_single_view() -> None:
    R = _rotation_matrix_z(np.pi / 4)
    result = so3_chordal_mean(R[None, :, :], np.array([1.0]))
    np.testing.assert_allclose(result, R, atol=1e-9)


@pytest.mark.unit
def test_so3_chordal_mean_identity_for_two_equal_rotations() -> None:
    R = _rotation_matrix_z(np.pi / 6)
    result = so3_chordal_mean(np.stack([R, R]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(result, R, atol=1e-9)


@pytest.mark.unit
def test_so3_chordal_mean_skews_toward_heavy_view() -> None:
    R1 = np.eye(3)
    R2 = _rotation_matrix_z(np.pi / 2)
    # Weight 1.0 on R1, 0.0 on R2 → result is R1.
    result = so3_chordal_mean(np.stack([R1, R2]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(result, R1, atol=1e-9)


@pytest.mark.unit
def test_so3_chordal_mean_returns_proper_rotation() -> None:
    rng = np.random.default_rng(0)
    Rs = np.stack([_rotation_matrix_z(a) for a in rng.uniform(-1, 1, 5)])
    result = so3_chordal_mean(Rs, np.ones(5))
    assert np.linalg.det(result) > 0
    np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-9)


@pytest.mark.unit
def test_so3_chordal_mean_rejects_zero_weights() -> None:
    R = np.eye(3)
    with pytest.raises(ValueError):
        so3_chordal_mean(R[None, :, :], np.array([0.0]))


@pytest.mark.unit
def test_so3_geodesic_distance_known_angles() -> None:
    R1 = np.eye(3)
    R2 = _rotation_matrix_z(np.pi / 2)
    assert abs(so3_geodesic_distance(R1, R2) - np.pi / 2) < 1e-9
    assert so3_geodesic_distance(R1, R1) < 1e-9


@pytest.mark.unit
def test_so3_geodesic_distance_symmetric() -> None:
    R1 = _rotation_matrix_z(0.3)
    R2 = _rotation_matrix_z(1.1)
    d12 = so3_geodesic_distance(R1, R2)
    d21 = so3_geodesic_distance(R2, R1)
    assert abs(d12 - d21) < 1e-12
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_pose_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.pose_fusion'`

- [ ] **Step 3: Implement the SO(3) primitives**

```python
# src/utils/pose_fusion.py
"""Math primitives for cross-shot SMPL fusion.

Used by ``src.stages.refined_poses``. All functions are pure — no I/O,
no logging — and operate on numpy arrays. SO(3) operations follow the
chordal (Frobenius) cost so the weighted mean is the SVD-projection
of the Euclidean weighted mean back onto the rotation group.
"""

from __future__ import annotations

import numpy as np


def so3_chordal_mean(rotations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted chordal mean of a stack of SO(3) matrices.

    Minimises ``sum_v w_v ||R - R_v||_F^2`` over R in SO(3) via
    SVD-projection of the weighted Euclidean mean.

    Args:
        rotations: ``(V, 3, 3)`` proper rotation matrices.
        weights:   ``(V,)`` non-negative weights with sum > 0.

    Returns:
        The mean rotation as a ``(3, 3)`` proper rotation matrix.

    Raises:
        ValueError: if shapes mismatch or weights sum to zero.
    """
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"rotations must be (V, 3, 3); got {rotations.shape}")
    if weights.shape != (rotations.shape[0],):
        raise ValueError(f"weights must be ({rotations.shape[0]},); got {weights.shape}")
    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        raise ValueError("weights sum to zero")
    weighted_mean = (weights[:, None, None] * rotations).sum(axis=0) / w_sum
    U, _, Vt = np.linalg.svd(weighted_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # Flip the smallest-singular-value column so det(R) = +1.
        Vt[-1, :] *= -1
        R = U @ Vt
    return R


def so3_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance in radians between two SO(3) matrices."""
    cos_theta = (np.trace(R1.T @ R2) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_pose_fusion.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/pose_fusion.py tests/test_pose_fusion.py
git commit -m "feat(refined_poses): add SO(3) chordal mean and geodesic distance"
```

---

## Task 3: Robust translation fuse with MAD-based outlier drop

**Files:**
- Modify: `src/utils/pose_fusion.py`
- Modify: `tests/test_pose_fusion.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pose_fusion.py`)

```python
from src.utils.pose_fusion import robust_translation_fuse


@pytest.mark.unit
def test_robust_translation_fuse_two_views_passthrough() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    weights = np.array([1.0, 1.0])
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    np.testing.assert_allclose(fused, [2.0, 0.0, 0.0])
    assert kept.tolist() == [True, True]


@pytest.mark.unit
def test_robust_translation_fuse_drops_far_outlier() -> None:
    # Two views agree at x ≈ 1, third is 11 m away → dropped.
    positions = np.array(
        [[1.0, 0.0, 0.0], [1.05, 0.0, 0.0], [11.0, 0.0, 0.0]]
    )
    weights = np.ones(3)
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    assert kept.tolist() == [True, True, False]
    np.testing.assert_allclose(fused, [1.025, 0.0, 0.0])


@pytest.mark.unit
def test_robust_translation_fuse_weighted_after_drop() -> None:
    positions = np.array(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0]]
    )
    weights = np.array([1.0, 3.0, 1.0])
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    assert kept.tolist() == [True, True, False]
    # Weighted mean of [1, 3] with weights [1, 3] = (1 + 9) / 4 = 2.5.
    np.testing.assert_allclose(fused, [2.5, 0.0, 0.0])


@pytest.mark.unit
def test_robust_translation_fuse_no_drops_when_clustered() -> None:
    positions = np.array(
        [[1.0, 0.0, 0.0], [1.05, 0.0, 0.0], [0.95, 0.0, 0.0]]
    )
    weights = np.ones(3)
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    assert kept.tolist() == [True, True, True]
    np.testing.assert_allclose(fused, [1.0, 0.0, 0.0])


@pytest.mark.unit
def test_robust_translation_fuse_zero_weight_view_excluded_from_mean() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    weights = np.array([1.0, 0.0])
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    np.testing.assert_allclose(fused, [1.0, 0.0, 0.0])
    assert kept.tolist() == [True, True]


@pytest.mark.unit
def test_robust_translation_fuse_zero_total_weight_returns_zero() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    weights = np.zeros(2)
    fused, kept = robust_translation_fuse(positions, weights, k_sigma=3.0)
    np.testing.assert_allclose(fused, [0.0, 0.0, 0.0])
    assert kept.tolist() == [False, False]
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_pose_fusion.py -v -k robust_translation_fuse`
Expected: FAIL with `ImportError: cannot import name 'robust_translation_fuse'`

- [ ] **Step 3: Implement `robust_translation_fuse`** (append to `src/utils/pose_fusion.py`)

```python
def robust_translation_fuse(
    positions: np.ndarray,
    weights: np.ndarray,
    k_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean with MAD-based outlier drop.

    With V <= 2: returns the weighted mean and an all-True kept mask
    (no outlier check possible with fewer than 3 views).

    With V >= 3:
        1. Compute the weighted mean across all views.
        2. distances[v] = ||positions[v] - mean||
        3. MAD = median(|distances - median(distances)|)
        4. Drop views with distances > median + k_sigma * 1.4826 * MAD.
           (1.4826 makes MAD a consistent estimator of stddev for
           normally distributed data.)
        5. Re-compute the weighted mean over kept views.

    Edge cases:
        - All weights zero → returns (zeros, all-False mask).
        - MAD == 0 (cluster is a point) → no drops.
        - All views dropped → falls back to no-drop weighted mean.

    Args:
        positions: ``(V, 3)`` candidate positions in pitch metres.
        weights:   ``(V,)`` non-negative weights.
        k_sigma:   scale of the outlier threshold in MAD units.

    Returns:
        ``(fused_position (3,), kept_mask (V,) bool)``.
    """
    V = positions.shape[0]
    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        return np.zeros(3), np.zeros(V, dtype=bool)
    weighted_mean = (weights[:, None] * positions).sum(axis=0) / w_sum
    if V <= 2:
        return weighted_mean, np.ones(V, dtype=bool)
    distances = np.linalg.norm(positions - weighted_mean, axis=1)
    median_dist = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_dist)))
    if mad == 0.0:
        return weighted_mean, np.ones(V, dtype=bool)
    threshold = median_dist + k_sigma * 1.4826 * mad
    kept = distances <= threshold
    if not kept.any():
        return weighted_mean, np.ones(V, dtype=bool)
    kept_weights = weights * kept.astype(float)
    if kept_weights.sum() <= 0:
        return weighted_mean, np.ones(V, dtype=bool)
    fused = (kept_weights[:, None] * positions).sum(axis=0) / kept_weights.sum()
    return fused, kept
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_pose_fusion.py -v`
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/pose_fusion.py tests/test_pose_fusion.py
git commit -m "feat(refined_poses): add robust translation fuse with MAD outlier drop"
```

---

## Task 4: RefinedPosesStage — single-shot passthrough

Smallest end-to-end stage slice: discover players, write per-player NPZ + diagnostics for players that appear in only one shot, no fusion math yet.

**Files:**
- Create: `src/stages/refined_poses.py`
- Test: `tests/test_refined_poses_stage.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_refined_poses_stage.py
"""Tests for src.stages.refined_poses.RefinedPosesStage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose, RefinedPoseDiagnostics
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import Alignment, SyncMap
from src.stages.refined_poses import RefinedPosesStage


def _default_config() -> dict:
    return {
        "refined_poses": {
            "outlier_k_sigma": 3.0,
            "min_contributing_views": 1,
            "high_disagreement_pos_m": 0.5,
            "high_disagreement_rot_rad": 0.5,
            "savgol_window": 1,        # tests run with smoothing disabled
            "savgol_poly": 2,
            "smooth_rotations": False,
            "beta_aggregation": "weighted_mean",
            "beta_disagreement_warn": 0.3,
        }
    }


def _make_smpl_track(
    *,
    player_id: str,
    shot_id: str,
    n_frames: int,
    root_t_x_per_frame: float = 0.5,
    confidence: float = 1.0,
) -> SmplWorldTrack:
    frames = np.arange(n_frames, dtype=np.int64)
    return SmplWorldTrack(
        player_id=player_id,
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n_frames, 24, 3)),
        root_R=np.tile(np.eye(3), (n_frames, 1, 1)),
        root_t=np.column_stack(
            [frames * root_t_x_per_frame, np.zeros(n_frames), np.zeros(n_frames)]
        ),
        confidence=np.full(n_frames, confidence),
        shot_id=shot_id,
    )


def _write_sync_map(output_dir: Path, *, ref: str, offsets: dict[str, int]) -> None:
    alignments = [
        Alignment(shot_id=sid, frame_offset=off, method="manual", confidence=1.0)
        for sid, off in offsets.items()
    ]
    sm = SyncMap(reference_shot=ref, alignments=alignments)
    sm.save(output_dir / "shots" / "sync_map.json")


@pytest.mark.integration
def test_refined_poses_single_shot_passthrough(tmp_path: Path) -> None:
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="origi01", offsets={"origi01": 0})
    track = _make_smpl_track(player_id="P001", shot_id="origi01", n_frames=10)
    track.save(output_dir / "hmr_world" / "origi01__P001_smpl_world.npz")

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    assert stage.is_complete() is False
    stage.run()
    assert stage.is_complete() is True

    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    assert refined.player_id == "P001"
    assert refined.contributing_shots == ("origi01",)
    np.testing.assert_array_equal(refined.frames, track.frames)
    np.testing.assert_allclose(refined.root_t, track.root_t)
    assert refined.view_count.tolist() == [1] * 10

    diag = RefinedPoseDiagnostics.load(
        output_dir / "refined_poses" / "P001_diagnostics.json"
    )
    assert diag.contributing_shots == ("origi01",)
    assert all(f.low_coverage for f in diag.frames)
    assert all(not f.high_disagreement for f in diag.frames)

    summary = json.loads(
        (output_dir / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert summary["players_refined"] == 1
    assert summary["single_shot_players"] == 1
    assert summary["multi_shot_players"] == 0


@pytest.mark.integration
def test_refined_poses_is_complete_after_run(tmp_path: Path) -> None:
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="origi01", offsets={"origi01": 0})
    _make_smpl_track(player_id="P001", shot_id="origi01", n_frames=5).save(
        output_dir / "hmr_world" / "origi01__P001_smpl_world.npz"
    )

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    stage.run()
    assert stage.is_complete() is True

    # Removing a refined NPZ flips is_complete back to False.
    (output_dir / "refined_poses" / "P001_refined.npz").unlink()
    assert stage.is_complete() is False
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_refined_poses_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.stages.refined_poses'`

- [ ] **Step 3: Implement the stage with single-shot passthrough only**

```python
# src/stages/refined_poses.py
"""Stage 7-of-7 between hmr_world/ball and export.

Fuses each player's per-shot SMPL reconstructions into a single track
on the shared reference timeline. v1 contract:

  - identity is the ``player_id`` annotation set during tracking;
    no automatic re-id;
  - sync is taken from ``output/shots/sync_map.json`` (authoritative);
  - players appearing in only one shot pass through unchanged.

This module hosts the stage class. Pure math lives in
``src.utils.pose_fusion``; smoothing helpers come from
``src.utils.temporal_smoothing``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.refined_pose import (
    FrameDiagnostic,
    RefinedPose,
    RefinedPoseDiagnostics,
)
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import SyncMap

logger = logging.getLogger(__name__)


class RefinedPosesStage(BaseStage):
    name = "refined_poses"

    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        hmr_dir = self.output_dir / "hmr_world"
        if not hmr_dir.exists():
            return True
        out_dir = self.output_dir / "refined_poses"
        player_ids = self._discover_player_ids(hmr_dir)
        if not player_ids:
            return True
        return all((out_dir / f"{pid}_refined.npz").exists() for pid in player_ids)

    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = (self.config.get("refined_poses") or {})
        hmr_dir = self.output_dir / "hmr_world"
        out_dir = self.output_dir / "refined_poses"
        out_dir.mkdir(parents=True, exist_ok=True)

        sync_map = self._load_sync_map()
        contributions = self._gather_contributions(hmr_dir)

        summary: dict = {
            "players_refined": 0,
            "single_shot_players": 0,
            "multi_shot_players": 0,
            "total_fused_frames": 0,
            "single_view_frames": 0,
            "high_disagreement_frames": 0,
            "shots_missing_sync": [],
            "beta_disagreement_warnings": [],
        }
        known_shots = {a.shot_id for a in sync_map.alignments}

        for pid, contribs in sorted(contributions.items()):
            refined, diag = self._fuse_player(pid, contribs, sync_map, cfg)
            refined.save(out_dir / f"{pid}_refined.npz")
            diag.save(out_dir / f"{pid}_diagnostics.json")

            summary["players_refined"] += 1
            distinct_shots = {sid for sid, _ in contribs}
            if len(distinct_shots) <= 1:
                summary["single_shot_players"] += 1
            else:
                summary["multi_shot_players"] += 1
            summary["total_fused_frames"] += len(refined.frames)
            for fd in diag.frames:
                if fd.low_coverage:
                    summary["single_view_frames"] += 1
                if fd.high_disagreement:
                    summary["high_disagreement_frames"] += 1
            for sid in distinct_shots:
                if sid and sid not in known_shots and sid not in summary["shots_missing_sync"]:
                    summary["shots_missing_sync"].append(sid)

        (out_dir / "refined_poses_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        logger.info(
            "[refined_poses] %d player(s) refined, %d frames, %d high-disagreement",
            summary["players_refined"],
            summary["total_fused_frames"],
            summary["high_disagreement_frames"],
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _discover_player_ids(hmr_dir: Path) -> set[str]:
        return {RefinedPosesStage._parse_pid(p) for p in hmr_dir.glob("*_smpl_world.npz")}

    @staticmethod
    def _parse_pid(npz_path: Path) -> str:
        stem = npz_path.name.removesuffix("_smpl_world.npz")
        if "__" in stem:
            return stem.split("__", 1)[1]
        return stem  # legacy single-shot files have no shot prefix

    def _load_sync_map(self) -> SyncMap:
        sync_path = self.output_dir / "shots" / "sync_map.json"
        if sync_path.exists():
            return SyncMap.load(sync_path)
        logger.warning(
            "[refined_poses] sync_map.json missing; treating all offsets as 0"
        )
        return SyncMap(reference_shot="", alignments=[])

    def _gather_contributions(
        self, hmr_dir: Path
    ) -> dict[str, list[tuple[str, SmplWorldTrack]]]:
        out: dict[str, list[tuple[str, SmplWorldTrack]]] = {}
        for npz in sorted(hmr_dir.glob("*_smpl_world.npz")):
            track = SmplWorldTrack.load(npz)
            out.setdefault(track.player_id, []).append((track.shot_id, track))
        return out

    # ------------------------------------------------------------------

    def _fuse_player(
        self,
        player_id: str,
        contribs: list[tuple[str, SmplWorldTrack]],
        sync_map: SyncMap,
        cfg: dict,
    ) -> tuple[RefinedPose, RefinedPoseDiagnostics]:
        """v1: single-shot passthrough only. Multi-shot path lands in Task 5."""
        if len({sid for sid, _ in contribs}) != 1:
            raise NotImplementedError(
                "multi-shot fusion not yet wired (Task 5)"
            )
        shot_id, track = contribs[0]
        offset = sync_map.offset_for(shot_id) if shot_id else 0
        ref_frames = np.asarray(track.frames, dtype=np.int64) - offset
        n = len(ref_frames)

        refined = RefinedPose(
            player_id=player_id,
            frames=ref_frames,
            betas=np.asarray(track.betas, dtype=np.float64),
            thetas=np.asarray(track.thetas, dtype=np.float64),
            root_R=np.asarray(track.root_R, dtype=np.float64),
            root_t=np.asarray(track.root_t, dtype=np.float64),
            confidence=np.asarray(track.confidence, dtype=np.float64),
            view_count=np.ones(n, dtype=np.int32),
            contributing_shots=(shot_id,) if shot_id else (),
        )
        diag = RefinedPoseDiagnostics(
            player_id=player_id,
            contributing_shots=(shot_id,) if shot_id else (),
            frames=tuple(
                FrameDiagnostic(
                    frame=int(ref_frames[i]),
                    contributing_shots=(shot_id,) if shot_id else (),
                    dropped_shots=(),
                    pos_disagreement_m=0.0,
                    rot_disagreement_rad=0.0,
                    low_coverage=True,
                    high_disagreement=False,
                )
                for i in range(n)
            ),
            summary={
                "total_frames": n,
                "single_view_frames": n,
                "high_disagreement_frames": 0,
            },
        )
        return refined, diag
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_refined_poses_stage.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/stages/refined_poses.py tests/test_refined_poses_stage.py
git commit -m "feat(refined_poses): add stage skeleton with single-shot passthrough"
```

---

## Task 5: Multi-shot fusion — weighted mean across reference timeline

**Files:**
- Modify: `src/stages/refined_poses.py`
- Modify: `tests/test_refined_poses_stage.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_refined_poses_stage.py`)

```python
@pytest.mark.integration
def test_refined_poses_two_shots_fuses_root_t(tmp_path: Path) -> None:
    """Two shots, sync offset, both see player at the same wall-clock instant.

    Each shot's local frame f corresponds to reference frame f - offset.
    Shot A: offset 0,  local frames 0..9, root_t = [f, 0, 0]
    Shot B: offset 5,  local frames 5..14, root_t = [(f-5)+0.2, 0, 0]
                       i.e. on the reference timeline B sees ref_f = local_f - 5
                       at root_t_x = ref_f + 0.2.

    Fused root_t at ref_f = 0..4 has only A → x = 0..4.
    Fused root_t at ref_f = 5..9 has both → x = mean(ref_f, ref_f + 0.2) = ref_f + 0.1.
    Fused root_t at ref_f = 10..14 has only B (would correspond to A's local 10..14
    which doesn't exist) — not in this test (A only has 10 frames).
    """
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="A", offsets={"A": 0, "B": 5})

    a = _make_smpl_track(player_id="P001", shot_id="A", n_frames=10)
    a.save(output_dir / "hmr_world" / "A__P001_smpl_world.npz")

    # B's local frames 5..14 (10 frames). Shift +0.2 m on x relative to A.
    n_b = 10
    b_local_frames = np.arange(5, 5 + n_b, dtype=np.int64)
    b_root_t = np.column_stack(
        [
            (b_local_frames - 5) + 0.2,
            np.zeros(n_b),
            np.zeros(n_b),
        ]
    )
    b = SmplWorldTrack(
        player_id="P001",
        frames=b_local_frames,
        betas=np.zeros(10),
        thetas=np.zeros((n_b, 24, 3)),
        root_R=np.tile(np.eye(3), (n_b, 1, 1)),
        root_t=b_root_t,
        confidence=np.ones(n_b),
        shot_id="B",
    )
    b.save(output_dir / "hmr_world" / "B__P001_smpl_world.npz")

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    stage.run()
    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")

    # Reference timeline spans 0..14:
    #   A contributes at ref 0..9 (local 0..9 - offset 0)
    #   B contributes at ref 0..9 (local 5..14 - offset 5)
    np.testing.assert_array_equal(refined.frames, np.arange(0, 15))
    # ref 0..4: only A
    np.testing.assert_allclose(refined.root_t[0:5, 0], np.arange(0, 5))
    # ref 5..9: both, mean = ref + 0.1
    np.testing.assert_allclose(
        refined.root_t[5:10, 0], np.arange(5, 10) + 0.1, atol=1e-9
    )
    # ref 10..14: only B
    np.testing.assert_allclose(
        refined.root_t[10:15, 0], np.arange(5, 10) + 0.2, atol=1e-9
    )
    # view_count
    assert refined.view_count.tolist() == (
        [1, 1, 1, 1, 1] + [2, 2, 2, 2, 2] + [1, 1, 1, 1, 1]
    )
    assert set(refined.contributing_shots) == {"A", "B"}
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_refined_poses_stage.py::test_refined_poses_two_shots_fuses_root_t -v`
Expected: FAIL — `NotImplementedError: multi-shot fusion not yet wired (Task 5)`.

- [ ] **Step 3: Replace `_fuse_player` with the multi-shot implementation**

Replace the existing `_fuse_player` method in `src/stages/refined_poses.py` with this version. Imports at the top of the file gain:

```python
from src.utils.pose_fusion import so3_chordal_mean, so3_geodesic_distance
from src.utils.temporal_smoothing import savgol_axis, slerp_window
```

Method body:

```python
    def _fuse_player(
        self,
        player_id: str,
        contribs: list[tuple[str, SmplWorldTrack]],
        sync_map: SyncMap,
        cfg: dict,
    ) -> tuple[RefinedPose, RefinedPoseDiagnostics]:
        # Project each contribution onto the reference timeline.
        on_ref: list[tuple[str, SmplWorldTrack, np.ndarray]] = []
        for shot_id, track in contribs:
            offset = sync_map.offset_for(shot_id) if shot_id else 0
            ref_frames = np.asarray(track.frames, dtype=np.int64) - offset
            on_ref.append((shot_id, track, ref_frames))

        union = np.unique(np.concatenate([rf for _, _, rf in on_ref]))
        n = len(union)

        fused_root_t = np.zeros((n, 3))
        fused_root_R = np.tile(np.eye(3), (n, 1, 1))
        fused_thetas = np.zeros((n, 24, 3))
        fused_conf = np.zeros(n)
        view_count = np.zeros(n, dtype=np.int32)

        # Per-shot frame → row index lookup, for O(1) access at each ref frame.
        lookups: list[tuple[str, SmplWorldTrack, dict[int, int]]] = [
            (sid, tr, {int(f): i for i, f in enumerate(rf)})
            for (sid, tr, rf) in on_ref
        ]

        diag_frames: list[FrameDiagnostic] = []

        for i, ref in enumerate(union):
            views: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []
            for sid, tr, idx in lookups:
                local_i = idx.get(int(ref))
                if local_i is None:
                    continue
                conf = float(tr.confidence[local_i])
                if conf <= 0.0:
                    continue
                views.append(
                    (
                        sid,
                        np.asarray(tr.root_t[local_i], dtype=np.float64),
                        np.asarray(tr.root_R[local_i], dtype=np.float64),
                        np.asarray(tr.thetas[local_i], dtype=np.float64),
                        conf,
                    )
                )

            if not views:
                # All zero-confidence at this frame; skip recording it
                # by leaving the slot at zero and view_count=0. Filtered out below.
                continue

            sids = tuple(v[0] for v in views)
            if len(views) == 1:
                _, t, R, thetas, conf = views[0]
                fused_root_t[i] = t
                fused_root_R[i] = R
                fused_thetas[i] = thetas
                fused_conf[i] = conf
                view_count[i] = 1
                diag_frames.append(
                    FrameDiagnostic(
                        frame=int(ref),
                        contributing_shots=sids,
                        dropped_shots=(),
                        pos_disagreement_m=0.0,
                        rot_disagreement_rad=0.0,
                        low_coverage=True,
                        high_disagreement=False,
                    )
                )
                continue

            # Multi-view path. Outlier rejection wired in Task 6 — for now
            # treat all views as kept.
            kept = np.ones(len(views), dtype=bool)
            kept_sids = tuple(v[0] for v in views)
            dropped_sids: tuple[str, ...] = ()

            weights = np.array([v[4] for v in views])
            ts = np.stack([v[1] for v in views])
            Rs = np.stack([v[2] for v in views])
            thetass = np.stack([v[3] for v in views])

            fused_t = (weights[:, None] * ts).sum(axis=0) / weights.sum()
            fused_R = so3_chordal_mean(Rs, weights)

            # Per-joint chordal mean — convert axis-angle to SO(3) via Rodrigues.
            joint_R_per_view = np.stack(
                [_axis_angle_to_so3_batch(thetass[v]) for v in range(len(views))]
            )  # (V, 24, 3, 3)
            fused_joint_R = np.stack(
                [so3_chordal_mean(joint_R_per_view[:, j], weights) for j in range(24)]
            )  # (24, 3, 3)
            fused_theta = np.stack(
                [_so3_to_axis_angle(fused_joint_R[j]) for j in range(24)]
            )

            fused_root_t[i] = fused_t
            fused_root_R[i] = fused_R
            fused_thetas[i] = fused_theta
            fused_conf[i] = float(weights.sum())
            view_count[i] = int(kept.sum())

            # Per-frame disagreement: max pairwise distance among kept views.
            pos_dis = float(
                np.max(np.linalg.norm(ts - fused_t, axis=1)) if len(ts) > 0 else 0.0
            )
            rot_dis = float(
                max(so3_geodesic_distance(Rs[v], fused_R) for v in range(len(views)))
            )
            diag_frames.append(
                FrameDiagnostic(
                    frame=int(ref),
                    contributing_shots=kept_sids,
                    dropped_shots=dropped_sids,
                    pos_disagreement_m=pos_dis,
                    rot_disagreement_rad=rot_dis,
                    low_coverage=False,
                    high_disagreement=(
                        pos_dis > float(cfg.get("high_disagreement_pos_m", 0.5))
                        or rot_dis > float(cfg.get("high_disagreement_rot_rad", 0.5))
                    ),
                )
            )

        # Drop frames where view_count stayed 0 (all views had zero confidence).
        keep = view_count > 0
        union = union[keep]
        fused_root_t = fused_root_t[keep]
        fused_root_R = fused_root_R[keep]
        fused_thetas = fused_thetas[keep]
        fused_conf = fused_conf[keep]
        view_count = view_count[keep]

        # Beta fusion: weighted mean across all contributing tracks.
        beta_stack = np.stack([np.asarray(tr.betas) for _, tr in contribs])
        beta_weights = np.array(
            [float(np.asarray(tr.confidence).mean()) for _, tr in contribs]
        )
        if beta_weights.sum() > 0:
            fused_betas = (
                (beta_weights[:, None] * beta_stack).sum(axis=0) / beta_weights.sum()
            )
        else:
            fused_betas = beta_stack.mean(axis=0)

        # Beta disagreement check (warning-only).
        if len(contribs) > 1:
            max_beta_dist = float(
                np.max(
                    [
                        np.linalg.norm(beta_stack[i] - fused_betas)
                        for i in range(len(contribs))
                    ]
                )
            )
            if max_beta_dist > float(cfg.get("beta_disagreement_warn", 0.3)):
                logger.warning(
                    "[refined_poses] %s beta disagreement %.3f exceeds %.3f",
                    player_id,
                    max_beta_dist,
                    cfg.get("beta_disagreement_warn", 0.3),
                )

        # Smoothing — wired in Task 7.
        contributing_shots = tuple(
            sorted({sid for sid, _ in contribs if sid})
        )

        refined = RefinedPose(
            player_id=player_id,
            frames=union,
            betas=fused_betas,
            thetas=fused_thetas,
            root_R=fused_root_R,
            root_t=fused_root_t,
            confidence=fused_conf,
            view_count=view_count,
            contributing_shots=contributing_shots,
        )
        diag = RefinedPoseDiagnostics(
            player_id=player_id,
            contributing_shots=contributing_shots,
            frames=tuple(diag_frames),
            summary={
                "total_frames": int(len(union)),
                "single_view_frames": int(sum(1 for f in diag_frames if f.low_coverage)),
                "high_disagreement_frames": int(
                    sum(1 for f in diag_frames if f.high_disagreement)
                ),
            },
        )
        return refined, diag
```

Add helpers below the class (still in the same file):

```python
def _axis_angle_to_so3(omega: np.ndarray) -> np.ndarray:
    """Rodrigues. omega is (3,). Returns (3, 3)."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-9:
        return np.eye(3)
    k = omega / theta
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _axis_angle_to_so3_batch(thetas: np.ndarray) -> np.ndarray:
    """thetas: (24, 3) → (24, 3, 3)."""
    return np.stack([_axis_angle_to_so3(thetas[j]) for j in range(thetas.shape[0])])


def _so3_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Inverse of Rodrigues. Returns (3,)."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # Numerically stable axis extraction near pi.
        diag = np.diag(R)
        i = int(np.argmax(diag))
        e = np.zeros(3)
        e[i] = 1.0
        v = (R[:, i] + e) / np.sqrt(max(2.0 * (1.0 + diag[i]), 1e-12))
        return theta * v
    sin_theta = np.sin(theta)
    omega = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    return omega * theta / (2.0 * sin_theta)
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_refined_poses_stage.py -v`
Expected: 3 passed (single-shot tests still green + new two-shot test).

- [ ] **Step 5: Commit**

```bash
git add src/stages/refined_poses.py tests/test_refined_poses_stage.py
git commit -m "feat(refined_poses): fuse multi-shot views on reference timeline"
```

---

## Task 6: Outlier rejection on multi-view frames

**Files:**
- Modify: `src/stages/refined_poses.py`
- Modify: `tests/test_refined_poses_stage.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_refined_poses_stage.py`)

```python
@pytest.mark.integration
def test_refined_poses_outlier_view_dropped(tmp_path: Path) -> None:
    """Three shots overlap on a reference frame; one shot's root_t is 5 m off.

    Expected: outlier shot is dropped from that frame's contribution; fused
    root_t equals the weighted mean of the two agreeing shots; diagnostic
    records the dropped shot.
    """
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="A", offsets={"A": 0, "B": 0, "C": 0})

    n = 3
    frames = np.arange(n, dtype=np.int64)
    base = np.column_stack([frames * 1.0, np.zeros(n), np.zeros(n)])

    a = SmplWorldTrack(
        player_id="P001", frames=frames, betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=base.copy(),
        confidence=np.ones(n), shot_id="A",
    )
    b = SmplWorldTrack(
        player_id="P001", frames=frames, betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=base + np.array([0.05, 0.0, 0.0]),
        confidence=np.ones(n), shot_id="B",
    )
    # Shot C is wildly off at frame 1 only.
    c_root_t = base.copy()
    c_root_t[1] = [50.0, 0.0, 0.0]
    c = SmplWorldTrack(
        player_id="P001", frames=frames, betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=c_root_t,
        confidence=np.ones(n), shot_id="C",
    )
    a.save(output_dir / "hmr_world" / "A__P001_smpl_world.npz")
    b.save(output_dir / "hmr_world" / "B__P001_smpl_world.npz")
    c.save(output_dir / "hmr_world" / "C__P001_smpl_world.npz")

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    stage.run()

    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    diag = RefinedPoseDiagnostics.load(
        output_dir / "refined_poses" / "P001_diagnostics.json"
    )

    # Frame 1: C dropped, fused root_t ~= mean(A, B) at that frame.
    assert refined.view_count[1] == 2
    np.testing.assert_allclose(
        refined.root_t[1], [(1.0 + 1.05) / 2, 0.0, 0.0], atol=1e-9
    )
    fd1 = diag.frames[1]
    assert "C" in fd1.dropped_shots
    assert "A" in fd1.contributing_shots and "B" in fd1.contributing_shots

    # Frames 0 and 2: all three agree → no drops.
    assert refined.view_count[0] == 3
    assert refined.view_count[2] == 3
    assert diag.frames[0].dropped_shots == ()
    assert diag.frames[2].dropped_shots == ()
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_refined_poses_stage.py::test_refined_poses_outlier_view_dropped -v`
Expected: FAIL — view_count[1] is 3 instead of 2 (outlier rejection not yet applied).

- [ ] **Step 3: Wire `robust_translation_fuse` into the multi-view branch**

In `src/stages/refined_poses.py`, add the import:

```python
from src.utils.pose_fusion import (
    robust_translation_fuse,
    so3_chordal_mean,
    so3_geodesic_distance,
)
```

Replace the multi-view branch (the block starting `# Multi-view path.`) with:

```python
            # Multi-view path with MAD-based outlier rejection on positions.
            weights = np.array([v[4] for v in views])
            ts = np.stack([v[1] for v in views])
            Rs = np.stack([v[2] for v in views])
            thetass = np.stack([v[3] for v in views])

            fused_t, kept = robust_translation_fuse(
                ts, weights, k_sigma=float(cfg.get("outlier_k_sigma", 3.0))
            )
            kept_sids = tuple(v[0] for v, k in zip(views, kept) if k)
            dropped_sids = tuple(v[0] for v, k in zip(views, kept) if not k)
            kept_idx = np.where(kept)[0]
            kept_weights = weights[kept_idx]
            kept_Rs = Rs[kept_idx]
            kept_thetas = thetass[kept_idx]
            fused_R = so3_chordal_mean(kept_Rs, kept_weights)

            joint_R_per_view = np.stack(
                [_axis_angle_to_so3_batch(kept_thetas[v]) for v in range(len(kept_idx))]
            )  # (Vk, 24, 3, 3)
            fused_joint_R = np.stack(
                [
                    so3_chordal_mean(joint_R_per_view[:, j], kept_weights)
                    for j in range(24)
                ]
            )
            fused_theta = np.stack(
                [_so3_to_axis_angle(fused_joint_R[j]) for j in range(24)]
            )

            fused_root_t[i] = fused_t
            fused_root_R[i] = fused_R
            fused_thetas[i] = fused_theta
            fused_conf[i] = float(kept_weights.sum())
            view_count[i] = int(len(kept_idx))

            # Disagreement among kept views.
            kept_ts = ts[kept_idx]
            pos_dis = float(
                np.max(np.linalg.norm(kept_ts - fused_t, axis=1))
                if len(kept_ts) > 0
                else 0.0
            )
            rot_dis = float(
                max(so3_geodesic_distance(kept_Rs[v], fused_R) for v in range(len(kept_idx)))
                if len(kept_idx) > 0
                else 0.0
            )
            min_views = int(cfg.get("min_contributing_views", 1))
            diag_frames.append(
                FrameDiagnostic(
                    frame=int(ref),
                    contributing_shots=kept_sids,
                    dropped_shots=dropped_sids,
                    pos_disagreement_m=pos_dis,
                    rot_disagreement_rad=rot_dis,
                    low_coverage=(int(len(kept_idx)) < min_views),
                    high_disagreement=(
                        pos_dis > float(cfg.get("high_disagreement_pos_m", 0.5))
                        or rot_dis > float(cfg.get("high_disagreement_rot_rad", 0.5))
                    ),
                )
            )
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_refined_poses_stage.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stages/refined_poses.py tests/test_refined_poses_stage.py
git commit -m "feat(refined_poses): drop outlier views via robust_translation_fuse"
```

---

## Task 7: Post-fusion smoothing

Apply Savitzky-Golay to fused `root_t`, lie-algebra/SLERP smoothing to fused rotations using existing helpers in `src/utils/temporal_smoothing.py`.

**Files:**
- Modify: `src/stages/refined_poses.py`
- Modify: `tests/test_refined_poses_stage.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_refined_poses_stage.py`)

```python
@pytest.mark.integration
def test_refined_poses_savgol_smooths_root_t(tmp_path: Path) -> None:
    """Single shot, noisy root_t → smoothing reduces RMS error vs ground truth."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="A", offsets={"A": 0})

    n = 30
    frames = np.arange(n, dtype=np.int64)
    truth = np.column_stack([frames * 0.1, np.zeros(n), np.zeros(n)])
    rng = np.random.default_rng(0)
    noisy = truth + rng.normal(scale=0.05, size=truth.shape)
    track = SmplWorldTrack(
        player_id="P001", frames=frames, betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=noisy,
        confidence=np.ones(n), shot_id="A",
    )
    track.save(output_dir / "hmr_world" / "A__P001_smpl_world.npz")

    cfg = _default_config()
    cfg["refined_poses"]["savgol_window"] = 7
    cfg["refined_poses"]["savgol_poly"] = 3
    stage = RefinedPosesStage(config=cfg, output_dir=output_dir)
    stage.run()
    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")

    raw_rms = float(np.sqrt(np.mean((noisy - truth) ** 2)))
    smooth_rms = float(np.sqrt(np.mean((refined.root_t - truth) ** 2)))
    assert smooth_rms < raw_rms
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_refined_poses_stage.py::test_refined_poses_savgol_smooths_root_t -v`
Expected: FAIL — `smooth_rms < raw_rms` does not hold (smoothing is not yet applied).

- [ ] **Step 3: Wire smoothing in `_fuse_player`**

In `src/stages/refined_poses.py`, replace the line `# Smoothing — wired in Task 7.` and the lines that follow it (the block that builds `contributing_shots` and `refined`) with:

```python
        # Smoothing on the fused track.
        sw = int(cfg.get("savgol_window", 7))
        spo = int(cfg.get("savgol_poly", 3))
        if len(union) >= spo + 2 and sw > 1:
            fused_root_t = savgol_axis(fused_root_t, window=sw, order=spo, axis=0)
            fused_thetas = savgol_axis(fused_thetas, window=sw, order=spo, axis=0)

        if bool(cfg.get("smooth_rotations", True)) and len(union) >= 3:
            fused_root_R = slerp_window(fused_root_R, window=sw if sw > 1 else 5)

        contributing_shots = tuple(
            sorted({sid for sid, _ in contribs if sid})
        )
```

(The rest of the method — the `refined = RefinedPose(...)` and `diag = ...` blocks — is unchanged.)

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_refined_poses_stage.py -v`
Expected: 5 passed. The single-shot passthrough test still passes because it uses `savgol_window=1` and `smooth_rotations=False`.

- [ ] **Step 5: Commit**

```bash
git add src/stages/refined_poses.py tests/test_refined_poses_stage.py
git commit -m "feat(refined_poses): apply Savgol + SLERP smoothing post-fusion"
```

---

## Task 8: Register `refined_poses` in the runner

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_runner.py`)

```python
@pytest.mark.unit
def test_resolve_all_includes_refined_poses_between_ball_and_export() -> None:
    stages = resolve_stages("all", None)
    assert "refined_poses" in stages
    assert stages.index("refined_poses") == stages.index("ball") + 1
    assert stages.index("export") == stages.index("refined_poses") + 1


@pytest.mark.unit
def test_resolve_from_refined_poses_includes_export_only() -> None:
    assert resolve_stages("all", "refined_poses") == ["refined_poses", "export"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_runner.py -v -k refined_poses`
Expected: FAIL — `'refined_poses' in stages` is False.

- [ ] **Step 3: Add `refined_poses` to `_STAGE_NAMES` and `_stage_class`**

In `src/pipeline/runner.py`:

```python
_STAGE_NAMES: list[str] = [
    "prepare_shots",
    "tracking",
    "camera",
    "hmr_world",
    "ball",
    "refined_poses",
    "export",
]
```

Add the lazy import branch in `_stage_class()` (just before the `export` branch):

```python
    if name == "refined_poses":
        from src.stages.refined_poses import RefinedPosesStage
        return RefinedPosesStage
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: All previous tests still pass. The pre-existing `test_resolve_all` will FAIL because the expected list no longer matches. Update it:

```python
@pytest.mark.unit
def test_resolve_all() -> None:
    assert resolve_stages("all", None) == [
        "prepare_shots",
        "tracking",
        "camera",
        "hmr_world",
        "ball",
        "refined_poses",
        "export",
    ]
```

Re-run: `pytest tests/test_runner.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/runner.py tests/test_runner.py
git commit -m "feat(refined_poses): register stage between ball and export"
```

---

## Task 9: Export consumes `RefinedPose` (with fallback)

For each shot, `export` now loads `output/refined_poses/*_refined.npz`, slices each refined track to that shot's local timeline using `sync_map` (`f_local = f_ref + offset`), and feeds the resulting `SmplWorldTrack`-shaped per-shot view into the existing GLB builder. When `output/refined_poses/` is empty, fall back to the legacy per-shot `SmplWorldTrack` path.

**Files:**
- Modify: `src/stages/export.py`
- Modify: `tests/test_export_stage.py`

- [ ] **Step 1: Read the export internals to find the load site**

Run: `grep -n "_export_gltf\|hmr_world\|SmplWorldTrack" src/stages/export.py`

The relevant function is the per-shot loader around line 185–200 (`hmr_dir = self.output_dir / "hmr_world"`). The function returns a `list[SmplWorldTrack]` filtered by `shot_id`. We replace this list with one constructed from `RefinedPose` slices.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_export_stage.py`)

```python
import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import Alignment, SyncMap


@pytest.mark.integration
def test_export_consumes_refined_poses_when_present(tmp_path):
    """Export's per-shot player loader pulls from refined_poses/*.npz when that
    directory is non-empty, projecting frames back to the shot's local timeline.
    """
    from src.stages.export import ExportStage

    output_dir = tmp_path
    # Two shots, one player. Shot B has offset=5: ref 0..9 → B local 5..14.
    (output_dir / "shots").mkdir()
    SyncMap(
        reference_shot="A",
        alignments=[
            Alignment(shot_id="A", frame_offset=0),
            Alignment(shot_id="B", frame_offset=5),
        ],
    ).save(output_dir / "shots" / "sync_map.json")

    refined = RefinedPose(
        player_id="P001",
        frames=np.arange(10, dtype=np.int64),  # reference timeline 0..9
        betas=np.zeros(10),
        thetas=np.zeros((10, 24, 3)),
        root_R=np.tile(np.eye(3), (10, 1, 1)),
        root_t=np.column_stack(
            [np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)]
        ),
        confidence=np.ones(10),
        view_count=np.full(10, 2, dtype=np.int32),
        contributing_shots=("A", "B"),
    )
    (output_dir / "refined_poses").mkdir()
    refined.save(output_dir / "refined_poses" / "P001_refined.npz")

    # Build the helper that the new export uses (exposed as a module function).
    from src.stages.export import _per_shot_smpl_tracks

    a_tracks = _per_shot_smpl_tracks(output_dir, shot_id="A")
    assert len(a_tracks) == 1
    assert a_tracks[0].player_id == "P001"
    assert a_tracks[0].shot_id == "A"
    np.testing.assert_array_equal(a_tracks[0].frames, np.arange(10))

    b_tracks = _per_shot_smpl_tracks(output_dir, shot_id="B")
    assert len(b_tracks) == 1
    # Reference frames 0..9 → B-local frames 5..14.
    np.testing.assert_array_equal(b_tracks[0].frames, np.arange(5, 15))
    # Pose values themselves are unchanged — only the frame indices shift.
    np.testing.assert_allclose(
        b_tracks[0].root_t[:, 0], np.arange(10, dtype=np.float64)
    )


@pytest.mark.integration
def test_export_falls_back_to_smpl_world_when_refined_dir_empty(tmp_path):
    """When output/refined_poses/ is empty, export reads SmplWorldTracks
    from output/hmr_world/ as before.
    """
    from src.stages.export import _per_shot_smpl_tracks

    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    legacy = SmplWorldTrack(
        player_id="P001",
        frames=np.arange(5, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((5, 24, 3)),
        root_R=np.tile(np.eye(3), (5, 1, 1)),
        root_t=np.zeros((5, 3)),
        confidence=np.ones(5),
        shot_id="A",
    )
    legacy.save(output_dir / "hmr_world" / "A__P001_smpl_world.npz")

    tracks = _per_shot_smpl_tracks(output_dir, shot_id="A")
    assert len(tracks) == 1
    assert tracks[0].player_id == "P001"
    assert tracks[0].shot_id == "A"
```

- [ ] **Step 3: Run tests and verify they fail**

Run: `pytest tests/test_export_stage.py -v -k 'refined_poses or fallback'`
Expected: FAIL — `_per_shot_smpl_tracks` does not exist yet.

- [ ] **Step 4: Implement `_per_shot_smpl_tracks` in `src/stages/export.py`**

Add at module top:

```python
from src.schemas.refined_pose import RefinedPose
from src.schemas.sync_map import SyncMap
```

Add the helper function right above `class ExportStage:`:

```python
def _per_shot_smpl_tracks(
    output_dir: Path, *, shot_id: str | None
) -> list[SmplWorldTrack]:
    """Return per-shot SmplWorldTracks consumed by the export builder.

    Prefers ``output/refined_poses/*_refined.npz`` (one fused track per
    player on the reference timeline). For each refined track we slice
    the frames inside the shot's local window after applying the
    reverse sync offset (``f_local = f_ref + offset``) and emit a
    SmplWorldTrack-shaped view that the existing builder already knows
    how to consume.

    Falls back to ``output/hmr_world/*_smpl_world.npz`` when the
    refined directory is empty (e.g. user re-runs ``--stages export``
    without first running refined_poses).
    """
    refined_dir = output_dir / "refined_poses"
    refined_files = (
        sorted(refined_dir.glob("*_refined.npz")) if refined_dir.exists() else []
    )
    if refined_files:
        sync_path = output_dir / "shots" / "sync_map.json"
        sync_map = (
            SyncMap.load(sync_path)
            if sync_path.exists()
            else SyncMap(reference_shot="", alignments=[])
        )
        offset = sync_map.offset_for(shot_id) if shot_id else 0
        out: list[SmplWorldTrack] = []
        for p in refined_files:
            r = RefinedPose.load(p)
            ref_frames = np.asarray(r.frames, dtype=np.int64)
            local = ref_frames + offset
            # No clipping: even with a single shot, every refined frame
            # round-trips back to a defined local frame. If the export
            # caller wants a strict shot-window clip, that's a downstream
            # filter — not this loader's job.
            out.append(
                SmplWorldTrack(
                    player_id=r.player_id,
                    frames=local,
                    betas=np.asarray(r.betas),
                    thetas=np.asarray(r.thetas),
                    root_R=np.asarray(r.root_R),
                    root_t=np.asarray(r.root_t),
                    confidence=np.asarray(r.confidence),
                    shot_id=shot_id or "",
                )
            )
        return out

    # Legacy path.
    hmr_dir = output_dir / "hmr_world"
    if not hmr_dir.exists():
        return []
    tracks = [SmplWorldTrack.load(p) for p in sorted(hmr_dir.glob("*_smpl_world.npz"))]
    if shot_id is None:
        return tracks
    return [t for t in tracks if getattr(t, "shot_id", "") == shot_id]
```

Now **replace the inline player-loading block** inside `ExportStage._export_gltf` (the block from `hmr_dir = self.output_dir / "hmr_world"` through the construction of `players: list[SmplWorldTrack]`) with a single call:

```python
        players = _per_shot_smpl_tracks(self.output_dir, shot_id=shot_id)
```

(Read lines 184–202 of `src/stages/export.py` first to see exactly what they delete; the goal is to replace the whole `npz_files = sorted(...) → players = [...]` block.)

Do the equivalent replacement in the FBX manifest path (`_export_fbx`-side code around lines 295–305 of `src/stages/export.py` that currently iterates `npz_files`). Use the same helper:

```python
        all_players = _per_shot_smpl_tracks(self.output_dir, shot_id=None)
        # Restore the existing per-track loop — name_mapping etc — but
        # iterate `all_players` instead of `[SmplWorldTrack.load(p) for p in npz_files]`.
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `pytest tests/test_export_stage.py tests/test_export_stage_manifest.py -v`
Expected: all previously-green tests still pass + the two new ones pass.

- [ ] **Step 6: Commit**

```bash
git add src/stages/export.py tests/test_export_stage.py
git commit -m "feat(refined_poses): export consumes refined tracks with hmr_world fallback"
```

---

## Task 10: Quality report integration

**Files:**
- Modify: `src/pipeline/quality_report.py`
- Modify: `tests/test_quality_report.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_quality_report.py`)

```python
@pytest.mark.unit
def test_quality_report_includes_refined_poses_section(tmp_path) -> None:
    import json
    from src.pipeline.quality_report import write_quality_report

    refined_dir = tmp_path / "refined_poses"
    refined_dir.mkdir()
    summary = {
        "players_refined": 3,
        "single_shot_players": 1,
        "multi_shot_players": 2,
        "total_fused_frames": 100,
        "single_view_frames": 20,
        "high_disagreement_frames": 4,
        "shots_missing_sync": [],
        "beta_disagreement_warnings": [],
    }
    (refined_dir / "refined_poses_summary.json").write_text(json.dumps(summary))
    write_quality_report(tmp_path)
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report["refined_poses"]["players_refined"] == 3
    assert report["refined_poses"]["high_disagreement_frames"] == 4
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_quality_report.py -v -k refined_poses`
Expected: FAIL — `KeyError: 'refined_poses'`.

- [ ] **Step 3: Add the section in `src/pipeline/quality_report.py`**

Append immediately before `out = output_dir / "quality_report.json"` at the bottom of `write_quality_report`:

```python
    refined_summary_path = output_dir / "refined_poses" / "refined_poses_summary.json"
    if refined_summary_path.exists():
        report["refined_poses"] = json.loads(refined_summary_path.read_text())
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_quality_report.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/quality_report.py tests/test_quality_report.py
git commit -m "feat(refined_poses): surface refined_poses summary in quality_report"
```

---

## Task 11: Config defaults + dashboard panel addendum

**Files:**
- Modify: `config/default.yaml`
- Modify: `src/web/static/index.html`

- [ ] **Step 1: Add the config block**

Append to `config/default.yaml` (after the `ball:` block, before `export:`):

```yaml
refined_poses:
  # Outlier rejection (per frame, per view)
  outlier_k_sigma: 3.0
  min_contributing_views: 1
  high_disagreement_pos_m: 0.5
  high_disagreement_rot_rad: 0.5
  # Smoothing (reuses src/utils/temporal_smoothing.py)
  savgol_window: 7
  savgol_poly: 3
  smooth_rotations: true
  # Beta fusion
  beta_aggregation: weighted_mean
  beta_disagreement_warn: 0.3
```

- [ ] **Step 2: Add a Refined-Poses status row to the Multi-shot status panel**

In `src/web/static/index.html`, find the Multi-shot status panel (search for "Multi-shot status" or "shot picker"). Add one new row to its table that reads from `quality_report.refined_poses`:

```html
<!-- Inside the Multi-shot status table body -->
<tr>
  <td>Refined poses</td>
  <td colspan="5">
    <span id="refined-poses-summary">—</span>
  </td>
</tr>
```

In the JavaScript that renders the panel (search for the function that already updates per-shot rows), add a sibling read:

```js
fetch('/api/quality_report')
  .then(r => r.json())
  .then(report => {
    const rp = report.refined_poses;
    const span = document.getElementById('refined-poses-summary');
    if (!rp) {
      span.textContent = 'not run';
      return;
    }
    span.textContent =
      `${rp.players_refined} players (${rp.multi_shot_players} multi-shot, ` +
      `${rp.single_shot_players} single-shot), ` +
      `${rp.total_fused_frames} frames, ` +
      `${rp.high_disagreement_frames} flagged`;
  });
```

If the `/api/quality_report` endpoint does not yet exist, instead read directly from a static path the dashboard already exposes — `grep -n "quality_report" src/web/server.py` to confirm.

- [ ] **Step 3: Smoke test the page renders**

Run: `python -c "from pathlib import Path; from src.web.static import *"` (sanity import check) and then start the server with the existing CLI:

```bash
python recon.py serve --output ./output/ &
SERVER_PID=$!
sleep 1
curl -s http://127.0.0.1:8000/ > /dev/null && echo "OK"
kill $SERVER_PID
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add config/default.yaml src/web/static/index.html
git commit -m "feat(refined_poses): add config defaults and dashboard summary row"
```

---

## Task 12: End-to-end integration test

**Files:**
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_runner.py`)

```python
@pytest.mark.integration
def test_pipeline_refined_poses_end_to_end(tmp_path: Path) -> None:
    """Two shots, one shared player. Stage HMR outputs directly, then run
    refined_poses + export through run_pipeline. Assert one fused NPZ per
    player and one GLB per shot.
    """
    import json
    import numpy as np
    from src.pipeline.runner import run_pipeline
    from src.schemas.refined_pose import RefinedPose
    from src.schemas.shots import Shot, ShotsManifest
    from src.schemas.smpl_world import SmplWorldTrack
    from src.schemas.sync_map import Alignment, SyncMap

    out = tmp_path
    (out / "shots").mkdir()
    (out / "hmr_world").mkdir()
    (out / "camera").mkdir()

    ShotsManifest(
        source_file="",
        fps=30.0,
        total_frames=20,
        shots=[
            Shot(id="A", start_frame=0, end_frame=9, start_time=0.0,
                 end_time=0.3, clip_file="shots/A.mp4", speed_factor=1.0),
            Shot(id="B", start_frame=0, end_frame=9, start_time=0.0,
                 end_time=0.3, clip_file="shots/B.mp4", speed_factor=1.0),
        ],
    ).save(out / "shots" / "shots_manifest.json")
    SyncMap(
        reference_shot="A",
        alignments=[
            Alignment(shot_id="A", frame_offset=0),
            Alignment(shot_id="B", frame_offset=0),
        ],
    ).save(out / "shots" / "sync_map.json")

    # Identical poses in both shots → fused track equals either input.
    n = 10
    frames = np.arange(n, dtype=np.int64)
    base = np.column_stack([frames * 1.0, np.zeros(n), np.zeros(n)])
    for sid in ("A", "B"):
        SmplWorldTrack(
            player_id="P001", frames=frames, betas=np.zeros(10),
            thetas=np.zeros((n, 24, 3)),
            root_R=np.tile(np.eye(3), (n, 1, 1)),
            root_t=base.copy(),
            confidence=np.ones(n), shot_id=sid,
        ).save(out / "hmr_world" / f"{sid}__P001_smpl_world.npz")

    # Camera tracks are needed by export — write minimal stubs only if the
    # export gltf path requires them. (If the gltf builder errors here,
    # disable gltf in config and assert refined_poses outputs only.)
    config = {
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "ball": {},
        "export": {"gltf_enabled": False, "fbx_enabled": False},
        "refined_poses": {
            "outlier_k_sigma": 3.0,
            "min_contributing_views": 1,
            "high_disagreement_pos_m": 0.5,
            "high_disagreement_rot_rad": 0.5,
            "savgol_window": 1,
            "savgol_poly": 2,
            "smooth_rotations": False,
            "beta_aggregation": "weighted_mean",
            "beta_disagreement_warn": 0.3,
        },
    }
    run_pipeline(
        output_dir=out,
        stages="refined_poses,export",
        from_stage=None,
        config=config,
    )
    assert (out / "refined_poses" / "P001_refined.npz").exists()
    refined = RefinedPose.load(out / "refined_poses" / "P001_refined.npz")
    assert refined.contributing_shots == ("A", "B")
    np.testing.assert_array_equal(refined.frames, frames)
    np.testing.assert_allclose(refined.root_t, base)
    summary = json.loads(
        (out / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert summary["players_refined"] == 1
    assert summary["multi_shot_players"] == 1

    # quality_report.json contains the refined_poses section.
    report = json.loads((out / "quality_report.json").read_text())
    assert report["refined_poses"]["players_refined"] == 1
```

- [ ] **Step 2: Run the test and verify it passes**

Run: `pytest tests/test_runner.py::test_pipeline_refined_poses_end_to_end -v`
Expected: PASS.

- [ ] **Step 3: Run the full test suite once to confirm no regressions**

Run: `pytest -x -q`
Expected: All previously-green tests stay green; the 5+ new tests added across this plan pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_runner.py
git commit -m "test(refined_poses): end-to-end integration through run_pipeline"
```

---

## Self-review checklist

Run through these after the last commit:

- [ ] **Spec coverage** — every section of `docs/superpowers/specs/2026-05-09-refined-poses-design.md` (§§3.1–3.6, §4.1–4.4, §5.1–5.3, §6 edge cases, §7 unit/stage/integration tests, §8 quality-report keys, §9 config keys) is implemented and tested.
- [ ] **Outlier behaviour** — `tests/test_refined_poses_stage.py::test_refined_poses_outlier_view_dropped` exercises the V≥3 path. The V=2 passthrough is exercised in `test_refined_poses_two_shots_fuses_root_t`.
- [ ] **Sync edge cases** — exercise "sync_map missing entirely" and "shot missing from sync_map" via two added micro-tests if not already covered. Add them to Task 4 or Task 5 if you find a gap.
- [ ] **Type consistency** — `RefinedPose.contributing_shots` is `tuple[str, ...]` in every reference; `view_count` dtype is `int32` in every construction site.
- [ ] **No placeholders** — every step's code block compiles standalone; no "TODO", "TBD", or "implement later" remains.

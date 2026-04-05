# Sync Two-Pass Graph Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Stage 3 (Temporal Sync) so that all non-reference clip pairs are compared against each other (pass 2), then a weighted least-squares solve over the full pairwise graph produces globally consistent offsets.

**Architecture:** Pass 1 runs the existing star-topology signal stack (each clip vs reference). Pass 2 runs the same stack for all non-reference pairs with a narrowed search window derived from pass-1 offsets. A `_solve_offset_graph()` function solves the weighted least-squares system `o_j − o_i = e_ij` over all edges and returns globally consistent offsets.

**Tech Stack:** Python, NumPy (`numpy.linalg.lstsq`), existing signal functions in `src/stages/sync.py`, pytest.

---

## File Map

| File | Change |
|------|--------|
| `src/schemas/sync_map.py` | Add `graph_residual_frames: float \| None = None` to `Alignment` |
| `config/default.yaml` | Add three keys under `sync:` |
| `src/stages/sync.py` | Add `max_lag_frames` to `_align_audio`; add `max_lag` to `_visual_similarity_profile`, `_align_visual`, `_align_formation_spatial`; add `_solve_offset_graph`; add `_collect_pairwise_estimates`; refactor `run()` |
| `tests/test_sync.py` | New tests for `_solve_offset_graph` and `_collect_pairwise_estimates` |

---

## Task 1: Add `graph_residual_frames` to `Alignment` schema

**Files:**
- Modify: `src/schemas/sync_map.py`
- Test: `tests/test_schemas.py` (or inline in `tests/test_sync.py`)

- [ ] **Step 1: Write failing test**

Add to `tests/test_sync.py`:

```python
def test_alignment_loads_without_graph_residual():
    """Old sync_map.json files without graph_residual_frames must still load."""
    import json, tempfile
    from src.schemas.sync_map import Alignment, SyncMap

    data = {
        "reference_shot": "shot_001",
        "alignments": [
            {
                "shot_id": "shot_002",
                "frame_offset": 141,
                "confidence": 0.8,
                "method": "audio",
                "overlap_frames": [0, 365],
            }
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        p = Path(f.name)

    sm = SyncMap.load(p)
    assert sm.alignments[0].graph_residual_frames is None
    p.unlink()


def test_alignment_saves_and_loads_graph_residual():
    import json, tempfile
    from src.schemas.sync_map import Alignment, SyncMap

    sm = SyncMap(
        reference_shot="shot_001",
        alignments=[
            Alignment(
                shot_id="shot_002",
                frame_offset=141,
                confidence=0.8,
                method="audio",
                overlap_frames=[0, 365],
                graph_residual_frames=7.5,
            )
        ],
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = Path(f.name)
    sm.save(p)
    loaded = SyncMap.load(p)
    assert loaded.alignments[0].graph_residual_frames == pytest.approx(7.5)
    p.unlink()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/joebower/workplace/football-perspectives
pytest tests/test_sync.py::test_alignment_loads_without_graph_residual tests/test_sync.py::test_alignment_saves_and_loads_graph_residual -v
```

Expected: both FAIL (AttributeError or similar — `graph_residual_frames` not defined)

- [ ] **Step 3: Add `graph_residual_frames` field to `Alignment`**

In `src/schemas/sync_map.py`, change:

```python
@dataclass
class Alignment:
    shot_id: str
    frame_offset: int
    confidence: float
    method: str
    overlap_frames: list[int]
    graph_residual_frames: float | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sync.py::test_alignment_loads_without_graph_residual tests/test_sync.py::test_alignment_saves_and_loads_graph_residual -v
```

Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add src/schemas/sync_map.py tests/test_sync.py
git commit -m "feat: add graph_residual_frames field to Alignment schema"
```

---

## Task 2: Add config keys

**Files:**
- Modify: `config/default.yaml`

No new tests needed — config is read by `TemporalSyncStage.run()` which is tested end-to-end later. This is a prerequisite for Tasks 6 and 7.

- [ ] **Step 1: Add three keys under `sync:` in `config/default.yaml`**

Add after `trajectory_refinement_coarse_match_distance_m: 20.0`:

```yaml
  pass2_search_margin_frames: 30
  graph_confidence_alpha: 0.4
  graph_residual_flag_threshold_frames: 15
```

- [ ] **Step 2: Verify config loads cleanly**

```bash
python -c "import yaml; cfg = yaml.safe_load(open('config/default.yaml')); print(cfg['sync']['pass2_search_margin_frames'])"
```

Expected: `30`

- [ ] **Step 3: Commit**

```bash
git add config/default.yaml
git commit -m "chore: add pass2 and graph solver config keys to default.yaml"
```

---

## Task 3: Add `max_lag_frames` to `_align_audio()`

**Files:**
- Modify: `src/stages/sync.py` (function `_align_audio`, lines ~582–652)
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_sync.py`:

```python
def test_align_audio_max_lag_excludes_true_peak(tmp_path):
    """When the true lag exceeds max_lag_frames the result should not find it."""
    from src.stages.sync import _align_audio
    import wave, struct

    sample_rate = 16000
    fps = 25.0
    duration_s = 4.0
    n_samples = int(sample_rate * duration_s)

    # Burst signal at t=3s in ref, t=0.5s in shot → true lag = 2.5s = 62 frames
    ref_audio = np.zeros(n_samples, dtype=np.float32)
    ref_audio[int(sample_rate * 3.0) : int(sample_rate * 3.0) + 400] = 1.0
    shot_audio = np.zeros(n_samples, dtype=np.float32)
    shot_audio[int(sample_rate * 0.5) : int(sample_rate * 0.5) + 400] = 1.0

    def _write_wav(path: Path, data: np.ndarray) -> None:
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((data * 32767).astype(np.int16).tobytes())

    ref_path = tmp_path / "ref.wav"
    shot_path = tmp_path / "shot.wav"
    _write_wav(ref_path, ref_audio)
    _write_wav(shot_path, shot_audio)

    # Without window: should find peak near lag=62 frames
    est_no_window = _align_audio(ref_path, shot_path, fps)

    # With tight window (±5 frames around 0): peak at 62 is outside → different result
    est_windowed = _align_audio(ref_path, shot_path, fps, max_lag_frames=5)
    assert abs(est_windowed.offset) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sync.py::test_align_audio_max_lag_excludes_true_peak -v
```

Expected: FAIL (`TypeError` — unexpected keyword `max_lag_frames`)

- [ ] **Step 3: Add `max_lag_frames` parameter to `_align_audio()`**

In `src/stages/sync.py`, change the signature:

```python
def _align_audio(
    ref_clip: Path,
    shot_clip: Path,
    fps: float,
    sample_rate: int = 16000,
    min_zscore: float = 4.0,
    min_pearson_r: float = 0.4,
    max_lag_frames: int | None = None,
) -> AlignmentEstimate:
```

After the line `lags = np.arange(-(len(shot_audio) - 1), len(ref_audio))`, add:

```python
    if max_lag_frames is not None:
        max_lag_samples = int(round(max_lag_frames / fps * sample_rate))
        mask = np.abs(lags) <= max_lag_samples
        if np.any(mask):
            corr = corr[mask]
            lags = lags[mask]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_sync.py::test_align_audio_max_lag_excludes_true_peak -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
pytest tests/test_sync.py -v
```

Expected: all existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/stages/sync.py tests/test_sync.py
git commit -m "feat: add max_lag_frames param to _align_audio"
```

---

## Task 4: Add `max_lag` to `_visual_similarity_profile`, `_align_visual`, `_align_formation_spatial`

**Files:**
- Modify: `src/stages/sync.py` (functions `_visual_similarity_profile` ~294, `_align_visual` ~330, `_align_formation_spatial` ~505)
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_sync.py`:

```python
def test_visual_similarity_profile_max_lag_limits_offsets():
    from src.stages.sync import _visual_similarity_profile

    rng = np.random.default_rng(0)
    ref_desc = rng.random((20, 16)).astype(np.float32)
    shot_desc = rng.random((20, 16)).astype(np.float32)

    offsets_full, _ = _visual_similarity_profile(ref_desc, shot_desc, min_overlap=2)
    offsets_windowed, _ = _visual_similarity_profile(
        ref_desc, shot_desc, min_overlap=2, max_lag=5
    )

    assert len(offsets_windowed) < len(offsets_full)
    assert int(np.abs(offsets_windowed).max()) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sync.py::test_visual_similarity_profile_max_lag_limits_offsets -v
```

Expected: FAIL (`TypeError` — unexpected keyword `max_lag`)

- [ ] **Step 3: Add `max_lag` to `_visual_similarity_profile()`**

Change signature:

```python
def _visual_similarity_profile(
    ref_desc: np.ndarray,
    shot_desc: np.ndarray,
    min_overlap: int,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
```

After `offsets = np.arange(-(N_shot - 1), N_ref, dtype=np.int64)`, add:

```python
    if max_lag is not None:
        offsets = offsets[(offsets >= -max_lag) & (offsets <= max_lag)]
    scores = np.zeros(len(offsets), dtype=np.float64)
```

Remove the existing `scores = np.zeros(len(offsets), dtype=np.float64)` line that follows (it's being replaced above).

- [ ] **Step 4: Thread `max_lag_frames` through `_align_visual()`**

Change signature:

```python
def _align_visual(
    ref_clip: Path,
    shot_clip: Path,
    fps: float,
    min_overlap_frames: int,
    sample_fps: float,
    max_lag_frames: int | None = None,
) -> AlignmentEstimate:
```

After `frame_step = max(1, round(fps / sample_fps))` (inside the function, after `_extract_frame_descriptors`), add before the `_visual_similarity_profile` call:

```python
    max_lag_samples = None if max_lag_frames is None else max(1, max_lag_frames // frame_step)
```

Pass `max_lag=max_lag_samples` to `_visual_similarity_profile`.

- [ ] **Step 5: Thread `max_lag_frames` through `_align_formation_spatial()`**

Change signature:

```python
def _align_formation_spatial(
    ref_desc: np.ndarray,
    shot_desc: np.ndarray,
    frame_step: int,
    min_overlap_frames: int,
    max_lag_frames: int | None = None,
) -> AlignmentEstimate:
```

Add before `_visual_similarity_profile` call:

```python
    max_lag_samples = None if max_lag_frames is None else max(1, max_lag_frames // frame_step)
```

Pass `max_lag=max_lag_samples` to `_visual_similarity_profile`.

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_sync.py -v
```

Expected: new test PASS, all existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add src/stages/sync.py tests/test_sync.py
git commit -m "feat: add max_lag windowing to visual similarity and formation alignment"
```

---

## Task 5: Implement `_solve_offset_graph()`

**Files:**
- Modify: `src/stages/sync.py` (add new function after `_align_by_matched_pitch_trajectories`)
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sync.py`:

```python
from src.stages.sync import _solve_offset_graph


def test_solve_offset_graph_single_clip():
    """One clip with one pass-1 edge — solved offset equals the estimate."""
    solved = _solve_offset_graph(n_clips=2, edges=[(0, 1, 141.0, 0.8)])
    assert solved[0] == pytest.approx(0.0)
    assert solved[1] == pytest.approx(141.0)


def test_solve_offset_graph_consistent_triangle():
    """Three clips with consistent pass-1 + pass-2 edges — solution matches."""
    # o1 = 141, o2 = 393 → o2 - o1 = 252
    edges = [
        (0, 1, 141.0, 0.8),   # ref → clip1
        (0, 2, 393.0, 0.7),   # ref → clip2
        (1, 2, 252.0, 0.9),   # clip1 → clip2 (consistent)
    ]
    solved = _solve_offset_graph(n_clips=3, edges=edges)
    assert solved[0] == pytest.approx(0.0)
    assert solved[1] == pytest.approx(141.0, abs=1.0)
    assert solved[2] == pytest.approx(393.0, abs=1.0)


def test_solve_offset_graph_conflicting_edges_compromise():
    """Pass-2 edge contradicts pass-1 — solution is a weighted compromise."""
    # Pass 1 says clip1=100, clip2=200 → expected pair offset 100
    # Pass 2 (high confidence) says pair offset is 80 → pulls clip2 toward 180
    edges = [
        (0, 1, 100.0, 0.5),
        (0, 2, 200.0, 0.5),
        (1, 2, 80.0, 2.0),   # high-weight contradicting edge
    ]
    solved = _solve_offset_graph(n_clips=3, edges=edges)
    # clip2 should be pulled below 200 due to the contradicting high-weight edge
    assert solved[2] < 200.0


def test_solve_offset_graph_zero_weight_edges_ignored():
    """Zero-weight edges have no effect on the solution."""
    edges_with = [
        (0, 1, 141.0, 0.8),
        (0, 2, 393.0, 0.7),
        (1, 2, 999.0, 0.0),   # zero weight — should be ignored
    ]
    edges_without = [
        (0, 1, 141.0, 0.8),
        (0, 2, 393.0, 0.7),
    ]
    solved_with = _solve_offset_graph(n_clips=3, edges=edges_with)
    solved_without = _solve_offset_graph(n_clips=3, edges=edges_without)
    assert solved_with[1] == pytest.approx(solved_without[1], abs=0.1)
    assert solved_with[2] == pytest.approx(solved_without[2], abs=0.1)


def test_solve_offset_graph_reference_always_zero():
    """Reference clip (index 0) is always 0.0 regardless of edges."""
    solved = _solve_offset_graph(n_clips=3, edges=[(0, 1, 50.0, 1.0), (0, 2, 100.0, 1.0)])
    assert solved[0] == pytest.approx(0.0)


def test_solve_offset_graph_single_clip_returns_correct_shape():
    """n_clips=1 (only reference) — returns array of length 1."""
    solved = _solve_offset_graph(n_clips=1, edges=[])
    assert len(solved) == 1
    assert solved[0] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sync.py::test_solve_offset_graph_single_clip tests/test_sync.py::test_solve_offset_graph_consistent_triangle tests/test_sync.py::test_solve_offset_graph_conflicting_edges_compromise tests/test_sync.py::test_solve_offset_graph_zero_weight_edges_ignored tests/test_sync.py::test_solve_offset_graph_reference_always_zero tests/test_sync.py::test_solve_offset_graph_single_clip_returns_correct_shape -v
```

Expected: all FAIL (`ImportError` — `_solve_offset_graph` not defined)

- [ ] **Step 3: Implement `_solve_offset_graph()`**

Add after the `_align_celebration_signal` function (around line 1298) in `src/stages/sync.py`:

```python
def _solve_offset_graph(
    n_clips: int,
    edges: list[tuple[int, int, float, float]],
) -> np.ndarray:
    """
    Weighted least-squares solve for clip offsets.

    Clips are indexed 0..n_clips-1. Clip 0 is the reference (offset = 0, pinned).
    Non-reference clips are indexed 1..n_clips-1.

    Edge convention: o_j - o_i = offset_estimate.
    Each edge is (i, j, offset_estimate, weight).

    Returns float array of length n_clips (index 0 = 0.0).
    """
    n_vars = n_clips - 1  # reference is pinned at 0
    result = np.zeros(n_clips, dtype=np.float64)
    if n_vars <= 0:
        return result

    valid_edges = [(i, j, e, w) for i, j, e, w in edges if w > 0]
    if not valid_edges:
        return result

    n_edges = len(valid_edges)
    A = np.zeros((n_edges, n_vars), dtype=np.float64)
    b = np.zeros(n_edges, dtype=np.float64)
    W = np.zeros(n_edges, dtype=np.float64)

    for k, (i, j, e, w) in enumerate(valid_edges):
        # Constraint: o_j - o_i = e  (reference has o_0 = 0, not a variable)
        if i > 0:
            A[k, i - 1] = -1.0
        if j > 0:
            A[k, j - 1] = +1.0
        b[k] = float(e)
        W[k] = float(w)

    sqrt_W = np.sqrt(W)
    x, _, _, _ = np.linalg.lstsq(
        sqrt_W[:, None] * A,
        sqrt_W * b,
        rcond=None,
    )
    result[1:] = x
    return result
```

Also add `_solve_offset_graph` to the imports in `tests/test_sync.py`:

```python
from src.stages.sync import (
    ...existing imports...,
    _solve_offset_graph,
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sync.py::test_solve_offset_graph_single_clip tests/test_sync.py::test_solve_offset_graph_consistent_triangle tests/test_sync.py::test_solve_offset_graph_conflicting_edges_compromise tests/test_sync.py::test_solve_offset_graph_zero_weight_edges_ignored tests/test_sync.py::test_solve_offset_graph_reference_always_zero tests/test_sync.py::test_solve_offset_graph_single_clip_returns_correct_shape -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/stages/sync.py tests/test_sync.py
git commit -m "feat: implement _solve_offset_graph weighted least-squares solver"
```

---

## Task 6: Implement `_collect_pairwise_estimates()`

**Files:**
- Modify: `src/stages/sync.py` (add new function after `_solve_offset_graph`)
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_sync.py`:

```python
from src.stages.sync import _collect_pairwise_estimates
from src.schemas.shots import Shot, ShotsManifest


def test_collect_pairwise_estimates_returns_correct_pairs(tmp_path):
    """With 3 shots (1 ref + 2 non-ref), exactly 1 pair is returned."""
    from src.schemas.tracks import TracksResult

    # Create minimal dummy video files (1-frame black)
    for name in ("shot_001.mp4", "shot_002.mp4", "shot_003.mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(tmp_path / name), fourcc, 25.0, (64, 36))
        out.write(np.zeros((36, 64, 3), dtype=np.uint8))
        out.release()

    shots = [
        Shot(id="shot_001", clip_file="shot_001.mp4", start_frame=0, end_frame=25, start_time=0.0, end_time=1.0),
        Shot(id="shot_002", clip_file="shot_002.mp4", start_frame=0, end_frame=25, start_time=0.0, end_time=1.0),
        Shot(id="shot_003", clip_file="shot_003.mp4", start_frame=0, end_frame=25, start_time=0.0, end_time=1.0),
    ]
    clips = {s.id: tmp_path / s.clip_file for s in shots}
    tracks_by_shot: dict[str, TracksResult] = {}
    n_frames = {"shot_001": 25, "shot_002": 25, "shot_003": 25}
    pass1 = {
        "shot_002": AlignmentEstimate(offset=141, confidence=0.8, method="audio", valid=True),
        "shot_003": AlignmentEstimate(offset=393, confidence=0.7, method="audio", valid=True),
    }
    cfg = {"min_confidence": 0.3, "pass2_search_margin_frames": 30}

    results = _collect_pairwise_estimates(
        shots=shots,
        clips=clips,
        tracks_by_shot=tracks_by_shot,
        n_frames_by_shot=n_frames,
        pass1_estimates=pass1,
        poses_dir=tmp_path / "poses",
        fps=25.0,
        cfg=cfg,
    )

    # 2 non-reference shots → 1 pair
    assert len(results) == 1
    i, j, est = results[0]
    # i and j are indices in shots list: shot_002 = index 1, shot_003 = index 2
    assert i == 1
    assert j == 2
    assert isinstance(est, AlignmentEstimate)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sync.py::test_collect_pairwise_estimates_returns_correct_pairs -v
```

Expected: FAIL (`ImportError` — `_collect_pairwise_estimates` not defined)

- [ ] **Step 3: Implement `_collect_pairwise_estimates()`**

Add after `_solve_offset_graph` in `src/stages/sync.py`:

```python
def _collect_pairwise_estimates(
    shots: list,
    clips: dict[str, Path],
    tracks_by_shot: dict[str, "TracksResult"],
    n_frames_by_shot: dict[str, int],
    pass1_estimates: dict[str, AlignmentEstimate],
    poses_dir: Path,
    fps: float,
    cfg: dict,
) -> list[tuple[int, int, AlignmentEstimate]]:
    """
    Run the signal stack for all non-reference pairs.

    shots[0] is the reference. All pairs (i, j) with 1 <= i < j < len(shots)
    are evaluated. Returns list of (i, j, AlignmentEstimate).

    For pairs where both pass-1 confidences meet min_confidence, the search
    window is constrained to ±pass2_search_margin_frames around the expected
    relative offset. Otherwise, full search is used.
    """
    min_conf = float(cfg.get("min_confidence", 0.3))
    search_margin = int(cfg.get("pass2_search_margin_frames", 30))
    sample_fps = float(cfg.get("sample_fps", 5.0))
    min_overlap_frames = int(cfg.get("min_overlap_frames", 25))
    reid_min_track_frames = int(cfg.get("reid_min_track_frames", 20))
    reid_min_similarity = float(cfg.get("reid_min_similarity", 0.6))
    speed_factors = [float(s) for s in cfg.get("speed_factors", [1.0])]
    agreement_tolerance = int(cfg.get("agreement_tolerance_frames", 8))
    audio_min_zscore = float(cfg.get("audio_min_zscore", 4.0))

    non_ref = shots[1:]
    results: list[tuple[int, int, AlignmentEstimate]] = []

    for a_pos, shot_a in enumerate(non_ref):
        for b_pos in range(a_pos + 1, len(non_ref)):
            shot_b = non_ref[b_pos]
            i = a_pos + 1   # index in shots list
            j = b_pos + 1   # index in shots list

            est_a = pass1_estimates.get(shot_a.id)
            est_b = pass1_estimates.get(shot_b.id)

            # Compute search window
            if (
                est_a is not None and est_b is not None
                and est_a.confidence >= min_conf and est_b.confidence >= min_conf
            ):
                max_lag: int | None = search_margin
            else:
                max_lag = None

            clip_a = clips[shot_a.id]
            clip_b = clips[shot_b.id]
            n_a = n_frames_by_shot.get(shot_a.id, 1)
            n_b = n_frames_by_shot.get(shot_b.id, 1)
            tracks_a = tracks_by_shot.get(shot_a.id)
            tracks_b = tracks_by_shot.get(shot_b.id)

            logging.info(
                "  [sync/pass2] aligning %s ↔ %s (max_lag=%s)",
                shot_a.id, shot_b.id, max_lag,
            )

            # --- Audio ---
            audio_est = _align_audio(
                ref_clip=clip_a,
                shot_clip=clip_b,
                fps=fps,
                min_zscore=audio_min_zscore,
                max_lag_frames=max_lag,
            )

            # --- Celebration cross-correlation ---
            ref_celeb = _compute_celebration_signal(poses_dir, shot_a.id, n_a)
            shot_celeb = _compute_celebration_signal(poses_dir, shot_b.id, n_b)
            celeb_est = _align_celebration_signal(
                ref_celeb, shot_celeb, speed_factors=speed_factors,
            )

            # --- Player re-ID ---
            reid_est = AlignmentEstimate(
                offset=0, confidence=0.0, method="player_reid", valid=False,
            )
            if tracks_a is not None and tracks_b is not None:
                reid_est = _align_player_reid(
                    ref_clip=clip_a,
                    shot_clip=clip_b,
                    ref_tracks=tracks_a,
                    shot_tracks=tracks_b,
                    ref_n_frames=n_a,
                    shot_n_frames=n_b,
                    fps=fps,
                    min_track_frames=reid_min_track_frames,
                    min_similarity=reid_min_similarity,
                    speed_factors=speed_factors,
                    agreement_tolerance=agreement_tolerance,
                )

            # --- Fusion: audio first, then celebration, then reid, else invalid ---
            if audio_est.valid and audio_est.confidence >= 0.5:
                best = audio_est
            elif celeb_est.valid and celeb_est.confidence > 0.3:
                best = celeb_est
            elif reid_est.valid:
                best = reid_est
            elif audio_est.valid:
                best = audio_est
            else:
                best = AlignmentEstimate(
                    offset=0, confidence=0.0, method="low_confidence", valid=False,
                )

            results.append((i, j, best))

    return results
```

Also add `_collect_pairwise_estimates` to the imports in `tests/test_sync.py`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_sync.py::test_collect_pairwise_estimates_returns_correct_pairs -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stages/sync.py tests/test_sync.py
git commit -m "feat: implement _collect_pairwise_estimates for pass-2 pairwise alignment"
```

---

## Task 7: Wire pass 2 + graph solve into `TemporalSyncStage.run()`

**Files:**
- Modify: `src/stages/sync.py` (`TemporalSyncStage.run()`, from line ~1317)
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_sync.py`:

```python
def test_run_produces_graph_residual_field(tmp_path):
    """TemporalSyncStage.run() must populate graph_residual_frames on Alignment."""
    from src.stages.sync import TemporalSyncStage
    from src.schemas.sync_map import SyncMap

    # Build two minimal clips
    for name in ("shot_001.mp4", "shot_002.mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(tmp_path / "shots" / name), fourcc, 25.0, (64, 36))
        (tmp_path / "shots").mkdir(exist_ok=True)
        out = cv2.VideoWriter(
            str(tmp_path / "shots" / name), fourcc, 25.0, (64, 36)
        )
        for _ in range(50):
            out.write(np.zeros((36, 64, 3), dtype=np.uint8))
        out.release()

    manifest_data = {
        "source_file": "dummy.mp4",
        "fps": 25.0,
        "total_frames": 100,
        "shots": [
            {"id": "shot_001", "clip_file": "shots/shot_001.mp4",
             "start_frame": 0, "end_frame": 50, "start_time": 0.0, "end_time": 2.0},
            {"id": "shot_002", "clip_file": "shots/shot_002.mp4",
             "start_frame": 0, "end_frame": 50, "start_time": 0.0, "end_time": 2.0},
        ],
    }
    import json
    (tmp_path / "shots" / "shots_manifest.json").write_text(json.dumps(manifest_data))

    stage = TemporalSyncStage(config={"sync": {}}, output_dir=tmp_path)
    stage.run()

    sm = SyncMap.load(tmp_path / "sync" / "sync_map.json")
    assert len(sm.alignments) == 1
    # graph_residual_frames must be present (0.0 when only one non-ref clip)
    assert sm.alignments[0].graph_residual_frames is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sync.py::test_run_produces_graph_residual_field -v
```

Expected: FAIL (`AssertionError` — `graph_residual_frames` is None)

- [ ] **Step 3: Refactor `run()` — accumulate pass-1 estimates before appending alignments**

In `TemporalSyncStage.run()`, read the new config values after the existing config block (around line 1338):

```python
        pass2_margin = int(cfg.get("pass2_search_margin_frames", 30))
        graph_alpha = float(cfg.get("graph_confidence_alpha", 0.4))
        graph_residual_threshold = float(cfg.get("graph_residual_flag_threshold_frames", 15))
```

Replace `alignments: list[Alignment] = []` with:

```python
        pass1_estimates: dict[str, AlignmentEstimate] = {}
```

At the end of the per-shot loop (where `alignments.append(Alignment(...))` currently is), replace the append with:

```python
            pass1_estimates[shot.id] = best
            logging.info(
                "  [sync/pass1] %s → %s  offset=%+d  conf=%.2f  method=%s%s",
                reference, shot.id, offset, confidence, method, flag,
            )
            print(
                f"  -> {shot.id} (pass1) offset={offset:+d} frames, "
                f"confidence={confidence:.2f} ({method}){flag}"
            )
```

Remove the existing `logging.info` and `print` at the end of the loop (they're now replaced above).

- [ ] **Step 4: Add pass 2, graph solve, and alignment assembly after the loop**

After the pass-1 loop (before `SyncMap(...).save(...)`), add:

```python
        # --- Pass 2: pairwise non-reference comparisons ---
        pairwise = _collect_pairwise_estimates(
            shots=manifest.shots,
            clips={s.id: self.output_dir / s.clip_file for s in manifest.shots},
            tracks_by_shot=tracks_by_shot,
            n_frames_by_shot=n_frames_by_shot,
            pass1_estimates=pass1_estimates,
            poses_dir=poses_dir,
            fps=fps,
            cfg=cfg,
        )

        # --- Build edge list for solver ---
        # Shots are indexed 0..n-1; shots[0] is the reference (index 0)
        edges: list[tuple[int, int, float, float]] = []
        for idx, shot in enumerate(manifest.shots[1:], start=1):
            est = pass1_estimates[shot.id]
            edges.append((0, idx, float(est.offset), float(est.confidence)))
        for i, j, est in pairwise:
            edges.append((i, j, float(est.offset), float(est.confidence)))

        # --- Solve ---
        solved = _solve_offset_graph(len(manifest.shots), edges)

        # Build lookup: shot_id → list of incident pairwise edge confidences
        incident: dict[str, list[float]] = {s.id: [] for s in manifest.shots[1:]}
        for i, j, est in pairwise:
            if est.confidence > 0:
                incident[manifest.shots[i].id].append(est.confidence)
                incident[manifest.shots[j].id].append(est.confidence)

        # --- Assemble final alignments ---
        alignments: list[Alignment] = []
        for idx, shot in enumerate(manifest.shots[1:], start=1):
            solved_offset = int(round(solved[idx]))
            pass1_est = pass1_estimates[shot.id]
            residual = float(abs(solved_offset - pass1_est.offset))

            inc = incident[shot.id]
            mean_inc = float(np.mean(inc)) if inc else pass1_est.confidence
            confidence = float(min(
                1.0,
                pass1_est.confidence * graph_alpha + mean_inc * (1.0 - graph_alpha),
            ))

            method = pass1_est.method
            if residual > graph_residual_threshold:
                method = "graph_refined"

            shot_n_frames = n_frames_by_shot.get(shot.id, 1)
            start, end = _compute_overlap_frames(ref_n_frames, shot_n_frames, solved_offset)
            overlap = max(0, end - start)
            if confidence < min_conf or overlap < min_overlap_frames:
                method = "low_confidence"

            alignments.append(Alignment(
                shot_id=shot.id,
                frame_offset=solved_offset,
                confidence=confidence,
                method=method,
                overlap_frames=[start, end],
                graph_residual_frames=residual,
            ))

            flag = "" if confidence >= min_conf else " [WARNING] low confidence"
            logging.info(
                "  [sync] %s → %s  offset=%+d  conf=%.2f  method=%s  residual=%.1f%s",
                reference, shot.id, solved_offset, confidence, method, residual, flag,
            )
            print(
                f"  -> {shot.id} offset={solved_offset:+d} frames, "
                f"confidence={confidence:.2f} ({method}), residual={residual:.1f}f{flag}"
            )
```

- [ ] **Step 5: Run integration test**

```bash
pytest tests/test_sync.py::test_run_produces_graph_residual_field -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/test_sync.py -v
```

Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/stages/sync.py tests/test_sync.py
git commit -m "feat: wire two-pass graph solve into TemporalSyncStage.run()"
```

---

## Self-Review Checklist

After completing all tasks, run:

```bash
pytest tests/ -v --tb=short
```

Verify:
- [ ] `graph_residual_frames` field loads as `None` for old JSON files
- [ ] `_solve_offset_graph` handles n_clips=1, all-zero weights, and consistent triangles
- [ ] `_collect_pairwise_estimates` returns `(n-1)*(n-2)/2` results for n shots
- [ ] `run()` output includes `graph_residual_frames` on every `Alignment`
- [ ] `max_lag_frames` in `_align_audio` and `_align_visual`/`_align_formation_spatial` limits the search range

# Temporal Sync — Diagnostic Summary

## Target Offsets (ground truth)

| Shot | Timecode | Expected offset |
|------|----------|-----------------|
| origi02 | 4:21 (SS:FF @ 30fps) | **+141 frames** |
| origi03 | 13:03 | **+393 frames** |
| origi04 | 13:23 | **+413 frames** |

Reference clip: `origi01` (506 frames, 16.87s, wide broadcast view)

---

## Approaches Tried

### 1. Motion-speed cross-correlation (original)
**Signal**: median per-frame player speed from pitch coordinates (1D cross-correlation)

| Shot | Found offset | Error |
|------|-------------|-------|
| origi02 | +10 | 131fr |
| origi03 | -25 | 418fr |
| origi04 | +120 (conf=0.01) | 293fr |

**Why it fails**: origi03/04 are replays at different playback speeds and angles; speed values don't match. Low confidence on origi04 (no pitch coordinates — calibration failed for that camera).

---

### 2. Visual frame similarity (matrix cosine, grayscale thumbnails 64×36)
**Signal**: L2-normalised grayscale frame descriptors @ 5fps, matrix dot-product similarity

| Shot | Found offset | Error |
|------|-------------|-------|
| origi02 | (low confidence) | — |
| origi03 | (low confidence) | — |
| origi04 | (low confidence) | — |

**Why it fails**: broadcast football frames all look very similar (green pitch + same stadium). Peak z-score < 1 everywhere — visual confidence is universally weak.

---

### 3. Spatial formation histograms (pitch-coordinate grid)
**Signal**: player `pitch_position` values binned into 8×5 grid, L2-normalised, matrix similarity @ 5fps

| Shot | Found offset | Error |
|------|-------------|-------|
| origi02 | +10 | 131fr |
| origi03 | -25 | 418fr |
| origi04 | +120 | 293fr |

**Why it fails**: **Pitch coordinates are not in a globally consistent frame across shots.** Each camera is independently calibrated, producing incompatible local coordinate systems:

| Shot | x range (5th–95th pct) | y range |
|------|------------------------|---------|
| origi01 | −9242 → 5985 | 75 → 2981 |
| origi02 | −7682 → 5070 (+ outliers to ±1.8M) | broken |
| origi03 | 3 → 835 | −145 → 230 |
| origi04 | no pitch positions | — |

origi01 and origi03 have an ~8× scale difference and different origins — the same physical player appears at `x ≈ −8400` in origi01 and `x ≈ 430` in origi03.

---

### 4. Combined windowed pose + track descriptor
**Signal**: per-frame [normalized joint velocity, normalized player spread] with temporal window (window=11, sample_fps=15)

| Shot | Found offset | Error |
|------|-------------|-------|
| origi02 | +146 | **5fr ✓** |
| origi03 | +50 | 343fr |
| origi04 | +442 | 29fr |

**Why it fails for origi03/04**: Wide shot (origi01) and close-up replays show completely different player populations. At the expected alignment point, origi01 and origi03 have Pearson r ≈ −0.12 (anti-correlated) because the 10+ visible players in the wide shot are not the same subset as the 4–6 players in the close-up. The signal also fires false positives for the larger overlap windows at wrong offsets.

---

### 5. Pose joint angles — arm raise & knee flex
**Signals**:
- Arm raise angle: angle between (shoulder→wrist) and downward vertical, per player
- Knee flex angle at joint (hip, knee, ankle)
- Aggregates: max, mean, std, fraction above threshold, 75th percentile

Pearson r at expected offset:

| Shot / Feature | r @ expected | Note |
|----------------|-------------|------|
| origi03 / mean_arm | −0.528 | Anti-correlated |
| origi03 / frac>120° | −0.239 | Anti-correlated |
| origi03 / mean_knee | +0.523 | Moderate positive but false peaks elsewhere |
| origi04 / frac>120° | +0.487 | Positive, z=2.22 |
| origi04 / max_arm | +0.673 | Positive |

Best Pearson scan result:

| Shot | Found offset | Error | r@found | r@expected |
|------|-------------|-------|---------|------------|
| origi02 | — | — | — | — |
| origi03 | +226 | 167fr | 0.63 | −0.28 |
| origi04 | +430 | **17fr** | 0.81 | 0.50 |

**Why it fails for origi03**: At the expected overlap (origi01 frames 393–506), the wide shot shows players with mean arm raise ~20° while origi03 (close-up of goal area) starts with ~38°. The visible player populations differ — wide shot has 12+ players mostly not celebrating, close-up shows the immediate 4–6 celebrating players.

**origi04 partial progress**: arm raise signal peaks at +430fr (17fr off expected 413fr) with r=0.81, and scores reasonably at the expected offset (r=0.50, z=2.22). The offset error may partly reflect the user-provided timecode being approximate (within ~0.5s).

---

## Key Blockers

1. **No ball detections** in any shot (ball tracker missed all shots) — ball trajectory would be ideal for multi-view sync.

2. **Pitch coordinates not globally aligned** — each camera calibration has its own coordinate origin and scale. Cannot compare player positions across cameras in world space.

3. **No consistent player IDs across shots** — player matching (Stage 6) runs *after* sync (Stage 3), creating a chicken-and-egg dependency. Cannot directly match "player A in shot 1 = player B in shot 3".

4. **Fundamental cross-view problem for close-up replays**: origi03/04 show a tight crop of 4–8 players; origi01 shows 10–14. At the same physical timestamp, aggregate statistics (mean arm raise, mean speed, player spread) are dominated by different player populations. No aggregate feature reliably fires the same value from both views simultaneously.

---

## What's Available to Work With

| Signal | origi02 | origi03 | origi04 |
|--------|---------|---------|---------|
| Pitch positions (consistent?) | Broken (±1.8M outliers) | Different local frame | None |
| Image-space track bboxes | ✓ | ✓ | ✓ |
| Pose keypoints (COCO 17) | ✓ | ✓ | ✓ |
| Ball detections | ✗ | ✗ | ✗ |
| Arm raise signal works | — | ✗ | Partial (~17fr off) |

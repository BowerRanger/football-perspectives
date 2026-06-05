# Landmark-free camera tracking via learned field registration

- **Date:** 2026-06-05
- **Status:** Design — awaiting review
- **Branch:** `feat/landmark-free-camera`
- **Goal:** Achieve crisp, pixel-perfect detected-line camera tracking with **zero manually
  placed landmarks**, by auto-generating anchors from a vendored learned soccer-pitch
  registration model and feeding the existing sub-pixel solver stack unchanged.

## 1. Problem & success criteria

### The problem
The camera stage today **requires** the user to hand-click pitch landmarks in the web
anchor editor: at least one "rich" anchor with ≥6 non-coplanar point landmarks to recover
metric pose, plus enough anchors to cover the clip. This is slow (~15–30 min per minute of
footage) and the click noise (1–3 px) is itself a precision floor.

Critically, the **precision problem is already solved**. The sub-pixel painted-line
detector (`line_detector.py`: bright-ridge template + RANSAC + green-mask gating) plus the
static-camera line solve (`static_line_solver.py` + `static_c_profile.py`) already reach
**mean 0.95 px / median 0.81 px / 74% of frames sub-1 px** on the `gberch` clip — *when it
has a roughly-correct camera to start from*. Line detection works by searching a strip
around **projected** catalogue lines, so it needs an approximately-correct camera before it
can run at all.

The manual clicks do exactly two jobs, both of which must be automated to reach zero
landmarks:
1. **Rough per-frame pose** — a camera close enough that strip-search line detection locks on.
2. **Data association** — which detected image line/point is which world feature
   (`near_touchline`? `right_18yd_front`?).

Data association is genuinely hard because the FIFA pitch is **mirror-symmetric left/right
and near/far**, and box edges repeat: geometry alone cannot tell which half of the pitch is
in view. Today the human resolves this instantly by recognising the scene.

### Success criteria
- **Primary:** On `gberch` with **zero manual clicks**, the auto-anchored pipeline produces
  a camera track whose line-RMS is at parity with the manual-anchor baseline — target
  **mean ≤ 1.0 px** (vs today's 0.95 px), median ≤ ~0.9 px.
- The same holds (qualitatively, within tuning) on `origi01`/`origi02`.
- Manual anchoring remains fully functional as an override/correction path.
- No regression to existing camera-stage behaviour when the auto path is disabled or
  unavailable.

## 2. Approaches considered

| # | Approach | Cold-start pose | Data association | Symmetry | Reaches 0 clicks? |
|---|---|---|---|---|---|
| **A (chosen)** | Learned hybrid — PnLCalib keypoint/line heatmaps → metric pose via existing solver | Free (channel = named point) | Intrinsic (learned) + static-camera consensus | Learned from appearance | **Yes** |
| B | Classical CV only — LSD/Hough + vanishing points + RANSAC over catalogue-assignment hypotheses | Hand-rolled search | Brittle hypothesis search | Needs temporal + ~1 hint/clip | Rarely |
| C | Auto-suggest + 1-click confirm in editor | Detector + human | Human-in-loop | Human resolves | No (1 click) |

**Decision: Approach A.** The left/right + near/far mirror symmetry is fundamentally an
*appearance* problem (which crowd, ads, goal) that a SoccerNet-trained network resolves for
free, whereas classical geometry (B) cannot robustly disambiguate it and would reintroduce a
permanent assignment+symmetry battle. A also gives data association intrinsically (each
heatmap channel is a named point) and maps directly onto the existing `Anchor` schema, so
the entire downstream precision stack is reused unchanged. C does not reach the zero-click
goal.

### Model selection (verified 2026-06-05)
| Model | License | Maintained | Checkpoints | Notes |
|---|---|---|---|---|
| **PnLCalib** ([mguti97/PnLCalib](https://github.com/mguti97/PnLCalib)) | **GPL-2.0** | ✅ Mar 2026, 85★ | ✅ GitHub Releases (separate keypoint + line HRNet models; SoccerNet + WC14/TSWC/WorldPose finetunes) | SOTA successor to NBJW; intrinsic data association; PnL refinement + lens-distortion module |
| NBJW ([No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles)) | GPL-2.0 | 2024 | ✅ | Superseded by PnLCalib (same authors) |
| TVCalib ([MM4SPA/tvcalib](https://github.com/MM4SPA/tvcalib)) | MIT | 2024, 45★ | ✅ | Older; segmentation + differentiable optimisation; heavier; no free keypoint association |
| sportlight 2023 | None (all rights reserved) | — | — | Not legally reusable |

**Chosen model: PnLCalib.** GPL-2.0 accepted (decision below). It is the only actively
maintained SOTA option that emits *named* keypoints (association solved) with downloadable
offline checkpoints.

### Decisions locked with the user
- **License:** GPL-2.0 accepted. PnLCalib is vendored as an **arms-length git submodule
  invoked as a subprocess** (exactly like the existing GVHMR integration), keeping the
  boundary clean. This is a personal/research project; GPL distribution obligations are not
  expected to bite.
- **Integration depth:** **Auto-anchor generator.** The learned model emits semantic
  point+line correspondences; they become standard `Anchor` objects; the existing joint
  solve + static-camera C-profile + sub-pixel line solver produce the final fit. Maximum
  reuse; the static-camera constraint and all precision work are preserved.

## 3. Architecture

### 3.1 Where it slots
A new **pre-step** at the top of the camera stage's per-shot flow:

```
camera stage, per shot:
  ┌─ NEW: if auto_anchors.enabled and (no manual anchors  OR  mode == augment):
  │        anchors = auto_anchor.generate(clip, shot_id, cfg)   # learned model
  │        write anchors → output/camera/{shot}_anchors.json    # same file the editor uses
  └─ (unchanged from here)
     load {shot}_anchors.json
     qualifying filter (≥4 pts or ≥2 lines; ≥1 rich anchor)
     lens prior → solve_anchors_jointly → refine_with_shared_translation
     propagate (K,R,t) between anchors
     line_extraction refinement (static-C line solve)
     assemble + save camera_track.json
```

Because auto-anchors are written to the **normal anchors JSON**, the user can open the
anchor editor, *see* the auto-generated points/lines, and nudge / add / delete them —
graceful override is free. `mode: replace-when-empty` (default) only auto-generates when no
manual anchors exist; `mode: augment` unions auto + manual; `mode: force` regenerates.

### 3.2 Data flow
```
clip.mp4
  └─ auto_anchor.generate():
       1. sample candidate keyframes (uniform stride; configurable count)
       2. for each keyframe frame_bgr:
            PnLCalibProvider.register_frame() → FieldRegistrationResult
              { keypoints: {pnl_id → (image_xy, conf)},
                lines:     {pnl_id → (image_seg, conf)} }
       3. map via pnlcalib_catalogue_map:  pnl_id → (project world_xyz, catalogue name)
       4. confidence filter (drop low-conf detections)
       5. clip-level static-camera CONSENSUS:
            - solve a quick pose per keyframe from its mapped points
            - reject keyframes whose camera centre / orientation disagrees
              with the temporal+static-camera consensus (kills left/right flips)
       6. keep keyframes that qualify as anchors (≥4 pts or ≥2 lines;
          ensure ≥1 rich anchor with ≥6 non-coplanar pts overall)
       7. emit AnchorSet (image_size, clip_id, anchors[]) — identical to editor output
```

## 4. Components (new)

| Path | Responsibility |
|---|---|
| `third_party/pnlcalib/` | Vendored PnLCalib submodule |
| `third_party/pnlcalib/weights/` | `SV_kp`, `SV_lines` checkpoints (from GitHub Releases) |
| `src/utils/pnlcalib_provider.py` | Run PnLCalib inference via subprocess shim (GVHMR cwd / cuda-redirect pattern); parse keypoint + line heatmap outputs with per-detection confidence into `FieldRegistrationResult` |
| `src/utils/pnlcalib_catalogue_map.py` | **The bridge.** Static, unit-tested table: PnLCalib keypoint/line id → this project's world xyz + catalogue name. Handles the SoccerNet-template-coords → project-world-coords transform |
| `src/utils/auto_anchor.py` | Keyframe sampling → confidence filter → clip-level static-camera consensus → emit `AnchorSet` |
| `src/stages/camera.py` | Pre-step hook to invoke `auto_anchor` when enabled |
| `config/default.yaml` | `camera.auto_anchors.*` config block |

### 4.1 The catalogue bridge (primary integration risk)
PnLCalib operates in the SoccerNet pitch template: a 105×68 m model with its **origin at the
pitch centre** and a fixed keypoint dictionary. This project's world frame has its **origin
at the near-left corner** (x along the near touchline 0→105, y across 0→68, z up). A wrong
mapping mirrors or offsets *everything*, so the bridge is a dedicated, unit-tested module:

- A constant table `PNL_KEYPOINT_TO_WORLD[pnl_id] = (name, (x, y, z))` and the analogous
  line table, derived from PnLCalib's published template + this project's
  `pitch_landmarks.py` / `pitch_lines_catalogue.py`.
- Exact PnLCalib indices/coords are **derived and verified in Phase 1** against a known
  frame (not asserted blind here).
- Unit test: a handful of known PnLCalib ids map to the expected project world coords;
  round-trip a known camera and confirm projected points land on the right features.

### 4.2 Data association & symmetry
- **Intrinsic association:** each PnLCalib heatmap channel *is* a named point/line — no
  assignment problem. The model is trained on broadcast footage, so it resolves which
  goal/half from appearance.
- **Consensus safety net:** a clip-level check (`auto_anchor`) rejects any keyframe whose
  recovered pose disagrees with the temporal/static-camera consensus — all keyframes must
  share one camera centre (the constraint the static-camera path already enforces). This
  eliminates residual per-frame left/right flips that geometry alone cannot catch.

### 4.3 Config (`camera.auto_anchors`)
```yaml
camera:
  auto_anchors:
    enabled: false          # opt-in; off preserves today's behaviour exactly
    mode: replace_when_empty # replace_when_empty | augment | force
    keyframe_stride: 30      # sample every N frames as anchor candidates
    max_keyframes: 12        # cap on auto-anchor count
    min_keypoint_conf: 0.5   # drop detections below this
    min_points_per_anchor: 4 # qualify-as-anchor threshold (matches solver)
    consensus_max_centre_disagreement_m: 3.0  # reject flipped/discordant keyframes
    model:
      kp_weights: third_party/pnlcalib/weights/SV_kp
      line_weights: third_party/pnlcalib/weights/SV_lines
      device: cpu            # GVHMR-style CPU default on macOS; GPU on the Linux box
```

## 5. Error handling & fallback (graceful degradation)
- PnLCalib submodule/checkpoint missing → log a clear warning, fall back to today's
  manual-anchor behaviour (require `{shot}_anchors.json`). No crash.
- A clip yields too few confident keyframes (or zero rich anchors) → warn, fall back to
  manual; surface the shortfall in the quality report.
- The anchor editor stays fully functional. Auto-anchors written to the normal JSON are
  inspectable and editable.
- Per-keyframe + consensus confidences flow into `quality_report.json` and the dashboard
  confidence timeline.
- All external data validated at the boundary (provider outputs schema-checked before they
  become anchors).

## 6. Testing
- **Unit — bridge:** known PnLCalib ids → expected project world coords; projected
  round-trip lands on the right features.
- **Unit — consensus:** an injected flipped/discordant keyframe is rejected; concordant
  keyframes are kept.
- **Unit — provider parsing:** mocked subprocess output parses into `FieldRegistrationResult`
  with correct confidences; malformed output handled.
- **Unit — auto_anchor:** sampling + filtering produce a valid `AnchorSet` that the existing
  qualifying filter accepts (≥1 rich anchor).
- **Integration (headline):** on `gberch` with **zero manual clicks**, auto-anchors → existing
  solver → assert line-RMS mean ≤ 1.0 px (parity with the 0.95 px manual baseline).
- **Regression:** existing camera-stage tests stay green with the auto path disabled
  (default).

## 7. Phasing
1. **Vendor + provider + bridge.** Add the submodule, fetch checkpoints, build
   `pnlcalib_provider` and `pnlcalib_catalogue_map`, and a one-frame CLI smoke test that
   registers a frame and prints mapped correspondences — validate the bridge numerically and
   visually before anything else.
2. **Auto-anchor generation.** `auto_anchor.py` (sample → filter → consensus → emit
   `AnchorSet`), wired into the camera stage behind `auto_anchors.enabled`. Unit tests.
3. **End-to-end + tuning.** Run on `gberch`/`origi`; tune keyframe selection + confidence
   gates to hit RMS parity; integrate confidences into the quality report and editor overlay.

## 8. Scope guardrails (YAGNI)
- One concrete PnLCalib provider behind a thin function seam — **not** a plugin framework
  (the user chose PnLCalib, not "defer behind an interface").
- Uniform keyframe sampling + confidence gate for v1; adaptive keyframe selection is a future
  enhancement.
- `replace_when_empty` default; `augment` is a cheap union; no editor-side auto-suggest UI in
  v1 (the auto-anchors are already visible/editable in the editor).
- Goal-post / vertical-line detection beyond what PnLCalib already provides is out of scope.

## 9. Risks & open questions
- **Bridge correctness** is the dominant risk — mitigated by Phase 1 validation before any
  downstream wiring.
- **PnLCalib intrinsics vs ours:** we consume its *detections* (points/lines), not its camera,
  so its internal intrinsic model does not need to match ours. (Open: whether to also seed
  the lens prior from its detections or keep `lens_from_anchor` as-is — default keep.)
- **CPU inference speed** on macOS for HRNet-class models: acceptable because we run only on a
  sparse set of keyframes, not every frame. Confirm in Phase 1.
- **Coordinate/units** in PnLCalib's template (metres vs normalised; axis directions) —
  pinned down in Phase 1.
- **Rich-anchor guarantee:** if a clip never shows ≥6 non-coplanar points across keyframes
  (e.g. no goal/flag visible), metric pose is under-constrained — fall back to manual and
  warn. (Vertical features like goal posts give the non-coplanar z-spread.)

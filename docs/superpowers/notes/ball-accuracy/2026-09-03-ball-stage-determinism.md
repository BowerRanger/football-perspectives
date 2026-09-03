# Ball stage determinism investigation (2026-09-03)

**Trigger:** `run_touch_recall_validation.py` on gberch produced legacy
union recall 0.500 in one run and 0.375 in another, with byte-identical
config/code/inputs (previous agent's controlled A/B). Suspected locus per
the task brief: second-pass/foot-guided corridor crops triggering real
WASB (MPS) inference with results that vary run-to-run.

**Constraint:** battery-limited — no full ball-stage or validation runs
(each ~20+ min). Everything below is a static audit plus micro-experiments
(seconds to ~2 min each), run sequentially.

## Result summary

Could not reproduce the reported nondeterminism at either the Python
level or the WASB/MPS-forward-pass level on this machine (macOS, torch
2.1.2, the checked-in `wasb_soccer_finetuned_v1.pth.tar` checkpoint).
Both layers tested bit-identical across repeats. A regression test
(`tests/test_ball_stage_determinism.py`) now locks in the Python-level
finding. The leading remaining hypothesis for the original 0.500/0.375
swing is that the two compared runs did not actually hold the upstream
**camera track** fixed — PnLCalib-on-MPS nondeterminism is already
documented for this repo (`docs/superpowers/notes/camera-investigation/`,
CLAUDE.md) and was outside this investigation's scope to re-verify
without a camera-stage re-solve (a genuinely expensive operation, not a
cheap probe).

## 1. Python-level audit (static)

Grepped `src/stages/ball.py` + the ~40-module `src/utils/ball_*` family
for the classic footguns:

- `for x in <set/frozenset>` where the loop body does something
  order-sensitive (not just membership testing or idempotent
  pop/assignment). Found several `set`/`frozenset` declarations
  (`ball_auto_anchor.py`, `ball_kinematic_touch.py`,
  `ball_touch_attribution.py`, `ball.py` itself), but every one is either
  used for membership (`in`, `any(...)`) or the consuming loop already
  ends in an explicit `sorted(..., key=...)` with a total-order key
  (frame, then player_id, then bone — never left to tuple/dataclass
  default ordering). Representative examples already correct as found:
  - `ball_kinematic_touch.py::propose_touches` returns
    `sorted(out, key=lambda e: (e.frame, e.player_id, e.bone))`.
  - `ball_kinematic_touch.py::nms_touches` returns
    `sorted(kept, key=lambda e: (e.frame, e.player_id or "", e.bone or ""))`.
  - `ball_auto_anchor.py::_burst_nms` sorts candidates by `-score` for
    NMS, but the *pre-sort* candidate order (which is what stable-sort
    ties fall back to) is itself already deterministic — see below.
  - `ball_touch_attribution.py::refine_touch_attribution` picks the best
    relabel via `min(gaps.items(), key=lambda kv: (kv[1], kv[0]))` — gap
    value first, `(player_id, bone)` second, so exact-gap ties are still
    total-ordered.
  - `list(set(...))` (a direct, unsorted set-to-list conversion) does not
    occur anywhere in the family.
- No `random`/`np.random` usage, no threading/multiprocessing, in the
  entire `ball_*` family — ruling out a whole class of nondeterminism.
- `PlayerContext.joints_at()` (`ball_player_context.py`) — the shared
  per-frame joint iterator many of the above consume — returns joints in
  a fixed order: players from `sorted(set(refined_paths) | set(hmr_paths))`
  at load time, bones from the literal (insertion-ordered) dict
  `BONE_TO_SMPL_INDEX` in `ball_anchor_heights.py`. Not hash-seed
  dependent.

**Conclusion:** the static audit found the family already defensively
written (sorted-with-explicit-tiebreak is the norm, not added by this
investigation) — no fix was needed because no bug was found.

## 2. Python-level micro-experiment (empirical)

Per the task's suggested cheap test: replayed the **pure-Python**
event/anchor tail of the pipeline (`detect_events` → foot-guided merge →
`propose_touches`/`merge_touch_events` → `refine_touch_attribution` →
`generate_auto_anchors` → `merge_anchors`) directly from the
already-cached `<shot>_ball_observations.json` sidecar — no detector, no
video decode. `BallStage._detect_loop` is monkeypatched to return the
cached `(steps, raw_confidences, sources)` verbatim, and
`second_pass`/`foot_guided`/`detection_cache` are disabled in config (a
dummy detector is injected and asserts if ever called), isolating
exactly the "python-level" half of the investigation.

Ran this replay for gberch (`output/`), kroupi01 (`output-kroupi/`), and
origi01 (`output-origi/`) — three separate `python` subprocesses per
clip with `PYTHONHASHSEED` set to `0` and `4276993775` respectively.
Sanity-checked first that these two seeds actually do change native set
iteration order for representative strings from this codebase (player
ids, bone names):

```
PYTHONHASHSEED=0        -> ['P002', 'head', 'l_foot', 'P003', 'P001', 'r_foot']
PYTHONHASHSEED=4276993775 -> ['l_foot', 'P001', 'P002', 'P003', 'head', 'r_foot']
```

Result: **`<shot>_ball_anchors_auto.json` was byte-identical across the
two hash seeds for all three clips.** This is now
`tests/test_ball_stage_determinism.py`
(`test_anchor_pipeline_is_hashseed_independent`, `pytest.mark.integration`,
parametrized over the three clips, skips cleanly if the local fixture
output dirs are absent — consistent with `test_ball_anchor_accuracy.py`'s
convention). ~18s total for all three clips.

## 3. WASB/MPS detector-level micro-experiment (empirical)

`config/default.yaml` sets `ball.wasb.device: mps` (the code's own
`_pick_device` default for `"auto"` is actually CPU — MPS is an explicit
opt-in in this repo's config, done for eval speed; see the comment at
`config/default.yaml:869`). Every real ball-stage run on this Mac,
including `run_touch_recall_validation.py`'s (no `--config` override, so
defaults apply), runs WASB on MPS.

Three escalating probes against `WASBBallDetector` on real gberch frames
(`output/shots/gberch.mp4`), fine-tuned-v1 checkpoint:

1. **Isolated frames, same process, fresh detector + `reset()` per
   call** (9 frames around touch/anchor moments + a few grounded frames,
   3 repeats): `max|Δheatmap|` = 0 for every frame, decision-level
   (`heatmap_candidates` top-1) identical every repeat, on both `cpu` and
   `mps`.
2. **Isolated frames, three separate fresh processes** (same 9 frames,
   one heatmap-sweep each, saved to `.npy` and diffed pairwise): MPS
   `max|Δheatmap|` = 0 across all three processes, every frame.
3. **Full sequential clip, natural rolling 3-frame buffer** (all 429
   frames of gberch.mp4, decoded once via `cv2.VideoCapture.read()` in
   order — mirrors the real pass-1 detection loop, not the synthetic
   reset-per-frame case above), two separate fresh processes, MD5
   checksum of the raw heatmap array per frame: **0/429 checksum
   diffs, 0/429 top-1-candidate diffs** between the two full-clip runs.
   Also confirmed `cv2.VideoCapture` frame seeks (`CAP_PROP_POS_FRAMES`)
   are themselves byte-reproducible across 5 fresh-process trials (rules
   out a video-decode-seek explanation independently of MPS).
4. Checked for silently-random model weights: `load_wasb_model` logged
   zero "missing keys" / "unexpected keys" warnings loading the
   fine-tuned-v1 checkpoint — the full HRNet state dict is checkpoint-covered,
   so there is no unseeded `torch.nn.init` randomness left in play (which,
   if present, would itself have shown up as a cross-process diff in (2)
   and (3) above, since no seed is set anywhere in this code path).

**Timing (honest speed comparison, full 429-frame gberch clip, same
checkpoint):** MPS 26.2–26.8s (~62ms/frame) vs CPU 125.2s (~292ms/frame)
— **CPU is ~4.7x slower**, consistent with the existing config comment
("unset -> CPU, which makes dashboard runs take hours").

**Conclusion:** on this machine/torch version, WASB HRNet inference is
bit-for-bit deterministic given identical input frames, both
same-process and cross-process, both for isolated calls and a realistic
full-clip sequential decode with the model's natural rolling buffer. No
evidence of MPS forward-pass nondeterminism was found — this differs
from the documented PnLCalib-on-MPS situation (a different model/solver
entirely; RANSAC-style camera solving has explicit randomized steps that
WASB's forward pass does not).

This does **not** prove MPS is deterministic in every circumstance (a
~20-minute real run under real system thermal/memory load is a different
regime than these seconds-long idle-system probes), but it does rule out
"the WASB forward pass is inherently nondeterministic on MPS" as an
unconditional explanation — under the conditions actually tested, it
was not.

## 4. Cache-path audit (static)

`ball.detection_cache.enabled` defaults to `false`
(`config/default.yaml:586`), and `run_touch_recall_validation.py` never
overrides it (`load_config(None)` with no `--config` in the reported A/B)
— so `wrap_if_enabled` returns the raw detector unchanged and
`CachingBallDetector`/`detection_cache.json` are never in the loop for
the reported evidence. Ruled out as the cause of *that* specific report,
though the class itself was audited for completeness:

- Cache keys are **content hashes** of the actual frame/crop bytes
  (`hashlib.md5(frame[::4, ::4].tobytes())` + shape), not coordinates —
  a crop's cache key is a function of its pixel content, not of how the
  corridor that produced it was constructed. This means "float coords in
  keys" isn't a real risk here (unlike, say, keying on a rounded corridor
  center) — two calls with the same crop content always collide
  correctly regardless of construction order.
- `_detect_shot`'s pass-1 detect loop has no resume/reuse of stale
  `_ball_track.json`/`_ball_observations.json` from a previous script
  invocation — every ball-stage run re-detects from scratch (confirmed
  by reading `_detect_shot`; the only sidecar *read* mid-run is the
  observations file it just wrote earlier in the *same* run, for the
  flight-veto post-pass).

`ball.detection_cache.enabled: true` (an explicit opt-in some other
workflow, e.g. `scripts/eval_ball_accuracy.py --det-cache`, might use) was
out of scope for this report since it's not what
`run_touch_recall_validation.py` exercises by default.

## Fixes landed

None — no Python-level nondeterminism source was found to fix. The
family was already written defensively (explicit total-order sort keys
throughout the touch/anchor pipeline). The deliverable for this
investigation is the regression test below, which makes the "no bug
found" finding falsifiable and durable rather than a one-time claim.

## What's committed

- `tests/test_ball_stage_determinism.py` — new. Runs the cached-observation
  pure-Python event/anchor pipeline twice (subprocess, `PYTHONHASHSEED=0`
  vs `4276993775`) per clip (gberch/kroupi01/origi01) and asserts
  byte-identical `<shot>_ball_anchors_auto.json`. `pytest.mark.integration`,
  skips cleanly without the local `output*/` fixtures. This is the
  regression guard requested by the task — if a future change introduces
  set/dict-order dependence anywhere in the event/anchor pipeline, this
  test will catch it without needing a GPU or a real detector.

## Remaining variance sources / open questions

1. **Camera-track drift between runs (leading hypothesis, unverified
   here).** PnLCalib-on-MPS nondeterminism is already documented in this
   repo (CLAUDE.md's measurement-discipline note, the
   `camera-investigation` notes). If the previous agent's two compared
   runs did not hold `<shot>_camera_track.json` byte-identical (e.g. a
   broader pipeline invocation re-solved camera between the two ball-stage
   runs, or a concurrent process touched the file), even a sub-pixel `R`/`t`
   difference could shift second-pass/foot-guided corridor gates and
   kinematic-touch ray-gap thresholds (`contact_gap_m=0.30`,
   `kin_min_foot_speed_tight_gap_m=0.17`) enough to flip several
   borderline touch decisions — a small input perturbation, not a code
   bug. **Recommendation:** when comparing ball-stage-only runs for
   measurement purposes (touch recall, anchor accuracy), always reuse a
   single frozen `camera_track.json` rather than re-solving camera per
   run — consistent with the existing "judge camera quality separately,
   never eyeball a single PnLCalib run" discipline. This was not
   independently re-verified in this investigation (re-solving camera
   twice to compare is not a "cheap" probe under the battery constraint).
2. **MPS under real system load.** The MPS probes above ran on an
   otherwise-idle machine for seconds to ~30s each. A ~20-minute full
   validation run exercises far more forward passes under real thermal/
   memory conditions; this investigation's bound (26–27s, 429 frames,
   idle system) does not extend automatically to that regime. If the
   confirming full run (scheduled for later, per the task) reproduces
   variance with camera held fixed, the next-cheapest follow-up is
   re-running probe 3 above (full-clip checksum diff) immediately before
   and after a real ~20-minute stage run on the same machine, back to
   back, to see if sustained load changes the picture.
3. **`ball.wasb.device: cpu` as a determinism knob.** Already exists as a
   config key (no new knob needed) — `_pick_device` honors `"cpu"`
   directly. Given finding 3 above, forcing CPU is not currently
   justified by evidence of MPS nondeterminism, and the honest cost is
   ~4.7x slower detection (26s → 125s for one 429-frame shot at this
   detector's resolution) — worth paying only if the camera-track-drift
   hypothesis is ruled out by a controlled re-run with frozen camera and
   variance still appears.

## Test results

```
.venv311/bin/python -m pytest tests/test_ball_*.py -q
```

See commit message for the exact pass/fail count at the time of landing;
the only expected failure is the pre-existing, documented
`test_ball_stage.py::test_aerial_arc_promotes_grounded_run_to_flight`
(unrelated to this investigation — noted in `CLAUDE.md` as a known
failure on `main`).

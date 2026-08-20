# Ball sub-20cm: operator handoff — the last mile

**State (branch `ball-sub20cm-accuracy`, ~100 commits, updated after W7):**
60 of 881 measurable dense frames (6.8%) exceed 20 cm — down from 119
(13.4%) after the W7 off-ray touch-keyframe fix cleared origi01's four
biggest spans without operator input. The remaining spans are being
per-frame decomposed before any data-bound claim (the W7 lesson); this
doc lists the operator continuations for whatever survives that pass.

Totals per clip (same-day re-certification): origi01 18/355,
gberch 17/308, s013 13/172, kroupi01 12/46.

## Path A — anchors at flagged spans (~45 min, removes up to ~100 frames)

Place 1–2 ball anchors inside each span below (mid-span frame is fine;
pick the true state — `grounded` / `player_touch` with bone / `bounce`).
Spans sorted by impact. Anchoring the top 8 spans covers ~70% of the
residual.

**origi01** (updated 2026-08-20 after the W7 off-ray touch fix — was 77
failing, now 18; the [4–9]/[91–115]/[135–146]/[150–161]/[225–233] spans
cleared without operator input; serve `output-origi`):

| Span | Frames failing | Note |
|---|---|---|
| [369–374] | 6 | |
| [426–428] + [432–434] | 5 | |
| singletons 38, 207, 248, 263, 279, 298, 458 | 7 | one anchor each only if chasing the strict bar |

**gberch** (17 failing; serve `output`):
[182–189]×4, [395–401]×6, [322]×1, [336–337]×2, [382]×1, [388–389]×2, [408]×1

**s013** (13 failing; serve `output-japan`):
[202–205]×4, [148–151]×3, [133–136]×2, [209–211]×2, [36]×1, [117]×1

**kroupi01** (12 failing; serve `output-kroupi`):
[146–148]×3, [128–129]×2, singletons at 4, 10, 49, 111, 116, 121, 138

Workflow per clip:

```bash
python recon.py serve --output <dir>          # ball anchor editor
# place anchors, then:
python recon.py run --input <clip> --output <dir> --stages ball
# re-measure:
.venv311/bin/python scripts/eval_ball_accuracy.py --output <dir> --shot <shot> \
    --detector caching --det-cache docs/superpowers/notes/ball-accuracy/det_cache/<shot>.json
```

## Path B — s013 replay fixes (~30 min, absolute 3-D truth for s013 flights)

s013's group g02 has six replay partners. W6 proved landmark-free partner
cameras are globally wrong (fixes triangulate underground) — but the
machinery + physical-plausibility gate (947eb13) will exploit any
OPERATOR-REVIEWED partner automatically.

1. Un-exclude ONE partner in the dashboard (s012 is shortest, 94 frames;
   s008 is the wide live angle).
2. Open the camera anchor editor for that shot and fix its landmark
   anchors (auto-minted starting points appear; the wrongness is global
   placement — check the far-side landmark identities).
3. Re-run: `python recon.py run --input <reel> --output output-japan --stages camera,ball`
4. The ball stage triangulates s013 fixes, gates them physically, and the
   fix-arc solver consumes them. Junk solves self-reject (diag
   `cross_replay.partners.<shot>.rejected = "implausible_geometry"`).

## Path C — detector gold labels → fine-tune v3 (days)

Label ball centers on the failing spans above (they are exactly where the
detector is blind or noisy), rebuild the corpus, fine-tune:

```bash
.venv311/bin/python scripts/build_finetune_corpus.py ...
.venv311/bin/python scripts/finetune_wasb.py ...
```

v2 measured as a precision/coverage trade (kept for study at
`third_party/wasb_sbdt/pretrained_weights/wasb_soccer_finetuned_v2.pth.tar`);
v3 only pays off if labels target the failing spans.

## Re-certification

After any path, re-run the 4-clip certification block at the bottom of
`2026-08-17-baseline.md` and append the new table.

# Ball sub-20cm: operator handoff — the last mile

**State (branch `ball-sub20cm-accuracy`, 95 commits):** 119 of 885 measurable
dense frames (13.4%) exceed 20 cm. Every autonomous axis is exhausted and
measured (solver hypotheses certified; replay-pair path proven to need
operator cameras, W6; measurement noise floor probed moot). The residual is
data-bound. This doc turns the three continuations into concrete checklists.

Derived from current shipped artifacts on 2026-08-20; failing-frame total
reconciles exactly with the certification (17+77+13+12 = 119 over 885).

## Path A — anchors at flagged spans (~45 min, removes up to ~100 frames)

Place 1–2 ball anchors inside each span below (mid-span frame is fine;
pick the true state — `grounded` / `player_touch` with bone / `bounce`).
Spans sorted by impact. Anchoring the top 8 spans covers ~70% of the
residual.

**origi01** (77 failing — 65% of everything; serve `output-origi`):

| Span | Frames failing | Note |
|---|---|---|
| [91–115] | 24 | 23-frame anchor gap (88 → 111); 1–2 anchors ~f95/f105 |
| [135–146] | 11 | between anchors 140/149; bend the solver can't explain |
| [225–233] | 7 | inside 224/234 bracket — likely hidden touch |
| [367–373] | 6 | |
| [150–154] + [158–161] | 9 | flanks anchor 155 |
| [4–9] | 4 | clip head, before anchor 7 settles |
| smaller: [38]×1 [207]×1 [212–214]×2 [248]×1 [263]×1 [279]×1 [298–301]×3 [426–428]×3 [432–434]×2 [458]×1 | 16 | one anchor each only if chasing the strict bar |

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

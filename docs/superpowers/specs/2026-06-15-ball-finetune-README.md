# Ball-detector fine-tuning — workflow

- **Date:** 2026-06-15 (scaffold); runbook made real 2026-07-04 (Phase 4 execution).
- **Why:** Inference-only touch detection is capped at 0/8 recall on gberch because the **ball detector** (WASB) confidently misdetects at touch frames and can't disambiguate the ball (see [`project_ball_detection_rework`] memory + `2026-06-15-ball-detection-direction-changes-design.md`). Everything downstream — direction-change segmentation, pose attribution, foot-guided zoom, body-pinning — is correct and proven on synthetic data, and the pose is excellent (feet ~15px from the true ball). The unlock is a reliable detector.
- **Status:** Shipped. Fine-tuned checkpoint v1 is the config default (commit 5cb1237). The vendored WASB trainer turned out to be CUDA-locked (its `train_and_test` Hydra runner assumes a GPU-resident dataloader pipeline that doesn't run on this repo's CPU/MPS dev boxes), so it was replaced end-to-end with a repo-side corpus builder + trainer (`scripts/build_finetune_corpus.py`, `scripts/finetune_wasb.py`, `src/utils/ball_finetune_train.py`) that mirrors WASB's preprocessing and wBCE loss and holdout-evaluates every epoch. Measured results: coverage lift on origi01/origi02/kroupi01(holdout)/s013 and gberch union touch recall 0.250 → 0.625 with `foot_guided` + `touch_attribution` re-enabled — see `docs/superpowers/specs/2026-07-02-ball-stage-improvement-design.md` §4.3/§4.4 for the full tables.

## The asset: manual anchors are gold labels

The operator's `*_ball_anchors.json` records the **clicked ball pixel** per frame — exact ground-truth ball positions. `src/utils/ball_finetune_export.py` converts these to WASB's CVAT-style annotation XML (verified consumable by `third_party/wasb_sbdt/src/datasets/soccer.py:load_xml`). The format is identical to the soccer dataset, so training reuses the soccer loader with `dataset.root_dir` repointed — **no WASB-repo code changes**.

## Workflow

### 1. Label — and accumulate automatically (the flywheel)
Place/curate manual ball anchors via the web editor (`/ball-anchor-editor?shot=<clip>`). **Every save auto-appends** those gold labels to a growing WASB-format corpus at `output/ball_finetune/` (`POST /ball-anchors` → `ball_label_corpus.record_labels`; the save response reports `labels_recorded`). No extra step — the labelling you do for a good reconstruction *is* the training set, and it's targeted at the detector's failure frames (touches).

Check what's accumulated:
```bash
python scripts/ball_corpus.py status        # clips + total labels
```
For a real fine-tune, keep reconstructing clips until you have **hundreds–thousands** of labels (the status warns under 200). Labelling contiguous frames yields the densest training signal (WASB uses 3-frame stacks; only the centre frame needs a label, so sparse labels are fine — each is one sample).

### 2. Materialize frames (batch, before training)
Saving only writes labels (cheap). Extract the frames once before training:
```bash
python scripts/ball_corpus.py materialize --output ./output
# extracts frames/<clip>/*.png for every clip in the corpus manifest
```
(Or for a single clip outside the flywheel: `python scripts/export_ball_labels.py --clip-id <clip> --dataset-root ./output/ball_finetune`.)

### 3. Build the fine-tune corpus (repo-side, no GPU needed)
`scripts/build_finetune_corpus.py` extracts every clip frame plus a merged gold∪weak-label
annotation XML per clip (gold = operator anchors, weak = solved-track labels within
`--weak-window` frames of a gold anchor; gold wins on collision), and writes a manifest
recording the train/holdout split:
```bash
python scripts/build_finetune_corpus.py \
    --pairs output:gberch output-origi:origi01 \
            output-kroupi:kroupi01 output-japan:s013 \
    --corpus-root output/ball_finetune_corpus \
    --holdout kroupi01
```
Each `--pairs` entry is `OUTPUT_DIR:CLIP_ID`. `--holdout` names clips to exclude from
training and reserve for the epoch-by-epoch eval — kroupi01 above was never trained on,
which is why its coverage number in the measured results is meaningful signal, not just
in-sample fit. Useful flags: `--weak-window` (default 20), `--weak-min-conf` (default 0.5),
`--skip-frames` (reuse frames already materialized from a prior corpus build).

### 4. Fine-tune (repo-side trainer, no vendored-WASB CUDA path needed)
The vendored WASB Hydra trainer (`third_party/wasb_sbdt/src/main.py --config-name=train`)
turned out to be CUDA-locked and was **not used**. `scripts/finetune_wasb.py` drives a
repo-side harness (`src/utils/ball_finetune_train.py`) instead: it builds a train/val split
from the corpus manifest's `train` clips (random `val_frac` split, seed 0), evaluates
hit-rate on the manifest's `holdout` clips every epoch, and checkpoints by holdout hit-rate
(falling back to validation hit-rate when there's no holdout):
```bash
python scripts/finetune_wasb.py \
    --corpus-root output/ball_finetune_corpus \
    --run-dir output/ball_finetune_runs/run2 \
    --epochs 30 --batch 4 --lr 1e-4
```
Other flags: `--device` (`auto|cpu|cuda|mps`, default `auto` — note training prefers MPS on
macOS, unlike the detector's conservative cpu-only inference default), `--init` (checkpoint
to initialise from, default `wasb_soccer_best.pth.tar`), `--val-frac` (default 0.1),
`--limit-samples` (debug runs on a subset).

**Caveat validated 2026-07-04:** best-on-holdout selection is noisy on a small holdout split
and can pick epoch 0 (i.e. no training happened) as "best." The checkpoint that actually
shipped as v1 is the **last completed epoch's checkpoint** (`<run-dir>/last.pth.tar`, not the
holdout-argmin `best.pth.tar`) — inspect the run's `history.json`/progress log and use
judgement, don't blindly take whatever the harness calls "best."

### 5. Evaluate
Two harnesses, both usable without re-running the corpus build:
- **Detection coverage** per clip — re-run the ball stage (or `detection_coverage` sidecar
  inspection) with the candidate checkpoint swapped in via a config override (see below);
  compare `detection_coverage.total` in the diag sidecar / quality report against the stock
  baseline.
- **Touch recall (the real metric)** — forced `BallStage` runs plus
  `scripts/run_touch_recall_validation.py`, using a small checkpoint-override YAML merged
  over `config/default.yaml` (`src.pipeline.config.load_config` deep-merges it):
  ```yaml
  # /tmp/finetune_v1_checkpoint.yaml
  ball:
    wasb:
      checkpoint: output/ball_finetune_runs/run2/last.pth.tar
  ```
  ```bash
  python scripts/run_touch_recall_validation.py \
      --output output-gberch --shot gberch \
      --config /tmp/finetune_v1_checkpoint.yaml
  # --report-only re-prints the table from existing snapshots without a GPU run
  ```
  This runs the ball stage twice (kinematic-touch proposer off then on), snapshots
  `<shot>_ball_anchors_auto_{break_only,union}.json`, and prints the break-only /
  proposer-only / union recall table against the shot's manual anchors. To measure
  `foot_guided` / `touch_attribution` on top, add those keys to the same override YAML
  (`ball.foot_guided.enabled: true`, `ball.touch_attribution.enabled: true`) and re-run.

### 6. Promote the checkpoint
Once a candidate clears the acceptance bar (§4.3 of the improvement-design spec), copy it
into the tracked weights directory and flip the config default — this is exactly what
commit 5cb1237 did for v1:
```bash
cp output/ball_finetune_runs/run2/last.pth.tar \
   third_party/wasb_sbdt/pretrained_weights/wasb_soccer_finetuned_v1.pth.tar
```
```yaml
# config/default.yaml
ball:
  wasb:
    checkpoint: third_party/wasb_sbdt/pretrained_weights/wasb_soccer_finetuned_v1.pth.tar
    # checkpoint: third_party/wasb_sbdt/pretrained_weights/wasb_soccer_best.pth.tar  # stock fallback
```
With a reliable detector, the high-precision boosters are trustworthy again:
```yaml
ball:
  foot_guided:
    enabled: true   # ball-at-foot becomes trustworthy once detection is reliable
  touch_attribution:
    enabled: true   # relabelling now helps rather than overwrites correct labels
```

## What's built
- **Corpus + export:** `ball_finetune_export.anchors_to_cvat_xml` + `export_dataset`,
  `scripts/export_ball_labels.py`, `src/utils/ball_weak_labels.py`,
  `scripts/build_finetune_corpus.py` — all tested, validated against WASB's own `load_xml`.
- **Training:** `src/utils/ball_finetune_train.py` + `scripts/finetune_wasb.py` — repo-side,
  CPU/MPS/CUDA-portable, no vendored-WASB CUDA path required.
- **Evaluation:** `scripts/run_touch_recall_validation.py` (+ `scripts/ball_touch_recall_report.py`),
  `detection_coverage` sidecar/quality-report plumbing, `tests/test_ball_anchor_accuracy.py`.
- **Shipped:** fine-tuned checkpoint v1 promoted into `pretrained_weights/` and set as the
  config default (commit 5cb1237); `foot_guided` and `touch_attribution` re-enabled on top
  of it (this doc's history above).

[`project_ball_detection_rework`]: ../../../../.claude (memory)

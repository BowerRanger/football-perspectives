# Ball-detector fine-tuning — workflow scaffold

- **Date:** 2026-06-15
- **Why:** Inference-only touch detection is capped at 0/8 recall on gberch because the **ball detector** (WASB) confidently misdetects at touch frames and can't disambiguate the ball (see [`project_ball_detection_rework`] memory + `2026-06-15-ball-detection-direction-changes-design.md`). Everything downstream — direction-change segmentation, pose attribution, foot-guided zoom, body-pinning — is correct and proven on synthetic data, and the pose is excellent (feet ~15px from the true ball). The unlock is a reliable detector.
- **Status:** Scaffold. The label-export half is built, tested, and validated against WASB's own `load_xml`. Training itself needs labels at scale + a GPU and is the operator's step.

## The asset: manual anchors are gold labels

The operator's `*_ball_anchors.json` records the **clicked ball pixel** per frame — exact ground-truth ball positions. `src/utils/ball_finetune_export.py` converts these to WASB's CVAT-style annotation XML (verified consumable by `third_party/wasb_sbdt/src/datasets/soccer.py:load_xml`). The format is identical to the soccer dataset, so training reuses the soccer loader with `dataset.root_dir` repointed — **no WASB-repo code changes**.

## Workflow

### 1. Label
Place/curate manual ball anchors in `output/ball/<clip>_ball_anchors.json` via the web editor (`/ball-anchor-editor?shot=<clip>`). For a real fine-tune, label **several clips** and aim for **hundreds–thousands** of labelled frames — one clip's ~60 anchors is too thin to move the detector (the export script warns under 200). Densest yield comes from labelling contiguous frames (WASB trains on 3-frame stacks).

### 2. Export to WASB soccer format
```bash
python scripts/export_ball_labels.py --output ./output --clip-id gberch \
    --dataset-root ./output/ball_finetune
# repeat per clip; writes frames/<clip>/*.png + annos/<clip>.xml
```

### 3. Fine-tune (vendored WASB repo, GPU)
WASB is Hydra-configured (`third_party/wasb_sbdt/src/main.py`). Fine-tune = train with the soccer loader pointed at the export, **initialised from the pretrained checkpoint**:
```bash
cd third_party/wasb_sbdt/src
python3 main.py --config-name=train \
    dataset=soccer \
    dataset.root_dir=<repo>/output/ball_finetune \
    'dataset.train.videos=[gberch,origi01]' \
    'dataset.test.videos=[kroupi01]' \
    model=wasb \
    detector.model_path=../pretrained_weights/wasb_soccer_best.pth.tar \
    output_dir=<repo>/output/ball_finetune/runs
```
Notes:
- There is no `configs/train.yaml` in the vendored repo — create one mirroring `configs/eval.yaml` with `runner: train_and_test` (see `runners/`) plus a `loss` (e.g. `hm_wbce`) and `optimizer` (e.g. `adam_multistep`) from the existing `configs/loss|optimizer/`. The exact knobs (epochs, lr, frames_in=3) live in those configs.
- `detector.model_path` initialises from the soccer checkpoint → fine-tune, not train-from-scratch.

### 4. Evaluate
- Detector recall/precision: WASB eval on a held-out clip (`--config-name=eval … dataset.test.videos=[…] detector.model_path=<fine-tuned>`).
- Touch recall (the real metric): point the pipeline at the new checkpoint and re-run the gberch harness:
  ```python
  from src.utils.ball_touch_recall import touches_from_anchor_set, match_touches
  m = touches_from_anchor_set('/tmp/gberch_preregen_backup/gberch_ball_anchors.json')
  a = touches_from_anchor_set('output/ball/gberch_ball_anchors_auto.json')
  print(match_touches(m, a, frame_tol=2, require_bone=False))
  ```

### 5. Swap the checkpoint in
Point the pipeline at the fine-tuned weights and re-run the ball stage:
```yaml
# config/default.yaml
ball:
  wasb:
    checkpoint: <repo>/output/ball_finetune/runs/.../best.pth.tar
```
With a reliable detector, re-enable the high-precision booster:
```yaml
ball:
  foot_guided:
    enabled: true   # ball-at-foot becomes trustworthy once detection is reliable
```

## What's built vs. operator's step
- **Built + tested:** `ball_finetune_export.anchors_to_cvat_xml` + `export_dataset`; `scripts/export_ball_labels.py`; validated against WASB `load_xml`.
- **Operator's step:** curate labels at scale, author `configs/train.yaml`, run training on a GPU, eval, swap the checkpoint.
- **Then:** the already-built segmentation + pose-attribution + foot-guided stack should convert the now-reliable detections into touches at high recall.

[`project_ball_detection_rework`]: ../../../../.claude (memory)

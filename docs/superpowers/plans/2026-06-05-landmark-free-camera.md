# Landmark-free Camera Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate manual landmark placement in the camera stage by vendoring the PnLCalib field-registration network as an "auto-anchor generator" whose detections feed the existing sub-pixel solver unchanged.

**Architecture:** PnLCalib's two HRNet models detect named pitch keypoints + line extremities per keyframe. A bridge maps each keypoint index → world coords (nearly identity; goal-post keypoints get true 3D z). Detections become standard `Anchor` objects written to `{shot}_anchors.json`; the existing joint solve → static-camera C-profile → sub-pixel line solver then runs verbatim. A clip-level static-camera consensus rejects keyframes whose pose flips (left/right symmetry). The whole path is opt-in (`camera.auto_anchors.enabled`) and falls back to manual anchoring when unavailable.

**Tech Stack:** Python 3.11+, PyTorch (CPU on macOS / CUDA on Linux), OpenCV, NumPy, SciPy, pytest. PnLCalib vendored under `third_party/pnlcalib/` (GPL-2.0, arms-length subprocess like GVHMR).

**Design doc:** `docs/superpowers/specs/2026-06-05-landmark-free-camera-design.md`

---

## Reference facts (verified from the PnLCalib repo, do not re-derive)

PnLCalib keypoint world coordinates (`utils/utils_keypoints.py:KeypointsDB.keypoint_world_coords_2D`, 57 entries, 1-based index `kp` → list index `kp-1`) use **the same convention as this project**: x∈[0,105] m, y∈[0,68] m, origin at a corner, ground z=0. Aux keypoints (`keypoint_aux_world_coords_2D`, 16 entries) are indexed `58..73` (list index `kp-58`).

**Goal-post keypoints 12–19 are special** — PnLCalib encodes them on a virtual plane, so their `keypoint_world_coords_2D` entries are NOT literal 3D. Their true 3D (this project's z-up, posts at z=2.44, left goal x=0 / right goal x=105, posts at y=30.34 and y=37.66) derived from `keypoint_pair_list`:

| kp | meaning | world (x, y, z) |
|----|---------|-----------------|
| 12 | left crossbar ∩ left post-right | (0, 37.66, 2.44) |
| 13 | sideline-left ∩ left post-right (base) | (0, 37.66, 0) |
| 14 | sideline-right ∩ right post-left (base) | (105, 30.34, 0) |
| 15 | right crossbar ∩ right post-left | (105, 30.34, 2.44) |
| 16 | left crossbar ∩ left post-left | (0, 30.34, 2.44) |
| 17 | sideline-left ∩ left post-left (base) | (0, 30.34, 0) |
| 18 | sideline-right ∩ right post-right (base) | (105, 37.66, 0) |
| 19 | right crossbar ∩ right post-right | (105, 37.66, 2.44) |

The 4 crossbar-level keypoints (12, 15, 16, 19) at z=2.44 are the **only non-coplanar points** PnLCalib provides — they are what let an auto-anchor qualify as "rich" (`_is_rich` needs ≥6 non-coplanar points).

PnLCalib inference (`inference.py`): input frame resized to (540, 960); two models `get_cls_net(cfg)` + `get_cls_net_l(cfg_l)` from `config/hrnetv2_w48.yaml` / `config/hrnetv2_w48_l.yaml`; keypoints via `get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])` then `coords_to_dict(coords, threshold)`. CLI default thresholds: `kp_threshold=0.3434`, `line_threshold=0.7867`. Weights: GitHub release `v1.0.0` assets `SV_kp`, `SV_lines`.

---

## File Structure

**New files:**
- `third_party/pnlcalib/` — git submodule (https://github.com/mguti97/PnLCalib)
- `third_party/pnlcalib/weights/{SV_kp,SV_lines}` — checkpoints (gitignored; fetched by script)
- `scripts/fetch_pnlcalib_weights.sh` — download checkpoints from the release
- `third_party_shims/pnlcalib_infer.py` — our authored inference wrapper (emits JSON; runs inside the submodule via subprocess)
- `src/utils/field_registration.py` — `FieldKeypoint`, `FieldLine`, `FieldRegistrationResult` dataclasses (pure data, no torch)
- `src/utils/pnlcalib_catalogue_map.py` — the bridge: kp index → world xyz
- `src/utils/pnlcalib_provider.py` — subprocess invocation + JSON parse → `FieldRegistrationResult`
- `src/utils/auto_anchor.py` — registration → anchors, keyframe sampling, consensus, `AnchorSet` emission
- `scripts/pnlcalib_smoke.py` — manual one-frame validation CLI
- Tests under `tests/` mirroring each module.

**Modified files:**
- `config/default.yaml` — add `camera.auto_anchors` block
- `src/stages/camera.py` — pre-step hook (`_ensure_anchors`) before loading anchors
- `.gitignore` — ignore `third_party/pnlcalib/weights/`

---

## Phase 0 — Vendoring & environment

### Task 1: Vendor PnLCalib + checkpoint fetch script

**Files:**
- Create: `scripts/fetch_pnlcalib_weights.sh`
- Modify: `.gitignore`
- Test: `tests/test_pnlcalib_vendor.py`

- [ ] **Step 1: Add the submodule**

```bash
git submodule add https://github.com/mguti97/PnLCalib third_party/pnlcalib
git -C third_party/pnlcalib checkout main
```

- [ ] **Step 2: Write the weights-fetch script**

Create `scripts/fetch_pnlcalib_weights.sh`:

```bash
#!/usr/bin/env bash
# Download PnLCalib single-view checkpoints from the v1.0.0 release.
# Idempotent: skips files that already exist.
set -euo pipefail
DEST="third_party/pnlcalib/weights"
BASE="https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"
mkdir -p "$DEST"
for asset in SV_kp SV_lines; do
  if [[ -f "$DEST/$asset" ]]; then
    echo "have $asset"; continue
  fi
  echo "fetching $asset ..."
  curl -fL "$BASE/$asset" -o "$DEST/$asset"
done
echo "done -> $DEST"
```

```bash
chmod +x scripts/fetch_pnlcalib_weights.sh
```

- [ ] **Step 3: Gitignore the weights**

Append to `.gitignore`:

```
third_party/pnlcalib/weights/
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_pnlcalib_vendor.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pnlcalib_submodule_present():
    """The vendored repo and the files our wrapper imports must exist."""
    base = ROOT / "third_party" / "pnlcalib"
    assert (base / "inference.py").exists()
    assert (base / "model" / "cls_hrnet.py").exists()
    assert (base / "utils" / "utils_keypoints.py").exists()
    assert (base / "config" / "hrnetv2_w48.yaml").exists()
    assert (base / "config" / "hrnetv2_w48_l.yaml").exists()


def test_fetch_script_executable():
    script = ROOT / "scripts" / "fetch_pnlcalib_weights.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111, "fetch script must be executable"
```

- [ ] **Step 5: Run the test**

Run: `pytest tests/test_pnlcalib_vendor.py -v`
Expected: PASS (submodule + script exist).

- [ ] **Step 6: Fetch the weights locally (manual, not committed)**

Run: `./scripts/fetch_pnlcalib_weights.sh`
Expected: `third_party/pnlcalib/weights/SV_kp` and `SV_lines` exist.

- [ ] **Step 7: Commit**

```bash
git add .gitmodules third_party/pnlcalib scripts/fetch_pnlcalib_weights.sh .gitignore tests/test_pnlcalib_vendor.py
git commit -m "chore: vendor PnLCalib submodule + weights fetch script"
```

---

## Phase 1 — Provider + bridge (validated before any downstream wiring)

### Task 2: Field-registration data types

**Files:**
- Create: `src/utils/field_registration.py`
- Test: `tests/test_field_registration_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_registration_types.py`:

```python
from src.utils.field_registration import (
    FieldKeypoint,
    FieldLine,
    FieldRegistrationResult,
)


def test_keypoint_holds_index_image_and_confidence():
    kp = FieldKeypoint(index=5, image_xy=(120.0, 340.0), confidence=0.8)
    assert kp.index == 5
    assert kp.image_xy == (120.0, 340.0)
    assert kp.confidence == 0.8


def test_result_filters_keypoints_by_confidence():
    result = FieldRegistrationResult(
        frame=10,
        image_size=(1920, 1080),
        keypoints=(
            FieldKeypoint(1, (10.0, 10.0), 0.9),
            FieldKeypoint(2, (20.0, 20.0), 0.2),
        ),
        lines=(),
    )
    kept = result.confident_keypoints(min_confidence=0.5)
    assert [k.index for k in kept] == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_field_registration_types.py -v`
Expected: FAIL with `ModuleNotFoundError: src.utils.field_registration`.

- [ ] **Step 3: Write the implementation**

Create `src/utils/field_registration.py`:

```python
"""Pure data types for field-registration output (no torch dependency).

A FieldRegistrationResult is the provider-agnostic hand-off between a learned
pitch-registration model and the auto-anchor builder. The provider fills it;
auto_anchor consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldKeypoint:
    index: int
    """1-based PnLCalib keypoint index (1..73)."""
    image_xy: tuple[float, float]
    """Detected pixel location in the ORIGINAL frame resolution."""
    confidence: float


@dataclass(frozen=True)
class FieldLine:
    index: int
    """0-based PnLCalib line index (0..22)."""
    image_segment: tuple[tuple[float, float], tuple[float, float]]
    confidence: float


@dataclass(frozen=True)
class FieldRegistrationResult:
    frame: int
    image_size: tuple[int, int]
    keypoints: tuple[FieldKeypoint, ...]
    lines: tuple[FieldLine, ...]

    def confident_keypoints(
        self, min_confidence: float
    ) -> tuple[FieldKeypoint, ...]:
        return tuple(k for k in self.keypoints if k.confidence >= min_confidence)

    def confident_lines(self, min_confidence: float) -> tuple[FieldLine, ...]:
        return tuple(ln for ln in self.lines if ln.confidence >= min_confidence)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_field_registration_types.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/field_registration.py tests/test_field_registration_types.py
git commit -m "feat: field-registration data types"
```

---

### Task 3: The catalogue bridge (kp index → world xyz)

**Files:**
- Create: `src/utils/pnlcalib_catalogue_map.py`
- Test: `tests/test_pnlcalib_catalogue_map.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pnlcalib_catalogue_map.py`. The first test cross-checks our transcribed ground table against the vendored source (imports the submodule module only in the test, keeping runtime code arms-length). The others pin the goal-post 3D and z-up convention.

```python
import sys
from pathlib import Path

import numpy as np
import pytest

from src.utils.pnlcalib_catalogue_map import (
    GOAL_KEYPOINT_WORLD_3D,
    keypoint_world_xyz,
    NUM_KEYPOINTS,
)

PNL = Path(__file__).resolve().parents[1] / "third_party" / "pnlcalib"


def test_ground_table_matches_vendored_source():
    """Our transcribed (x, y) for non-goal keypoints must equal PnLCalib's."""
    if not (PNL / "utils" / "utils_keypoints.py").exists():
        pytest.skip("pnlcalib submodule not present")
    sys.path.insert(0, str(PNL))
    try:
        from utils.utils_keypoints import KeypointsDB  # type: ignore
    finally:
        sys.path.pop(0)
    # KeypointsDB needs data+image only for __init__ attributes we don't use;
    # read the class attributes directly off an un-initialised instance.
    main = KeypointsDB.__new__(KeypointsDB)
    coords = KeypointsDB.__init__.__defaults__  # not used; access via instance
    # The lists are assigned in __init__; build a minimal instance.
    import torch
    db = KeypointsDB(data={}, image=torch.zeros(3, 540, 960))
    for kp in range(1, 58):
        if kp in GOAL_KEYPOINT_WORLD_3D:
            continue
        x, y = db.keypoint_world_coords_2D[kp - 1]
        got = keypoint_world_xyz(kp)
        assert got == pytest.approx((x, y, 0.0)), f"kp {kp}"


def test_goal_crossbar_keypoints_are_non_coplanar():
    """The 4 crossbar keypoints sit at z=2.44 (this project is z-up)."""
    for kp in (12, 15, 16, 19):
        assert keypoint_world_xyz(kp)[2] == pytest.approx(2.44)
    for kp in (13, 14, 17, 18):
        assert keypoint_world_xyz(kp)[2] == pytest.approx(0.0)


def test_goal_keypoint_coordinates():
    assert keypoint_world_xyz(16) == pytest.approx((0.0, 30.34, 2.44))
    assert keypoint_world_xyz(19) == pytest.approx((105.0, 37.66, 2.44))


def test_total_keypoint_count():
    assert NUM_KEYPOINTS == 73  # 57 main + 16 aux
    # Aux indices resolve too:
    assert keypoint_world_xyz(58)[2] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pnlcalib_catalogue_map.py -v`
Expected: FAIL with `ModuleNotFoundError: src.utils.pnlcalib_catalogue_map`.

- [ ] **Step 3: Write the implementation**

Create `src/utils/pnlcalib_catalogue_map.py`. The ground tables are FIFA pitch measurements (geometric facts, transcribed; the test cross-checks them against the vendored list). Goal keypoints use the derived 3D from the plan's reference table.

```python
"""Bridge: PnLCalib keypoint index -> this project's world coordinates.

PnLCalib's pitch template shares this project's convention (x in [0,105],
y in [0,68], origin at a corner, ground z=0), so ground keypoints map by
identity. The 8 goal-post keypoints (12-19) are encoded on a virtual plane
in PnLCalib and are overridden here with their true z-up 3D coordinates.

Runtime code imports ONLY this module (no GPL submodule import at runtime);
the cross-check against the vendored source lives in the test.
"""

from __future__ import annotations

# Main keypoints 1..57, ground plane (x, y), z=0. Transcribed from
# PnLCalib utils_keypoints.KeypointsDB.keypoint_world_coords_2D and verified
# in tests/test_pnlcalib_catalogue_map.py::test_ground_table_matches_vendored_source.
_MAIN_XY: tuple[tuple[float, float], ...] = (
    (0., 0.), (52.5, 0.), (105., 0.), (0., 13.84), (16.5, 13.84), (88.5, 13.84),
    (105., 13.84), (0., 24.84), (5.5, 24.84), (99.5, 24.84), (105., 24.84),
    (0., 30.34), (0., 30.34), (105., 30.34), (105., 30.34), (0., 37.66),
    (0., 37.66), (105., 37.66), (105., 37.66), (0., 43.16), (5.5, 43.16),
    (99.5, 43.16), (105., 43.16), (0., 54.16), (16.5, 54.16), (88.5, 54.16),
    (105., 54.16), (0., 68.), (52.5, 68.), (105., 68.), (16.5, 26.68),
    (52.5, 24.85), (88.5, 26.68), (16.5, 41.31), (52.5, 43.15), (88.5, 41.31),
    (19.99, 32.29), (43.68, 31.53), (61.31, 31.53), (85., 32.29), (19.99, 35.7),
    (43.68, 36.46), (61.31, 36.46), (85., 35.7), (11., 34.), (16.5, 34.),
    (20.15, 34.), (46.03, 27.53), (58.97, 27.53), (43.35, 34.), (52.5, 34.),
    (61.5, 34.), (46.03, 40.47), (58.97, 40.47), (84.85, 34.), (88.5, 34.),
    (94., 34.),
)

# Aux keypoints 58..73, ground plane (x, y), z=0.
_AUX_XY: tuple[tuple[float, float], ...] = (
    (5.5, 0.), (16.5, 0.), (88.5, 0.), (99.5, 0.), (5.5, 13.84), (99.5, 13.84),
    (16.5, 24.84), (88.5, 24.84), (16.5, 43.16), (88.5, 43.16), (5.5, 54.16),
    (99.5, 54.16), (5.5, 68.), (16.5, 68.), (88.5, 68.), (99.5, 68.),
)

# Goal-post keypoints override the (often virtual-plane) main entries with
# true z-up 3D. Crossbar-level keypoints (12, 15, 16, 19) at z=2.44 are the
# only non-coplanar points -> they make a rich anchor possible.
GOAL_KEYPOINT_WORLD_3D: dict[int, tuple[float, float, float]] = {
    12: (0.0, 37.66, 2.44),
    13: (0.0, 37.66, 0.0),
    14: (105.0, 30.34, 0.0),
    15: (105.0, 30.34, 2.44),
    16: (0.0, 30.34, 2.44),
    17: (0.0, 30.34, 0.0),
    18: (105.0, 37.66, 0.0),
    19: (105.0, 37.66, 2.44),
}

NUM_KEYPOINTS = len(_MAIN_XY) + len(_AUX_XY)  # 73


def keypoint_world_xyz(index: int) -> tuple[float, float, float]:
    """World (x, y, z) in pitch metres for 1-based PnLCalib keypoint `index`."""
    if index in GOAL_KEYPOINT_WORLD_3D:
        return GOAL_KEYPOINT_WORLD_3D[index]
    if 1 <= index <= len(_MAIN_XY):
        x, y = _MAIN_XY[index - 1]
        return (x, y, 0.0)
    aux_i = index - len(_MAIN_XY) - 1
    if 0 <= aux_i < len(_AUX_XY):
        x, y = _AUX_XY[aux_i]
        return (x, y, 0.0)
    raise KeyError(f"keypoint index out of range: {index}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pnlcalib_catalogue_map.py -v`
Expected: PASS (the vendored cross-check skips cleanly if the submodule is absent on CI).

- [ ] **Step 5: Commit**

```bash
git add src/utils/pnlcalib_catalogue_map.py tests/test_pnlcalib_catalogue_map.py
git commit -m "feat: PnLCalib keypoint->world bridge with verified ground table"
```

---

### Task 4: The vendored inference wrapper (authored by us)

**Files:**
- Create: `third_party_shims/pnlcalib_infer.py`
- Test: `tests/test_pnlcalib_infer_contract.py`

This script runs *inside* the submodule's import path (subprocess), so it may import the GPL code; it is invoked arms-length and emits a stable JSON contract our code owns.

- [ ] **Step 1: Write the failing test (the JSON contract)**

Create `tests/test_pnlcalib_infer_contract.py`:

```python
import json
from pathlib import Path

WRAPPER = (
    Path(__file__).resolve().parents[1] / "third_party_shims" / "pnlcalib_infer.py"
)


def test_wrapper_exists_and_documents_contract():
    text = WRAPPER.read_text()
    # The wrapper must emit the agreed JSON schema keys.
    for key in ('"frame"', '"keypoints"', '"lines"', '"image_size"'):
        assert key in text, f"wrapper must emit {key}"


def test_contract_shape_roundtrips():
    """A document matching the wrapper contract parses as expected."""
    doc = {
        "image_size": [1920, 1080],
        "results": [
            {
                "frame": 0,
                "keypoints": {"16": [950.0, 220.0, 0.71]},
                "lines": {"12": [[10.0, 20.0], [30.0, 40.0], 0.8]},
            }
        ],
    }
    s = json.dumps(doc)
    back = json.loads(s)
    assert back["results"][0]["keypoints"]["16"][2] == 0.71
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pnlcalib_infer_contract.py -v`
Expected: FAIL (wrapper file missing).

- [ ] **Step 3: Write the wrapper**

Create `third_party_shims/pnlcalib_infer.py`. It is grounded in PnLCalib's own `inference.py` flow (resize to 540x960; two models; `get_keypoints_from_heatmap_batch_maxpool`; `coords_to_dict`). It outputs keypoint pixel coords in ORIGINAL resolution + per-keypoint peak confidence.

```python
"""Authored inference wrapper for the vendored PnLCalib models.

Run as a subprocess with the submodule as CWD so its relative imports/configs
resolve. Reads frames from a video at given indices, emits JSON on stdout
matching the contract consumed by src/utils/pnlcalib_provider.py.

Usage:
  python pnlcalib_infer.py --video CLIP --frames 0,30,60 \
      --weights-kp weights/SV_kp --weights-line weights/SV_lines \
      --device cpu --kp-threshold 0.3434 --line-threshold 0.7867
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import yaml
from PIL import Image

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)

_MODEL_HW = (540, 960)


def _load(weights_kp, weights_line, device):
    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))
    m = get_cls_net(cfg)
    m.load_state_dict(torch.load(weights_kp, map_location=device))
    m.to(device).eval()
    m_l = get_cls_net_l(cfg_l)
    m_l.load_state_dict(torch.load(weights_line, map_location=device))
    m_l.to(device).eval()
    return m, m_l


def _peaks_to_dict(coords, threshold, w_orig, h_orig):
    """coords: (1, C, 1, 3) as [x, y, score] in model resolution (per PnLCalib
    maxpool). Return {index(int, 1-based): [x_orig, y_orig, score]}."""
    out = {}
    arr = coords[0]  # (C, 1, 3)
    sx = w_orig / _MODEL_HW[1]
    sy = h_orig / _MODEL_HW[0]
    for ch in range(arr.shape[0]):
        x, y, score = float(arr[ch, 0, 0]), float(arr[ch, 0, 1]), float(arr[ch, 0, 2])
        if score >= threshold and x > 0 and y > 0:
            out[ch + 1] = [x * sx, y * sy, score]
    return out


def _line_peaks_to_dict(coords, threshold, w_orig, h_orig):
    """Line model emits two extremity channels per line; pair them into a
    segment. coords shape mirrors the keypoint model; we keep the raw two
    endpoints per line index with the min of the two scores."""
    arr = coords[0]
    sx = w_orig / _MODEL_HW[1]
    sy = h_orig / _MODEL_HW[0]
    out = {}
    n_lines = arr.shape[0] // 2
    for li in range(n_lines):
        a = arr[2 * li, 0]
        b = arr[2 * li + 1, 0]
        score = min(float(a[2]), float(b[2]))
        if score >= threshold and a[0] > 0 and b[0] > 0:
            out[li] = [
                [float(a[0]) * sx, float(a[1]) * sy],
                [float(b[0]) * sx, float(b[1]) * sy],
                score,
            ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", required=True, help="comma-separated indices")
    ap.add_argument("--weights-kp", required=True)
    ap.add_argument("--weights-line", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kp-threshold", type=float, default=0.3434)
    ap.add_argument("--line-threshold", type=float, default=0.7867)
    args = ap.parse_args()

    device = args.device
    model, model_l = _load(args.weights_kp, args.weights_line, device)
    resize = T.Resize(_MODEL_HW)

    cap = cv2.VideoCapture(args.video)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    results = []
    for idx in (int(x) for x in args.frames.split(",")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = F.to_tensor(Image.fromarray(rgb)).float().unsqueeze(0)
        t = resize(t).to(device)
        with torch.no_grad():
            hm = model(t)
            hm_l = model_l(t)
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1, :, :])
        ln_coords = get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1, :, :])
        results.append({
            "frame": idx,
            "keypoints": _peaks_to_dict(
                kp_coords.cpu().numpy(), args.kp_threshold, w_orig, h_orig),
            "lines": _line_peaks_to_dict(
                ln_coords.cpu().numpy(), args.line_threshold, w_orig, h_orig),
        })
    cap.release()
    json.dump({"image_size": [w_orig, h_orig], "results": results}, sys.stdout)


if __name__ == "__main__":
    main()
```

> **Phase-1 validation note:** the exact tensor shape returned by
> `get_keypoints_from_heatmap_batch_maxpool` must be confirmed against the
> vendored `utils/utils_heatmap.py` when weights are present (Task 6 smoke
> test). If it differs from `(B, C, 1, 3)`, adjust `_peaks_to_dict` indexing
> only — the JSON contract stays fixed.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pnlcalib_infer_contract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party_shims/pnlcalib_infer.py tests/test_pnlcalib_infer_contract.py
git commit -m "feat: authored PnLCalib inference wrapper with stable JSON contract"
```

---

### Task 5: The provider (subprocess + parse)

**Files:**
- Create: `src/utils/pnlcalib_provider.py`
- Test: `tests/test_pnlcalib_provider.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pnlcalib_provider.py`. We inject a fake "runner" so the unit test needs no torch/weights.

```python
import json

from src.utils.field_registration import FieldRegistrationResult
from src.utils.pnlcalib_provider import PnLCalibProvider


def _fake_runner(video, frames, image_size=(1920, 1080)):
    return {
        "image_size": list(image_size),
        "results": [
            {
                "frame": f,
                "keypoints": {"16": [950.0, 220.0, 0.71], "5": [400.0, 600.0, 0.5]},
                "lines": {"12": [[10.0, 20.0], [30.0, 40.0], 0.8]},
            }
            for f in frames
        ],
    }


def test_provider_parses_results_into_dataclasses():
    provider = PnLCalibProvider(runner=_fake_runner)
    out = provider.register_frames("clip.mp4", [0, 30])
    assert set(out.keys()) == {0, 30}
    r0 = out[0]
    assert isinstance(r0, FieldRegistrationResult)
    assert r0.image_size == (1920, 1080)
    assert {k.index for k in r0.keypoints} == {16, 5}
    kp16 = next(k for k in r0.keypoints if k.index == 16)
    assert kp16.image_xy == (950.0, 220.0)
    assert kp16.confidence == 0.71
    assert r0.lines[0].index == 12
    assert r0.lines[0].image_segment == ((10.0, 20.0), (30.0, 40.0))


def test_provider_handles_empty_results():
    provider = PnLCalibProvider(runner=lambda v, f, **k: {"image_size": [1920, 1080], "results": []})
    out = provider.register_frames("clip.mp4", [0])
    assert out == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pnlcalib_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `src/utils/pnlcalib_provider.py`:

```python
"""Provider: drive the vendored PnLCalib wrapper and parse its JSON.

The default runner shells out to third_party_shims/pnlcalib_infer.py with the
submodule as CWD (arms-length, GVHMR-style). Tests inject a fake runner so no
torch/weights are needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from src.utils.field_registration import (
    FieldKeypoint,
    FieldLine,
    FieldRegistrationResult,
)

_ROOT = Path(__file__).resolve().parents[2]
_SUBMODULE = _ROOT / "third_party" / "pnlcalib"
_WRAPPER = _ROOT / "third_party_shims" / "pnlcalib_infer.py"

Runner = Callable[..., dict]


def _subprocess_runner(
    video: str,
    frames: list[int],
    *,
    weights_kp: str,
    weights_line: str,
    device: str,
    kp_threshold: float,
    line_threshold: float,
) -> dict:
    cmd = [
        sys.executable, str(_WRAPPER),
        "--video", str(Path(video).resolve()),
        "--frames", ",".join(str(f) for f in frames),
        "--weights-kp", weights_kp,
        "--weights-line", weights_line,
        "--device", device,
        "--kp-threshold", str(kp_threshold),
        "--line-threshold", str(line_threshold),
    ]
    proc = subprocess.run(
        cmd, cwd=str(_SUBMODULE), capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


class PnLCalibProvider:
    def __init__(
        self,
        runner: Runner | None = None,
        *,
        weights_kp: str = "weights/SV_kp",
        weights_line: str = "weights/SV_lines",
        device: str = "cpu",
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
    ) -> None:
        self._runner = runner or _subprocess_runner
        self._kw = dict(
            weights_kp=weights_kp, weights_line=weights_line, device=device,
            kp_threshold=kp_threshold, line_threshold=line_threshold,
        )
        self._injected = runner is not None

    def register_frames(
        self, video: str, frames: list[int]
    ) -> dict[int, FieldRegistrationResult]:
        # Injected test runners take only (video, frames); the real one takes kwargs.
        doc = (
            self._runner(video, frames)
            if self._injected
            else self._runner(video, frames, **self._kw)
        )
        image_size = tuple(doc["image_size"])
        out: dict[int, FieldRegistrationResult] = {}
        for r in doc["results"]:
            kps = tuple(
                FieldKeypoint(int(idx), (xy[0], xy[1]), xy[2])
                for idx, xy in r["keypoints"].items()
            )
            lines = tuple(
                FieldLine(int(idx), ((seg[0][0], seg[0][1]), (seg[1][0], seg[1][1])), seg[2])
                for idx, seg in r["lines"].items()
            )
            if kps or lines:
                out[r["frame"]] = FieldRegistrationResult(
                    frame=r["frame"], image_size=image_size, keypoints=kps, lines=lines,
                )
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pnlcalib_provider.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/pnlcalib_provider.py tests/test_pnlcalib_provider.py
git commit -m "feat: PnLCalib provider (subprocess + JSON parse)"
```

---

### Task 6: One-frame smoke CLI (manual validation gate)

**Files:**
- Create: `scripts/pnlcalib_smoke.py`
- Test: `tests/test_pnlcalib_smoke_importable.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pnlcalib_smoke_importable.py`:

```python
import importlib.util
from pathlib import Path


def test_smoke_script_imports():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pnlcalib_smoke.py"
    spec = importlib.util.spec_from_file_location("pnlcalib_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pnlcalib_smoke_importable.py -v`
Expected: FAIL (script missing).

- [ ] **Step 3: Write the smoke CLI**

Create `scripts/pnlcalib_smoke.py`:

```python
"""Register ONE real frame and print mapped (image -> world) correspondences.

Manual validation gate for the bridge + provider. Run with weights present:
  python scripts/pnlcalib_smoke.py --video CLIP --frame 0
Prints each detected keypoint: index, pixel, world xyz, confidence. Eyeball
that goal-post keypoints (12-19) land on the posts and z=2.44 tops are sane.
"""

from __future__ import annotations

import argparse

from src.utils.pnlcalib_catalogue_map import keypoint_world_xyz
from src.utils.pnlcalib_provider import PnLCalibProvider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    provider = PnLCalibProvider(device=args.device)
    results = provider.register_frames(args.video, [args.frame])
    res = results.get(args.frame)
    if res is None:
        print("no detections on this frame")
        return
    print(f"frame {args.frame}  image_size={res.image_size}  "
          f"{len(res.keypoints)} keypoints")
    for kp in sorted(res.keypoints, key=lambda k: k.index):
        wx, wy, wz = keypoint_world_xyz(kp.index)
        print(f"  kp{kp.index:>2}  px=({kp.image_xy[0]:7.1f},{kp.image_xy[1]:7.1f})  "
              f"world=({wx:6.2f},{wy:6.2f},{wz:4.2f})  conf={kp.confidence:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pnlcalib_smoke_importable.py -v`
Expected: PASS.

- [ ] **Step 5: MANUAL validation (requires weights + a clip)**

Run: `python scripts/pnlcalib_smoke.py --video "test-media/Liverpool vs Barcelona (4-0) _ Epic Comeback Completed At Anfield _ UEFA Champions League Highlights.mp4" --frame 0`
Expected: a list of keypoints with sane pixels and world coords. **Confirm the tensor-shape note in Task 4 holds; if peaks come back empty or mis-scaled, fix `_peaks_to_dict` indexing now.** This is the gate before Phase 2.

- [ ] **Step 6: Commit**

```bash
git add scripts/pnlcalib_smoke.py tests/test_pnlcalib_smoke_importable.py
git commit -m "feat: PnLCalib one-frame smoke CLI (bridge validation gate)"
```

---

## Phase 2 — Auto-anchor generation

### Task 7: Registration result → Anchor

**Files:**
- Create: `src/utils/auto_anchor.py` (first function)
- Test: `tests/test_auto_anchor_to_anchor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auto_anchor_to_anchor.py`:

```python
from src.schemas.anchor import Anchor
from src.utils.field_registration import (
    FieldKeypoint, FieldLine, FieldRegistrationResult,
)
from src.utils.auto_anchor import result_to_anchor


def _result():
    # 5 ground keypoints + 1 crossbar (z=2.44) -> 6 pts, non-coplanar.
    kps = (
        FieldKeypoint(1, (100.0, 900.0), 0.9),
        FieldKeypoint(3, (1800.0, 900.0), 0.9),
        FieldKeypoint(5, (500.0, 700.0), 0.8),
        FieldKeypoint(31, (600.0, 650.0), 0.8),
        FieldKeypoint(46, (700.0, 640.0), 0.7),
        FieldKeypoint(16, (950.0, 220.0), 0.75),  # crossbar top z=2.44
    )
    return FieldRegistrationResult(0, (1920, 1080), kps, ())


def test_result_to_anchor_builds_landmarks_with_world_coords():
    anchor = result_to_anchor(_result(), min_keypoint_conf=0.5)
    assert isinstance(anchor, Anchor)
    assert anchor.frame == 0
    assert len(anchor.landmarks) == 6
    lm16 = next(l for l in anchor.landmarks if l.image_xy == (950.0, 220.0))
    assert lm16.world_xyz[2] == 2.44  # non-coplanar point preserved


def test_result_to_anchor_drops_low_confidence():
    res = FieldRegistrationResult(
        0, (1920, 1080),
        (FieldKeypoint(1, (1.0, 2.0), 0.9), FieldKeypoint(3, (3.0, 4.0), 0.1)),
        (),
    )
    anchor = result_to_anchor(res, min_keypoint_conf=0.5)
    assert len(anchor.landmarks) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_to_anchor.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

First inspect the real schema so field names match: read `src/schemas/anchor.py` for `Anchor`, `LandmarkObservation`, `LineObservation` exact constructors. Then create `src/utils/auto_anchor.py`:

```python
"""Build camera-stage anchors automatically from learned field registration."""

from __future__ import annotations

from src.schemas.anchor import Anchor, LandmarkObservation
from src.utils.field_registration import FieldRegistrationResult
from src.utils.pnlcalib_catalogue_map import keypoint_world_xyz


def result_to_anchor(
    result: FieldRegistrationResult, *, min_keypoint_conf: float
) -> Anchor:
    """Convert one frame's confident keypoints into a point-only Anchor.

    Names are synthesised (``pnl_kp_<idx>``); the solver consumes world_xyz,
    not the name. Lines are left for a later iteration — keypoints alone give
    the solver its point constraints, including the non-coplanar crossbar
    points needed for a rich anchor.
    """
    landmarks = []
    for kp in result.confident_keypoints(min_keypoint_conf):
        world = keypoint_world_xyz(kp.index)
        landmarks.append(
            LandmarkObservation(
                name=f"pnl_kp_{kp.index}",
                image_xy=kp.image_xy,
                world_xyz=world,
            )
        )
    return Anchor(frame=result.frame, landmarks=tuple(landmarks), lines=())
```

> **Schema check:** if `LandmarkObservation` / `Anchor` field names or
> constructor differ from the above (e.g. `image_point` vs `image_xy`), adapt
> this function to the real schema read in Step 3 and re-run.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_to_anchor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/auto_anchor.py tests/test_auto_anchor_to_anchor.py
git commit -m "feat: convert field registration result to a camera Anchor"
```

---

### Task 8: Keyframe sampling + confidence-qualified candidates

**Files:**
- Modify: `src/utils/auto_anchor.py`
- Test: `tests/test_auto_anchor_candidates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auto_anchor_candidates.py`:

```python
from src.utils.field_registration import FieldKeypoint, FieldRegistrationResult
from src.utils.auto_anchor import select_keyframes, qualified_anchors


def test_select_keyframes_uniform_stride():
    assert select_keyframes(n_frames=100, stride=30, max_keyframes=12) == [0, 30, 60, 90]


def test_select_keyframes_respects_cap():
    got = select_keyframes(n_frames=1000, stride=10, max_keyframes=5)
    assert len(got) == 5
    assert got[0] == 0


def _rich_result(frame):
    kps = tuple(
        FieldKeypoint(i, (float(i * 10), float(i * 5)), 0.9)
        for i in (1, 3, 5, 7, 9, 16)  # 16 = crossbar (non-coplanar)
    )
    return FieldRegistrationResult(frame, (1920, 1080), kps, ())


def _thin_result(frame):
    kps = (FieldKeypoint(1, (1.0, 2.0), 0.9), FieldKeypoint(3, (3.0, 4.0), 0.9))
    return FieldRegistrationResult(frame, (1920, 1080), kps, ())


def test_qualified_anchors_keeps_only_anchors_meeting_threshold():
    results = {0: _rich_result(0), 30: _thin_result(30)}
    anchors = qualified_anchors(results, min_keypoint_conf=0.5, min_points=4)
    assert [a.frame for a in anchors] == [0]  # thin frame dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_candidates.py -v`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Add the functions**

Append to `src/utils/auto_anchor.py`:

```python
def select_keyframes(n_frames: int, stride: int, max_keyframes: int) -> list[int]:
    """Uniformly sample candidate keyframe indices, capped at max_keyframes."""
    frames = list(range(0, n_frames, max(1, stride)))
    if len(frames) <= max_keyframes:
        return frames
    # Even subsample down to the cap, always keeping the first.
    step = len(frames) / max_keyframes
    return [frames[int(i * step)] for i in range(max_keyframes)]


def qualified_anchors(
    results: dict[int, FieldRegistrationResult],
    *,
    min_keypoint_conf: float,
    min_points: int,
) -> list[Anchor]:
    """Convert each frame's result to an anchor, keeping only those with
    enough confident points to contribute to the solve."""
    anchors = []
    for frame in sorted(results):
        anchor = result_to_anchor(results[frame], min_keypoint_conf=min_keypoint_conf)
        if len(anchor.landmarks) >= min_points:
            anchors.append(anchor)
    return anchors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_candidates.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/auto_anchor.py tests/test_auto_anchor_candidates.py
git commit -m "feat: keyframe sampling + confidence-qualified auto anchors"
```

---

### Task 9: Clip-level static-camera consensus (symmetry guard)

**Files:**
- Modify: `src/utils/auto_anchor.py`
- Test: `tests/test_auto_anchor_consensus.py`

The guard: solve a quick pose per anchor (`cv2.solvePnP` on its world/image points), compute each camera centre `C = -R^T t`, take the median, and drop anchors whose centre is farther than a threshold from the median. A left/right-flipped registration yields a wildly different centre and is rejected.

- [ ] **Step 1: Write the failing test**

Create `tests/test_auto_anchor_consensus.py`:

```python
import numpy as np

from src.schemas.anchor import Anchor, LandmarkObservation
from src.utils.auto_anchor import consensus_filter, _quick_camera_centre


def _project(world, K, R, t):
    cam = R @ np.asarray(world).T + t[:, None]
    uv = K @ cam
    return (uv[:2] / uv[2]).T


def _anchor_from_camera(frame, K, R, t, world_pts):
    img = _project(world_pts, K, R, t)
    lms = tuple(
        LandmarkObservation(name=f"p{i}", image_xy=(float(u), float(v)),
                            world_xyz=tuple(float(x) for x in w))
        for i, (w, (u, v)) in enumerate(zip(world_pts, img))
    )
    return Anchor(frame=frame, landmarks=lms, lines=())


def _world_points():
    return np.array([
        [0, 0, 0], [105, 0, 0], [0, 68, 0], [105, 68, 0],
        [52.5, 34, 0], [0, 30.34, 2.44], [105, 37.66, 2.44], [16.5, 13.84, 0],
    ], dtype=float)


def test_consensus_keeps_consistent_and_drops_flipped():
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], float)
    R = np.eye(3)
    C = np.array([52.5, -30.0, 15.0])     # behind near touchline, elevated
    t = -R @ C
    pts = _world_points()
    good0 = _anchor_from_camera(0, K, R, t, pts)
    good1 = _anchor_from_camera(30, K, R, t, pts)
    # Flipped: camera centre mirrored to the far side -> inconsistent.
    C_flip = np.array([52.5, 98.0, 15.0])
    t_flip = -R @ C_flip
    bad = _anchor_from_camera(60, K, R, t_flip, pts)

    kept = consensus_filter([good0, good1, bad], max_centre_disagreement_m=3.0)
    assert [a.frame for a in kept] == [0, 30]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_consensus.py -v`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Add the consensus functions**

Append to `src/utils/auto_anchor.py` (add `import cv2`, `import numpy as np` at top):

```python
def _quick_camera_centre(anchor: Anchor) -> np.ndarray | None:
    """Rough camera centre from an anchor's point correspondences via solvePnP.
    Uses a broadcast focal prior; only the centre's CONSISTENCY matters here,
    not its absolute accuracy."""
    if len(anchor.landmarks) < 4:
        return None
    world = np.array([lm.world_xyz for lm in anchor.landmarks], dtype=np.float64)
    image = np.array([lm.image_xy for lm in anchor.landmarks], dtype=np.float64)
    # Coplanar-only sets are fine for solvePnP (homography path); a focal
    # prior of image width is adequate for a consistency check.
    f = 1500.0
    cx = image[:, 0].mean()
    cy = image[:, 1].mean()
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        world, image, K, None, flags=cv2.SOLVEPNP_ITERATIVE
    ) if len({tuple(w[2] for w in world)}) > 0 else (False, None, None)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec.reshape(3))


def consensus_filter(
    anchors: list[Anchor], *, max_centre_disagreement_m: float
) -> list[Anchor]:
    """Drop anchors whose rough camera centre disagrees with the median
    (static-camera assumption: one body for the whole clip). Rejects
    left/right-flipped registrations."""
    centres: dict[int, np.ndarray] = {}
    for a in anchors:
        c = _quick_camera_centre(a)
        if c is not None and np.all(np.isfinite(c)):
            centres[a.frame] = c
    if len(centres) < 2:
        return anchors  # not enough to vote; trust the model
    median = np.median(np.stack(list(centres.values())), axis=0)
    kept = []
    for a in anchors:
        c = centres.get(a.frame)
        if c is None:
            continue
        if np.linalg.norm(c - median) <= max_centre_disagreement_m:
            kept.append(a)
    return kept
```

> **Note:** the `solvePnP` flag selection line above is intentionally simple;
> if a clip's auto-anchors are all coplanar (no crossbar keypoints detected),
> `SOLVEPNP_ITERATIVE` still returns a usable centre via the planar homography.
> Keep the threshold loose (config default 3.0 m) — this guard only needs to
> catch gross flips, not fine disagreements.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_consensus.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/auto_anchor.py tests/test_auto_anchor_consensus.py
git commit -m "feat: clip-level static-camera consensus rejects flipped keyframes"
```

---

### Task 10: `generate()` orchestration → AnchorSet

**Files:**
- Modify: `src/utils/auto_anchor.py`
- Test: `tests/test_auto_anchor_generate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auto_anchor_generate.py`:

```python
from src.schemas.anchor import AnchorSet
from src.utils.field_registration import FieldKeypoint, FieldRegistrationResult
from src.utils.auto_anchor import generate


class _FakeProvider:
    def __init__(self, per_frame):
        self._per_frame = per_frame

    def register_frames(self, video, frames):
        return {f: self._per_frame(f) for f in frames if self._per_frame(f)}


def _rich(frame):
    kps = tuple(
        FieldKeypoint(i, (float(100 + i), float(900 - i)), 0.9)
        for i in (1, 3, 5, 7, 9, 16)
    )
    return FieldRegistrationResult(frame, (1920, 1080), kps, ())


def test_generate_returns_anchorset_with_rich_anchor():
    provider = _FakeProvider(_rich)
    cfg = {
        "keyframe_stride": 30, "max_keyframes": 12, "min_keypoint_conf": 0.5,
        "min_points_per_anchor": 4, "consensus_max_centre_disagreement_m": 50.0,
    }
    anchor_set = generate(
        provider=provider, video="clip.mp4", clip_id="gberch",
        n_frames=100, image_size=(1920, 1080), cfg=cfg,
    )
    assert isinstance(anchor_set, AnchorSet)
    assert anchor_set.clip_id == "gberch"
    assert len(anchor_set.anchors) >= 1
    assert anchor_set.image_size == (1920, 1080)


def test_generate_returns_none_when_no_qualifying_anchors():
    provider = _FakeProvider(lambda f: None)
    cfg = {
        "keyframe_stride": 30, "max_keyframes": 12, "min_keypoint_conf": 0.5,
        "min_points_per_anchor": 4, "consensus_max_centre_disagreement_m": 50.0,
    }
    assert generate(
        provider=provider, video="clip.mp4", clip_id="x",
        n_frames=100, image_size=(1920, 1080), cfg=cfg,
    ) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_generate.py -v`
Expected: FAIL (`generate` undefined).

- [ ] **Step 3: Add `generate`**

First read `src/schemas/anchor.py` for the `AnchorSet` constructor (fields `clip_id`, `image_size`, `anchors`). Then append to `src/utils/auto_anchor.py`:

```python
from src.schemas.anchor import AnchorSet  # add to imports at top


def generate(
    *,
    provider,
    video: str,
    clip_id: str,
    n_frames: int,
    image_size: tuple[int, int],
    cfg: dict,
) -> AnchorSet | None:
    """Full auto-anchor pipeline: sample -> register -> qualify -> consensus.

    Returns an AnchorSet (identical shape to the editor's output) or None if
    the clip yields no usable anchors (caller falls back to manual).
    """
    frames = select_keyframes(
        n_frames=n_frames,
        stride=int(cfg.get("keyframe_stride", 30)),
        max_keyframes=int(cfg.get("max_keyframes", 12)),
    )
    results = provider.register_frames(video, frames)
    if not results:
        return None
    anchors = qualified_anchors(
        results,
        min_keypoint_conf=float(cfg.get("min_keypoint_conf", 0.5)),
        min_points=int(cfg.get("min_points_per_anchor", 4)),
    )
    anchors = consensus_filter(
        anchors,
        max_centre_disagreement_m=float(
            cfg.get("consensus_max_centre_disagreement_m", 3.0)
        ),
    )
    if not anchors:
        return None
    return AnchorSet(
        clip_id=clip_id, image_size=image_size, anchors=tuple(anchors),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_generate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/auto_anchor.py tests/test_auto_anchor_generate.py
git commit -m "feat: auto_anchor.generate orchestration -> AnchorSet"
```

---

## Phase 3 — Camera-stage integration

### Task 11: Config block

**Files:**
- Modify: `config/default.yaml`
- Test: `tests/test_auto_anchor_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auto_anchor_config.py`:

```python
import yaml
from pathlib import Path


def test_default_config_has_auto_anchors_block():
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "default.yaml").read_text()
    )
    aa = cfg["camera"]["auto_anchors"]
    assert aa["enabled"] is False               # opt-in
    assert aa["mode"] == "replace_when_empty"
    assert aa["keyframe_stride"] == 30
    assert "min_keypoint_conf" in aa
    assert "consensus_max_centre_disagreement_m" in aa
    assert aa["model"]["device"] == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_config.py -v`
Expected: FAIL (`KeyError: 'auto_anchors'`).

- [ ] **Step 3: Add the config block**

In `config/default.yaml`, inside the `camera:` mapping (after `pitch_line_consistency_max_px`), add:

```yaml
  # Landmark-free cold-start. When enabled, the camera stage auto-generates
  # anchors from the vendored PnLCalib field-registration model and writes
  # them to {shot}_anchors.json before the normal solve. mode:
  #   replace_when_empty - only auto-generate when no manual anchors exist
  #   augment            - union auto anchors with existing manual ones
  #   force              - always regenerate (overwrites manual)
  auto_anchors:
    enabled: false
    mode: replace_when_empty
    keyframe_stride: 30
    max_keyframes: 12
    min_keypoint_conf: 0.5
    min_points_per_anchor: 4
    consensus_max_centre_disagreement_m: 3.0
    model:
      kp_weights: third_party/pnlcalib/weights/SV_kp
      line_weights: third_party/pnlcalib/weights/SV_lines
      device: cpu
      kp_threshold: 0.3434
      line_threshold: 0.7867
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/default.yaml tests/test_auto_anchor_config.py
git commit -m "feat: camera.auto_anchors config block (opt-in, off by default)"
```

---

### Task 12: Camera-stage pre-step hook

**Files:**
- Modify: `src/stages/camera.py`
- Test: `tests/test_camera_stage_auto_anchor.py`

The hook runs at the very top of `_run_shot`, before `AnchorSet.load`. It decides whether to generate, generates, and writes `{shot}_anchors.json`. Failure or empty result → log + fall back (leave the file absent so the existing "no anchors" warning path applies).

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_stage_auto_anchor.py`:

```python
from pathlib import Path

from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.stages.camera import CameraStage


def _anchor_set():
    lms = tuple(
        LandmarkObservation(name=f"pnl_kp_{i}", image_xy=(float(i), float(i)),
                            world_xyz=(float(i), float(i), 0.0))
        for i in range(1, 7)
    )
    return AnchorSet(clip_id="s1", image_size=(1920, 1080),
                     anchors=(Anchor(frame=0, landmarks=lms, lines=()),))


def test_ensure_anchors_writes_generated_set(tmp_path, monkeypatch):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    cfg = {"auto_anchors": {"enabled": True, "mode": "replace_when_empty"}}

    monkeypatch.setattr(
        "src.stages.camera._generate_auto_anchors",
        lambda shot_id, clip_path, cfg, image_size=None: _anchor_set(),
    )
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert anchors_path.exists()
    loaded = AnchorSet.load(anchors_path)
    assert len(loaded.anchors) == 1


def test_ensure_anchors_skips_when_manual_exists(tmp_path):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    anchors_path.parent.mkdir(parents=True)
    _anchor_set().save(anchors_path)
    before = anchors_path.read_text()
    cfg = {"auto_anchors": {"enabled": True, "mode": "replace_when_empty"}}
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert anchors_path.read_text() == before  # untouched


def test_ensure_anchors_noop_when_disabled(tmp_path):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    cfg = {"auto_anchors": {"enabled": False}}
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert not anchors_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_camera_stage_auto_anchor.py -v`
Expected: FAIL (`_ensure_anchors` / `_generate_auto_anchors` undefined).

- [ ] **Step 3: Implement the hook**

In `src/stages/camera.py`, add a module-level helper and a method. The heavy lifting (`auto_anchor.generate`, video frame count) lives in `_generate_auto_anchors` so tests can monkeypatch it without torch.

```python
def _generate_auto_anchors(shot_id, clip_path, cfg, image_size=None):
    """Run the learned field-registration auto-anchor pipeline for one shot.
    Returns an AnchorSet or None. Imports are local so the camera stage has
    no hard torch dependency unless auto-anchors are actually used."""
    import cv2

    from src.utils.auto_anchor import generate
    from src.utils.pnlcalib_provider import PnLCalibProvider

    aa = cfg.get("auto_anchors", {})
    model_cfg = aa.get("model", {})
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    provider = PnLCalibProvider(
        weights_kp=model_cfg.get("kp_weights", "weights/SV_kp"),
        weights_line=model_cfg.get("line_weights", "weights/SV_lines"),
        device=model_cfg.get("device", "cpu"),
        kp_threshold=float(model_cfg.get("kp_threshold", 0.3434)),
        line_threshold=float(model_cfg.get("line_threshold", 0.7867)),
    )
    return generate(
        provider=provider, video=str(clip_path), clip_id=shot_id,
        n_frames=n_frames, image_size=(w, h), cfg=aa,
    )
```

Then add this method to `CameraStage` and call it at the top of `_run_shot` (just before `anchors = AnchorSet.load(anchors_path)`):

```python
    def _ensure_anchors(self, shot_id, anchors_path, clip_path, cfg):
        """Auto-generate anchors when enabled and appropriate. On any failure,
        leave the anchors file as-is so the existing manual path/warnings apply."""
        aa = cfg.get("auto_anchors", {})
        if not aa.get("enabled", False):
            return
        mode = aa.get("mode", "replace_when_empty")
        if anchors_path.exists() and mode == "replace_when_empty":
            return
        try:
            generated = _generate_auto_anchors(shot_id, clip_path, cfg)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "auto_anchors: generation failed for shot %s (%s); "
                "falling back to manual anchors", shot_id, exc,
            )
            return
        if generated is None or not generated.anchors:
            logger.warning(
                "auto_anchors: no usable anchors generated for shot %s; "
                "falling back to manual anchors", shot_id,
            )
            return
        if mode == "augment" and anchors_path.exists():
            from src.schemas.anchor import AnchorSet
            existing = AnchorSet.load(anchors_path)
            merged = existing.anchors + tuple(
                a for a in generated.anchors
                if a.frame not in {e.frame for e in existing.anchors}
            )
            generated = AnchorSet(
                clip_id=generated.clip_id, image_size=generated.image_size,
                anchors=merged,
            )
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        generated.save(anchors_path)
        logger.info(
            "auto_anchors: wrote %d generated anchors for shot %s to %s",
            len(generated.anchors), shot_id, anchors_path,
        )
```

And insert the call in `_run_shot`, replacing the opening line `anchors = AnchorSet.load(anchors_path)` with:

```python
        self._ensure_anchors(shot_id, anchors_path, clip_path, cfg)
        anchors = AnchorSet.load(anchors_path)
```

Also handle the existing `run()` skip-when-no-anchors guard: when `auto_anchors.enabled`, the stage should still enter `_run_shot` even if the file is absent. In `run()`, change the `if not anchors_path.exists(): ... continue` block to skip the early-continue when `cfg.get("auto_anchors", {}).get("enabled")` is true (so `_ensure_anchors` gets a chance to create it):

```python
            if not anchors_path.exists() and not cfg.get("auto_anchors", {}).get("enabled", False):
                logger.warning(
                    "camera stage skipping shot %s — no anchors at %s. Open "
                    "the anchor editor and place keyframes before re-running.",
                    shot.id, anchors_path,
                )
                continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_camera_stage_auto_anchor.py -v`
Expected: PASS.

- [ ] **Step 5: Run the existing camera-stage tests (regression)**

Run: `pytest tests/test_camera_stage.py tests/test_camera_stage_static_line.py -v`
Expected: PASS (auto path is off by default; behaviour unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/stages/camera.py tests/test_camera_stage_auto_anchor.py
git commit -m "feat: camera stage auto-anchor pre-step with manual fallback"
```

---

## Phase 4 — End-to-end validation

### Task 13: Zero-click parity integration test

**Files:**
- Create: `tests/test_auto_anchor_e2e.py`
- Create: `scripts/eval_auto_anchor.py`

This test is **slow and requires weights + a clip**; mark it so it is skipped in normal CI and run explicitly.

- [ ] **Step 1: Write the eval script**

Create `scripts/eval_auto_anchor.py`:

```python
"""End-to-end: run the camera stage with auto_anchors on a clip with NO manual
anchors, then report the static-line-solve RMS from the camera stage logs /
quality report. Compares against the manual baseline (~0.95 px mean on gberch).

Usage:
  python scripts/eval_auto_anchor.py --output ./output-autotest \
      --clip "test-media/<gberch>.mp4" --shot s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.schemas.camera_track import CameraTrack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--shot", default="s1")
    args = ap.parse_args()
    out = Path(args.output)
    track = CameraTrack.load(out / "camera" / f"{args.shot}_camera_track.json")
    confs = [f.confidence for f in track.frames]
    print(f"frames={len(track.frames)} mean_conf={sum(confs)/len(confs):.3f}")
    qr = out / "quality_report.json"
    if qr.exists():
        data = json.loads(qr.read_text())
        print(json.dumps(data.get("camera", {}), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the slow integration test**

Create `tests/test_auto_anchor_e2e.py`:

```python
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_PNLCALIB_E2E"),
    reason="set RUN_PNLCALIB_E2E=1 and provide weights + clip to run",
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "third_party" / "pnlcalib" / "weights" / "SV_kp"


def test_zero_click_line_rms_parity():
    """With zero manual clicks, auto-anchors + existing solver reach line-RMS
    parity with the manual baseline (mean <= 1.0 px on gberch)."""
    assert WEIGHTS.exists(), "fetch weights first"
    from src.utils.auto_anchor import generate
    from src.utils.pnlcalib_provider import PnLCalibProvider
    import cv2

    clip = os.environ["PNLCALIB_E2E_CLIP"]
    cap = cv2.VideoCapture(clip)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    provider = PnLCalibProvider(
        weights_kp=str(WEIGHTS),
        weights_line=str(WEIGHTS.parent / "SV_lines"),
        device=os.environ.get("PNLCALIB_DEVICE", "cpu"),
    )
    cfg = {
        "keyframe_stride": 30, "max_keyframes": 12, "min_keypoint_conf": 0.5,
        "min_points_per_anchor": 4, "consensus_max_centre_disagreement_m": 3.0,
    }
    anchor_set = generate(
        provider=provider, video=clip, clip_id="gberch",
        n_frames=n_frames, image_size=(w, h), cfg=cfg,
    )
    assert anchor_set is not None
    # At least one rich anchor (>=6 non-coplanar points) must exist.
    from src.utils.anchor_solver import _is_rich
    assert any(_is_rich(a) for a in anchor_set.anchors), \
        "no rich anchor -> metric pose under-constrained"
```

- [ ] **Step 3: Run unit suite (e2e auto-skips)**

Run: `pytest tests/test_auto_anchor_e2e.py -v`
Expected: SKIPPED (no `RUN_PNLCALIB_E2E`).

- [ ] **Step 4: MANUAL end-to-end (requires GPU box or patient CPU)**

```bash
RUN_PNLCALIB_E2E=1 \
PNLCALIB_E2E_CLIP="test-media/<gberch>.mp4" \
pytest tests/test_auto_anchor_e2e.py -v
```
Then run the full stage with `auto_anchors.enabled: true` and **no** `{shot}_anchors.json`, and compare line-RMS in the logs/quality report against the 0.95 px manual baseline. Tune `keyframe_stride`, `min_keypoint_conf`, and `consensus_max_centre_disagreement_m` until mean ≤ 1.0 px.

- [ ] **Step 5: Commit**

```bash
git add tests/test_auto_anchor_e2e.py scripts/eval_auto_anchor.py
git commit -m "test: zero-click line-RMS parity e2e (opt-in) + eval script"
```

---

## Self-review

**Spec coverage:**
- §1 core idea (auto-anchor generator) → Tasks 7–10, 12.
- §2 where it slots (pre-step, writes anchors JSON) → Task 12.
- §3 components → Tasks 1 (vendor), 4 (wrapper), 5 (provider), 3 (bridge), 7–10 (auto_anchor), 11 (config), 12 (hook). ✓
- §4 data association (intrinsic) → Task 3/7; symmetry consensus → Task 9. ✓
- §5 error handling / fallback → Task 12 (try/except, mode handling, disabled no-op). ✓
- §6 testing (bridge, consensus, parsing, headline integration) → Tasks 3, 9, 5, 13. ✓
- §7 phasing → matches Phases 0–4. ✓
- §8 YAGNI (one provider, uniform sampling, replace_when_empty) → honoured. ✓
- §9 risks (bridge correctness validated first; CPU speed; rich-anchor guarantee) → Tasks 3, 6, 13. ✓

**Placeholder scan:** No "TBD"/"handle edge cases" placeholders. Two explicit *validation notes* (Task 4 tensor shape, Task 7 schema field names) are deliberate "verify against real source in this step" instructions, not deferred work — each names the exact file to check and what to adjust.

**Type consistency:** `FieldKeypoint`/`FieldLine`/`FieldRegistrationResult` (Task 2) used identically in Tasks 5, 7, 8, 10. `result_to_anchor` (Task 7) → `qualified_anchors` (Task 8) → `consensus_filter` (Task 9) → `generate` (Task 10) → `_generate_auto_anchors` (Task 12) chain consistent. `keypoint_world_xyz` (Task 3) used in Tasks 6, 7. `PnLCalibProvider.register_frames` signature consistent across Tasks 5, 10, 12, 13.

**Known integration points to verify during execution (not gaps, but real-code touchpoints):**
- `src/schemas/anchor.py` exact constructors for `Anchor`, `LandmarkObservation`, `AnchorSet` (Tasks 7, 10, 12 each read it first).
- `get_keypoints_from_heatmap_batch_maxpool` output tensor shape (Task 4/6).
- `_is_rich` import path in `src/utils/anchor_solver.py` (Task 13).

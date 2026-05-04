# macOS Dependency Handoff (2026-04-04)

## Context
Goal was to get Stage 4 (`pose`) running in a single Python environment on macOS (Apple Silicon), with MMPose/OpenMMLab stack working end-to-end.

Primary command under test:

```bash
python recon.py run --input test-media/origi-vs-barcelona.mp4 --output ./output --stages pose
```

## What Was Changed In Repo
1. Updated dependency bounds in `pyproject.toml` to a tighter OpenMMLab lane and added missing runtime deps (`mmdet`, `xtcocotools`).
2. Added pinned constraints file:
   - `constraints/macos-py311-openmmlab.txt`
3. Updated install instructions in `README.md` to use constrained installs and health checks.
4. Fixed constraints file format issue:
   - Replaced `scenedetect[opencv]==...` with `scenedetect==...` because pip constraints do not allow extras.

## Major Failure Sequence Observed
1. Initial OpenMMLab install/build instability around `mmcv` and `chumpy` style build behavior.
2. `openxlab`/`opendatalab` conflict chain caused `tqdm` pin mismatch with project requirements.
3. `xtcocotools` import repeatedly failed with ABI mismatch:
   - `ValueError: numpy.dtype size changed, may indicate binary incompatibility`
4. `torch` required `setuptools<82`, causing environment constraints.
5. After partial recovery, `mmdet` missing caused MMPose import failure.
6. Installing `mmdet==3.3.0` exposed strict version guard:
   - `mmdet` rejected `mmcv==2.2.0`; expects `<2.2.0`.
7. Attempted `mmcv==2.1.0` install failed from source build path due setuptools/pkg_resources/build-isolation issues in this environment.
8. Latest install attempt used Python 3.14 environment (`.venv`), which forced source builds for SciPy/NumPy stack and failed due to missing Fortran compiler (`gfortran`).

## Confirmed Root Causes
1. Environment drift across mixed Python versions (`.venv` on 3.14 vs intended 3.11 lane).
2. Unconstrained resolver upgrades causing incompatible combinations of:
   - `numpy`, `opencv-python`, `mmcv`, `mmdet`, `mmpose`, `xtcocotools`
3. Source-build fallback on macOS without required toolchain (Fortran) when wheels are unavailable for selected Python/version combo.
4. Tight OpenMMLab inter-package compatibility constraints (`mmdet` <-> `mmcv`) not satisfied by ad-hoc installs.

## Current Status At Handoff
1. Repo now contains a documented constrained install lane and revised metadata.
2. User attempted constrained install in `.venv` (Python 3.14), which failed at SciPy metadata/build due to missing Fortran compiler.
3. Actionable guidance already provided: use Python 3.11 environment (`.venv311`) for the pinned lane.

## Recommended Next Steps (For Claude Code)
1. Use a fresh Python 3.11 venv only:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -V
python -m pip install -U pip
python -m pip install -c constraints/macos-py311-openmmlab.txt -e .
python -m pip check
```

2. Run import smoke checks in `.venv311`:

```bash
python -c "import numpy, cv2, torch, mmcv, mmdet, mmpose; print('imports ok')"
python -c "import xtcocotools._mask; print('xtcocotools ok')"
```

3. If `mmcv` still tries source build/fails:
   - ensure no Python 3.14 env is active
   - inspect wheel availability for exact Python ABI and platform
   - if needed, select a tested torch/mmcv/mmdet/mmpose quartet with published wheels for cp311 macOS arm64.

4. Re-run stage command once imports pass:

```bash
python recon.py run --input test-media/origi-vs-barcelona.mp4 --output ./output --stages pose
```

## Notes For Investigation
1. Validate whether pinned `torch==2.11.0` / `torchvision==0.22.0` is intentional and wheel-available for target Python 3.11 on this host.
2. Confirm `mmpose` selected release and `mmdet==3.3.0` remain compatible with chosen `mmcv` build variant.
3. Keep all installs constrained; avoid in-place unconstrained upgrades of OpenMMLab packages.

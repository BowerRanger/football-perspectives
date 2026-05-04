#!/usr/bin/env bash
#
# Setup script for GVHMR checkpoints.
#
# Most checkpoints are on Google Drive and require manual download.
# This script creates the directory structure, symlinks files we already
# have, and prints instructions for what needs to be downloaded manually.
#
# Google Drive folder:
#   https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD
#
# SMPL/SMPLX body models (free registration required):
#   SMPL:  https://smpl.is.tue.mpg.de/
#   SMPLX: https://smpl-x.is.tue.mpg.de/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GVHMR_ROOT="$REPO_ROOT/third_party/gvhmr"
CKPT_DIR="$GVHMR_ROOT/inputs/checkpoints"

echo "=== GVHMR Checkpoint Setup ==="
echo "Repo root: $REPO_ROOT"
echo "Checkpoint dir: $CKPT_DIR"
echo

# Create directory structure
mkdir -p "$CKPT_DIR/gvhmr"
mkdir -p "$CKPT_DIR/hmr2"
mkdir -p "$CKPT_DIR/vitpose"
mkdir -p "$CKPT_DIR/yolo"
mkdir -p "$CKPT_DIR/body_models/smpl"
mkdir -p "$CKPT_DIR/body_models/smplx"
mkdir -p "$GVHMR_ROOT/inputs"
mkdir -p "$GVHMR_ROOT/outputs"

# Symlink YOLOv8x if we have it
if [ -f "$REPO_ROOT/yolov8x.pt" ] && [ ! -f "$CKPT_DIR/yolo/yolov8x.pt" ]; then
    ln -sf "$REPO_ROOT/yolov8x.pt" "$CKPT_DIR/yolo/yolov8x.pt"
    echo "[OK] Symlinked yolov8x.pt"
fi

# Symlink SMPL neutral if we have it
if [ -f "$REPO_ROOT/data/smpl/SMPL_NEUTRAL.pkl" ] && [ ! -f "$CKPT_DIR/body_models/smpl/SMPL_NEUTRAL.pkl" ]; then
    ln -sf "$REPO_ROOT/data/smpl/SMPL_NEUTRAL.pkl" "$CKPT_DIR/body_models/smpl/SMPL_NEUTRAL.pkl"
    echo "[OK] Symlinked SMPL_NEUTRAL.pkl"
fi

echo
echo "=== Status ==="

check_file() {
    local path="$1"
    local name="$2"
    if [ -f "$path" ]; then
        echo "  [OK]      $name"
    else
        echo "  [MISSING] $name"
    fi
}

check_file "$CKPT_DIR/gvhmr/gvhmr_siga24_release.ckpt"     "GVHMR model"
check_file "$CKPT_DIR/hmr2/epoch=10-step=25000.ckpt"        "HMR2.0 ViT features"
check_file "$CKPT_DIR/vitpose/vitpose-h-multi-coco.pth"     "ViTPose-H"
check_file "$CKPT_DIR/yolo/yolov8x.pt"                      "YOLOv8x"
check_file "$CKPT_DIR/body_models/smpl/SMPL_NEUTRAL.pkl"    "SMPL neutral"
check_file "$CKPT_DIR/body_models/smplx/SMPLX_NEUTRAL.npz"  "SMPLX neutral"

echo
echo "=== Download Instructions ==="
echo
echo "1. GVHMR + HMR2 + ViTPose checkpoints (Google Drive):"
echo "   https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
echo
echo "   Download and place:"
echo "     gvhmr_siga24_release.ckpt   → $CKPT_DIR/gvhmr/"
echo "     epoch=10-step=25000.ckpt    → $CKPT_DIR/hmr2/"
echo "     vitpose-h-multi-coco.pth    → $CKPT_DIR/vitpose/"
echo
echo "2. SMPL body model (free registration):"
echo "   https://smpl.is.tue.mpg.de/"
echo "   Download SMPL_NEUTRAL.pkl → $CKPT_DIR/body_models/smpl/"
echo
echo "3. SMPLX body model (free registration):"
echo "   https://smpl-x.is.tue.mpg.de/"
echo "   Download SMPLX_NEUTRAL.npz → $CKPT_DIR/body_models/smplx/"
echo

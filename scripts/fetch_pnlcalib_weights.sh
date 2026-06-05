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

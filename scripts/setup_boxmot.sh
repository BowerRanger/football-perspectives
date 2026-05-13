#!/usr/bin/env bash
#
# Setup script for BoxMOT ReID weights.
#
# Downloads OSNet x0.25 trained on MSMT17 into
# third_party/boxmot/osnet_x0_25_msmt17.pt — the default ReID
# embedding model used by BoT-SORT when tracking.tracker = botsort
# in config/default.yaml.
#
# BoxMOT hosts these weights on Google Drive. We delegate the
# download to gdown (which is installed as a boxmot dependency)
# rather than curl/wget so the Drive "is this a real user?"
# interstitial page is handled correctly.
#
# The Drive ID mirrors what boxmot.appearance.reid.config.URLS uses
# internally, so re-running this against a newer boxmot release will
# still work as long as the file name doesn't change.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$REPO_ROOT/third_party/boxmot"

# Default to OSNet-AIN x1.0 (the model referenced from config/default.yaml).
# Override by setting WEIGHTS_FILE before invoking, e.g.:
#   WEIGHTS_FILE=osnet_x0_25_msmt17.pt scripts/setup_boxmot.sh
WEIGHTS_FILE="${WEIGHTS_FILE:-osnet_ain_x1_0_msmt17.pt}"
DEST_PATH="$DEST_DIR/$WEIGHTS_FILE"

# Drive IDs mirror boxmot/appearance/reid/config.py TRAINED_URLS.
# Add more entries here if you want to experiment with other ReID
# backbones — the IDs are stable across boxmot 12.x.
case "$WEIGHTS_FILE" in
    osnet_x0_25_msmt17.pt)    DRIVE_ID="1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF" ;;
    osnet_x0_5_msmt17.pt)     DRIVE_ID="1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv" ;;
    osnet_x0_75_msmt17.pt)    DRIVE_ID="1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc" ;;
    osnet_x1_0_msmt17.pt)     DRIVE_ID="112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M" ;;
    osnet_ain_x1_0_msmt17.pt) DRIVE_ID="1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal" ;;
    osnet_ibn_x1_0_msmt17.pt) DRIVE_ID="1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ" ;;
    *)
        echo "Unknown WEIGHTS_FILE: $WEIGHTS_FILE" >&2
        echo "Edit scripts/setup_boxmot.sh to add the Google Drive ID" >&2
        echo "(see boxmot/appearance/reid/config.py for the catalogue)" >&2
        exit 1
        ;;
esac
DRIVE_URL="https://drive.google.com/uc?id=$DRIVE_ID"

echo "=== BoxMOT ReID Checkpoint Setup ==="
echo "Repo root:       $REPO_ROOT"
echo "Destination:     $DEST_PATH"
echo

mkdir -p "$DEST_DIR"

if [ -f "$DEST_PATH" ]; then
    echo "Already present — $DEST_PATH"
    echo "(delete it and re-run to refresh)"
    exit 0
fi

# Prefer the venv's Python if one is active or sitting next to us.
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$REPO_ROOT/.venv311/bin/python" ]; then
        PYTHON_BIN="$REPO_ROOT/.venv311/bin/python"
    elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "No python interpreter found. Install one (or activate the venv) and re-run." >&2
        exit 1
    fi
fi

echo "Downloading $WEIGHTS_FILE from Google Drive via gdown..."
if ! "$PYTHON_BIN" -c "
import sys
try:
    import gdown
except ImportError:
    sys.stderr.write('gdown is not installed in this Python environment.\n')
    sys.stderr.write('Install boxmot first (pip install boxmot) or pip install gdown.\n')
    sys.exit(2)
ok = gdown.download(id='$DRIVE_ID', output='$DEST_PATH', quiet=False)
sys.exit(0 if ok else 3)
"; then
    rm -f "$DEST_PATH"
    echo "Download failed." >&2
    echo "Manual fallback: visit $DRIVE_URL in a browser and save to $DEST_PATH" >&2
    exit 1
fi

if [ ! -s "$DEST_PATH" ]; then
    echo "Download succeeded but the file is empty — removing." >&2
    rm -f "$DEST_PATH"
    exit 1
fi

echo
echo "Done. ReID weights ready at $DEST_PATH"

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

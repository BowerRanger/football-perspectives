import importlib.util
from pathlib import Path


def test_smoke_script_imports():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pnlcalib_smoke.py"
    spec = importlib.util.spec_from_file_location("pnlcalib_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")

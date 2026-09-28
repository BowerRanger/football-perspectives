"""Helpers for asserting dashboard features exist in the React frontend.

The dashboard is a React SPA: its source lives in ``frontend/src`` and the
committed production build in ``src/web/static/app``. Feature tests grep
both — the source proves the feature was written, the bundle proves the
committed build (what ``recon.py serve`` actually ships) includes it.
String literals such as endpoint URLs and payload keys survive
minification, so they make stable markers.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"
BUILD_DIR = REPO_ROOT / "src" / "web" / "static" / "app"


def source_text(*relative_dirs: str) -> str:
    """Concatenate every .ts/.tsx file under the given ``frontend/src`` dirs."""
    chunks: list[str] = []
    for rel in relative_dirs:
        root = FRONTEND_SRC / rel
        paths = [root] if root.is_file() else sorted(root.rglob("*.ts*"))
        chunks.extend(p.read_text(encoding="utf-8") for p in paths)
    if not chunks:
        raise AssertionError(f"no frontend source found under {relative_dirs}")
    return "\n".join(chunks)


def bundle_text() -> str:
    """Concatenate every JS chunk of the committed production build."""
    paths = sorted((BUILD_DIR / "assets").glob("*.js"))
    if not paths:
        raise AssertionError("no committed build under src/web/static/app/assets — run `npm run build` in frontend/")
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


def assert_markers(text: str, markers: list[str], where: str) -> None:
    missing = [m for m in markers if m not in text]
    assert not missing, f"{where} is missing feature markers: {missing}"

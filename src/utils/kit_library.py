"""Kit library — club kits shipped as ``config/kits/<club_slug>.yaml``.

File shape::

    club: Liverpool
    aliases: [Liverpool FC, LFC]
    kits:
      "2025-26/home": {shirt: ..., shorts: ..., socks: ..., sleeves: short}
      "2025-26/gk":   {...}

A kit is addressed by ref ``<club_slug>/<season>/<name>`` (the slug is the
file stem), e.g. ``liverpool/2025-26/home``. ``referees.yaml`` has the same
shape with ``club: Referees`` and plain names (``referees/black``).

Kits whose name contains ``gk`` are goalkeeper kits; ``referees/*`` are
referee kits; everything else is an outfield team kit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.schemas.kit import KitSpecError, normalise_kit_spec

logger = logging.getLogger(__name__)

DEFAULT_LIBRARY_DIR = Path(__file__).resolve().parents[2] / "config" / "kits"
REFEREE_SLUG = "referees"


def _norm_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())
    s = re.sub(r"\b(fc|afc|cf|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def season_for_date(date: str | None) -> str | None:
    """``"2026-05-12"`` -> ``"2025-26"`` (seasons roll over in July)."""
    m = re.match(r"^(\d{4})-(\d{2})", date or "")
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    start = year if month >= 7 else year - 1
    return f"{start}-{(start + 1) % 100:02d}"


@dataclass(frozen=True)
class LibraryKit:
    ref: str
    slug: str
    season: str | None
    name: str
    spec: dict

    @property
    def is_keeper(self) -> bool:
        return "gk" in self.name.lower()

    @property
    def is_referee(self) -> bool:
        return self.slug == REFEREE_SLUG


@dataclass
class KitLibrary:
    kits: dict[str, LibraryKit] = field(default_factory=dict)
    clubs: dict[str, dict] = field(default_factory=dict)  # slug -> {club, aliases}

    def get(self, ref: str) -> dict:
        try:
            return dict(self.kits[ref].spec)
        except KeyError:
            raise KitSpecError(f"unknown kit ref {ref!r}") from None

    def club_slug(self, name: str | None) -> str | None:
        target = _norm_name(name or "")
        if not target:
            return None
        for slug, meta in self.clubs.items():
            names = [slug.replace("_", " "), meta.get("club", ""), *meta.get("aliases", [])]
            if target in {_norm_name(n) for n in names}:
                return slug
        return None

    def select(self, *, slugs: list[str] | None = None, season: str | None = None,
               keepers: bool = False, referees: bool = False) -> dict[str, dict]:
        """Candidate ``ref -> spec`` pool.

        ``slugs`` limits to those clubs; ``season`` keeps kits of that season
        (kits with no season always pass). Outfield pools exclude keeper and
        referee kits unless asked.
        """
        out: dict[str, dict] = {}
        for ref, kit in self.kits.items():
            if referees != kit.is_referee:
                continue
            if not referees and kit.is_keeper != keepers:
                continue
            if slugs is not None and kit.slug not in slugs:
                continue
            if season and kit.season and kit.season != season:
                continue
            out[ref] = dict(kit.spec)
        return out


def _load_file(path: Path, lib: KitLibrary) -> None:
    raw = yaml.safe_load(path.read_text()) or {}
    slug = path.stem
    lib.clubs[slug] = {"club": raw.get("club", slug), "aliases": list(raw.get("aliases") or [])}
    for key, spec in (raw.get("kits") or {}).items():
        parts = str(key).split("/")
        season, name = (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])
        try:
            norm = normalise_kit_spec(spec)
        except KitSpecError as exc:
            logger.warning("[kit_library] %s:%s invalid kit: %s", path.name, key, exc)
            continue
        ref = f"{slug}/{key}"
        lib.kits[ref] = LibraryKit(ref, slug, season, name, norm)


_CACHE: dict[Path, KitLibrary] = {}


def load_library(root: Path | None = None) -> KitLibrary:
    root = Path(root) if root else DEFAULT_LIBRARY_DIR
    if root in _CACHE:
        return _CACHE[root]
    lib = KitLibrary()
    if root.is_dir():
        for path in sorted(root.glob("*.yaml")):
            try:
                _load_file(path, lib)
            except (yaml.YAMLError, OSError) as exc:
                logger.warning("[kit_library] could not load %s: %s", path, exc)
    _CACHE[root] = lib
    return lib


def resolve_kit(value: Any, library: KitLibrary | None = None) -> dict:
    """A library ref string or an inline KitSpec dict -> normalised KitSpec."""
    if isinstance(value, str):
        lib = library or load_library()
        spec = lib.get(value)
        spec.setdefault("ref", value)
        return spec
    if isinstance(value, Mapping):
        ref = value.get("ref")
        base: dict = {}
        if isinstance(ref, str) and "shirt" not in value:
            base = (library or load_library()).get(ref)
        return normalise_kit_spec({**base, **value})
    raise KitSpecError(f"cannot resolve kit from {value!r}")

"""Per-stadium pitch metadata loader.

Backs the user's "stadium" selection in the anchor editor. The only
field consumed today is ``mowing`` — see
``src/utils/pitch_lines_catalogue.py`` for the static catalogue and
``mow_stripe_lines`` below for the dynamic, stadium-derived entries
that get merged into ``/pitch_lines`` when a stadium is selected.

Why this lives outside the static catalogue: stripe widths and origins
are stadium-specific (groundsmen mow at different cadences on different
pitches), so a fixed catalogue can't cover them. Sourcing them from a
registry that the user picks once per clip keeps the static catalogue
honest while still giving thin frames an extra translation constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


_PITCH_LEN_M = 105.0
_PITCH_WID_M = 68.0
_DEFAULT_REGISTRY_PATH = (
    Path(__file__).parent.parent.parent / "config" / "stadiums.yaml"
)

Orientation = Literal["along_touchline", "along_goalline", "grid"]


@dataclass(frozen=True)
class MowingPattern:
    """Geometry of a stadium's mowing stripes.

    ``orientation``:
      - ``along_touchline`` — stripes parallel to the touchlines
        (world x-axis). Boundaries occur at ``origin_y_m + k*stripe_width_m``
        for integer k, clipped to the pitch's y-range.
      - ``along_goalline`` — stripes parallel to the goal lines
        (world y-axis). Boundaries occur at successive x values.
      - ``grid`` — both sets generated.
    """

    orientation: Orientation
    stripe_width_m: float
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0


@dataclass(frozen=True)
class StadiumConfig:
    id: str
    display_name: str
    mowing: MowingPattern | None = None


class StadiumConfigError(ValueError):
    """Raised when the stadium registry YAML can't be parsed cleanly."""


_VALID_ORIENTATIONS: tuple[Orientation, ...] = (
    "along_touchline", "along_goalline", "grid",
)


def load_stadiums(path: Path | None = None) -> dict[str, StadiumConfig]:
    """Parse ``config/stadiums.yaml`` (or ``path`` if given) into a registry.

    Returns a mapping ``{stadium_id: StadiumConfig}``. An empty file or a
    missing ``stadiums:`` key resolves to an empty mapping (clips with
    no stadium reference behave as today).
    """
    target = path if path is not None else _DEFAULT_REGISTRY_PATH
    if not target.exists():
        return {}
    with open(target) as f:
        raw = yaml.safe_load(f) or {}
    stadiums_raw = raw.get("stadiums") or {}
    if not isinstance(stadiums_raw, dict):
        raise StadiumConfigError(
            f"{target}: top-level 'stadiums' must be a mapping, "
            f"got {type(stadiums_raw).__name__}"
        )
    out: dict[str, StadiumConfig] = {}
    for sid, body in stadiums_raw.items():
        if not isinstance(body, dict):
            raise StadiumConfigError(
                f"{target}: stadium '{sid}' must be a mapping, "
                f"got {type(body).__name__}"
            )
        out[sid] = _parse_one(sid, body, target)
    return out


def _parse_one(sid: str, body: dict, src: Path) -> StadiumConfig:
    display_name = body.get("display_name", sid)
    mowing_raw = body.get("mowing")
    mowing = _parse_mowing(sid, mowing_raw, src) if mowing_raw else None
    return StadiumConfig(id=sid, display_name=display_name, mowing=mowing)


def _parse_mowing(sid: str, body: dict, src: Path) -> MowingPattern:
    if not isinstance(body, dict):
        raise StadiumConfigError(
            f"{src}: stadium '{sid}': 'mowing' must be a mapping"
        )
    orientation = body.get("orientation")
    if orientation not in _VALID_ORIENTATIONS:
        raise StadiumConfigError(
            f"{src}: stadium '{sid}': invalid mowing.orientation "
            f"{orientation!r}; expected one of {_VALID_ORIENTATIONS}"
        )
    width = body.get("stripe_width_m")
    if not isinstance(width, (int, float)) or width <= 0:
        raise StadiumConfigError(
            f"{src}: stadium '{sid}': mowing.stripe_width_m must be a "
            f"positive number, got {width!r}"
        )
    return MowingPattern(
        orientation=orientation,
        stripe_width_m=float(width),
        origin_x_m=float(body.get("origin_x_m", 0.0)),
        origin_y_m=float(body.get("origin_y_m", 0.0)),
    )


def mow_stripe_lines(
    stadium: StadiumConfig,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Generate stadium-specific position-known mow-stripe entries.

    Each entry mirrors ``LINE_CATALOGUE``'s shape: a name mapped to two
    world endpoints. Boundary lines are emitted at every multiple of
    ``stripe_width_m`` from the origin that falls strictly inside the
    pitch (excluding the endpoints, which would coincide with the
    touchlines / goal lines and don't add information beyond what the
    existing painted entries already provide).

    Naming uses the world coordinate so the user can match the picked
    palette entry against the visible mow seam by counting:
    ``mow_y_15.0`` is the boundary at y=15.0 m.

    Returns ``{}`` when the stadium has no mowing block.
    """
    if stadium.mowing is None:
        return {}
    out: dict[
        str, tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = {}
    pat = stadium.mowing
    if pat.orientation in ("along_touchline", "grid"):
        for y in _stripe_boundaries(pat.origin_y_m, pat.stripe_width_m, _PITCH_WID_M):
            name = f"mow_y_{y:.1f}"
            out[name] = ((0.0, y, 0.0), (_PITCH_LEN_M, y, 0.0))
    if pat.orientation in ("along_goalline", "grid"):
        for x in _stripe_boundaries(pat.origin_x_m, pat.stripe_width_m, _PITCH_LEN_M):
            name = f"mow_x_{x:.1f}"
            out[name] = ((x, 0.0, 0.0), (x, _PITCH_WID_M, 0.0))
    return out


def _stripe_boundaries(origin: float, width: float, max_extent: float) -> list[float]:
    """Stripe-boundary coordinates strictly inside (0, max_extent).

    Walks both directions from ``origin`` so origins outside the pitch
    still produce the right interior set. Endpoints (0 / max_extent) are
    excluded because they overlap with the touchlines / goal lines that
    are already in the static catalogue.
    """
    if width <= 0:
        return []
    out: list[float] = []
    # Start at the smallest k such that origin + k*width > 0.
    k_min = int(((0.0 - origin) // width) + 1)
    k = k_min
    while True:
        v = origin + k * width
        if v >= max_extent:
            break
        if v > 0.0:
            out.append(round(v, 1))
        k += 1
    return out

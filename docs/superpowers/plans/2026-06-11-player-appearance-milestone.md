# Player Appearance Milestone — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable per-player modular skeletal-mesh assembly and fully data-driven appearance (kit colours, skin tone, face, hair colour/hairstyle) that persist across re-imports and are editable in the EUW.

**Architecture:** A pure-Python `appearance.py` sidecar module (mirrors `team_overrides.py`) owns load/save/resolve. `player_parts.py` holds asset-path constants for every modular part mesh. `kit_colors.py` gains per-part `KitPrimary`/`KitSecondary` helpers. `load_reconstruction.py` applies appearance on import and exposes `set_appearance`/`player_appearance_rows` entry points for the EUW. In-editor: `BP_PlayerActor` is rebuilt with five leader-pose part components and `BodyVariant`/`HeadVariant` Blueprint variables; new/updated materials expose `KitPrimary`, `KitSecondary`, `SkinTint`, `FaceTexture`, `TattooEnabled`, and `HairTint` parameters. `EUW_LoadReconstruction` is extended with a per-player appearance panel that auto-applies on change.

**Tech Stack:** Python 3.x (UE editor Python + pytest for offline tests), Unreal Engine 5.8 Blueprint/Material/UMG editor authoring.

---

## Tickets

| Ticket | Title | Dependency |
|--------|-------|------------|
| BOW-85 | Appearance data layer | — |
| BOW-84 | Modular player assembly | BOW-85 (data schema) |
| BOW-93 | Parameterised kit materials | BOW-84 |
| BOW-86 | Skin tone & face levers | BOW-84, BOW-85 |
| BOW-87 | Hair colour & hairstyle levers | BOW-84, BOW-85 |
| BOW-94 | EUW appearance panel | all above |

---

## File Structure

**Python — create:**
- `Content/Python/football_perspectives/appearance.py` — sidecar load/save/resolve (BOW-85)
- `Content/Python/football_perspectives/player_parts.py` — asset-path registry + slot map (BOW-84/93)
- `Content/Python/tests/test_appearance.py` — offline unit tests

**Python — modify:**
- `Content/Python/football_perspectives/kit_colors.py` — per-part `KitPrimary`/`KitSecondary` helpers (BOW-93)
- `Content/Python/football_perspectives/load_reconstruction.py` — apply appearance on load; `set_appearance`, `player_appearance_rows` entry points (BOW-94)

**UE editor (in-editor authoring, no Python equivalent):**
- Create `/Game/Materials/M_Kit` — parameterised kit master material (BOW-93)
- Modify `/Game/Football_player/Materials/M_Football_player_BODY1` (and BODY2*) — add `SkinTint`, `FaceTexture`, `TattooEnabled` (BOW-86)
- Modify `/Game/Football_player/Materials/M_Football_player_HAIR` — add `HairTint` (BOW-87)
- Modify `/Game/Players/BP_PlayerActor` — leader-pose part components, `BodyVariant`/`HeadVariant` variables (BOW-84)
- Modify `EUW_LoadReconstruction` — per-player appearance panel (BOW-94)

---

## Task 1: Appearance data layer (BOW-85)

**Files:**
- Create: `Content/Python/football_perspectives/appearance.py`
- Test: `Content/Python/tests/test_appearance.py`

- [ ] **Step 1.1 — Write failing tests**

Create `Content/Python/tests/test_appearance.py`:

```python
from __future__ import annotations

import json
import pytest
from pathlib import Path
from football_perspectives import appearance


def _write_sidecar(tmp_path, players):
    export = tmp_path / "export"
    export.mkdir(exist_ok=True)
    path = export / "gberch_appearance.json"
    path.write_text(json.dumps({"schema": 1, "clip": "gberch", "players": players}))
    return tmp_path


def test_load_missing_returns_empty(tmp_path):
    assert appearance.load_appearances(tmp_path, "gberch") == {}


def test_load_returns_stored_fields(tmp_path):
    _write_sidecar(tmp_path, {"P001": {"body_variant": 3, "skin_tint": "#ff0000"}})
    result = appearance.load_appearances(tmp_path, "gberch")
    assert result["P001"]["body_variant"] == 3
    assert result["P001"]["skin_tint"] == "#ff0000"


def test_resolve_fills_all_defaults_for_absent_player():
    rec = appearance.resolve_appearance("P001", {})
    assert rec == {
        "body_variant": 1,
        "head_variant": "head",
        "face_variant": 1,
        "skin_tint": "",
        "tattoo": False,
        "hair_color": "",
        "hair_style": "head",
    }


def test_resolve_merges_stored_with_defaults(tmp_path):
    _write_sidecar(tmp_path, {"P001": {"body_variant": 4, "tattoo": True}})
    apps = appearance.load_appearances(tmp_path, "gberch")
    rec = appearance.resolve_appearance("P001", apps)
    assert rec["body_variant"] == 4
    assert rec["tattoo"] is True
    assert rec["hair_color"] == ""  # default fills in


def test_body_variant_clamped_to_range(tmp_path):
    _write_sidecar(tmp_path, {"P001": {"body_variant": 99}})
    apps = appearance.load_appearances(tmp_path, "gberch")
    assert apps["P001"]["body_variant"] == 6


def test_body_variant_clamped_low(tmp_path):
    _write_sidecar(tmp_path, {"P001": {"body_variant": 0}})
    apps = appearance.load_appearances(tmp_path, "gberch")
    assert apps["P001"]["body_variant"] == 1


def test_invalid_head_variant_ignored(tmp_path):
    _write_sidecar(tmp_path, {"P001": {"head_variant": "invalid"}})
    apps = appearance.load_appearances(tmp_path, "gberch")
    assert "head_variant" not in apps.get("P001", {})


def test_face_variant_clamped_to_range(tmp_path):
    _write_sidecar(tmp_path, {"P001": {"face_variant": 9}})
    apps = appearance.load_appearances(tmp_path, "gberch")
    assert apps["P001"]["face_variant"] == 3


def test_save_and_reload_roundtrip(tmp_path):
    apps = {"P001": {"body_variant": 2, "hair_color": "#330000"}}
    appearance.save_appearances(tmp_path, "gberch", apps)
    reloaded = appearance.load_appearances(tmp_path, "gberch")
    assert reloaded["P001"]["body_variant"] == 2
    assert reloaded["P001"]["hair_color"] == "#330000"


def test_save_drops_empty_records(tmp_path):
    apps: dict = {"P001": {}, "P002": {"body_variant": 3}}
    appearance.save_appearances(tmp_path, "gberch", apps)
    reloaded = appearance.load_appearances(tmp_path, "gberch")
    assert "P001" not in reloaded
    assert "P002" in reloaded


def test_save_stable_key_order(tmp_path):
    apps = {"P003": {"body_variant": 1}, "P001": {"body_variant": 2}}
    appearance.save_appearances(tmp_path, "gberch", apps)
    raw = json.loads((tmp_path / "export" / "gberch_appearance.json").read_text())
    assert list(raw["players"].keys()) == ["P001", "P003"]  # sorted


def test_set_player_appearance_persists(tmp_path):
    rec = appearance.set_player_appearance(tmp_path, "gberch", "P001", body_variant=5)
    assert rec["body_variant"] == 5
    apps = appearance.load_appearances(tmp_path, "gberch")
    assert apps["P001"]["body_variant"] == 5


def test_set_player_appearance_merges_partial_updates(tmp_path):
    appearance.set_player_appearance(tmp_path, "gberch", "P001", body_variant=2)
    rec = appearance.set_player_appearance(tmp_path, "gberch", "P001", skin_tint="#abcdef")
    assert rec["body_variant"] == 2   # preserved from prior call
    assert rec["skin_tint"] == "#abcdef"


def test_set_player_appearance_ignores_unknown_keys(tmp_path):
    rec = appearance.set_player_appearance(tmp_path, "gberch", "P001", unknown_field="x")
    assert "unknown_field" not in rec
    assert rec["body_variant"] == 1   # default
```

- [ ] **Step 1.2 — Run tests, confirm FAIL**

```bash
cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_appearance.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'football_perspectives.appearance'`

- [ ] **Step 1.3 — Create appearance.py**

Create `Content/Python/football_perspectives/appearance.py`:

```python
"""Per-player appearance overrides: body/head variant, skin tone, face, hair.

Same pattern as team_overrides.py — sidecar JSON, no unreal import, fully
unit-testable outside the editor.

Sidecar: export/<clip>_appearance.json
Schema:
    {
      "schema": 1,
      "clip": "gberch",
      "players": {
        "P001": {
          "body_variant": 2,
          "head_variant": "head1",
          "face_variant": 1,
          "skin_tint": "#8B6F47",
          "tattoo": false,
          "hair_color": "#1a0a00",
          "hair_style": "head1"
        }
      }
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

SCHEMA = 1

_VALID_HEAD_VARIANTS = frozenset(("head", "head1"))
_VALID_HAIR_STYLES = frozenset(("head", "head1"))

_DEFAULTS: dict = {
    "body_variant": 1,
    "head_variant": "head",
    "face_variant": 1,
    "skin_tint": "",
    "tattoo": False,
    "hair_color": "",
    "hair_style": "head",
}

PlayerAppearances = Dict[str, dict]


def appearance_path(base_dir: Path, clip: str) -> Path:
    """Sidecar location: <base_dir>/export/<clip>_appearance.json."""
    return Path(base_dir) / "export" / f"{clip}_appearance.json"


def load_appearances(base_dir: Path, clip: str) -> PlayerAppearances:
    """Load appearance overrides from sidecar. Returns {} on missing/invalid file."""
    path = appearance_path(base_dir, clip)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    players = raw.get("players")
    if not isinstance(players, dict):
        return {}
    out: PlayerAppearances = {}
    for pid, rec in players.items():
        if not isinstance(rec, dict):
            continue
        clean: dict = {}
        bv = rec.get("body_variant")
        if isinstance(bv, int):
            clean["body_variant"] = max(1, min(6, bv))
        hv = rec.get("head_variant")
        if isinstance(hv, str) and hv in _VALID_HEAD_VARIANTS:
            clean["head_variant"] = hv
        fv = rec.get("face_variant")
        if isinstance(fv, int):
            clean["face_variant"] = max(1, min(3, fv))
        st = rec.get("skin_tint")
        if isinstance(st, str):
            clean["skin_tint"] = st
        if "tattoo" in rec:
            clean["tattoo"] = bool(rec["tattoo"])
        hc = rec.get("hair_color")
        if isinstance(hc, str):
            clean["hair_color"] = hc
        hs = rec.get("hair_style")
        if isinstance(hs, str) and hs in _VALID_HAIR_STYLES:
            clean["hair_style"] = hs
        out[pid] = clean
    return out


def save_appearances(base_dir: Path, clip: str, appearances: PlayerAppearances) -> Path:
    """Write the sidecar. Drops empty per-player entries; sorts by player_id."""
    path = appearance_path(base_dir, clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    players = {pid: rec for pid, rec in sorted(appearances.items()) if rec}
    payload = {"schema": SCHEMA, "clip": clip, "players": players}
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def resolve_appearance(player_id: str, appearances: PlayerAppearances) -> dict:
    """Merge stored record with defaults. Always returns a complete record."""
    rec = appearances.get(player_id) or {}
    return {**_DEFAULTS, **rec}


def set_player_appearance(
    base_dir: Path,
    clip: str,
    player_id: str,
    **kwargs,
) -> dict:
    """Update one player's appearance fields and persist. Returns resolved record.

    Accepts any subset of schema keys (_DEFAULTS). Unknown keys are ignored.
    """
    appearances = load_appearances(base_dir, clip)
    current = dict(appearances.get(player_id) or {})
    for key, val in kwargs.items():
        if key in _DEFAULTS:
            current[key] = val
    appearances[player_id] = current
    save_appearances(base_dir, clip, appearances)
    return resolve_appearance(player_id, appearances)
```

- [ ] **Step 1.4 — Run tests, confirm all PASS**

```bash
cd "/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python"
python -m pytest tests/test_appearance.py -v
```

Expected: 14 tests, all PASS.

---

## Task 2: Player part asset registry (BOW-84 Python side)

**Files:**
- Create: `Content/Python/football_perspectives/player_parts.py`

- [ ] **Step 2.1 — Create player_parts.py**

Create `Content/Python/football_perspectives/player_parts.py`:

```python
"""Asset-path registry for modular player part skeletal meshes.

All paths are /Game/... content-browser paths for unreal.load_asset().
Verify and correct these by browsing to Football_player in the UE Content
Browser. The exact subfolder structure (Mesh/, Meshes/, etc.) varies by
how the pack was imported.

Run discover_slots() in the UE Python console (Output Log → Python mode)
to print the Maya-exported material slot names for each mesh, then fill
in SLOT_MAP with the results.
"""
from __future__ import annotations

# Skeletal body variants (SK_football_player1–6)
BODY_VARIANT_PATHS: dict[int, str] = {
    1: "/Game/Football_player/Mesh/SK_football_player1",
    2: "/Game/Football_player/Mesh/SK_football_player2",
    3: "/Game/Football_player/Mesh/SK_football_player3",
    4: "/Game/Football_player/Mesh/SK_football_player4",
    5: "/Game/Football_player/Mesh/SK_football_player5",
    6: "/Game/Football_player/Mesh/SK_football_player6",
}

# Head/hair variants
HEAD_VARIANT_PATHS: dict[str, str] = {
    "head": "/Game/Football_player/Mesh/SK_football_player_head",
    "head1": "/Game/Football_player/Mesh/SK_football_player_head1",
}

# Clothing part meshes
KIT_PART_PATHS: dict[str, str] = {
    "shirt": "/Game/Football_player/Mesh/SK_football_player_t_shirt",
    "shorts": "/Game/Football_player/Mesh/SK_football_player_shorts",
    "socks": "/Game/Football_player/Mesh/SK_football_player_socks",
    "sneakers": "/Game/Football_player/Mesh/SK_football_player_sneakers",
}

# M_Kit master material (created in Task 4)
M_KIT_PATH = "/Game/Materials/M_Kit"

# Body material (for SkinTint / FaceTexture / TattooEnabled params)
BODY_MATERIAL_PATH = "/Game/Football_player/Materials/M_Football_player_BODY1"

# Hair material (for HairTint param)
HAIR_MATERIAL_PATH = "/Game/Football_player/Materials/M_Football_player_HAIR"

# Face texture variants
FACE_TEXTURE_PATHS: dict[int, str] = {
    1: "/Game/Football_player/Textures/T_Football_player_Body2_BaseColo6_face1",
    2: "/Game/Football_player/Textures/T_Football_player_Body2_BaseColo6_face2",
    3: "/Game/Football_player/Textures/T_Football_player_Body2_BaseColo6_face3",
}

# Maya-exported material slot names per part → kit colour role.
# "primary" = main kit colour; "secondary" = trim/accent.
# FILL IN after running discover_slots() in the UE Python console (Task 3).
# Format: {part_name: {maya_slot_name: "primary" | "secondary"}}
SLOT_MAP: dict[str, dict[str, str]] = {
    "shirt": {},     # e.g. {"blinn16": "primary", "blinn14": "secondary"}
    "shorts": {},    # e.g. {"phong1": "primary"}
    "socks": {},     # e.g. {"blinn12": "primary"}
    "sneakers": {},  # shoe slots — not kit-coloured by default
}

# BP_PlayerActor component names for each kit part (set in Task 7)
PART_COMPONENT_NAMES: dict[str, str] = {
    "shirt": "ShirtMesh",
    "shorts": "ShortsMesh",
    "socks": "SocksMesh",
    "sneakers": "SneakersMesh",
    "head": "HeadMesh",
}


def discover_slots() -> None:
    """Print material slot names for each part mesh.

    Run in the UE Python console (Output Log → Python command type):
        import importlib, football_perspectives.player_parts as pp
        importlib.reload(pp); pp.discover_slots()

    Copy-paste the output into SLOT_MAP above.
    """
    import unreal  # noqa: PLC0415 — only available inside UE

    all_meshes = {
        **{f"body_{i}": p for i, p in BODY_VARIANT_PATHS.items() if i == 1},
        **HEAD_VARIANT_PATHS,
        **KIT_PART_PATHS,
    }
    for label, path in all_meshes.items():
        mesh = unreal.load_asset(path)
        if mesh is None:
            print(f"  {label}: NOT FOUND at {path}")
            continue
        slot_names: list[str] = []
        for getter in ("get_editor_property('materials')",):
            try:
                mats = mesh.get_editor_property("materials")
                for m in (mats or []):
                    try:
                        slot_names.append(str(m.get_editor_property("material_slot_name")))
                    except Exception:
                        slot_names.append("?")
                break
            except Exception:
                pass
        print(f"  {label}: {slot_names}")
```

No offline tests needed for this file (pure constants). The `discover_slots` function is exercised in Task 3.

---

## Task 3: Discover material slot names (in-editor — no code written)

- [ ] **Step 3.1 — Open UE editor and run discovery**

In the UE editor **Output Log** (Tools → Output Log), switch command type to **Python** and run:

```python
import importlib, sys
for mod in list(sys.modules):
    if "player_parts" in mod:
        del sys.modules[mod]
from football_perspectives import player_parts
player_parts.discover_slots()
```

Note the printed slot names. Example output (actual values vary by pack import):
```
  body_1: ['Body', 'Body_LOD0']
  head: ['Head', 'Hair']
  head1: ['Head', 'Hair']
  shirt: ['blinn16', 'blinn14']
  shorts: ['phong1']
  socks: ['blinn12']
  sneakers: ['blinn10', 'blinn11']
```

- [ ] **Step 3.2 — Verify content-browser paths**

In the Content Browser, navigate to the football player asset folder. Confirm each path in `BODY_VARIANT_PATHS`, `HEAD_VARIANT_PATHS`, `KIT_PART_PATHS`, and `FACE_TEXTURE_PATHS` matches the actual asset location. Update `player_parts.py` if any path differs (e.g. `Meshes/` not `Mesh/`).

- [ ] **Step 3.3 — Fill in SLOT_MAP in player_parts.py**

Open `Content/Python/football_perspectives/player_parts.py`. Replace the empty dicts in `SLOT_MAP` with the discovered slot names. The "primary" slot is the main body-colour area of each mesh — inspect each mesh's material assignments in the Skeletal Mesh editor to confirm which slot is the kit colour. Example:

```python
SLOT_MAP: dict[str, dict[str, str]] = {
    "shirt": {
        "blinn16": "primary",
        "blinn14": "secondary",
    },
    "shorts": {
        "phong1": "primary",
    },
    "socks": {
        "blinn12": "primary",
    },
    "sneakers": {},  # not kit-coloured
}
```

Save `player_parts.py`.

---

## Task 4: Create M_Kit master material (BOW-93 — in-editor)

**Files (UE editor):** Create `/Game/Materials/M_Kit`

- [ ] **Step 4.1 — Create the material asset**

In Content Browser → right-click under `/Game/Materials/` (create the folder if absent) → **Material** → name it `M_Kit`. Double-click to open the Material Editor.

- [ ] **Step 4.2 — Add colour and pattern parameters**

Right-click in the graph → **Vector Parameter** → name `KitPrimary` → set default `(0.2, 0.3, 0.8, 1.0)` (blue). Repeat for:

| Type | Name | Default |
|------|------|---------|
| Vector Parameter | `KitPrimary` | (0.2, 0.3, 0.8, 1.0) |
| Vector Parameter | `KitSecondary` | (1.0, 1.0, 1.0, 1.0) |
| Scalar Parameter | `PatternBlend` | 0.0 |

- [ ] **Step 4.3 — Wire the graph**

Add a **LinearInterpolate** node (Lerp):
- A = `KitPrimary` output
- B = `KitSecondary` output
- Alpha = `PatternBlend` output
- Lerp output → **Base Color** pin on the Material Result node

If the pack includes cloth textures (search Content Browser for `T_Football_player_Clothes_Normal`):
- Add **Texture Parameter** named `ClothNormal` → default = that texture
- Add **TextureSample** node fed by `ClothNormal` → Normal output → **Normal** pin

Set **Blend Mode** = Opaque, **Shading Model** = Default Lit in the Details panel.

- [ ] **Step 4.4 — Save M_Kit**

Click **Apply** then **Save** in the Material Editor toolbar.

- [ ] **Step 4.5 — Assign M_Kit to kit part meshes**

For each of `SK_football_player_t_shirt`, `_shorts`, `_socks`:

1. Double-click the mesh → opens Skeletal Mesh editor
2. In the **Material Slots** section (right panel), find the "primary" slot (from SLOT_MAP in Task 3)
3. Click the slot picker → type `M_Kit` → select it
4. Click **Save** (Ctrl+S)

- [ ] **Step 4.6 — Spot-check**

Drag `SK_football_player_t_shirt` into a test level viewport. Select it → Details panel → **Materials** → the primary slot shows `M_Kit`. Change `KitPrimary` default to red (1,0,0,1) in M_Kit — shirt turns red in viewport. Revert to blue. Delete the test actor.

---

## Task 5: Add skin/face/tattoo params to body materials (BOW-86 — in-editor)

**Files (UE editor):** Modify `/Game/Football_player/Materials/M_Football_player_BODY1` (and all `BODY2*` variants)

- [ ] **Step 5.1 — Open M_Football_player_BODY1**

Double-click to open in Material Editor.

- [ ] **Step 5.2 — Add parameters**

Add these nodes to the graph:

| Type | Name | Default |
|------|------|---------|
| Vector Parameter | `SkinTint` | (1.0, 1.0, 1.0, 1.0) — neutral multiply |
| Texture Parameter | `FaceTexture` | `T_Football_player_Body2_BaseColo6_face1` |
| Scalar Parameter | `TattooEnabled` | 0.0 |

- [ ] **Step 5.3 — Wire SkinTint**

Find the existing base colour node chain (the TextureSample feeding Base Color or the existing Multiply chain). Insert a new **Multiply** node:
- A = existing base colour value (before the Material output)
- B = `SkinTint`
- Multiply output → **Base Color** on the Material Result

- [ ] **Step 5.4 — Wire FaceTexture**

If the mesh has a separate face UV region in its texture: find where the face texture is sampled (e.g. a `TextureSample` using `T_Football_player_Body2_BaseColo6_face1` as its texture). Replace the hardcoded texture reference with the `FaceTexture` parameter node feeding a `TextureSample`.

If the face area blends into the body texture via a mask: wire `FaceTexture` → `TextureSample` → feed into a `Lerp` using the existing face mask as Alpha, then route into the base colour chain.

If you cannot identify the face region in the graph, add `FaceTexture` as a parameter only (no graph connection yet); Python will call `set_texture_parameter_value("FaceTexture", ...)` to swap it, and the parameter just needs to exist.

- [ ] **Step 5.5 — Note TattooEnabled**

`TattooEnabled` is a scalar parameter that Python reads to decide which material to apply to the body mesh's tattoo slot. It does not need to be connected to the graph output — its presence as a named parameter lets Python's DMI code reference it as a flag. Add the node but leave it unconnected.

- [ ] **Step 5.6 — Repeat for all BODY2* materials**

Open each `M_Football_player_BODY2*` material. Add the same three parameters (`SkinTint`, `FaceTexture`, `TattooEnabled`) wired identically. Save each.

- [ ] **Step 5.7 — Save and verify compile**

Save all modified body materials. Confirm no red errors in the Material Editor stats bar.

---

## Task 6: Add HairTint param to hair material (BOW-87 — in-editor)

**Files (UE editor):** Modify `/Game/Football_player/Materials/M_Football_player_HAIR`

- [ ] **Step 6.1 — Open M_Football_player_HAIR**

Double-click in the Material Editor.

- [ ] **Step 6.2 — Add HairTint parameter**

Add a **Vector Parameter** named `HairTint` with default `(1.0, 1.0, 1.0, 1.0)`.

- [ ] **Step 6.3 — Wire HairTint into base colour**

Find the existing `TextureSample` for `T_Football_player_hair` (the hair colour texture). Insert a **Multiply** node:
- A = `TextureSample` RGB output (or existing colour chain)
- B = `HairTint`
- Multiply output → **Base Color** on Material Result

- [ ] **Step 6.4 — Save and spot-check**

Save `M_Football_player_HAIR`. Drag `SK_football_player_head` into a test level → change `HairTint` default to dark red (0.3, 0.0, 0.0, 1.0) → hair darkens. Revert to white (1,1,1,1). Delete test actor.

---

## Task 7: Modular assembly in BP_PlayerActor (BOW-84 — in-editor)

**Files (UE editor):** Modify `/Game/Players/BP_PlayerActor`

- [ ] **Step 7.1 — Safety copy**

In Content Browser, right-click `BP_PlayerActor` → **Duplicate** → name it `BP_PlayerActor_Monolithic_Backup`. This is a non-destructive reference in case the original needs to be reverted.

- [ ] **Step 7.2 — Open BP_PlayerActor**

Double-click → Blueprint editor opens.

- [ ] **Step 7.3 — Add Blueprint variables**

In **My Blueprint** panel → **Variables** section → click **+** for each:

| Name | Type | Instance Editable | Tooltip |
|------|------|-------------------|---------|
| `BodyVariant` | Integer | ✓ | Body skeleton variant 1–6 |
| `HeadVariant` | String | ✓ | Head/hair mesh: "head" or "head1" |

- [ ] **Step 7.4 — Add part mesh components**

In the **Components** panel, locate the existing body `SkeletalMeshComponent` (note its name, e.g. `MeshComp`). Add five **SkeletalMeshComponent** children:

For each, add as a child of the body mesh component (drag onto it in the hierarchy):

| Component name | Default skeletal mesh | Leader Pose |
|---------------|----------------------|-------------|
| `ShirtMesh` | `SK_football_player_t_shirt` | `MeshComp` |
| `ShortsMesh` | `SK_football_player_shorts` | `MeshComp` |
| `SocksMesh` | `SK_football_player_socks` | `MeshComp` |
| `SneakersMesh` | `SK_football_player_sneakers` | `MeshComp` |
| `HeadMesh` | `SK_football_player_head` | `MeshComp` |

For each new component, in the **Details** panel under **Mesh → Leader Pose Component**, set to the body `MeshComp`. This makes each part adopt the body's animated pose without re-evaluating the skeleton.

- [ ] **Step 7.5 — Add Construction Script for body variant**

Open the **Construction Script** graph (tab at top of Blueprint editor). After the existing content (if any):

1. Drag `BodyVariant` variable onto the graph → **Get BodyVariant**
2. Add a **Switch on Int** node. Connect `BodyVariant` to the Selection pin. Add cases 1–6.
3. For each case, add a **Set Skeletal Mesh Asset** node targeting `MeshComp`. Connect the corresponding skeletal mesh asset (drag from Content Browser into graph as an asset reference):
   - Case 1 → `SK_football_player1`
   - Case 2 → `SK_football_player2`
   - …
   - Case 6 → `SK_football_player6`

4. Drag `HeadVariant` variable → **Get HeadVariant**
5. Add a **Switch on String** node. Add cases `"head"` and `"head1"`.
6. Each case: **Set Skeletal Mesh Asset** targeting `HeadMesh`:
   - `"head"` → `SK_football_player_head`
   - `"head1"` → `SK_football_player_head1`

- [ ] **Step 7.6 — Compile and verify**

Click **Compile** (top toolbar). Zero errors. Click **Save**.

Drag `BP_PlayerActor` into a test level. In the Details panel, set `BodyVariant = 3` → body mesh changes to `SK_football_player3`. Set `HeadVariant = "head1"` → head mesh changes. Assign a retargeted anim → press **Simulate** → confirm shirt/shorts/socks/head all animate together with the body. Delete test actor.

---

## Task 8: Per-part kit colour helpers in kit_colors.py (BOW-93 Python side)

**Files:**
- Modify: `Content/Python/football_perspectives/kit_colors.py`

These functions are `unreal`-dependent and cannot be unit-tested offline. They are verified via the smoke test in Task 9.7.

- [ ] **Step 8.1 — Add import for player_parts**

At the top of `kit_colors.py`, after the existing imports, add:

```python
from football_perspectives import player_parts
```

- [ ] **Step 8.2 — Add KitPrimary/KitSecondary constants**

After the existing `_VECTOR_PARAM = "TeamColour"` line, add:

```python
_KIT_PRIMARY_PARAM = "KitPrimary"
_KIT_SECONDARY_PARAM = "KitSecondary"
```

- [ ] **Step 8.3 — Add _get_part_component helper**

After the existing `_try_set_live_actor_color` function, add:

```python
def _get_part_component(
    actor, component_name: str
) -> "Optional[unreal.SkeletalMeshComponent]":
    """Return a named SkeletalMeshComponent from a spawned actor, or None."""
    try:
        comps = actor.get_components_by_class(unreal.SkeletalMeshComponent)
        for c in comps:
            try:
                if c.get_name() == component_name:
                    return c
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return None


def _slot_index_map(comp: "unreal.SkeletalMeshComponent") -> "dict[str, int]":
    """Return {slot_name: material_index} for a mesh component."""
    result: dict[str, int] = {}
    try:
        n = comp.get_num_materials()
        for idx in range(n):
            try:
                name = str(comp.get_material_slot_name(idx))
                result[name] = idx
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return result
```

- [ ] **Step 8.4 — Add set_part_kit_colors**

```python
def set_part_kit_colors(
    actor,
    primary: unreal.LinearColor,
    secondary: Optional[unreal.LinearColor],
) -> int:
    """Push KitPrimary/KitSecondary onto the shirt/shorts/socks components.

    Creates or reuses a DMI from M_Kit on each kit-part slot identified in
    player_parts.SLOT_MAP. Returns the count of material slots updated.
    Used for the live-actor immediate re-tint path.
    """
    updated = 0
    for part_name, slot_map in player_parts.SLOT_MAP.items():
        if not slot_map:
            continue
        comp_name = player_parts.PART_COMPONENT_NAMES.get(part_name)
        if not comp_name:
            continue
        comp = _get_part_component(actor, comp_name)
        if comp is None:
            continue
        by_slot = _slot_index_map(comp)
        for slot_name, role in slot_map.items():
            idx = by_slot.get(slot_name)
            if idx is None:
                continue
            try:
                dmi = comp.create_dynamic_material_instance(idx)
                if dmi is None:
                    continue
                if role == "primary":
                    dmi.set_vector_parameter_value(_KIT_PRIMARY_PARAM, primary)
                elif role == "secondary" and secondary is not None:
                    dmi.set_vector_parameter_value(_KIT_SECONDARY_PARAM, secondary)
                updated += 1
            except Exception as exc:  # noqa: BLE001
                unreal.log_warning(
                    f"[kit_colors] {part_name}.{slot_name}: {exc!r}"
                )
    return updated
```

- [ ] **Step 8.5 — Add set_template_kit_colors**

```python
def set_template_kit_colors(
    binding: "unreal.MovieSceneBindingProxy",
    primary: unreal.LinearColor,
    secondary: Optional[unreal.LinearColor],
) -> bool:
    """Write KitPrimary/KitSecondary onto a spawnable's object template.

    Mirrors _set_template_color but targets the M_Kit params on the new
    modular actor. The legacy TeamColour path (apply_to_sequence) continues
    to work for BP_PlayerActor_No_Retarget which still uses M_PlayerKit.
    """
    template = _binding_template(binding)
    if template is None:
        return False
    ok = False
    try:
        template.set_editor_property(_KIT_PRIMARY_PARAM, primary)
        ok = True
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(f"[kit_colors] template {_KIT_PRIMARY_PARAM}: {exc!r}")
    if secondary is not None:
        try:
            template.set_editor_property(_KIT_SECONDARY_PARAM, secondary)
        except Exception as exc:  # noqa: BLE001
            unreal.log_warning(f"[kit_colors] template {_KIT_SECONDARY_PARAM}: {exc!r}")
    return ok
```

Save `kit_colors.py`.

---

## Task 9: Wire appearance into load_reconstruction.py (BOW-84/85/86/87/93/94 integration)

**Files:**
- Modify: `Content/Python/football_perspectives/load_reconstruction.py`

- [ ] **Step 9.1 — Add imports**

In `load_reconstruction.py`, add to the existing import block:

```python
from football_perspectives import appearance, player_parts
```

- [ ] **Step 9.2 — Add _apply_appearance_to_template**

After the existing `_resolve_sequence` function, add:

```python
def _apply_appearance_to_template(
    binding,
    rec: dict,
    primary: "unreal.LinearColor | None",
    secondary: "unreal.LinearColor | None",
) -> None:
    """Write appearance fields onto one player spawnable's object template.

    Sets BodyVariant + HeadVariant Blueprint variables from the appearance
    record, and writes KitPrimary/KitSecondary material params when a colour
    is available. All calls are guarded — failures log and skip.
    """
    template = kit_colors._binding_template(binding)
    if template is None:
        return

    for prop, val in (
        ("BodyVariant", int(rec.get("body_variant", 1))),
        ("HeadVariant", str(rec.get("head_variant", "head"))),
    ):
        try:
            template.set_editor_property(prop, val)
        except Exception as exc:  # noqa: BLE001
            unreal.log_warning(
                f"[football_perspectives] template {prop}={val!r}: {exc!r}"
            )

    if primary is not None:
        kit_colors.set_template_kit_colors(binding, primary, secondary)
```

- [ ] **Step 9.3 — Add _apply_skin_hair_to_actor**

```python
def _apply_skin_hair_to_actor(actor, rec: dict) -> None:
    """Push SkinTint, FaceTexture, HairTint to a live spawned actor.

    Best-effort — any failure logs and skips. Only runs on the live-actor
    path (EUW auto-apply); template writes handle the durable spawn state.
    """
    skin_tint = rec.get("skin_tint", "")
    hair_color = rec.get("hair_color", "")
    face_variant = int(rec.get("face_variant", 1))

    try:
        comps = actor.get_components_by_class(unreal.SkeletalMeshComponent)
    except Exception:  # noqa: BLE001
        return

    for comp in (comps or []):
        try:
            mesh = comp.get_editor_property("skeletal_mesh")
            if mesh is None:
                continue
            mesh_name = mesh.get_name().lower()
        except Exception:  # noqa: BLE001
            continue

        is_body = any(
            kw in mesh_name for kw in ("player1", "player2", "player3",
                                        "player4", "player5", "player6")
        )
        is_head = any(kw in mesh_name for kw in ("head", "hair"))

        if is_body and skin_tint:
            _push_param_to_all_slots(comp, "SkinTint",
                                     kit_colors.hex_to_linear_color(skin_tint))
        if is_body:
            face_path = player_parts.FACE_TEXTURE_PATHS.get(face_variant)
            if face_path:
                face_tex = unreal.load_asset(face_path)
                if face_tex is not None:
                    _push_texture_to_all_slots(comp, "FaceTexture", face_tex)
        if is_head and hair_color:
            _push_param_to_all_slots(comp, "HairTint",
                                     kit_colors.hex_to_linear_color(hair_color))


def _push_param_to_all_slots(
    comp: "unreal.SkeletalMeshComponent",
    param: str,
    value,
) -> None:
    try:
        n = comp.get_num_materials()
        for idx in range(n):
            dmi = comp.create_dynamic_material_instance(idx)
            if dmi is not None:
                dmi.set_vector_parameter_value(param, value)
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(f"[football_perspectives] {param}: {exc!r}")


def _push_texture_to_all_slots(
    comp: "unreal.SkeletalMeshComponent",
    param: str,
    texture,
) -> None:
    try:
        n = comp.get_num_materials()
        for idx in range(n):
            dmi = comp.create_dynamic_material_instance(idx)
            if dmi is not None:
                dmi.set_texture_parameter_value(param, texture)
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(f"[football_perspectives] {param}: {exc!r}")
```

- [ ] **Step 9.4 — Load appearances in load()**

In the `load()` function, find the block that resolves `overrides` and `color_by_player`:

```python
    overrides = team_overrides.load_overrides(base, clip)
    name_by_player = team_overrides.load_player_names(base)
    color_by_player = kit_colors.colors_by_player(m, overrides)
```

Add one line after:

```python
    apps = appearance.load_appearances(base, clip)
```

Then, after the existing `seq = build_sequence.build(...)` call, add the appearance application loop:

```python
    # Apply modular appearance (variant selection + per-part kit colours)
    # to each player spawnable's object template.
    name_to_pid = {
        (name_by_player.get(p.player_id) or p.player_id): p.player_id
        for p in m.players
    }
    for binding in seq.get_bindings():
        label = kit_colors._binding_display_name(binding)
        pid = name_to_pid.get(label) or label
        if not any(p.player_id == pid for p in m.players):
            continue  # ball / camera binding
        rec = appearance.resolve_appearance(pid, apps)
        primary = color_by_player.get(pid)
        secondary = None  # secondary colour: future enhancement
        _apply_appearance_to_template(binding, rec, primary, secondary)
```

Do the same inside `load_smpl()` (add `apps = appearance.load_appearances(base, clip)` and the same post-build loop).

- [ ] **Step 9.5 — Add set_appearance EUW entry point**

After the existing `reapply_overrides` function, add:

```python
def set_appearance(
    pipeline_output_dir: str,
    player_id: str,
    **kwargs,
) -> None:
    """Override one player's appearance and apply immediately.

    Entry point called by EUW on widget change (auto-apply). Persists to
    the appearance sidecar and pushes changes to the open sequence template
    and (best-effort) the live spawned actor.

    Accepts any appearance field kwargs: body_variant (int), head_variant
    (str "head"/"head1"), face_variant (int 1-3), skin_tint (str "#rrggbb"),
    tattoo (bool), hair_color (str "#rrggbb"), hair_style (str).
    """
    base = Path(str(pipeline_output_dir)).expanduser()
    try:
        m = manifest.load(base / "export" / "ue_manifest.json")
    except manifest.UeManifestError as exc:
        _fail(f"Manifest invalid:\n{exc}")
        return

    clip = m.clip_name
    pid = str(player_id)
    rec = appearance.set_player_appearance(base, clip, pid, **kwargs)

    seq = _resolve_sequence(clip)
    if seq is not None:
        name_by_player = team_overrides.load_player_names(base)
        label = name_by_player.get(pid) or pid
        overrides = team_overrides.load_overrides(base, clip)
        role = team_overrides.role_for_player(
            pid, overrides, next(
                (p.kit_role for p in m.players if p.player_id == pid), "unknown"
            )
        )
        primary = kit_colors.color_for_role(m, role)
        for binding in seq.get_bindings():
            if kit_colors._binding_display_name(binding) == label:
                _apply_appearance_to_template(binding, rec, primary, None)
                break
        unreal.EditorAssetLibrary.save_asset(seq.get_path_name())

    # Best-effort live-actor push (skin/hair only — mesh swap happens on
    # next spawn when the template is re-read)
    try:
        label = (team_overrides.load_player_names(base).get(pid) or pid)
        eas = unreal.EditorActorSubsystem()
        for actor in eas.get_all_level_actors():
            if actor.get_actor_label() == label:
                _apply_skin_hair_to_actor(actor, rec)
                break
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(
            f"[football_perspectives] live appearance push: {exc!r}"
        )

    unreal.log(f"[football_perspectives] appearance updated for {pid}: {rec}")
```

- [ ] **Step 9.6 — Add player_appearance_rows EUW entry point**

After `player_team_rows`, add:

```python
def player_appearance_rows(pipeline_output_dir: str) -> list[dict]:
    """Per-player rows for the EUW appearance panel.

    Returns a list of dicts, one per player, with all team override fields
    and all appearance fields resolved (merged with defaults). The widget
    uses this to populate its per-player controls on Refresh.

    Row keys: player_id, name, role, team, is_gk, is_ref,
              body_variant, head_variant, face_variant, skin_tint,
              tattoo, hair_color, hair_style.
    """
    base = Path(str(pipeline_output_dir)).expanduser()
    m = manifest.load(base / "export" / "ue_manifest.json")
    overrides = team_overrides.load_overrides(base, m.clip_name)
    apps = appearance.load_appearances(base, m.clip_name)
    name_by_player = team_overrides.load_player_names(base)
    rows: list[dict] = []
    for p in m.players:
        pid = p.player_id
        ov = overrides.get(pid)
        role = team_overrides.role_for_player(pid, overrides, p.kit_role)
        rec = appearance.resolve_appearance(pid, apps)
        rows.append({
            "player_id": pid,
            "name": name_by_player.get(pid) or pid,
            "role": role,
            "team": (ov or {}).get("team", ""),
            "is_gk": bool((ov or {}).get("is_gk", False)),
            "is_ref": bool((ov or {}).get("is_ref", False)),
            **rec,
        })
    return rows
```

- [ ] **Step 9.7 — Smoke test in UE Python console**

Open UE editor with a loaded reconstruction. In the Python console:

```python
import importlib, sys
for mod in list(sys.modules):
    if "football_perspectives" in mod:
        del sys.modules[mod]
from football_perspectives import load_reconstruction

OUTPUT = "/Users/joebower/workplace/output-kroupi"  # or the actual output dir

rows = load_reconstruction.player_appearance_rows(OUTPUT)
for r in rows:
    print(r["player_id"], r["body_variant"], r["role"])
```

Expected: one row per player, `body_variant=1`, roles from manifest, no exception.

Test `set_appearance`:

```python
load_reconstruction.set_appearance(OUTPUT, rows[0]["player_id"], body_variant=2)
```

Expected: log line `appearance updated for P0XX`, sidecar written to `export/<clip>_appearance.json`, no exception.

---

## Task 10: EUW appearance panel (BOW-94 — in-editor UMG)

**Files (UE editor):** Modify `EUW_LoadReconstruction`

This also closes the pending §8 team-overrides UI wiring from the kit-overrides design (the Python backend was done; the widget wiring was not).

- [ ] **Step 10.1 — Open EUW_LoadReconstruction**

In Content Browser, double-click `EUW_LoadReconstruction` → UMG Designer.

- [ ] **Step 10.2 — Create WBP_PlayerRow child widget (recommended)**

To keep the main widget manageable, create a child **Widget Blueprint** named `WBP_PlayerRow` (right-click in Content Browser → User Interface → Widget Blueprint). This widget represents one player row and owns all its controls.

Add a **Horizontal Box** root. Inside add:

| Control | Type | Width |
|---------|------|-------|
| Player name | Text Block | 120 |
| Team | ComboBox String | 80 |
| GK | CheckBox | 40 |
| Ref | CheckBox | 40 |
| Body | SpinBox (int, 1–6) | 50 |
| Head | ComboBox String ("head","head1") | 80 |
| Face | SpinBox (int, 1–3) | 50 |
| Skin | Editable Text (hex) | 80 |
| Hair col | Editable Text (hex) | 80 |
| Tattoo | CheckBox | 40 |

Add a **Variable** named `PlayerID` (String) and `OutputDir` (String) to `WBP_PlayerRow`. These are set by the parent before the row is added.

- [ ] **Step 10.3 — Wire onChange events in WBP_PlayerRow**

For each control, bind the appropriate changed event to an **Execute Python Script** node. Use the player row's `PlayerID` and `OutputDir` variables.

Team ComboBox `OnSelectionChanged` (SelectedItem: String):
```python
from football_perspectives import load_reconstruction
load_reconstruction.set_team(
    "{OutputDir}", "{PlayerID}",
    team=("home" if "{SelectedItem}" == "Home" else
          "away" if "{SelectedItem}" == "Away" else ""),
)
```

GK CheckBox `OnCheckStateChanged` (bIsChecked: Boolean):
```python
from football_perspectives import load_reconstruction, team_overrides
from pathlib import Path
base = Path("{OutputDir}")
m_clip = __import__("football_perspectives.manifest", fromlist=["manifest"]).load(
    base / "export" / "ue_manifest.json").clip_name
ov = team_overrides.load_overrides(base, m_clip).get("{PlayerID}", {})
load_reconstruction.set_team(
    "{OutputDir}", "{PlayerID}",
    team=ov.get("team", ""),
    is_gk={bIsChecked},
    is_ref=bool(ov.get("is_ref", False)),
)
```

Body SpinBox `OnValueChanged` (InValue: Float):
```python
from football_perspectives import load_reconstruction
load_reconstruction.set_appearance("{OutputDir}", "{PlayerID}", body_variant=int({InValue}))
```

Head ComboBox `OnSelectionChanged` (SelectedItem: String):
```python
from football_perspectives import load_reconstruction
load_reconstruction.set_appearance("{OutputDir}", "{PlayerID}", head_variant="{SelectedItem}")
```

Face SpinBox `OnValueChanged`:
```python
from football_perspectives import load_reconstruction
load_reconstruction.set_appearance("{OutputDir}", "{PlayerID}", face_variant=int({InValue}))
```

Skin text `OnTextCommitted` (Text: Text, CommitMethod: ETextCommit):
```python
from football_perspectives import load_reconstruction
load_reconstruction.set_appearance("{OutputDir}", "{PlayerID}", skin_tint="{Text}")
```

Hair Color text `OnTextCommitted`:
```python
from football_perspectives import load_reconstruction
load_reconstruction.set_appearance("{OutputDir}", "{PlayerID}", hair_color="{Text}")
```

Tattoo CheckBox `OnCheckStateChanged`:
```python
from football_perspectives import load_reconstruction
load_reconstruction.set_appearance("{OutputDir}", "{PlayerID}", tattoo={bIsChecked})
```

- [ ] **Step 10.4 — Add Refresh Player List to EUW_LoadReconstruction**

In the main `EUW_LoadReconstruction` widget:

1. Add a **Scroll Box** below the existing Load buttons (or to the right in a horizontal split)
2. Add a **Button** labelled "Refresh Player List" above the Scroll Box
3. Bind the button's `OnClicked` to an **Execute Python Script** node:

```python
import importlib, sys
for mod in list(sys.modules.keys()):
    if "football_perspectives" in mod:
        del sys.modules[mod]
from football_perspectives import load_reconstruction

rows = load_reconstruction.player_appearance_rows("{OutputDir}")
# Clear and repopulate the scroll box via WBP_PlayerRow children.
# (Wire the Clear + Create Widget + Add Child nodes in Blueprint graph
# adjacent to this Execute Python Script node.)
```

After the Python node, add Blueprint nodes to:
1. Call **Remove All Children** on the Scroll Box
2. `ForEach` over the Python-returned rows (cast via `Make Array` of dicts or iterate via a Blueprint-side variable)
3. **Create Widget** (`WBP_PlayerRow`) → set `PlayerID` and `OutputDir` from the row data → **Add Child** to Scroll Box

> Note: Returning a Python list of dicts to Blueprint requires UE's Python-BP bridge. An alternative is to run `player_appearance_rows` from Python in the same Execute Python Script node, store results in a `unreal.Array` via `unreal.Array(unreal.StructBase)` if a struct is defined, or store to a temp JSON file and read it back from Blueprint. The simplest approach: define a `refresh_player_list_widget(euw_ref, output_dir)` Python function that creates and adds the WBP_PlayerRow widgets itself using `euw_ref.scroll_box_players.add_child_to_vertical_box(...)` — then the Blueprint just calls that one Python function.

- [ ] **Step 10.5 — End-to-end verification**

1. Open the UE editor. Load a reconstruction (Load Reconstruction button).
2. Click **Refresh Player List** — player rows appear in the scroll box.
3. Change **Body** spinner for P001 from 1 to 3 — log line appears `appearance updated for P001`, sidecar written.
4. Change **Hair** text field to `#8B0000` → log line, sidecar updated.
5. Change **Team** to Away → player kit changes colour immediately.
6. Click **Load Reconstruction** again (re-import) → all changes persist (body variant 3, hair tint, team colour all applied from sidecars).

---

## Self-Review

**Spec coverage check:**

| Requirement | Task(s) |
|-------------|---------|
| BOW-84: Modular assembly body+parts with leader pose | Task 7 (BP), Task 2/3 (Python paths), Task 9 (wiring) |
| BOW-84: BodyVariant/HeadVariant vars set from Python | Task 7 (BP vars), Task 9 (`_apply_appearance_to_template`) |
| BOW-85: Appearance sidecar load/save/resolve | Task 1 (appearance.py + tests) |
| BOW-85: Applied on import | Task 9.4 |
| BOW-86: Skin tint (vector param, multiply) | Task 5 (material), Task 9 (`_apply_skin_hair_to_actor`) |
| BOW-86: Face texture variant selection | Task 5 (FaceTexture param), Task 9.3 |
| BOW-86: Tattoo toggle | Task 5 (TattooEnabled param), Task 9.3 |
| BOW-87: Hair colour tint | Task 6 (material), Task 9.3 |
| BOW-87: Hairstyle via head-swap | Task 7 (HeadMesh + HeadVariant), Task 9 (template write) |
| BOW-93: M_Kit with KitPrimary/KitSecondary | Task 4 (material), Task 8 (kit_colors.py) |
| BOW-93: Per-part MIDs (shirt/shorts/socks separate) | Task 7 (separate components), Task 8 (set_part_kit_colors) |
| BOW-93: Slot-mapping table | Task 2 (player_parts.SLOT_MAP), Task 3 (discovery) |
| BOW-94: Player list with team + appearance controls | Task 10 (UMG) |
| BOW-94: Auto-apply on change | Task 10.3 (onChange wiring) |
| BOW-94: Persist via sidecars | Task 9.5 (set_appearance persists), Task 10 (wires to it) |
| BOW-94: §8 team-overrides UI wiring | Task 10 (Team/GK/Ref controls wire to set_team) |

**Placeholder check:** No TBD/TODO in code blocks. All function bodies are complete. `SLOT_MAP` is intentionally empty — filled in Task 3 after in-editor discovery.

**Type consistency:** `appearance.resolve_appearance()` returns `dict`. `_apply_appearance_to_template` consumes it with `.get()`. `set_player_appearance` returns `resolve_appearance(...)` output. Consistent throughout.

**Backward compat:** `build_sequence.build()` continues to call `kit_colors.apply_to_sequence` (writes `TeamColour`) — this remains correct for `BP_PlayerActor_No_Retarget` which still uses `M_PlayerKit`. The new `_apply_appearance_to_template` call (after `build()`) additionally writes `KitPrimary` via `set_template_kit_colors` for the main `BP_PlayerActor` which now uses `M_Kit`. If `M_Kit` doesn't expose `TeamColour`, the old call logs a warning and skips — non-fatal.

# In-Engine Team Assignment & Kit Colour Overrides — Design

- **Date:** 2026-05-29
- **Author:** Joe Bower (with Claude)
- **Status:** Implemented (Python backend) — in-editor UI wiring pending (see §8)
- **Related:** `src/utils/team_roles.py`, UE project `Content/Python/football_perspectives/` (`kit_colors.py`, `load_reconstruction.py`, `manifest.py`)

## 1. Problem

The pipeline already classifies each player track into a *kit role*
(`home` / `away` / `home_gk` / `away_gk` / `referee` / `unknown`) and emits a
role → hex-colour palette into `export/ue_manifest.json` (`UeManifest.kits`).
On import, `kit_colors.apply_to_sequence` writes the resolved colour onto each
player spawnable's `TeamColour` material parameter.

Two gaps remain:

1. **Team assignment from broadcast footage is error-prone.** The pipeline's
   team A/B and goalkeeper/referee classification is frequently wrong, so a
   player ends up in the wrong kit (or `unknown` grey).
2. **There is no in-engine way to fix it.** Today the only remedy is editing
   the pipeline output and re-exporting. The user wants to set/override a
   player's team *in the editor*, have the kit colour update, and have that
   choice **persist** and be **easily overwriteable**.

## 2. Goals / Non-Goals

**Goals**
- Per-player override of **team** (Home/Away) plus a **goalkeeper** and a
  **referee** toggle, resolving to a canonical kit role.
- Overridden role drives the kit colour via the existing `manifest.kits`
  palette and `M_PlayerKit`'s `TeamColour` parameter.
- Overrides **persist** to a sidecar JSON in the pipeline output dir and are
  honoured on subsequent imports.
- Changing an override **auto-applies** (re-tints) without a separate button.

**Non-Goals**
- Editing the kit *colours themselves* in-engine (colours come from match
  metadata / the manifest palette; tweak there or on `M_PlayerKit`).
- Round-tripping overrides back into the pipeline's `players.json` (out of
  scope; the sidecar is the source of truth for the in-engine layer).
- Per-frame team changes (a player's team is constant for the clip).

## 3. Decisions (captured from brainstorming)

| Question | Decision |
|---|---|
| Where to edit overrides | The `EUW_LoadReconstruction` editor widget |
| Pick granularity | **Team (Home/Away) + GK toggle + Referee toggle** |
| Persistence | **Sidecar JSON** in the output dir (`export/<clip>_team_overrides.json`) |
| Apply trigger | **Auto-apply on edit** (instant re-tint of the open sequence) |
| Current state | In-engine colouring not yet working for this clip |

## 4. Approaches considered

**A. Python-backed override layer (extends existing `kit_colors` path) — CHOSEN.**
A small, `unreal`-free `team_overrides` module owns the sidecar (load/save) and
the `(team, is_gk, is_ref) → kit_role` resolution. `kit_colors.colors_by_player`
gains an optional `overrides` argument; when present, the override-derived role
wins over the manifest role, then the colour is looked up in the same palette.
A `set_team(...)` entry point updates the sidecar and re-tints. The widget calls
these Python functions.
*Pros:* reuses the working colour path; the only real logic is pure and unit
-testable; minimal binary-asset surgery (the material/BP already read
`TeamColour`). *Cons:* the polished per-player widget UI still needs UMG work.

**B. Actor-property-driven (Approach 2).** Add instance-editable
`Team`/`bIsGoalkeeper`/`bIsReferee` variables + a `KitColors` map to
`BP_PlayerActor`, and an OnConstruction graph that re-tints from those. The
widget just sets the properties.
*Pros:* single source of truth on the actor; Details-panel editing for free;
construction script gives robust editor re-tint. *Cons:* heavy Blueprint-graph
surgery; **not achievable via the current MCP toolset** (see §8) and risky to
author blind on a non-version-controlled project. Recorded as a future
enhancement.

**C. Central DataAsset.** A per-clip DataAsset lists every track → team; the
actor reads it; the widget edits it. *Pros:* bulk editing. *Cons:* more
indirection than the user wants; still needs binary-asset authoring.

## 5. Architecture (chosen)

```
EUW_LoadReconstruction (UMG)            ← in-editor surface (UI wiring pending)
        │  set_team(base, player_id, team, gk, ref)   on row edit
        ▼
team_overrides.py        (no `unreal` dep, unit-testable)
   • OVERRIDE_FILENAME = export/<clip>_team_overrides.json
   • load_overrides(base, clip) -> {player_id: {team,is_gk,is_ref}}
   • save_overrides(base, clip, overrides)
   • set_player_override(base, clip, pid, team, gk, ref) -> kit_role
   • kit_role_from_override(team, is_gk, is_ref)  (mirror of pipeline)
        │
        ▼
kit_colors.py
   • colors_by_player(m, overrides=None) -> {player_id: LinearColor}
       override role (if any) wins over manifest kit_role; colour from kits map
   • apply_to_sequence(seq, colors)        (existing; template param write)
   • apply_player_color(seq, pid, color)   (new; single player + best-effort
                                            live-actor DMI push)
        │
        ▼
M_PlayerKit.TeamColour  ──(DMI in BP_PlayerActor)──►  player kit material
```

Canonical mapping (shared contract, mirrored in pipeline + UE):

```
is_ref            → "referee"
team home + gk    → "home_gk"     team home → "home"
team away + gk    → "away_gk"     team away → "away"
otherwise         → "unknown"
```

This reuses `src/utils/team_roles.derive_kit_role` semantics: referee class
wins; goalkeeper promotes a side's outfield role to its keeper role.

## 6. Data: sidecar override file

`export/<clip>_team_overrides.json`:

```json
{
  "schema": 1,
  "clip": "gberch",
  "players": {
    "P001": {"team": "away"},
    "P003": {"team": "home", "is_gk": true},
    "P007": {"is_ref": true}
  }
}
```

- Only overridden players appear; absent players keep the pipeline role.
- `team` ∈ {`home`,`away`}; `is_gk`/`is_ref` default `false`.
- `is_ref: true` ignores `team`.
- Read at import (`load_reconstruction.load`) and merged before colouring.

## 7. Application semantics

- **On import:** load sidecar → `colors_by_player(m, overrides)` →
  `apply_to_sequence` writes templates (existing behaviour, now override-aware).
- **On edit (auto-apply):** widget row change → `set_team(...)` → updates
  sidecar, recomputes that player's colour, calls `apply_player_color` to set
  the spawnable template and (best-effort) the live spawned actor's DMI.
- Robustness: all `unreal` API calls are wrapped; a single failed player logs
  and is skipped rather than aborting the load (matches existing `kit_colors`
  style). Bad hex falls back to grey.

## 8. Implementation constraint (important)

The Unreal MCP server in use exposes tools via a dynamic *toolset-loading*
mechanism (`list_toolsets`/`load_toolset`/`describe_toolset`). In the current
Claude Code session the loaded tool schemas register for **reading** but the
individual tools (`find_assets`, `set_properties`, `create_node`, material/BP
editors, …) are **not invokable** through the harness bridge — only the three
management tools are. Consequently this iteration implements everything that
lives in **Python text files on disk** (fully reviewable/revertible) and does
**not** perform live Blueprint/Material/UMG edits.

Remaining in-editor work (needs working MCP tools or hands-on editor time):
1. **`EUW_LoadReconstruction` UI:** a scrollable per-player list with a Team
   dropdown + GK/Ref checkboxes, each row calling `set_team(...)` on change.
   (UMG designer authoring is not covered by the available MCP toolsets.)
2. **Optional Approach-B upgrade:** instance-editable role properties +
   OnConstruction re-tint on `BP_PlayerActor` for Details-panel editing and
   crisp editor-viewport feedback.
3. **Verify** `M_PlayerKit` truly has a `TeamColour` vector param and that
   `BP_PlayerActor` builds its DMI from it (code comments assert this; confirm
   in-editor).

Until the UI exists, the feature is usable by editing the sidecar JSON and
re-running Load Reconstruction (or calling `set_team` from the Python console),
which satisfies "easily overwriteable + persisted".

## 9. Testing

- Pipeline `kit_role_from_override` covered by unit tests in
  `tests/test_team_roles.py` (parametrised over team × gk × ref, including
  referee-wins-over-team and unknown fallback).
- The UE `team_overrides` mirror is intentionally a thin, no-deps copy of the
  same mapping; `unreal`-dependent glue can only be exercised in-editor.

## 10. Risks

- **Manifest schema drift:** the camera-perspectives work may bump the pipeline
  manifest to v3 while the UE reader pins v2 (`manifest.SCHEMA_VERSION`), which
  would make `load()` hard-fail for the whole clip (and thus block colouring).
  Out of scope here but flagged — verify the gberch export's `schema_version`.
- **Editor live re-tint:** spawnable templates update reliably; pushing to an
  already-spawned editor-preview actor's DMI is best-effort (BeginPlay does not
  run in editor preview). Approach-B's construction script is the durable fix.
- **No VCS on the UE project:** `FootballPerspectives 5.8` is not a git repo, so
  UE-side Python changes have no automatic safety net — kept additive and
  backward-compatible (`overrides` defaults to `None` = prior behaviour).

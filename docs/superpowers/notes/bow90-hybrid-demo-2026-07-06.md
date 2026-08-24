# BOW-90 Hybrid Player Models — Retarget Chain Validation (2026-07-06)

**Status: the Mannequin→MetaHuman retarget chain is BUILT and VALIDATED** against
the gberch reconstruction, cloud-free, using the MetaHuman body template. The
foot-to-ball offset risk (the issue's top open risk) is closed: **≤7 cm error,
typically ~2 cm** (was 1.5–1.9 m before the pelvis fix).

## What exists now

- `/Game/Skeleton/IKR_Mannequin_to_MetaHuman` — IK Retargeter,
  source `IK_Mannequin` (5 chains) → target `IK_MH_IKRig` (engine,
  `/MetaHumanCharacter/Animation/Retargeting/`), target preview mesh
  `SKM_Body` (`/MetaHumanCharacter/Body/IdentityTemplate/`).
- `/Game/Reconstructions/gberch/PlayersMH/P006_anim_mh` — Gravenberch's anim on
  the MetaHuman skeleton (`SMPL → Manny → MH`, second hop via
  `IKRetargetBatchOperation.duplicate_and_retarget`).
- LS_gberch demo binding "Gravenberch (MetaHuman)" — SkeletalMeshActor spawnable
  with `SKM_Body` playing `P006_anim_mh` alongside 21 pack players.
- `appearance.py` `model` field: `"pack" | "metahuman:<Asset>"` +
  `metahuman_asset_name()` helper; 86/86 offline tests pass.

## UE 5.8 retargeter gotchas (op-based rework)

1. A fresh `IKRetargeter` has an EMPTY op stack — `auto_map_chains` silently
   maps nothing until `controller.add_default_ops()` is called.
2. `AutoMapChainType.FUZZY` mis-maps chains that have no source counterpart
   (Root←RightArm, Head←LeftArm). Null everything except the five real pairs:
   Spine, Left/RightArm, Left/RightLeg.
3. **Pelvis op**: default proportional scaling shrank root translation by ~0.94
   (MH template height ratio) → players drifted ~1.5 m off their pitch
   positions. Fix: `blend_to_source_translation = 1.0` with per-axis weights
   `(1, 1, 0)` — XY copied 1:1 from source (reconstruction truth), Z stays
   proportional.
4. **"Run IK Rig" op must be DISABLED**: with unset/default IK goals it drags a
   leg sideways into a giant "fin" and hunches the torso. "Root Motion" op also
   disabled (pelvis blend handles world motion; root bone unused downstream).
5. Deleting an AnimSequence while an open Sequencer references it crashes the
   editor (packed-ref referencer scan). `close_level_sequence()` first.

## Known cosmetic gaps (follow-ups)

- Head/neck motion is dropped: `IK_Mannequin` has no Neck/Head chains, so the
  MH head holds the retarget pose. Add Neck + Head chains to `IK_Mannequin`
  (spine_05→neck_01→head exists on both skeletons) and remap.
- Fingers hold ref pose (SMPL has no fingers — same as pack, fine).
- MH body renders as the grey template — full hero body needs the assembled
  character (blocked on Epic sign-in, below).

## Cloud blocker for full hero characters

`/Game/MetaHumans/Hero01` (MetaHumanCharacter asset) exists, but auto-rigging
and texture synthesis are Epic **cloud** calls; with no signed-in Epic session
the blocking call asserts (`WaitForCloudRequests` → invalid SharedPtr) and
kills the editor. **One-time user step: open the MetaHuman Character editor and
sign in to the Epic account**, then `request_auto_rigging` / 
`request_texture_sources` / `build_meta_human(OPTIMIZED, MEDIUM)` complete the
hero pipeline (see `test_character_assembly.py` in the MetaHumanCharacter
plugin for the exact flow).

## Remaining build scope (from the Linear issue)

- [ ] `build_sequence`: branch player spawnable on `appearance` `model` field
      (SkeletalMeshActor + MH body + `<pid>_anim_mh` instead of BP_PlayerActor)
- [ ] `load_reconstruction`: retarget-on-load for `metahuman:` players (reuse
      `retarget.py` pattern with `IKR_Mannequin_to_MetaHuman`)
- [ ] Hero01 assembly after Epic sign-in; swap demo binding to assembled body
- [ ] Kit re-skin onto MH body (artist work) / kit-coloured bodysuit interim
- [ ] Neck/Head chains in IK_Mannequin
- [ ] KeenTools likeness pipeline (artist work, legal review first)

## Update 2026-07-07 — Hero01 assembled; Sequencer integration blocked two ways

**Hero01 is fully assembled and saved**: Epic sign-in done (auth persists across
editor restarts); rig + textures fetched via **non-blocking** cloud requests
(`blocking=True` is unusable over remote execution — the wait pumps FTSTicker,
which re-enters the Python remote-exec ticker mid-command and asserts). Poll
`can_build_meta_human` instead (~1 min). `build_meta_human(OPTIMIZED, MEDIUM)`
output: `SKM_Hero01_BodyMesh`, `SKM_Hero01_FaceMesh`, `BP_Hero01` under
`/Game/MetaHumans/Hero01_Build/` — **save the build directory immediately**
(first build was lost to a later crash).

**Both Sequencer integration routes for the full BP_Hero01 are blocked in 5.8.0:**
1. **Spawnable**: MetaHuman BPs build components in construction scripts →
   the spawnable-destroy path crashes (`DestroySpawnedObject` →
   `IsCreatedByConstructionScript` SIGSEGV) whenever the binding respawns.
2. **Possessable**: possessing the actor (or its Body component) poisons the
   whole sequence evaluation — the Face's ARKit control-rig mapping fails a
   skeleton-identity check (`AS_MetaHuman_ARKit_Mapping: Provided Skeleton
   Face_Archetype_Skeleton does not match bound ControlRigObject`), open
   errors, and **zero spawnables evaluate** until the bindings are removed
   AND the editor restarts.

**Workaround options for the demo** (untested, next session):
- Level-actor + direct component animation (`ANIMATION_SINGLE_NODE`,
  `play_animation` + `set_position`) — no Sequencer involvement; first attempt
  didn't render visibly before context ran out, needs a tick/pose refresh
  investigation.
- Body-only SkeletalMeshActor possessable (headless — pair with a separate
  Face attachment later).
- Strip the Face component from a duplicated BP_Hero01 variant for sequencing.

**Current state**: LS_gberch restored to pack-only (Hero bindings removed,
saved); `Hero01_Gravenberch` level actor parked at the pitch offset origin;
`P006_anim_mh` valid (feet ≤7 cm vs pack).

## Update 2026-07-07 (later) — POSSESSABLE ROUTE WORKS; yesterday's "poisoning" was stale-LS state

**The level-actor possessable route is VALIDATED on 5.8.0.** BP_Hero01 spawned
as a level actor (`Hero01_Gravenberch`), `seq.add_possessable(actor)` with a
`MovieSceneSkeletalAnimationTrack` playing `P006_anim_mh` **on the actor
binding** (no Body-component sub-binding needed — the track finds the Body
comp), plus the standard constant transform offset (−3400, −5250, yaw 90).
Result at frames 200/300: hero Body pelvis within **5 cm** of the pack
Gravenberch's pelvis, root motion carried, full running pose, all 22 pack
spawnables + ball + cameras evaluating normally alongside. No ARKit errors.

**Re-diagnosis of yesterday's blocker:** the "possessable poisons all
spawnables" symptom was (at least today) reproducible with NO Hero binding at
all — LS_gberch itself was carrying stale packed object refs. Closing that
sequencer crashed with the known `GetObjectDataFromId` assert
(`ACameraActor::GetDefaultAttachComponent` during SpawnRegister cleanup).
Recovery per protocol: delete `LS_gberch.uasset`, relaunch, rebuild the LS.
After the rebuild the possessable route worked first try. The 5.8.0 SPAWNABLE
construction-script crash remains real — keep using possessables for MetaHumans.

**Debugging traps hit today (do not repeat):**
- `EditorActorSubsystem.get_all_level_actors()` does NOT list
  Sequencer-spawned transients in 5.8 — "zero spawnables" from that probe is a
  false signal. Use `LevelSequenceEditorBlueprintLibrary.get_bound_objects`
  (build `MovieSceneObjectBindingID` with `guid = binding.get_id()`).
- Verify player positions from a bound object's pelvis socket, not from a
  remembered foot coordinate — screenshots of empty pitch were just a
  mis-aimed camera.
- `load()` re-runs wipe `/Game/Reconstructions/<clip>` INCLUDING `PlayersMH/`;
  regen via the batch retarget (source `SKM_Manny_Simple`, target
  `/MetaHumanCharacter/Body/IdentityTemplate/SKM_Body`, suffix `_mh`) takes
  seconds.

**Cosmetic gaps in the possessable demo:** the Face component does not follow
the Body while Sequencer drives it (stays at the actor transform in ref pose)
— currently `set_visibility(False)` on the level actor, so the hero is
headless up close; head/neck retarget chains still missing anyway. Kit is the
grey/skin default MetaHuman body (no shirt/shorts).

**Saved state:** LS_gberch (rebuilt, pack + "Gravenberch (Hero01)"
possessable) and FootballStadium level (with `Hero01_Gravenberch` actor, Face
hidden) both saved; editor healthy.

## Update 2026-07-07 (evening) — MetaHuman-by-default Load Reconstruction SHIPPED

Load Reconstruction now renders every player as a MetaHuman in a
parameterised football kit (gberch: 22/22 through the real `load()` path,
verified at frames 200/300 — positions/pose match pack within ~5 cm, heads
animate).

**What was built:**
- `appearance.py`: `model` default flipped `pack` → `metahuman` (bare token =
  template body); `metahuman:<Asset>` reserved for future likenesses; `pack`
  is the per-player opt-out. 87/87 offline tests pass.
- `retarget.py`: `retarget_player_anim_mh()` — Manny→MH second hop via
  `IKR_Mannequin_to_MetaHuman` into `PlayersMH/<pid>_anim_mh` (shared
  `_run_batch_retarget` core; deletes an existing dest before rename).
- `build_sequence.py`: `_add_player_spawnable_mh()` — body = SkeletalMeshActor
  spawnable with `SKM_Body` (template) + kit MIC override; face = second
  SkeletalMeshActor spawnable (`SKM_Face`) on a MovieScene3DAttachTrack to the
  body's `head` bone with constant relative transform = inverse head ref-pose
  (`_MH_FACE_REL_LOC/_ROT`, recompute if template changes). Face slots get the
  same kit MIC (Z-bands paint head skin / drape shirt). Per-player fallback to
  pack when the MH anim is missing.
- `mh_kits.py` (new): per-player `MI_Kit_<pid>` under
  `/Game/Reconstructions/<clip>/Kits/`, parent `M_FootballKit`, Shirt+Socks
  from kit_colors team colour, shorts white.
- `M_FootballKit` (/Game/Players/MetaHuman/): procedural bodysuit kit —
  PreSkinnedPosition (via **VertexInterpolator** — pixel shader can't read it
  directly) Z-band cascade socks→knee-skin→shorts→shirt→neck-skin + lateral-X
  arm override (sleeve above `SleeveZ`, skin below). Params: ShirtColor,
  ShortsColor, SocksColor, SkinTone, SockTop, ShortsBottom, ShortsTop,
  CollarZ, SleeveZ, TorsoHalfWidth, ArmMinZ, Roughness, PatternTex,
  PatternTiling, PatternStrength — team kit colours/patterns plug in via MIC
  params, no code change.
- **Neck/Head chains fixed through BOTH hops**: IK_SMPL Spine trimmed to
  spine3 + Neck/Head chains; IK_Mannequin Spine→spine_05 + Neck
  (single-bone neck_01 — two-bone chain was rejected) + Head; both
  retargeters remapped (first one needed per-op `run_op_initial_setup(i)`
  before `set_source_chain` would accept the new chains). Result: MH head
  local-rotation spread 0°→34°, foot delta vs pre-change ≤2.4 cm.

**Crash lessons (hit twice today):** do NOT rebuild/recompile a material
while a Level Sequence whose spawnables use it is open — spawnable destroy
crashes (`IsCreatedByConstructionScript` / later `ACineCameraActor::Tick`
GetObjectDataFromId on relaunch). Remedy per protocol: delete LS .uasset,
relaunch, rebuild LS from saved assets (fast — anims/MICs survive; use the
rebuild-from-assets script pattern, no re-import needed).

**Editor-automation gotchas:** `get_all_level_actors()` does NOT list
Sequencer spawnables (use `get_bound_objects`); viewport screenshots only
render when the editor app is FOREGROUND (osascript frontmost first); never
`open UnrealEditor.app` bare — it spawns a project-less second instance that
hijacks remote exec.

**Cosmetic follow-ups:** face drape shadow reads as a collar (fine); sleeves
short (tune `SleeveZ`); bald grey-skin heads — real hair/faces arrive with
per-player MetaHuman assets (`metahuman:<Asset>`); shorts/socks per-team
colours + pattern textures need a team kit spec feeding the existing params.

## Update 2026-07-07 (late) — male default + neck band investigation

User notes: players should be MALE by default, and the head/torso junction
shows a dark band ("gap"); ideal is a single body+head mesh with clothes on
top.

**Single combined mesh: not attainable in 5.8.** OPTIMIZED builds always
emit separate Body+Face meshes; UEFN pipeline (single-mesh) requires a UEFN
project file; SkeletalMergingLibrary can't merge across the two different
skeletons. The practical equivalent is a MATCHED built pair (Hero01-style)
whose neck seam is designed to fit.

**Male body via Python: blocked — every scripted route no-ops in 5.8.**
`set_body_constraints`+`commit_body_state` (incl. Masculine/Feminine ±2,
Height): readback stays at defaults, builds identical.
`import_body_whole_rig` and `get_mesh_for_body_conforming_from_dna` return
INVALID_INPUT_DATA (the FixedCompatibility DNAs — m_tal_nrw etc. — carry
joints but no mesh geometry; Epic's own test references an
ArchetypeDNA/SKM_Face.dna that no longer ships). `set_body_joints` returns
True and invalidates the rig but the built mesh is unchanged.
`SetMetaHumanBodyType` (the UI's path) is not exposed to Python.
**⇒ ONE-TIME UI STEP NEEDED: open `/Game/MetaHumans/PlayerBaseM` in the
MetaHuman Character editor, pick a male body preset (tall/normal), then
Build (OPTIMIZED/MEDIUM into /Game/MetaHumans/PlayerBaseM_Build,
name PlayerBaseM).** Rig+textures re-fetch automatically (sign-in persists;
use non-blocking + poll if scripted). Then swap the pipeline defaults:
`retarget.py:_SKM_MH_BODY_PATH` and `build_sequence.py:_SKM_MH_BODY/_FACE`
→ the PlayerBaseM meshes, recompute `_MH_FACE_REL_*` from the new body's
head-socket ref transform, re-run load() (does the 22 retargets), verify.

**Dark neck band root causes found (archetype face on SkeletalMeshActor):**
1. `M_Hide` slots (eyeshell/saliva/cartilage/eyelashes) render OPAQUE BLACK
   on a raw SkeletalMeshActor — MetaHuman BPs hide those sections in
   component logic. Fixed: shared `MI_KitHidden` (fully-clipped kit
   instance) overrides all M_Hide slots (`mh_kits.ensure_hidden_instance`).
2. Kit master is now BLEND_MASKED with a `FaceClipZ` opacity clip;
   per-player `MI_Kit_<pid>_Face` instances clip the face's neck/clavicle
   drape (default 143 = neck base). Verified working on a level actor at
   every LOD (clean head, eyes, no band).
3. A residual dark band still shows IN THE SEQUENCE at gameplay distance —
   face LOD sections there resolve to a slot outside the override set
   (clip has no effect on it even at 500). Not worth further archetype
   hacking: the built matched pairs (PlayerBaseM after the UI step) use
   baked per-LOD materials with no M_Hide and a fitted seam, which
   eliminates the whole class. Revisit only if the band survives the mesh
   swap.

Editor-stability rule reconfirmed the hard way: sequencer left open across
remote-exec calls in an asset-churn session → `ACineCameraActor::Tick`
GetObjectDataFromId crash (3× today). Verify pattern that works: fresh-ish
editor, open LS + set time + screenshot in ONE call, close in the next.

## Update 2026-07-07 (final) — MALE DEFAULT SHIPPED (PlayerBaseM pair)

User did the 30-second UI step (male tall preset on
`/Game/MetaHumans/PlayerBaseM` — note: the preset sets Masculine/Feminine
NEGATIVE, my scripted +2.0 was the wrong direction). Then scripted:
non-blocking auto-rig + texture synthesis (poll `can_build_meta_human`),
OPTIMIZED/MEDIUM build → `PlayerBaseM_Build/PlayerBaseM/{Body,Face}` male
meshes (body top 149.6 vs female 142.4), saved to disk immediately.

Pipeline swap: `retarget._SKM_MH_BODY_PATH` and
`build_sequence._SKM_MH_BODY/_SKM_MH_FACE` → PlayerBaseM meshes;
`_MH_FACE_REL_LOC=(-162.575,-0.663,0)`; kit bands retuned to male
proportions (SockTop 46 / ShortsBottom 61 / ShortsTop 109 / CollarZ 147 /
SleeveZ 122 / ArmMinZ 66). Full `load()` re-run: 22/22 MetaHuman players,
male bodies, realistic baked faces, kit colours, **no black bib** (built
faces have no M_Hide slots — that whole hack class only applies to the
archetype template face, the policy in `_add_mh_face_spawnable` now
auto-detects which face type it has).

New gotchas from this pass:
- The dirty `AS_MetaHuman_ARKit_Mapping` asset a MetaHuman build leaves in
  memory fails validation on ANY bulk save/delete (`load()`'s wipe died on
  it). Fix: `EditorLoadingAndSavingUtils.reload_packages` on that package
  to clear the dirty flag before load().
- Optimized-build FACE meshes keep source LOD3-7 screen-size thresholds →
  LOD selection culls them at normal camera distances (players looked
  headless; standalone actor rendered fine). Fix: `forced_lod_model=1` on
  the face component (LOD0 IS source-LOD3, a few k tris).
- Optimized-build BODY meshes include merged HEAD geometry in their lowest
  LODs (distant "kit-coloured heads" before the face fix). Harmless now the
  face always renders; if it resurfaces, clamp body max LOD.
- MetaHumanCharacter assets created via script are IN-MEMORY ONLY until
  save_loaded_asset succeeds — an editor crash erases them (PlayerBaseM
  vanished once this way). Save immediately after create.

Remaining polish: neck seam shows the face drape's baked grey tank-top
edge over the shirt (reads as an undershirt collar — acceptable); real
per-player likenesses + hair still future work via metahuman:<Asset>.

## Update 2026-07-07 (evening 2) — playback fixes: head smear + camera crash

Two defects only visible DURING PLAYBACK (static scrub verification missed
both — components don't tick on a scrub):

1. **Heads ~30 m from bodies, joined by smeared geometry**: built face
   meshes ship a post-process AnimBP (neck correction) that expects the
   MetaHuman BP component setup; on a bare attached SkeletalMeshActor its
   tick evaluates bone targets in the wrong space and drags face bones
   toward the binding's corner-origin offset. Fix:
   `disable_post_process_blueprint=True` on the face template component.

2. **Editor crash a few seconds into playback** (`ACineCameraActor::Tick`
   → `GetObjectDataFromId` packed-ref assert — the same signature that
   plagued the whole day, including yesterday's crash-table rows 1–3). It
   reproduces on a fresh session playing a freshly rebuilt LS, so it is
   NOT stale-asset churn: the 5.8.0 Mac CineCamera tick bug is simply
   back. Disabling template tick does not help (Sequencer re-enables tick
   on spawned actors). Fix: **all camera spawnables are plain CameraActor
   again** (broadcast keys FieldOfView via `_make_fov_track`
   [hfov = 2·atan(w/2fx)]; named POV/OTS cameras set constant
   `CameraComponent.field_of_view` on the template; `_set_camera_focal_length`
   and `_template_sensor_width`/`_make_focal_track` now unused). 30 s
   playback survival + full-sequence visual verified. Note the old caveat
   stands: plain CameraActor as MRQ camera-cut target SIGSEGVs during MRQ
   warm-up on 5.8.0 — but spawnable-camera MRQ is already broken (crash
   table, last row); renders need the possessable-camera route either way.

Editor-restore trap worth remembering: after any crash-relaunch the editor
RESTORES the Sequencer tab with the LS open — spawned cameras tick from
startup. First remote-exec action after every relaunch should be
`close_level_sequence()`.

## Update 2026-07-07 (evening 3) — neck seam closed

User still saw a head/body disconnect. Body LODs were probed for a merged
head (single-mesh route) — ALL male body LODs are headless, so the
two-mesh setup stays. The disconnect was three stacked cues, all fixed:
1. **Skin tone mismatch**: kit `SkinTone` default now matches the face
   bake ((0.62, 0.38, 0.27) linear, sampled from T_Head_LOD3_BC render).
2. **Grey tank-top drape bake over the shirt**: new
   `/Game/Players/MetaHuman/M_FaceSkinClip` — samples the face's baked
   skin texture but is BLEND_MASKED + two-sided with a `NeckClipZ`
   pre-skinned clip (default 150.5) that discards the clavicle drape;
   build_sequence overrides built-face head slots with it (archetype
   faces keep the old kit-MIC + M_Hide policy via `is_archetype_face`).
3. **Dark notch at the collar** (hollow body interior showing through the
   neck opening): kit shirt now runs to the neck cut (`CollarZ` 151 =
   crew-neck look) and `M_FootballKit` is two-sided so interior faces
   render shirt colour.

**HARD RULE confirmed twice more:** after ANY master-material rebuild,
REBUILD THE LS before opening/playing it — playback then survives
(including sequence-finish spawn cleanup). Playing an LS whose spawn
templates predate a material recompile crashes in
`FMovieSceneSpawnRegister` destroy paths. Verify playback with
`lsl.play()` + survival check, not static scrubs.

Residual (accepted): a small grey trim at the shoulder/neck junction —
reads as collar detail at broadcast distance; disappears with per-player
likeness builds whose bakes have no garment.

## Update 2026-07-07 (night) — skinned face-follow: findings + pending BP step

User: bust detaches from body in motion (rigid head-socket attach can't
deform with the neck — correct diagnosis; best practice is the face
POSE-COPYING the body like MetaHuman BPs do). Built `BP_MHPlayer`
(/Game/Players/) — minimal Actor BP, Body + Face skeletal components, no
construction logic — and hit every scripting wall in 5.8:

- **leader_pose_component set on the BP's SCS template does NOT remap** on
  spawn: instances point at `Body_GEN_VARIABLE` (the archetype) and the
  face collapses to the actor origin. TWeakObjectPtr component refs are
  not instanced-remapped when set from Python.
- **BP spawn templates have no components** (`get_components_by_class`
  empty, `get_editor_property('Body')` None) — SCS components only exist
  post-construction, so per-template wiring (leader OR per-player kit
  override_materials) is impossible for BP spawnables. The pack path
  solves this with actor VARIABLES + construction script (TeamColour).
- **ABP_Face** (assigned as Face anim_class — same as Epic's own built BP)
  does not follow our Body: its copy-pose source is wired by the owning
  MetaHuman BP's graphs, which is exactly the part that crashes Sequencer.
- **SetLeaderPoseComponent on live spawned instances is silently rejected**
  (reads back None) across metahuman_base_skel → Face_Archetype_Skeleton.

⇒ `_USE_BP_MH_PLAYER = False` in build_sequence parks the BP route; the
LS is rebuilt on the proven two-actor rigid-attach rig (verified again:
playback survives, heads attached, crew-neck seam).

## Update 2026-07-08 (FINAL) — SKINNED FACE-FOLLOW SHIPPED (BP route live)

`_USE_BP_MH_PLAYER = True`; every player is ONE `BP_MHPlayer` spawnable.
End state verified (play + pause mid-motion + screenshots): faces track the
body bone-for-bone (no detachment possible), per-player team kits apply,
playback stable, 87/87 offline tests, all assets saved.

The complete working recipe (all scripted; user only created the variable +
initial construction script):
- **Face follow**: Face comp runs `/Game/Players/ABP_MHFaceFollow` =
  ABP_Face duplicate with CopyPoseFromMesh `use_attached_parent=True`
  (Copy-Pose maps by NAME — leader pose maps by INDEX and explodes the
  face; never use leader pose across body/face skeletons).
- **Body forced LOD0** (`forced_lod_model=1` on Body): at gameplay
  distance the body's higher LODs cull corrective/twist bones from the
  evaluated pose, so copy-pose left ~550 face bones at the actor origin →
  skin "beams" stretching to the pitch corner. Full LOD0 pose fixes it.
- **Face slots**: `head_*` → M_FaceSkinClip (baked skin + NeckClipZ
  drape clip); every other slot (teeth/eyes/shells) → MI_KitHidden —
  their bones have no body counterpart at all.
- **Per-player kits**: `KitMaterial` BP variable (default M_FootballKit;
  must be INSTANCE EDITABLE — non-editable variables silently discard
  template values, and a None default + construction-script SetMaterial =
  naked players). build_sequence sets it per player on the spawn template
  (pack-player pattern); construction script applies it to Body slot 0.
- **Crash discipline** (editor crashed ~10× during this): BP
  compile/save ONLY as the FIRST action of a fresh editor session (a
  compile with the LS in memory reinstances the class under live
  templates and dies); after any BP/material change REBUILD the LS before
  opening it; close the restored Sequencer tab immediately after every
  relaunch. `set_variable_instance_editable` etc. exist in the MCP
  BlueprintTools when the flag needs setting programmatically.

## Update 2026-07-08 — model dropdown on Load Reconstruction

`EUW_LoadReconstruction` now has a **ModelSelector** ComboBoxString
(MetaHuman | Pack | SMPL, default MetaHuman) next to the Load button, and
`load_reconstruction.load()` takes `player_model=` accordingly:
- metahuman — BP_MHPlayer pair (per-player appearance `model` overrides
  still win when explicitly set in the sidecar);
- pack — BP_PlayerActor kit players, MH second hop skipped;
- smpl — BP_PlayerActor_No_Retarget with raw SMPL anims, no retargeting.
All three verified end-to-end through the real dropdown flow (spawned the
EUW, set the combo, executed the button script): "as pack 22/0 MH",
"as smpl 22/0", "as metahuman 22/22". LS name stays `LS_<clip>` for all
modes (the separate Generate-SMPL-LS button still makes `LS_<clip>_SMPL`).

Implementation notes (all scripted): the widget was added to the tree via
python (`unreal.load_object(None, WBP_path + ":WidgetTree")`, widgets are
subobjects like `...:WidgetTree.LoadButton`, `new_object(ComboBoxString,
outer=tree)` + `panel.add_child`). The graph was NOT touched (the read-DSL
isn't perfectly round-trippable — a whole-graph rewrite failed on
`LoadAsset_Blocking`); instead the `Load Recon Python Script` string
VARIABLE (CDO property, name has spaces: "Load Recon Python Script") now
reads the combo itself: `EditorUtilitySubsystem.
find_utility_widget_from_blueprint(wbp)` →
`inst.find_child_widget_by_name("ModelSelector")` (NOT
get_widget_from_name — not python-exposed) → `get_selected_option()` →
`load(output_dir, player_model=...)`. Widget `bIsVariable` is not
python-settable; find_child_widget_by_name avoids needing it.

## (history) Update 2026-07-08 — skinned face: ABP route authored via MCP toolsets

Joe added the construction script (leader-pose + kit). Result: kit variable
worked, but **leader pose across body→face skeletons EXPLODES the unmatched
facial bones** (giant skin beams — leader pose is by-INDEX; the old
kit-part crash row was the same disease). So leader pose is off the table
entirely for the face.

Joe's feedback taken: the unreal-mcp server has far more than remote exec —
`BlueprintTools.read/write_graph_dsl` (full K2 authoring, s-expr DSL),
find_nodes/get_node_infos/set_pin_value etc. Invoke via
`Scripts/mcp_call.py list|call` (graph refs are subobject paths like
`/Game/Players/BP_MHPlayer.BP_MHPlayer:UserConstructionScript`; DSL param
is `code`).

Done via MCP + ue_py (all scripted, no user steps):
1. Construction script rewritten KIT-ONLY:
   `(fn ConstructionScript () (Rendering|Material|SetMaterial (GetBody) 0 (GetKitMaterial)))`
2. `ABP_Face`'s AnimGraph is just CopyPoseFromMesh→Root with its
   SourceMeshComponent pin EMPTY (Epic wires it at game time — why faces
   never followed in Sequencer).
3. Created `/Game/Players/ABP_MHFaceFollow` = ABP_Face duplicate with
   `use_attached_parent=True` on the CopyPoseFromMesh node (set via ue_py
   on the AnimGraphNode's `node` struct + compile). BP_MHPlayer Face now
   uses it (ANIMATION_BLUEPRINT mode, post-process still disabled).

**BLOCKED at verification by a marathon editor boot**: the scheduled DDC
maintenance purge + the day's material churn triggered a full in-process
Metal shader recompile with no SCW parallelism — startup pinned >1.5 h
(log frozen after "DerivedDataCache: Maintenance finished"; `sample`
shows ShaderCompilingThread::CompilingLoop hot). Progress persists in the
DDC across restarts, so let it finish. NOTE: BP_MHPlayer.uasset +
ABP_MHFaceFollow.uasset were moved to **/tmp/ue_quarantine/** during
wedge-triage (they were innocent) — **RESTORE them to
Content/Players/ before rebuilding**, then: close seq → rebuild LS
(_USE_BP_MH_PLAYER=True is set) → probe Face follows Body during PLAY
(scrub is insufficient — copy-pose ticks only during playback) → verify
kits (KitMaterial template var + construction script) → play-survival →
save.

Superseded (kept for history):
**Pending ONE-TIME BP EDIT (user, ~2 min) to go fully skinned + fix
per-player kits on the BP route** — open `/Game/Players/BP_MHPlayer`:
1. Add variable `KitMaterial` (Material Interface), default =
   M_FootballKit, tick Instance Editable.
2. Construction Script: `Face → Set Leader Pose Component (New Leader
   Bone Component = Body)` and `Body → Set Material (Element 0,
   Material = KitMaterial)`.
3. Compile + save.
Then flip `_USE_BP_MH_PLAYER = True` and change the kit hookup in
`_add_player_spawnable_mh` to set the `KitMaterial` template property
(pack-style) instead of touching components; rebuild. If the instance-
level leader rejection above turns out to hold in construction scripts
too, plan B is a custom Face AnimBP (CopyPoseFromMesh, bUseAttachedParent
=true — one node, also a BP-editor job).

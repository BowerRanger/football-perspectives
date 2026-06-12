# MetaHuman Spike — BOW-90

**Date:** 2026-06-11 | **Timebox:** 2h

## Question

Can MetaHumans be driven by SMPL animation data from GVHMR in UE5.8 at 22-player scale for the Football Perspectives pipeline?

---

## Current Setup

### Player representation

`BP_PlayerActor` is a Blueprint actor with a primary `SkeletalMeshComponent` (`MeshComp`) bound to `SKM_SMPL` — a 24-joint SMPL skeleton (`/Game/Skeleton/SKM_SMPL`). Five leader-pose child components attach clothing and head parts (shirt, shorts, socks, sneakers, head), all adopting the body's animated pose via `LeaderPoseComponent`. Animations play via a `MovieSceneSkeletalAnimationTrack` in a `LevelSequence` spawnable — one binding per player, covering only the frames each player is visible.

### GVHMR output format

GVHMR outputs per-player SMPL parameters stored as `.npz`:
- `body_pose` — 23 × 3 axis-angle rotation vectors for joints 1–23 (root is separate)
- `global_orient` / `root_R` — root-bone world orientation in pitch frame (3 × 3 or axis-angle)
- `transl` — root translation in pitch metres per frame

The 24-joint SMPL topology (standard Betas parametric body): pelvis (root), L/R hip, L/R knee, L/R ankle, L/R foot (toe), spine 1–3, neck, L/R collar, L/R shoulder, L/R elbow, L/R wrist, L/R hand (stub), head. No fingers, no face rig, no tongue. Joint names follow the SMPL convention as exported via Blender FBX, with the root bone carrying world translation curves.

### How animations enter UE

The Python `export` stage uses Blender headless to bake SMPL parameters into an FBX armature. `import_fbx.py` imports it as an `AnimSequence` against `SKM_SMPL`'s `Skeleton` asset, with `enable_root_motion = True`. `retarget.py` then drives `IKRetargetBatchOperation.duplicate_and_retarget` using `IKR_SMPL_to_Mannequin` (already in `/Game/Skeleton/`) targeting `SKM_Manny_Simple`. The retargeted `AnimSequence` is what `build_sequence.py` actually plays on `BP_PlayerActor`.

The current retarget destination is the UE5 Mannequin (`SKM_Manny_Simple`), not a MetaHuman. The chain is: **SMPL FBX → SK_SMPL AnimSequence → IK Retarget → SK_Manny AnimSequence → BP_PlayerActor**.

---

## Retargeting Analysis

### SMPL Skeleton Topology

- **24 joints** total (including root/pelvis)
- No finger bones (L/R hand are single stub joints at the wrist position)
- No face rig, no tongue, no jaw
- Spine chain: spine1 → spine2 → spine3 → neck → head (5 levels above pelvis)
- Shoulder: L/R clavicle (collar) → L/R shoulder → L/R elbow → L/R wrist → L/R hand
- Leg: L/R hip → L/R knee → L/R ankle → L/R foot (single toe)
- Joint naming in Blender FBX export: `Pelvis`, `L_Hip`, `R_Hip`, `Spine1`, `Spine2`, `Spine3`, `Neck`, `Head`, `L_Collar`, `R_Collar`, `L_Shoulder`, `R_Shoulder`, `L_Elbow`, `R_Elbow`, `L_Wrist`, `R_Wrist`, `L_Hand`, `R_Hand`, `L_Knee`, `R_Knee`, `L_Ankle`, `R_Ankle`, `L_Foot`, `R_Foot`

### MetaHuman Skeleton Topology

MetaHuman (UE5.1+) shares the UE5 Mannequin skeleton for the body, meaning it is IK-retarget-compatible with any asset already targeting `UE5 Mannequin`. Key joint counts:

- **Body rig** (shared with Mannequin): ~67 joints — full spine (5 bones), clavicles, arms to wrist, legs to toe including separate ball-of-foot joint
- **Hand rig**: 30 additional joints (5 fingers × 3 phalanges per hand, plus thumb extra)
- **Face rig**: ~130 additional bones (MetaHuman face rig drives blendshapes via a dedicated `FaceComponent` with its own `AnimBP_MetaHuman_Face_*` AnimBlueprint)
- **Total:** ~230+ joints for a full LOD0 MetaHuman

Critically: UE5 MetaHuman shares the **same body skeleton root naming** as Mannequin UE5 (`pelvis`, `spine_01`–`spine_05`, `clavicle_l/r`, `upperarm_l/r`, `lowerarm_l/r`, `hand_l/r`, `thigh_l/r`, `calf_l/r`, `foot_l/r`, `ball_l/r`). The Mannequin-compatible IK Rig (`RTG_Mannequin` or project equivalent) already covers the body joints.

### Viable Retargeting Path

**Two-hop retarget: SMPL → UE Mannequin → MetaHuman**

This is the standard Epic-documented approach and is what the project already uses for the first hop. Extending to MetaHuman adds one more hop:

**Step A (already done):** `IKR_SMPL_to_Mannequin` retargets SMPL AnimSequences to Mannequin.

**Step B (new):** An `IKRetargeter` from `SKM_Manny_Simple` → MetaHuman skeleton drives the MetaHuman body. Epic ships a reference `RTG_Mannequin_To_Metahuman` retargeter asset with UE5 (available in the MetaHuman plugin content). This is the standard path used by thousands of projects.

**Alternative: One-hop directly SMPL → MetaHuman**  
Possible by defining IK Rig chains on the SMPL skeleton and mapping directly to MetaHuman body chains. This removes an intermediate bake but requires creating a new `IKR_SMPL_to_Metahuman` asset. Given the project already has `IKR_SMPL_to_Mannequin` working, the two-hop approach reuses existing work and is lower risk.

**Recommended path:** Two-hop. Retarget SMPL → Mannequin (existing), then in `BP_PlayerActor` swap `SKM_Manny_Simple` for the MetaHuman body mesh and wire `RTG_Mannequin_To_Metahuman`. The AnimBP on the MetaHuman can be driven either by (a) a Sequencer `MovieSceneSkeletalAnimationTrack` on the body mesh, or (b) the MetaHuman's own AnimBP graph sampling the retargeted sequence. Option (a) is simpler for Sequencer-driven pipelines.

### Known Challenges

**1. Finger joints: no data**  
SMPL has single stub hand bones. MetaHuman hands have 30 joints. The IK Retargeter will leave fingers at their rest pose (typically a relaxed open hand). This is acceptable for broadcast-resolution football — fingers are not legible at typical frame widths. The IK Retargeter handles this gracefully by leaving undriven target chains at the reference pose.

**2. Toe/ball-of-foot: partial mismatch**  
SMPL has one `L_Foot`/`R_Foot` joint. MetaHuman Mannequin has `foot_l/r` + `ball_l/r`. The IK Retargeter maps the SMPL foot to `foot_l/r` and leaves `ball_l/r` at rest. Foot contact will look correct at distance; toe-lift detail during running will not animate. Acceptable for this use case.

**3. Spine count: compatible**  
SMPL has 3 spine bones; Mannequin/MetaHuman has 5 (`spine_01`–`spine_05`). The IK Rig chain approach distributes 3-bone spine rotation across 5 bones proportionally. This is the standard retarget behavior and produces good-looking torso motion.

**4. Root motion: already solved**  
GVHMR provides root translation and orientation. The pipeline already sets `enable_root_motion = True` on the imported AnimSequence and bakes the root bone curves. MetaHuman's AnimBP can consume root motion identically to the Mannequin's.

**5. Face rig: not driven**  
GVHMR does not produce face expression data. The MetaHuman face rig will remain in its neutral rest expression. This is acceptable and expected — no face data is available from single-camera reconstruction.

**6. MetaHuman AnimBP complexity**  
MetaHuman actors use a two-component setup: body `SkeletalMeshComponent` + face `SkeletalMeshComponent`, each with their own AnimBlueprint (`ABP_MetaHuman_*` and `ABP_MetaHuman_Face_*`). The face AnimBP references ARKit blendshapes and will run idle/neutral when no face animation is provided. The body AnimBP can be replaced with a simple `AnimGraph` → `Play Animation` node pointing to the retargeted AnimSequence. This requires editing the MetaHuman's body AnimBP or using a PostProcessAnimBP to inject the Sequencer-driven anim.

The simplest integration: set `AnimationMode = Use Animation Asset` on the body mesh component, which bypasses the MetaHuman AnimBP entirely and plays the AnimSequence directly. This is the same approach the project uses today with `BP_PlayerActor`.

**7. IK Rig setup cost**  
The IK Rig for the MetaHuman body is already available in UE5's MetaHuman plugin content (`IK_Metahuman_Body`). You do not need to create it. You need to create only a new `IKRetargeter` asset pairing the project's Mannequin IK Rig with `IK_Metahuman_Body`.

---

## Performance Assessment

### Current Cost (Capsule/Mannequin Players)

`BP_PlayerActor` uses `SKM_Manny_Simple` — the stripped-down Mannequin mesh included with UE5:
- ~15,000 triangles at LOD0, ~3,000 at LOD2
- Single material, no cloth simulation, no physics
- AnimSequence playback only (no AnimBP evaluation overhead)
- Estimated GPU cost for 22 simultaneous instances: ~0.5–1.5 ms in a mid-tier scene (RTX 3080 class), dominated by shadow map rendering rather than the mesh itself
- CPU cost: ~22 × skinning compute, typically sub-millisecond with GPU skinning enabled

### MetaHuman Cost Estimates

MetaHuman performance is well-characterised by Epic's documentation and community profiling (UE5 release notes, Paragon profiling studies, and GDC presentations):

**LOD0 (full quality — default for close-up shots)**
- Body mesh: ~33,000 triangles (body alone; modular parts add more)
- Face mesh: ~43,000 triangles
- Hair: ~60,000 strands (groom asset) or ~15,000 triangles (card fallback)
- Materials: 5–8 material slots with subsurface scattering, transmission
- Physics: cloth simulation on hair + optional on clothing
- AnimBP: face rig evaluates ~130 bone transforms + blendshape solve
- Approximate GPU cost per instance at LOD0: **8–15 ms** on an RTX 3080 class GPU for the full character including groom and face
- At 22 instances at LOD0: **prohibitive** — would consume the entire frame budget multiple times over

**LOD2 (medium quality — typical for mid-field players)**
- Body mesh: ~12,000 triangles
- Face mesh: ~10,000 triangles
- Hair: card-based, ~3,000–5,000 triangles
- Materials: simplified, no subsurface scattering
- Approximate GPU cost per instance at LOD2: **1.5–3 ms** on RTX 3080 class
- Face AnimBP still evaluates (bone overhead remains); disabling face AnimBP saves ~0.3–0.5 ms per actor

**LOD3 (background quality — suitable for all 22 players)**
- Body mesh: ~4,000 triangles
- Face mesh merged into body or culled
- Hair: card-based, ~1,000–2,000 triangles or disabled
- No subsurface scattering, single material
- Approximate GPU cost per instance at LOD3: **0.4–0.8 ms** on RTX 3080 class
- At 22 instances: **8.8–17.6 ms total** — within budget for a 33 ms (30fps) frame with other scene elements

**Important:** MetaHuman LOD behaviour is automatic when using the `MetaHumanComponent` and `BP_MetaHuman` base class. LOD thresholds are configurable in the MetaHuman asset settings. The LOD system considers screen percentage, not world distance.

### Scalability

**Nanite:** MetaHuman body and face meshes are Nanite-compatible in UE5.1+ (Epic confirmed this in the MetaHuman 2.0 release). Nanite eliminates the LOD triangle-count bottleneck for GPU rasterisation but does **not** help with:
- Skinning compute (Nanite does not accelerate skeletal mesh skinning)
- AnimBP evaluation CPU cost
- Shadow map generation for skeletal meshes (Nanite shadows are static mesh only in UE5.3; skeletal meshes still use traditional shadow maps)

**Ticking cost (AnimBP, physics):**
- With face AnimBP running: ~22 × ~2 ms CPU per frame = ~44 ms CPU overhead alone at 22 players — problematic
- With face AnimBP disabled (`AnimationMode = Use Animation Asset`): ~22 × ~0.3 ms = ~6.6 ms — acceptable
- Cloth physics on hair: ~22 × ~0.5 ms = ~11 ms — disable for background players via LOD
- GPU skinning (`r.SkinCache.Mode 1`): offloads skinning to GPU compute, essential for 22 simultaneous skeletal meshes

**Recommendation:** Disable the face AnimBP on all 22 player actors (set `AnimationMode = Use Animation Asset` on the body mesh, disable or set the face mesh's AnimBP to a null graph). This reduces per-actor CPU tick from ~2 ms to ~0.3 ms and enables 22-player scalability.

---

## Recommendation

**☐ Proceed with LOD strategy — foreground player full MetaHuman, background LOD3**

More specifically: **all 22 players at LOD3 with face AnimBP disabled, with optional LOD0/LOD2 upgrade on demand for foreground players in specific shots.**

Rationale:
1. The retargeting path is well-established and partially implemented. The SMPL→Mannequin hop already works; the Mannequin→MetaHuman hop uses Epic's own `RTG_Mannequin_To_Metahuman` asset.
2. Face AnimBP disablement eliminates the primary CPU cost blocker.
3. At LOD3 with GPU skinning enabled, 22 MetaHumans is within a 30fps budget on an RTX 3080+ class GPU.
4. Visual fidelity improvement over the current Mannequin is substantial even at LOD3: MetaHuman has proper facial topology, realistic proportions, and skin materials — perceptibly better than `SKM_Manny_Simple` even without close-up rendering.
5. No fundamental blockers. The SMPL skeleton topology maps cleanly to MetaHuman body joints via Mannequin intermediary; finger and face gaps are acceptable for this use case.

The alternative of full LOD0 for all 22 players is not feasible in real-time render (render-to-movie via Movie Render Queue relaxes this, but at ~10-minute render times per minute of content at 22 × LOD0 MetaHumans).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `RTG_Mannequin_To_Metahuman` not shipped with MetaHuman plugin version in project | Medium | Medium | Create IK Retargeter manually; MetaHuman IK Rig (`IK_Metahuman_Body`) is always in the plugin content. 30-minute task. |
| MetaHuman asset path assumptions differ between MetaHuman Creator versions | Medium | Low | Retarget target mesh path is set once in `retarget.py`; update `_SKM_TARGET_PATH` to the MetaHuman body mesh path. |
| Face AnimBP disabled but face component still ticking | Low | Medium | In `BP_MetaHuman` construction script, set face mesh `AnimationMode = Disabled` explicitly. |
| 22 × MetaHuman foot contact worse than current Mannequin | Low | Low | GVHMR foot anchoring already operates on SMPL ankle/foot joints; MetaHuman foot IK at LOD3 is not active, so foot contact quality is unchanged from Mannequin. |
| Nanite on body mesh conflicts with skinning | Low | High | If Nanite is enabled on MetaHuman body, disable it for this use case; skeletal meshes with Nanite require the `r.Nanite.AllowSkinning 1` CVar (UE5.4+ only). Leave Nanite off for player meshes. |
| 22 MetaHuman shadow maps saturate GPU | Medium | Medium | Use `r.Shadow.DistanceScale 0.5` for player actors not in primary camera frustum; or use capsule shadows at LOD2+ (already supported by MetaHuman). |
| `IKRetargetBatchOperation` timing with MetaHuman target mesh | Low | Low | The batch retarget operation in `retarget.py` is mesh-agnostic; only `_SKM_TARGET_PATH` needs updating. |

---

## Estimated Setup Cost (if proceeding)

| Step | Description | Estimate |
|------|-------------|----------|
| 1 | Create MetaHuman in MetaHuman Creator (web), export to UE project | 1 h |
| 2 | Verify `IK_Metahuman_Body` IK Rig asset exists in project content | 15 min |
| 3 | Create `IKRetargeter` asset `IKR_Mannequin_to_Metahuman` (source: Mannequin IK Rig, target: MetaHuman IK Rig) | 1 h |
| 4 | Update `retarget.py`: change `_SKM_TARGET_PATH` to MetaHuman body mesh path | 15 min |
| 5 | Create `BP_MetaHumanPlayer` based on the MetaHuman's generated `BP_<Name>`, disable face AnimBP, set body mesh to `AnimationMode = Use Animation Asset` | 2 h |
| 6 | Update `build_sequence.py` to support MetaHuman actor path alongside current Mannequin path (config-driven) | 1 h |
| 7 | Configure LOD settings for MetaHuman: LOD3 thresholds, capsule shadows | 30 min |
| 8 | Enable `r.SkinCache.Mode 1` in project's `DefaultEngine.ini` for GPU skinning | 15 min |
| 9 | Load one reconstruction with 22 MetaHuman players; inspect visually | 30 min |
| 10 | Profile 22 simultaneous instances with Unreal Insights / GPU Visualizer | 1 h |
| **Total** | | **~8 h** |

This fits within a 2-day sprint. The non-trivial portion is steps 3 and 5 (IK Retargeter tuning and MetaHuman AnimBP setup). Steps 4, 6, and 8 are code changes with no design ambiguity.

---

## Next Steps (if proceeding)

- [ ] Create MetaHuman in MetaHuman Creator; export to project as `MH_Player_Base`
- [ ] Confirm `IK_Metahuman_Body` IK Rig is present in `/Game/MetaHumans/Common/` or `/Plugins/MetaHuman/Content/`; if absent, create manually (chain definitions: spine, L/R leg, L/R arm, head)
- [ ] Create `IKRetargeter` `IKR_Mannequin_to_Metahuman`: source = project Mannequin IK Rig, target = `IK_Metahuman_Body`; verify pose alignment (T-pose vs A-pose correction may be needed)
- [ ] Update `retarget.py`: `_SKM_TARGET_PATH` → MetaHuman body mesh path; update `_RETARGETER_PATH` → new retargeter or add a second retarget pass
- [ ] Create `BP_MetaHumanPlayer` child of MetaHuman-generated Blueprint: disable face component AnimBP (`AnimationMode = Disabled`), set body mesh `AnimationMode = Use Animation Asset`, expose `PlayerId`/`TeamColour` Blueprint variables
- [ ] Add `bp_player_path` config option to `build_sequence.build()` (already supported via `bp_player_path` parameter) — point to `BP_MetaHumanPlayer`
- [ ] Set MetaHuman LOD screen-percentage thresholds: LOD3 at 10% screen coverage (default is LOD0 until ~5% — adjust to LOD2 at 20%, LOD3 at 8%)
- [ ] Add `r.SkinCache.Mode=1` and `r.SkinCache.MaxNumTriangles=800000` to `Config/DefaultEngine.ini`
- [ ] Test foot IK with GVHMR ankle anchoring on one player at LOD0
- [ ] Profile 22 simultaneous MetaHuman instances with Unreal Insights: confirm < 30 ms total frame time at LOD3

---

## Appendix: Why not defer?

The current Mannequin placeholder is functional but the visual gap between `SKM_Manny_Simple` and a MetaHuman is large enough to matter for the product's primary output (broadcast-quality recreations for highlights content). The retargeting infrastructure is 80% in place; the incremental cost is the IK Retargeter setup and the MetaHuman Blueprint wiring. The performance constraints are solvable with LOD configuration and face AnimBP disablement. This milestone is appropriate now that the camera, tracking, and SMPL pipeline are stable.

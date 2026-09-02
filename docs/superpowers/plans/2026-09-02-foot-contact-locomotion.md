# Foot-Contact-Aware Locomotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate foot-floor clipping and foot sliding in reconstructed player
animations by making both extraction (`hmr_world`) and cleanup (`refined_poses`)
foot-contact-aware.

**Architecture:** Per-foot ground contacts are detected from ray-cast ViTPose
ankle pixels (camera-grounded, image-faithful). `hmr_world` anchors the root with
posed-FK offsets and pins stance feet via a smooth correction channel δ over a
dense carrier path. `refined_poses` replaces the blanket ground snap with a
contact-aware version and finishes with a foot-lock IK pass + penetration guard
that run after ALL smoothing. A foot-quality eval harness gates every change.

**Tech Stack:** Python 3.11 (`.venv311`), numpy + scipy only for all new
numerics. pytest. No torch, no GVHMR re-runs (all local validation goes through
`scripts/reanchor_hmr_world.py`).

**Spec:** `docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md`

## Global Constraints

- Run everything with `.venv311/bin/python`; tests via `.venv311/bin/python -m pytest`.
- New numerics: numpy/scipy only (refined_poses light-venv contract).
- FK convention everywhere: `thetas[:, 0]` IGNORED; `root_R` carries root world
  orientation; local chain `rot[j] = rot[parent] @ Rl(theta[j])` (matches
  `src/utils/smpl_skeleton.py`).
- Immutability: never mutate `SmplWorldTrack` / `RefinedPose` in place — build new.
- No schema changes to npz; contacts travel via `{shot}__{pid}_foot_contacts.json` sidecar.
- Pre-existing failures on main (NOT ours): `test_ball_stage.py::test_aerial_arc_promotes_grounded_run_to_flight`,
  `test_blender_export_smpl_skeleton.py::test_player_fbx_has_24_bones_and_full_keyframes`.
- SMPL indices: hips 1/2, knees 4/5, ankles 7/8, feet(toes) 10/11; COCO ankles 15/16.
- Ground plane z=0; ankle-plane z 0.05 (`_FOOT_PLANE_Z`); pitch metres.
- Commit after each green task with conventional-commit messages (no attribution footer).

---

### Task 1: Canonical-FK helper + synthetic gait fixture + shared foundations

**Files:**
- Modify: `src/utils/smpl_skeleton.py` (add `compute_canonical_joints_batch`; move
  `_beta_adjusted_rest_joints` + `_load_smpl_neutral_model` here from
  `src/stages/refined_poses.py` as public `beta_adjusted_rest_joints` /
  `load_smpl_neutral_model`, with re-imports in refined_poses — zero behavior change)
- Create: `tests/helpers/synthetic_gait.py` (+ empty `tests/helpers/__init__.py` if missing)
- Create: `src/utils/foot_contact.py` — in THIS task only the frozen `FootContacts`
  dataclass + `ContactSpan` + `to_json`/`from_json`/`shifted()` (see Task 3's
  interface block), so Tasks 3 and 4 can run in parallel against it
- Test: `tests/test_smpl_skeleton.py` (append), `tests/test_synthetic_gait.py`,
  `tests/test_foot_contact.py` (round-trip/shift tests only at this point)

**Interfaces:**
- Produces: `compute_canonical_joints_batch(thetas, rest_joints=None) -> np.ndarray  # (F, 24, 3)`
  — root-relative canonical y-up joint positions (pelvis at origin), i.e. the
  `global_pos` of `compute_all_joint_worlds_batch` BEFORE the world transform.
- Produces: `beta_adjusted_rest_joints(betas, smpl_model)` / `load_smpl_neutral_model()`
  in `src/utils/smpl_skeleton.py` (moved, public); `tests/test_refined_poses_*.py`
  must stay green after the move.
- Produces: `FootContacts` (the single contact currency for all later tasks).
- Produces: `synthetic_gait.make_walk(n_frames=120, fps=25.0, speed=2.0, stride_s=0.6, direction_deg=0.0) -> GaitTrack`
  where `GaitTrack` is a NamedTuple of `frames (F,), thetas (F,24,3), root_R (F,3,3),
  root_t (F,3), betas (10,), contacts_true (F,2) bool, fps float`. The generator
  builds an analytic alternating-stance walk in pitch frame: stance foot world
  position constant during its span (exact), swing foot advances 2·speed, pelvis
  moves at `speed`, feet at z≈0 during stance and lifting to 0.12 m mid-swing.
  It does NOT need anatomically-pretty thetas — it works backwards: choose foot
  world targets, keep legs straight (thetas zero) and instead vary `root_t`/`root_R`
  is NOT acceptable for IK tests, so: generate hip/knee axis-angles by simple
  2-link planar IK in the sagittal plane so FK feet EXACTLY hit the analytic foot
  targets (use the same law-of-cosines math Task 7 implements — write it here
  first as the test oracle, in the fixture, independent of `src/`).
- Consumes: `SMPL_PARENTS`, `SMPL_REST_JOINTS_YUP`, `axis_angle_to_matrix` from `src/utils/smpl_skeleton.py`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_smpl_skeleton.py (append)
def test_canonical_joints_batch_matches_world_batch():
    rng = np.random.default_rng(0)
    thetas = rng.normal(0, 0.3, (5, 24, 3))
    root_R = Rotation.random(5, random_state=1).as_matrix()
    root_t = rng.normal(0, 2, (5, 3))
    canon = compute_canonical_joints_batch(thetas)
    world = compute_all_joint_worlds_batch(thetas, root_R, root_t)
    rebuilt = np.einsum("fba,fja->fjb", root_R, canon) + root_t[:, None, :]
    np.testing.assert_allclose(rebuilt, world, atol=1e-9)

def test_canonical_joints_batch_pelvis_at_origin():
    canon = compute_canonical_joints_batch(np.zeros((3, 24, 3)))
    np.testing.assert_allclose(canon[:, 0], 0.0, atol=1e-12)
```

```python
# tests/test_synthetic_gait.py
def test_walk_stance_feet_are_stationary():
    g = make_walk(n_frames=100)
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    for side, joint in ((0, 10), (1, 11)):
        for a, b in _spans(g.contacts_true[:, side]):
            span = fw[a:b, joint, :2]
            assert np.linalg.norm(span - span[0], axis=1).max() < 1e-6

def test_walk_root_advances_at_speed():
    g = make_walk(n_frames=100, speed=2.0)
    dist = np.linalg.norm(g.root_t[-1, :2] - g.root_t[0, :2])
    assert abs(dist / ((len(g.frames) - 1) / g.fps) - 2.0) < 0.15

def test_walk_swing_foot_lifts():
    g = make_walk(n_frames=100)
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    swing_z = fw[~g.contacts_true[:, 0], 10, 2]
    assert swing_z.max() > 0.08
```

- [ ] **Step 2: Run to verify failure** — `.venv311/bin/python -m pytest tests/test_smpl_skeleton.py -k canonical tests/test_synthetic_gait.py -q` → import errors.
- [ ] **Step 3: Implement.** `compute_canonical_joints_batch` = extract the
  canonical loop from `compute_all_joint_worlds_batch` (share it: have the world
  version call the canonical one, then apply the world transform — keeps them
  provably identical). Fixture: analytic foot targets + sagittal 2-link IK
  (hip pitch + knee flexion around local x) with leg lengths from
  `SMPL_REST_JOINTS_YUP`; `root_R` = yaw(direction) so canonical −z faces travel;
  root height chosen so stance ankle reaches z=0.05.
- [ ] **Step 4: Run tests green**, plus `.venv311/bin/python -m pytest tests/test_smpl_skeleton.py tests/test_render_fk*.py -q` (FK parity guard).
- [ ] **Step 5: Commit** `feat(smpl): canonical-FK batch helper + synthetic gait fixture`

---

### Task 2: Foot-quality metrics + eval CLI + committed gberch baseline

**Files:**
- Create: `src/utils/foot_quality.py`
- Create: `scripts/eval_foot_quality.py`
- Test: `tests/test_foot_quality.py`

**Interfaces:**
- Produces: `foot_quality_metrics(*, frames, betas, thetas, root_R, root_t, fps, contacts=None, kp2d=None, cameras=None) -> dict`
  with keys `penetration` (pct_frames_sole_below_0, max_depth_cm, mean_depth_cm —
  sole proxy = foot joint z − `sole_clearance_m` 0.025), `lower_foot_z`
  (mean/p05/p50/p95), `skate` (per-foot mean/p50/p95 m/s measured **within
  contact spans** when `contacts` given, else while foot z < 0.10), `spans`
  (count, mean/max XY path length m), `flight` (pct frames both feet > 0.05 m),
  `contact_ratio`, and when kp2d+cameras given: `ankle_reproj_px`
  (mean/p95 over frames with COCO-ankle conf ≥ 0.5).
  `contacts` is `(F, 2) bool` or `None`. `cameras` is a dict
  `{frame: (K, R, t)}`; distortion handled by caller-undistorted kp2d — v1
  projects with pinhole only and documents it.
- Produces: CLI `.venv311/bin/python scripts/eval_foot_quality.py --output output [--players P001,P002] [--stage refined|hmr|both] [--json PATH]`
  → prints a compact table per player and writes JSON. Loads contacts sidecar
  `output/hmr_world/{shot}__{pid}_foot_contacts.json` when present (tolerates absence).
- Consumes: `compute_all_joint_worlds_batch`, `_beta_adjusted_rest_joints` +
  `_load_smpl_neutral_model` (import from `src.stages.refined_poses`).

- [ ] **Step 1: Write failing tests** — on the synthetic walk (ground truth known):

```python
def test_metrics_on_clean_walk_report_no_skate_no_penetration():
    g = make_walk(n_frames=120)
    m = foot_quality_metrics(frames=g.frames, betas=g.betas, thetas=g.thetas,
                             root_R=g.root_R, root_t=g.root_t, fps=g.fps,
                             contacts=g.contacts_true)
    assert m["skate"]["L"]["mean_mps"] < 0.05
    assert m["penetration"]["pct_frames_sole_below_0"] == 0.0
    assert 0.3 < m["contact_ratio"] < 0.9

def test_metrics_detect_injected_skate():
    g = make_walk(n_frames=120)
    slid = g.root_t.copy(); slid[:, 0] += np.linspace(0, 3.0, len(slid))  # +0.63 m/s drift
    m = foot_quality_metrics(frames=g.frames, betas=g.betas, thetas=g.thetas,
                             root_R=g.root_R, root_t=slid, fps=g.fps,
                             contacts=g.contacts_true)
    assert m["skate"]["L"]["mean_mps"] > 0.4

def test_metrics_detect_injected_penetration():
    g = make_walk(n_frames=60)
    sunk = g.root_t.copy(); sunk[:, 2] -= 0.06
    m = foot_quality_metrics(frames=g.frames, betas=g.betas, thetas=g.thetas,
                             root_R=g.root_R, root_t=sunk, fps=g.fps)
    assert m["penetration"]["pct_frames_sole_below_0"] > 50.0
```

- [ ] **Step 2: Verify fail.** — module missing.
- [ ] **Step 3: Implement** metrics (vectorised FK once, then pure array math;
  reuse the span-walk logic from the design probe). CLI mirrors
  `scripts/eval_anchor_clicks.py` structure (argparse, prints table, `--json`).
- [ ] **Step 4: Green** + run CLI on gberch: `.venv311/bin/python scripts/eval_foot_quality.py --output output --players P001,P002,P003 --json output/foot_quality_baseline.json`
  — eyeball numbers match the spec's baseline table (skate mean 2–3.3 m/s etc.).
- [ ] **Step 5: Commit** `feat(eval): foot-quality metrics, eval CLI, gberch baseline snapshot` (include the baseline JSON).

---

### Task 3: Contact detection (`src/utils/foot_contact.py`)

**Files:**
- Create: `src/utils/foot_contact.py`
- Test: `tests/test_foot_contact.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class FootContacts:
    n_frames: int
    in_contact: np.ndarray          # (F, 2) bool  [L, R]
    quality: np.ndarray             # (F, 2) float in [0, 1]
    spans: tuple[ContactSpan, ...]  # ContactSpan(side:int, start:int, end:int, pin:np.ndarray(3,))
    def to_json(self) -> dict; @classmethod def from_json(cls, d) -> "FootContacts"
    def shifted(self, offset: int) -> "FootContacts"   # frame-index shift (sync_map)

def detect_contacts(*, kp2d, frame_indices, per_frame_K, per_frame_R, per_frame_t,
                    distortion, thetas, root_R, betas, fps, cfg) -> FootContacts
def derive_contacts_from_fk(*, thetas, root_R, root_t, betas, fps,
                            speed_enter=0.6, speed_exit=1.2, max_height=0.12,
                            min_span_frames=4) -> FootContacts   # for refined_poses fallback
```

- Algorithm (`detect_contacts`), per foot (L: COCO 15/SMPL ankle 7/foot 10; R: 16/8/11):
  1. Ray-cast each confident (conf ≥ 0.3) ankle pixel via
     `foot_anchor.ankle_ray_to_pitch(..., plane_z=0.05)` → `w[f] (3,)`, NaN elsewhere.
  2. NaN-aware 3-frame median filter → `w_s`; central-difference speed
     `v[f] = |w_s[f+1] − w_s[f−1]|·fps/2` (NaN-propagating).
  3. Adaptive floor: `scale[f]` = ‖ray(uv+(0,1)) − ray(uv)‖ (metres per vertical px);
     `v_enter_eff = max(cfg.speed_enter_m_s, 0.5·cfg.px_noise·scale·fps)`,
     `v_exit_eff = max(cfg.speed_exit_m_s, 1.0·cfg.px_noise·scale·fps)`.
  4. Hysteresis state machine over v (enter below `v_enter_eff`, exit above `v_exit_eff`,
     NaN exits immediately).
  5. FK lower-foot gate via `compute_canonical_joints_batch` lifted by `root_R`
     (no root_t needed — compare the two feet's world-z relative to each other):
     contact only if this foot's z ≤ other foot's z + 0.05.
  6. Spans: contiguous runs ≥ `min_span_frames`; `pin = nanmedian(w_s[span])` with
     `pin[2] = 0.05`; reject span if p90 ‖w_s − pin‖ > `max_pin_spread_m`.
     `quality[f, side] = min(ankle_conf, 1 − v/v_exit_eff)` clipped to [0,1] inside spans, 0 outside.
- `derive_contacts_from_fk`: same hysteresis+span machinery, but the signal is the
  FK foot-joint world track (needs root_t) and a height gate `z < max_height`.
  Factor the shared span/hysteresis code into module-private helpers.

- [ ] **Step 1: Failing tests** — synthetic walk projected through a synthetic camera:

```python
def _project(K, R, t, pts):  # world → px, helper in test file
    cam = pts @ R.T + t
    uv = cam[:, :2] / cam[:, 2:3]
    return uv @ K[:2, :2].T + K[:2, 2]

def test_detect_contacts_recovers_true_stance_spans():
    g = make_walk(n_frames=120)
    K, R, t = make_broadcast_camera()          # fixture: 30 m back, 12 m up, fx≈2000
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, g.root_t)
    kp2d = np.zeros((120, 17, 3)); kp2d[..., 2] = 0.9
    kp2d[:, 15, :2] = _project(K, R, t, fw[:, 7]); kp2d[:, 16, :2] = _project(K, R, t, fw[:, 8])
    fc = detect_contacts(kp2d=kp2d, frame_indices=g.frames,
                         per_frame_K={f: K for f in g.frames}, per_frame_R={f: R for f in g.frames},
                         per_frame_t={f: t for f in g.frames}, distortion=(0.0, 0.0),
                         thetas=g.thetas, root_R=g.root_R, betas=g.betas, fps=g.fps, cfg=default_cfg())
    agree = (fc.in_contact == g.contacts_true).mean()
    assert agree > 0.85                        # edges may differ by a frame or two
    for span in fc.spans:                      # pins land on the true stance ankle (XY)
        true_xy = true_pin_for_span(g, span)   # fixture helper
        assert np.linalg.norm(span.pin[:2] - true_xy) < 0.08

def test_detect_contacts_pixel_noise_no_false_stance_when_far():
    # tiny far player: noise floor swamps signal → NO spans rather than wrong spans
    ... (same setup, camera 120 m back, add 2px N(0,1) noise, assert fc.spans == () or
         all spans satisfy the spread gate)

def test_low_confidence_frames_never_in_contact(): ...   # conf 0.1 → in_contact all False there
def test_min_span_and_spread_gates_reject_kick(): ...    # 3-frame dip → no span
def test_json_round_trip_and_shift(): ...
def test_derive_contacts_from_fk_matches_truth_on_walk(): ...
```

- [ ] **Step 2: Verify fail.** — module missing.
- [ ] **Step 3: Implement** per algorithm above (vectorised where easy; the
  hysteresis loop may be plain Python — F ≈ 500).
- [ ] **Step 4: Green.** `.venv311/bin/python -m pytest tests/test_foot_contact.py -q`
- [ ] **Step 5: Commit** `feat(contact): per-foot ray-cast contact detection + FK fallback`

---

### Task 4: Stance-pinned root solve (`src/utils/foot_lock.py`, part 1)

**Files:**
- Create: `src/utils/foot_lock.py`
- Test: `tests/test_foot_lock.py`

**Interfaces:**
- Produces:

```python
def solve_root_with_pins(*, root_carrier, root_R, thetas, betas, contacts,
                         fps, max_correction_m=0.5, decay_s=0.6,
                         rest_joints=None) -> tuple[np.ndarray, dict]
```

  Returns `(root_t (F,3), stats)`; stats: `constrained_frames`, `mean_delta_m`,
  `max_delta_m`, `clamped_frames`.
- Algorithm:
  1. `canon = compute_canonical_joints_batch(thetas, rest_joints)`;
     ankle offsets `off[f, side] = root_R[f] @ canon[f, ankle_idx]`.
  2. Per constrained frame: implied root per stance foot = `span.pin − off[f, side]`;
     multi-foot → quality-weighted mean.
  3. `δ[f] = implied − root_carrier[f]`; clamp ‖δ‖ ≤ `max_correction_m`
     (count clamps).
  4. Interpolate δ per-axis over all F with `scipy.interpolate.PchipInterpolator`
     on constrained indices (≥2 points; 1 point → constant). Before the first /
     after the last constrained frame: linear decay from edge δ to 0 over
     `decay_s·fps` frames, 0 beyond.
  5. Return `root_carrier + δ_dense`.

- [ ] **Step 1: Failing tests**

```python
def test_pinned_solve_zeroes_stance_skate_on_noisy_carrier():
    g = make_walk(n_frames=120)
    rng = np.random.default_rng(3)
    carrier = g.root_t + rng.normal(0, 0.05, g.root_t.shape)   # 5 cm anchor wobble
    fc = contacts_from_truth(g)                                # fixture → FootContacts w/ exact pins
    solved, stats = solve_root_with_pins(root_carrier=carrier, root_R=g.root_R,
                                         thetas=g.thetas, betas=g.betas,
                                         contacts=fc, fps=g.fps)
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, solved)
    for span in fc.spans:
        j = 7 if span.side == 0 else 8
        xy = fw[span.start:span.end, j, :2]
        assert np.linalg.norm(xy - span.pin[:2], axis=1).max() < 0.02

def test_delta_decays_to_carrier_outside_contacts(): ...  # frames far from any span → solved == carrier
def test_delta_clamped(): ...                             # pin 2 m off → ‖δ‖ == 0.5, clamped_frames > 0
def test_smooth_no_velocity_spikes_at_span_edges(): ...   # |Δ²(solved)| p99 < 3× |Δ²(truth)| p99
```

- [ ] **Step 2: Verify fail** → **Step 3: Implement** → **Step 4: Green.**
- [ ] **Step 5: Commit** `feat(contact): stance-pinned root solve with smooth delta channel`

---

### Task 5: hmr_world integration + reanchor script (mid-point eval)

**Files:**
- Modify: `src/stages/hmr_world.py` (step 5 of `process_player`, ~lines 637–741)
- Create: `scripts/reanchor_hmr_world.py`
- Test: `tests/test_hmr_world_stage.py` (append), `tests/test_reanchor_hmr_world.py`

**Interfaces:**
- Consumes: `detect_contacts`, `solve_root_with_pins`, `compute_canonical_joints_batch`.
- Produces: module-level

```python
def anchor_root_translation(*, kp2d, frame_indices, per_frame_K, per_frame_R,
                            per_frame_t, distortion, thetas, root_R, betas, cfg,
                            fps) -> tuple[np.ndarray, np.ndarray, FootContacts | None]
    # returns (root_t (F,3), confidence (F,), contacts or None-when-ankle_mid)
```

  factored out of `process_player` so the stage AND the reanchor script call the
  identical code. Behavior:
  - Carrier = existing per-frame ankle-mid ray-cast BUT the offset is the
    posed-FK mid-ankle: `0.5·(canon[f,7]+canon[f,8])` with lateral (x) zeroed
    in canonical frame, rotated by `root_R[f]` (replaces `_ANKLE_IN_ROOT` use;
    keep the constant for `anchor_mode: ankle_mid` bit-parity).
  - `cfg["anchor_mode"]`: `"contact"` (default) runs detect_contacts +
    solve_root_with_pins on the carrier; `"ankle_mid"` reproduces today's
    canonical-offset path EXACTLY (bit-parity test below).
  - Existing behaviors preserved in both modes: low-conf hold-last, lean
    correction, confidence computation, trailing Savgol on root_t (Savgol runs
    on the carrier BEFORE δ is added, so pins are not smeared).
  - `fps` from the shot's camera track (`CameraTrack.fps`), plumbed through
    `process_player`.
  - After solving, stage writes sidecar `{out_key}_foot_contacts.json`
    (`FootContacts.to_json()` + `{"shot_id":…, "player_id":…, "anchor_mode":…}`).
- Produces: CLI `.venv311/bin/python scripts/reanchor_hmr_world.py --output output [--shot gberch] [--players P001,…] [--mode contact|ankle_mid] [--suffix _reanchored | --in-place]`
  loads npz + kp2d sidecar + camera track, calls `anchor_root_translation`,
  writes npz (+ contacts sidecar). SAFETY: default is non-destructive
  (`--suffix _reanchored`); `--in-place` first writes a one-time
  `*.npz.pre_reanchor.bak` next to the original and NEVER overwrites an
  existing .bak — the GVHMR originals cannot be regenerated on this Mac.
  Sidecar shape documented in `src/schemas/foot_contacts.py` per repo convention.

- [ ] **Step 1: Failing tests**

```python
def test_ankle_mid_mode_bit_parity_with_legacy():
    # fixture: 40-frame synthetic track through the old code path (copy the old
    # loop into the test as _legacy_anchor for parity) → assert allclose(atol=1e-10)
def test_contact_mode_pins_stance_feet_on_synthetic_track(): ...  # skate < 0.3 m/s in spans
def test_sidecar_written_and_round_trips(tmp_path): ...
def test_reanchor_script_rewrites_root_t_only(tmp_path): ...      # thetas/root_R/betas/frames byte-identical
```

- [ ] **Step 2: Verify fail** → **Step 3: Implement** (extract function, wire
  cfg + fps, write sidecar; keep `process_player`'s public signature — add
  `fps: float = 25.0` kwarg).
- [ ] **Step 4: Green** + regression: `.venv311/bin/python -m pytest tests/test_hmr_world_stage.py tests/test_reanchor_hmr_world.py tests/test_foot_anchor.py -q`
- [ ] **Step 4b: Mid-point eval on gberch:**

```bash
.venv311/bin/python scripts/reanchor_hmr_world.py --output output --shot gberch --mode contact
.venv311/bin/python scripts/eval_foot_quality.py --output output --stage hmr --json output/foot_quality_reanchored_hmr.json
```

  Expect at hmr level: stance skate (within sidecar spans) mean < 0.5 m/s;
  floating median gone (lower-foot z p50 < 0.06 m); no acceptance gate yet —
  record numbers in the task report.
- [ ] **Step 5: Commit** `feat(hmr_world): contact-aware posed-FK root anchoring + reanchor script`

---

### Task 6: Contact-aware ground-z in refined_poses

**Files:**
- Modify: `src/stages/refined_poses.py` (`_ground_snap`, `_clean_single_track`, `run`)
- Test: `tests/test_refined_poses_stage.py`, `tests/test_refined_poses_cleanup.py` (adjust + extend)

**Interfaces:**
- Consumes: `FootContacts.from_json` / `.shifted`, `derive_contacts_from_fk`.
- Produces: `_ground_snap(root_R, root_t, thetas, *, target_foot_z, max_snap_distance, rest_joints, contacts=None) -> np.ndarray`
  — same name, new `contacts` kwarg:
  - `contacts` given: snap ONLY frames where either foot is in contact
    (targeting the in-contact foot's z, not the blanket lower foot); frames
    with no contact are untouched (flight preserved).
  - `contacts is None`: legacy blanket behavior (existing tests keep passing).
- Produces: stage plumbing — `_clean_single_track(..., contacts=None)`;
  `run()` loads `{shot}__{pid}_foot_contacts.json` when present, else
  `derive_contacts_from_fk` on the incoming track, and passes it through.
  Contacts indices are hmr_world track-array indices; after the trim slice
  (`sl`), shift with `contacts.shifted(-i_first)` — write a test proving
  span/trim alignment.

- [ ] **Step 1: Failing tests**

```python
def test_ground_snap_with_contacts_preserves_flight():
    g = make_walk(n_frames=120)
    lifted = g.root_t.copy(); lifted[:, 2] += 0.08          # everything floats 8 cm
    fc = contacts_from_truth(g)
    snapped = _ground_snap(g.root_R, lifted, g.thetas, target_foot_z=0.02,
                           max_snap_distance=0.30, rest_joints=None,
                           contacts=fc.in_contact)
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, snapped)
    stance_z = fw[g.contacts_true[:, 0], 10, 2]
    assert np.percentile(np.abs(stance_z - 0.02), 95) < 0.03
    both_air = ~g.contacts_true.any(axis=1)
    if both_air.any():                                       # flight frames NOT dragged down
        np.testing.assert_allclose(snapped[both_air], lifted[both_air])

def test_ground_snap_without_contacts_is_legacy(): ...       # None → old behavior byte-identical
def test_stage_prefers_sidecar_over_fk_fallback(tmp_path): ...
def test_contact_indices_survive_trim(): ...
```

- [ ] **Step 2: Verify fail** → **Step 3: Implement** → **Step 4: Green**
  (`tests/test_refined_poses_*.py` all green).
- [ ] **Step 5: Commit** `feat(refined_poses): contact-aware ground snap (flight preserved)`

---

### Task 7: Foot-lock IK + penetration guard (`src/utils/foot_lock.py`, part 2)

**Files:**
- Modify: `src/utils/foot_lock.py`
- Test: `tests/test_foot_lock.py` (append)

**Interfaces:**
- Produces:

```python
def lock_feet_ik(*, thetas, root_R, root_t, betas, contacts, fps,
                 target_foot_z=0.02, ik_max_joint_delta_deg=10.0,
                 max_residual_correction_m=0.15, edge_ease_frames=3,
                 rest_joints=None) -> tuple[np.ndarray, np.ndarray, dict]
    # returns (thetas', root_t', stats) — root_R untouched
def penetration_guard(*, thetas, root_R, root_t, betas, sole_clearance_m=0.025,
                      rest_joints=None) -> tuple[np.ndarray, dict]
```

- `lock_feet_ik` algorithm, per contact span (on the FINAL smoothed track):
  1. Pin: XY = median FK foot-joint XY over span, z = `target_foot_z`.
  2. Per frame, ease weight `w(f)` = 1 inside, linear 0→1 over `edge_ease_frames`
     at each span edge.
  3. Root micro-correction: per frame, mean over active spans of
     `(pin − foot_fk)·w`, low-passed with a 5-frame triangular kernel, clamped
     to `max_residual_correction_m`; add to `root_t` (XY and Z).
  4. Two-bone leg IK per stance side to land the ANKLE at
     `pin + (ankle_fk − foot_fk)` (preserves the ankle→toe vector):
     - Work in root-local canonical frame: `T_local = root_R.Tᵀ… = root_R[f].T @ (target_world − root_t[f])`.
     - Hip pivot `H = canon_rest_chain FK hip position`, lengths
       `L1 = ‖rest[knee] − rest[hip]‖`, `L2 = ‖rest[ankle] − rest[knee]‖`.
     - `d = clip(‖T_local − H‖, |L1−L2| + 1e-4, L1 + L2 − 1e-4)`.
     - Knee interior angle from law of cosines; knee bend axis = current FK knee
       bend axis (fall back to local +x if leg straight); hip rotation = minimal
       rotation taking current knee direction to the required one THEN aiming the
       ankle at `T_local` (two-step: aim, then flex).
     - Convert to local axis-angle: `Rl_new = R_global_parentᵀ @ R_global_needed`;
       `theta_new = as_rotvec`.
     - Clamp per-joint |Δrotvec| ≤ `ik_max_joint_delta_deg` (as radians); if the
       clamped solve leaves the foot > 4 cm from pin, SKIP the whole span
       (`stats["spans_skipped"] += 1`) and restore its original thetas.
     - Counter-rotate the ankle joint (`theta[7 or 8]`) so the foot's GLOBAL
       orientation is unchanged: `Rl_ankle_new = R_thigh_shin_newᵀ @ R_thigh_shin_old @ Rl_ankle_old`.
     - Blend theta edits by `w(f)` (slerp on the rotvec: `theta = (1−w)·old + w·new`
       is acceptable for ≤10° deltas).
  5. stats: `spans_locked`, `spans_skipped`, `mean_pin_err_m_before/after`,
     `max_root_corr_m`, `max_joint_delta_deg`.
- `penetration_guard`: batch FK feet joints; `deficit[f] = max(0, sole_clearance_m − min(lz, rz))`;
  `raise = triangular_smooth(rolling_max(deficit, ±2), width 3)`; `root_t[:, 2] += raise`;
  stats: `frames_raised`, `max_raise_cm`. Never lowers.

- [ ] **Step 1: Failing tests**

```python
def test_lock_feet_ik_lands_feet_on_pins():
    g = make_walk(n_frames=120)
    noisy = g.root_t + np.random.default_rng(5).normal(0, 0.03, g.root_t.shape)
    fc = contacts_from_truth(g)
    th2, rt2, stats = lock_feet_ik(thetas=g.thetas, root_R=g.root_R, root_t=noisy,
                                   betas=g.betas, contacts=fc, fps=g.fps)
    fw = compute_all_joint_worlds_batch(th2, g.root_R, rt2)
    for span in fc.spans:
        j = 10 if span.side == 0 else 11
        core = slice(span.start + 3, max(span.start + 3, span.end - 3))
        err = np.linalg.norm(fw[core, j, :2] - span.pin[:2], axis=1)
        if err.size: assert err.max() < 0.03

def test_lock_feet_ik_respects_joint_clamp(): ...        # pin 0.5 m off → span skipped, thetas restored
def test_lock_feet_ik_preserves_foot_global_orientation(): ...  # global R of foot joint before≈after (≤2°)
def test_penetration_guard_raises_only_and_clears_ground():
    g = make_walk(n_frames=60)
    sunk = g.root_t.copy(); sunk[:, 2] -= 0.05
    rt2, stats = penetration_guard(thetas=g.thetas, root_R=g.root_R, root_t=sunk, betas=g.betas)
    fw = compute_all_joint_worlds_batch(g.thetas, g.root_R, rt2)
    assert fw[:, [10, 11], 2].min() >= 0.025 - 1e-6
    assert (rt2[:, 2] >= sunk[:, 2] - 1e-9).all()
def test_penetration_guard_noop_when_clear(): ...
```

- [ ] **Step 2: Verify fail** → **Step 3: Implement** → **Step 4: Green.**
- [ ] **Step 5: Commit** `feat(contact): foot-lock two-bone IK + penetration guard`

---

### Task 8: refined_poses finale integration + config

**Files:**
- Modify: `src/stages/refined_poses.py` (`run()` step 3–4 area, ~line 1342), `config/default.yaml`
- Test: `tests/test_refined_poses_stage.py` (append)

**Interfaces:**
- Consumes: `lock_feet_ik`, `penetration_guard`, contacts already plumbed in Task 6.
- Produces: in `run()`, immediately after `_smooth_track` per (shot, player) and
  BEFORE `_assemble_player`: when `cfg["foot_lock"]["enabled"]` (default true),
  apply `lock_feet_ik` then `penetration_guard` using that track's contacts
  (trim-shifted). Nothing modifies thetas/root_t afterwards (single-shot
  assembly is pass-through; multi-shot merged players skip the finale —
  documented limitation). Summary gains `"foot_lock": {…stats…}` in
  `refined_poses_summary.json`; `quality_report.json` picks it up via the
  existing summary passthrough.
- Config block (`config/default.yaml` under `refined_poses:`) and hmr block —
  copy EXACTLY from spec §4 (`anchor_mode`, `contact:` under `hmr_world`;
  `foot_lock:` under `refined_poses`); wire every key with the spec's defaults.

- [ ] **Step 1: Failing test** — stage-level fixture run:

```python
def test_stage_foot_lock_reduces_skate_and_clears_penetration(tmp_path):
    write_synthetic_hmr_world_fixture(tmp_path)   # helper: walk + noisy carrier root, sidecar
    stage = RefinedPosesStage(output_dir=tmp_path, config={"refined_poses": {"foot_lock": {"enabled": True}}})
    stage.run()
    rp = RefinedPose.load(tmp_path / "refined_poses" / "P001_refined.npz")
    m = foot_quality_metrics(...)
    assert m["skate"]["L"]["mean_mps"] < 0.3
    assert m["penetration"]["pct_frames_sole_below_0"] < 0.5
    summary = json.loads((tmp_path / "refined_poses" / "refined_poses_summary.json").read_text())
    assert summary["foot_lock"]["spans_locked"] > 0

def test_stage_foot_lock_disabled_matches_previous_pipeline(tmp_path): ...
```

- [ ] **Step 2: Verify fail** → **Step 3: Implement + config** → **Step 4: Green**
  (all `tests/test_refined_poses*.py`).
- [ ] **Step 5: Commit** `feat(refined_poses): foot-lock IK finale + penetration guard, config wiring`

---

### Task 9: End-to-end acceptance on gberch + docs

**Files:**
- Modify: `CLAUDE.md` (Configuration bullet for `hmr_world.anchor_mode`/`contact.*` + `refined_poses.foot_lock.*`), `output/foot_quality_*.json` artifacts
- No src changes expected (fix-forward only if gates fail).

- [ ] **Step 1: Full local pipeline rebuild on gberch:**

```bash
.venv311/bin/python scripts/reanchor_hmr_world.py --output output --shot gberch --mode contact --in-place
.venv311/bin/python recon.py run --input test-media/cleaned_up/gberch.mp4 --output ./output/ --stages refined_poses
.venv311/bin/python scripts/eval_foot_quality.py --output output --players P001,P002,P003 --json output/foot_quality_after.json
```

  (Verify `output/hmr_world/*.npz.pre_reanchor.bak` exist after the reanchor.
  If recon.py's input arg is awkward for a stage-only rerun, invoke the stage
  the way `tests/test_refined_poses_stage.py` does — document which was used.)
- [ ] **Step 2: Acceptance gates (spec §6):** stance skate mean < 0.3 / p95 < 0.8 m/s;
  sole penetration < 0.5 % frames, max < 1 cm; P001 lower-foot z p95 ≥ 0.10 m;
  ankle reprojection within +10 % of baseline. Iterate thresholds/params until
  green (parameter changes only — algorithm changes reopen the relevant task).
- [ ] **Step 3: Cross-clip report-only:** run the eval CLI on `output-origi`,
  `output-japan`, `output-kroupi` (skip gracefully where sidecars are missing)
  and record results in the final report.
- [ ] **Step 4: Full default suite:** `.venv311/bin/python -m pytest tests/ -q`
  — green minus the two known-failing tests.
- [ ] **Step 5: Commit** `feat(animation): contact-aware locomotion validated on gberch`
  (artifacts + CLAUDE.md), then dispatch **fp-qa** for the independent verdict.

## Delegation & Waves (fp-lead decomposition, adopted)

- **Wave 1:** Task 1 (fp-pipeline-3d) ∥ config+docs pre-landing (fp-generalist:
  spec §4 keys into `config/default.yaml` + CLAUDE.md bullet, no behavior change)
- **Wave 2:** Task 3 (detection) ∥ Task 4 (solver) — disjoint files, both fp-pipeline-3d
- **Wave 3:** Task 5 (hmr_world+reanchor) ∥ Tasks 6–8 refined_poses side — disjoint files
- **Wave 4:** Task 9 E2E validation/tuning
- **Wave 5:** fp-qa verification gate (full suite, independent eval re-run,
  parity tests, .bak existence, invariant sweep, cross-clip read-only sanity,
  explicit GPU-box deferral flags)
- Parallel agents must commit with explicit pathspecs (`git commit -m … -- <files>`)
  to avoid cross-staging races.

## Self-Review Notes

- Spec coverage: [A]→Tasks 1–2, [B]→Task 3, [C]→Tasks 4–5, [D]→Tasks 6–8,
  validation §6→Task 9. Sidecar + `shifted()` cover the trim/sync alignment
  question the spec left implicit.
- Deferred (documented in spec §6): GPU-box GVHMR fresh runs, ball touch-recall
  revalidation.
- Type consistency: `FootContacts` is the single contact currency everywhere;
  `contacts.in_contact` (bool array) is what `_ground_snap` takes, the full
  object elsewhere — signatures above are authoritative.

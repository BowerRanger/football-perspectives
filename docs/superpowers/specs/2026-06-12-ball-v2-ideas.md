# Ball v2: ideas + prototypes for shot-agnostic, physically-real ball motion

Status: **proposal for review** — not an approved design. Prototype code in
`prototypes/` (untracked); run with `.venv/bin/python`.

Goal (from operator): realistic ball striking, spin, flight, and contacts
(feet, body, head, keeper hands, posts, crossbar, net) with minimal user
input, mirroring what actually happened in 3D — pipeline-generated before UE
export preferred, art-generated in engine acceptable.

---

## 1. What is actually wrong today (evidence, not vibes)

The auto-physics rework (branch `ball-auto-physics`) fixed continuity —
catastrophic frame jumps went from 138 m to ≤ 7 m — but three upstream
problems dominate everything that still looks bad:

**a. Detection sparsity is the bottleneck.** origi02: 44 % of frames have any
detection; 51 consecutive frames missing around the climax. Even origi01's
"long flight" segment 2 (69 frames) contains only **9 raw detections at
conf ≥ 0.3**. No physics solver can recover what was never measured.

**b. Mode mis-segmentation.** The event detector fired side-net impacts at
frames 451/460, a bounce at 465 and velocity breaks at 470/475 (origi01,
the goalmouth sequence) — yet the solver fit **one** ballistic span 454–488,
flagged underconstrained at 30 px. Greedy event classification + split-and-
retry under-uses the solver's own event evidence.

**c. Monocular depth ambiguity.** Cross-checking against the origi02 replay
(Prototype A below): typical depth disagreement on flights is ~2 m; worst
observed 8 m (frame 328: solved track says grounded z = 0.22 m, two-view
triangulation with 0.15 m ray-miss says z = 8.3 m — right where the solver
flagged a marginal restitution).

Also missing entirely: automatic spin (presets only), any ball rotation
channel in export, and any engine-side reaction (net, posts, keeper hands)
to contacts.

**Industry context** (research sweep, see §5): published single-camera ball
accuracy plateaus ≈ 0.6 m (Yandex 2025); Hawk-Eye/Beyond Sports/Viz Libero all
render *physics re-simulations constrained by sparse measurements*, never raw
measured curves. Our anchors+primitives architecture is the right skeleton —
it needs better evidence, better segmentation, and an engine realism layer.

---

## 2. Prototype results (real data, this repo)

### Prototype A — replays as an ad-hoc multi-view rig
`prototypes/replay_triangulation.py`

origi01 (live) + origi02 (replay) are the same event, already grouped and
sync-mapped by `prepare_shots`. The two solved cameras sit **37.5 m apart →
11–44° parallax** (SoccerNet-v3D's replay-triangulation failure was near-zero
parallax; we don't have that problem). Triangulating WASB detections across
the synced pair:

- **11/21 frame pairs triangulate with < 1 m ray-miss**; best pairs 2–15 cm.
- Triangulated heights independently confirm the solved track where it's
  good (z agreement to ~0.1–0.4 m) and **expose an 8 m depth error** where
  it's bad.
- **The ball is a better sync signal than motion-energy NCC**: scanning
  offsets for minimum median ray-miss corrects the saved sync by 2 frames
  (median miss 3.03 m → 0.93 m). This is VisualSync-lite, for free.
- The climactic shot segment received **5 triangulated fixes** — enough to
  pin an entire arc's depth with zero user input.

### Prototype B — spin identifiability on a real flight
`prototypes/spin_fit_with_fixes.py`

Refitting the flagged shot span (454–488) three ways:

| fit | residual | verdict |
|---|---|---|
| parabola, monocular | 190 px | unphysical (98 m/s launch) — span is not one arc |
| 9-dof Magnus, monocular, unbounded | 137 px | **diverges** (512 km/s) — spin from short monocular arcs is ill-conditioned, as the gray-box literature warns |
| Magnus + 5 replay fixes (soft, 30 px/m) | 125 px | physically sane scale (22 m/s, 4 m apex) but no single arc fits all fixes — because the span really contains net-impact + bounce events |

Conclusions: (1) replay fixes massively regularize depth/scale; (2) spin must
be fitted **bounded**, on **clean single-regime segments**, ideally coupled
through bounces — the current preset+bounded-refinement design is right, the
segmentation feeding it is what fails; (3) fitting machinery cannot rescue
wrong segmentation.

---

## 3. Proposed ideas, ranked

### Idea 1 — Ball Evidence Booster (attack detection sparsity first)
Highest leverage, least glamour. Make the 2D evidence dense before anything
3D: (a) two-pass detection — run WASB once, then re-run low-threshold
detection inside a trajectory-gated corridor predicted by the IMM, re-scoring
weak blobs by track consistency instead of raw confidence; (b) pool evidence
across replay shots of the same event through the sync map (a frame missing
in the live view often has a confident replay detection); (c) tile/zoom
inference on far-camera frames where the ball is < 8 px; (d) optionally
fine-tune WASB on our own hard frames harvested via the dashboard.
*Effort: M. Risk: low. No user input. Directly converts origi02-class clips
from "detector-limited" to solvable.*

### Idea 2 — Cross-replay triangulation stage (Prototype A, productionized)
New sub-stage after `camera`: for each sync group, refine the frame offset by
minimizing ball ray-miss (sub-frame, VisualSync-lite), then triangulate all
well-conditioned detection pairs (parallax > ~8°, ray-miss < 1 m) into
**sparse 3D fixes with covariance**, persisted per shot. The piecewise solver
consumes fixes as soft constraints alongside anchors (operator anchors still
win). Kills monocular depth ambiguity exactly on the events that matter most
— shots, crosses, free kicks — because those are what broadcasts replay.
*Effort: M (infra exists: groups, sync_map, cameras). Risk: low — fixes are
gated by ray-miss, degrade to no-op when absent. No user input.*

### Idea 3 — Global mode-sequence solve (replace greedy split-and-retry)
Reframe the per-shot solve as a single search over mode sequences
{rolling, flight, possessed-by-player-X, impact(post/net/body), out}, beam-
searched along the camera rays (Yandex 2025 validates exactly this framing on
soccer broadcast: ≈ 0.6 m mean 3D error, kick F1 0.74). Events become *scored
hypotheses*, not greedy decisions; anchors and triangulated fixes are hard/
soft constraints; current physics primitives remain the flight/roll models.
Directly fixes the 454–488 class of failure (one span vs. impact+bounce+
scramble) and the 40-velocity-break mis-parse (201–282).
*Effort: L (the big one). Risk: medium — but it subsumes split-and-retry
rather than discarding the physics layer.*

### Idea 4 — Spin as a first-class, bounded, coupled state + rotation export
(a) Fit ω per flight segment with hard bounds (|ω| ≤ ~15 rev/s), seeded by
touch type; (b) couple consecutive segments through bounces (spin changes
post-bounce tangential velocity predictably — strong identifiability, per
gray-box studies and TT3D); (c) type the touches with a SoccerNet-style
action spotter (header vs volley vs instep — sets both contact joint and
spin prior) — replaces nothing, feeds existing presets automatically;
(d) **export ω**: add a rotation channel to glTF/FBX and Sequencer keys so
the ball visibly spins/curls in UE (today: position only).
*Effort: M. Risk: low-medium. No user input. This is what makes free kicks
and whipped crosses read as real.*

### Idea 5 — Engine realism layer: kinematic ball, simulated consequences
The industry-standard cheat (EA's net patent, Hawk-Eye virtual replays,
Beyond Sports): the solved ball stays authoritative and **kinematic** in
Sequencer; secondaries simulate against it and are cached deterministically:
- goal net = Chaos Cloth pinned to the frame, kinematic sphere collider →
  automatic bulge/ripple on our solver's net-impact events;
- post/crossbar ping + slight camera-visible vibration triggered by
  goal_impact events;
- keeper-hand contacts: IK-snap the keeper's wrist to the ball at the catch/
  parry anchor frame (SMPL wrists already exist in refined_poses);
- ball rotation from the ω channel (Idea 4) with contact-consistent rolling.
*Effort: M (UE-side, mostly orthogonal). Risk: low. Pure visual win — no new
reconstruction accuracy required.*

### Idea 6 — Learned 2D→3D lifting prior as proposal, not authority
Retrain a Where-Is-The-Ball-style canonical-ray lifter on synthetic
trajectories generated by our own drag+Magnus ODE under our solved-camera
distribution. Use it only to *propose* 3D infills for spans the solver flags
unexplained; accept solely when the proposal survives reprojection + physics
gates. Converts today's "needs a manual anchor" flags into auto-resolutions.
*Effort: L. Risk: medium (training pipeline). Defer until 1–3 land.*

---

## 4. Suggested phasing

1. **Phase 1 (evidence):** Idea 1 + Idea 2. Both zero-user-input, low risk,
   and they compound: more detections → more triangulated fixes. Re-validate
   on kroupi/origi/Liverpool reel; expect origi02 to flip from detector-
   limited to solvable.
2. **Phase 2 (solve):** Idea 3, consuming Phase 1 fixes. Acceptance: the
   454–488 and 201–282 spans segment correctly with no manual anchors.
3. **Phase 3 (look):** Idea 4 + Idea 5. Spin in export, net/post/keeper
   reactions in UE.
4. **Phase 4 (long tail):** Idea 6 for the remaining flagged spans.

## 5. Sources (research sweep highlights)

- Vorobev et al. 2025, single-camera soccer ball localization (beam search
  over modes along rays) — arXiv 2506.07981
- Gutierrez-Perez et al. 2025, SoccerNet-v3D (replay triangulation; parallax
  warning) — arXiv 2504.10106
- Ponglertnapakorn 2025, Where Is The Ball (learned monocular lifting) —
  arXiv 2506.05763
- Gossard et al. 2025, TT3D (contact-state-parameterized ODE fits; spin from
  curvature) — CVPRW
- Achterhold et al. 2023, gray-box ball dynamics (spin identifiability via
  bounce coupling) — arXiv 2305.15189
- SoccerNet Ball Action Spotting 2024 (T-DEED, 12 touch classes, 73 mAP@1)
- EA patent US 8000947 (Verlet cloth net vs kinematic projectile);
  UE5 Chaos Cloth kinematic colliders + cache system

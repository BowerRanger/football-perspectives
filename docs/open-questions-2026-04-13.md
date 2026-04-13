# Open questions — 2026-04-13 calibration + ball work

This document tracks judgment calls I made during the autonomous
implementation of (1) calibration debug overlay, (2) mowing-stripe and
advertising-board calibration cues, (4) cross-shot calibration
alignment, and (5) ball tracking with parabolic flight reconstruction.

For each question, I've included the answer I went with and the
reasoning. We'll review on your return.

## Format

Each entry: **Question**, **What I picked**, **Why**.

---

## 2026-04-13 — Calibration debug overlay findings

**Calibration quality verdict from the debug overlay**: PnLCalib's
recovered calibration is **substantially wrong** on both `origi01`
and `origi02`. Projected pitch lines miss the painted markings by
many metres in some frames. Examples:

- `origi01/frame_00378.jpg`: left-goal-line projection lands several
  metres into the pitch interior; goal posts (magenta) are not
  visible (they project off-screen).
- `origi01/frame_00504.jpg`: zoomed goal shot; goal-line and 6-yard
  box are roughly aligned but goal frame is missing.
- `origi02/frame_00250.jpg`: wide shot; left 18-yard box (green)
  projects onto the **centre** of the pitch instead of the left side.
  Left penalty arc projects on the right side. The whole calibration
  is rotated/translated by a large amount.
- `origi01/frame_00156.jpg`: tactical wide; centre circle roughly
  aligned but touchlines are off.

**Implication for triangulation**: bird's-eye output is "kind of
right" only because the same wrong calibration is used to back-project
both the pose pixels *and* (implicitly) the pitch background — the
errors partially cancel within a shot. Across shots they don't, which
is why the user sees a discrete shift around frame 250 (origi02
kicking in with a different bias than origi01).

**Plan implication**: #2 (mowing stripes + advertising boards) and
#4 (cross-shot alignment) are doing real work — they're not band-aids
on a small bias, they're addressing a fundamentally weak calibration.
Acceptance criterion for #2: re-render the debug overlay and require
the projected pitch lines to fall within ~5 px of the painted lines
on at least 80 % of the keyframes per shot.

---

## 2026-04-13 — Ball reconstruction judgment calls

**Q: Should ball tracking re-use the player ByteTracker or run a separate one?**
Picked: re-use the player ByteTracker.  Removed the `class_name != "ball"`
filter in `tracking.py` so balls flow through the same tracker as players.
Why: the player detector already emits ball detections; ByteTrack handles
mixed classes fine and the downstream code only looks at `class_name` to
split the streams.  Avoids running a second tracker.  Risk: ball gets a
team label from the team classifier (likely "unknown"); harmless but
worth re-checking.

**Q: What ball radius / ground plane?**
Picked: `_BALL_RADIUS = 0.11 m` (FIFA standard ~22 cm diameter).
Single-shot ground projection back-projects to `z = 0.11`, not `z = 0`,
so the ball centre sits on top of the pitch surface.

**Q: How to detect when the ball is in flight vs grounded?**
Picked: pixel-velocity heuristic.  Compute per-frame ball pixel velocity
(magnitude in image coordinates).  Contiguous frames with velocity above
`_FLIGHT_PX_VELOCITY = 25 px/frame` AND currently using ground-projection
form a candidate flight segment.  Segments shorter than 4 frames or
longer than 60 frames are not refined (under 4 frames there isn't enough
constraint for a 6-DOF parabola; over 60 frames is implausible for a
single arc).
Why: simple, single-shot, no extra ML.  Risk: fast ground passes can
look like flight; the parabolic fit will land them at z ≈ 0 anyway,
so the worst case is a wasted LM solve.

**Q: How to seed the parabola for LM?**
Picked: take the first and last segment frames' ground-projected
positions as start and end, derive horizontal velocity from the (end -
start) / duration, and seed vertical velocity at `0.5 * g * duration`
(the value that returns the ball to its starting z by the end of the
segment).  Why: gives the optimiser a sensible apex without needing a
prior on shot height.  LM converges quickly from any sensible seed.

**Q: When two shots see the ball at the same global frame, what
weighting does the DLT use?**
Picked: equal weights (1.0 each).  No detection confidence per ball
because the YOLO ball is a single best detection per frame after
tracking.  Could revisit if cross-shot disagreement turns out to be a
problem.

**Q: Where to render the ball in the bird's-eye view?**
Picked: white circle scaled by world height, with a thin vertical
drop-line connecting it to its ground projection.  Why: the bird's eye
is a top-down 2D view but the ball can be in flight; a drop-line
preserves the height information without confusing the (x,y) position.
The ball circle floats at `(x, y - z*4)` in canvas pixels (~4 pixels per
metre of height).  Risk: looks weird if the calibration is wrong (the
ball ends up off the pitch); not a correctness issue.

## 2026-04-13 — Cross-shot calibration alignment judgment calls

**Q: Joint optimisation across shots, or sequential alignment?**
Picked: sequential.  Each non-reference shot gets aligned independently
to the reference shot via 2D yaw + planar translation Procrustes.  Why:
joint optimisation needs an existing bundle adjuster (`bundle_adjust.py`
has one but it has different assumptions and isn't currently wired in)
and the sequential path is simpler to verify.  We can layer joint
adjustment on top later if pairwise alignment is insufficient.

**Q: Free 6-DOF rigid transform, or constrained to yaw + planar
translation?**
Picked: yaw + planar translation only.  Why: the static-camera fuser
already gave us a robust camera position per shot; an out-of-plane or
pitch-tilt component to the alignment would imply we got the camera
height wrong, which the line-refinement step doesn't touch.  Restricting
to in-plane keeps the alignment focused on what we believe is
recoverable from foot correspondences.

**Q: Sanity bounds for alignment?**
Picked: |yaw| ≤ 25° and ‖translation‖ ≤ 30 m.  Why: any larger correction
implies the input calibration is fundamentally broken, not just slightly
off.  Better to skip than to apply a huge correction we can't verify.
The user can lower these in config if they want stricter behaviour.

## 2026-04-13 — Calibration line refinement judgment calls

**Q: Two-VP closed-form solve, or single-VP rotation refinement?**
Picked: single-VP rotation refinement.  Tried two-VP first
(`vp_calibration.calibration_from_vanishing_points` recovers focal
length and full rotation from two orthogonal VPs in closed form), but
in practice the line detector finds only one dominant family per frame
(mowing stripes parallel to touchlines).  The "second cluster" was
noise variations on the same direction, leading to wildly wrong focal
length estimates.  Switched to: take the dominant cluster, compute its
single VP, apply the minimum rotation that aligns PnLCalib's projected
touchline direction to that VP.  Keeps PnLCalib's tilt/roll/focal
length intact and corrects only the dominant pan error.

**Q: Acceptance threshold for the refined rotation?**
Picked: angular VP-consistency residual must strictly decrease.  Why:
no false positives — if the refinement makes things worse on the
detected line cluster, we don't accept it.  Initial implementation
also rejected refinements that rotated PnLCalib's R by more than 30°,
since pan errors that big are usually a misclassified cluster.  Loosened
to 60° because some real-world frames have larger pan errors.

**Q: Should we also refine focal length?**
Picked: no, not for now.  Refining focal length needs a second VP that
we don't reliably have.  An iterative-closest-line refinement against
the painted markings (centre circle, penalty boxes) could give us the
focal length but it's a bigger lift — leaving for a future iteration.

**Q: Where to run the refinement?**
Picked: inside the calibration stage, immediately after PnLCalib's per-
shot fuse.  Why: this is the natural seam — the line refinement needs
each shot's already-fused (R, t) as initialisation, and downstream
stages don't care whether the rotations were touched.

---

(entries appended below as decisions are made — newest at the top)

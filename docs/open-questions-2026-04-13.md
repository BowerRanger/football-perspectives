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

(entries appended below as decisions are made — newest at the top)

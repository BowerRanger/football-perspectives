"""Per-foot ground-contact detection — the shared currency between the
foot-contact-aware locomotion components (see
docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md,
Task 3 of the implementation plan).

Two detectors share this module:

- :func:`detect_contacts` — the primary, image-faithful detector. Per
  foot, ray-casts the confident COCO-ankle pixel to the pitch ground
  plane every frame (via :func:`src.utils.foot_anchor.ankle_ray_to_pitch`)
  to get a world-frame track that is physically stationary during true
  stance and sweeps fast during swing; a pixel-noise-adaptive hysteresis
  state machine turns that into per-frame stance flags, gated by an FK
  check that the candidate foot is actually the lower one.
- :func:`derive_contacts_from_fk` — a torch-free fallback for
  ``refined_poses`` when no ``detect_contacts`` sidecar is available: the
  same hysteresis+span machinery, but driven by the already-solved FK
  foot-joint world track (no camera/kp2d needed) plus a height gate.

``FootContacts`` is meant to travel as a JSON sidecar
(``{shot}__{pid}_foot_contacts.json``, see ``src/schemas/foot_contacts.py``
once Task 5 adds it) alongside the ``hmr_world`` / ``refined_poses`` npz
tracks it was computed from.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from src.utils.foot_anchor import ankle_ray_to_pitch
from src.utils.smpl_skeleton import (
    beta_adjusted_rest_joints,
    compute_all_joint_worlds_batch,
    compute_canonical_joints_batch,
    load_smpl_neutral_model,
)

# COCO-17 ankle keypoint indices (left, right) — matches
# ``src.stages.hmr_world._COCO_LEFT_ANKLE`` / ``_COCO_RIGHT_ANKLE``.
_COCO_ANKLE_IDX = (15, 16)

# SMPL joint indices for the foot/toe (left, right) — CLAUDE.md's joint
# index table: hips 1/2, knees 4/5, ankles 7/8, feet(toes) 10/11. The
# FK-driven signals in this module (the lower-foot gate, and
# ``derive_contacts_from_fk``'s tracked position) use the *foot* joint,
# not the ankle: it is the joint the two-bone leg IK actually pins
# (``src.utils.foot_lock.lock_feet_ik``), and the one whose world
# position is invariant under root translation drift during a true
# stance span. The ankle, one rotation earlier in the chain, is NOT
# invariant even during genuine stance: as the body advances over a
# planted foot the tibia rotates about the (stationary) toe — heel
# lift — so the ankle sweeps through several centimetres even while the
# toe never moves (this is what the Wave-4 root-cause diagnosis found:
# pinning the ankle forced the anatomically-stationary toe to sweep
# instead). See ``derive_contacts_from_fk``'s docstring for the same
# point, worked through algebraically. The ray-cast in
# :func:`_ray_cast_ankles` still targets the COCO *ankle* pixel (15/16),
# because that is the keypoint GVHMR's ViTPose actually annotates and
# it drives the (ankle-based) speed signal well — but ``detect_contacts``
# now AUGMENTS that ray-cast with the current pose's rigid ankle->toe
# offset before taking the per-span pin (see the toe-offset step in
# :func:`detect_contacts`), so the pin itself estimates the toe, and is
# consumed as such by
# ``src.utils.foot_lock.solve_root_with_pins``'s root->foot offset).
_SMPL_FOOT_IDX = (10, 11)

# SMPL joint indices for the ankle (left, right) — the ray-cast target
# (matches the COCO ankle keypoint GVHMR annotates) and the base point
# the per-frame ankle->toe offset below is measured from.
_SMPL_ANKLE_IDX = (7, 8)

# Ankle-confidence cutoff below which a keypoint never anchors a
# ray-cast. Matches ``src.stages.hmr_world._ANKLE_CONF_MIN`` (both trace
# to the spec's keypoint-confidence threshold) — duplicated rather than
# imported because ``hmr_world`` imports FROM this module's siblings,
# not the other way around, and the constant is small enough that a
# cross-module import would add coupling for no real reuse.
_ANKLE_CONF_MIN = 0.3

# Pitch-frame z of the ray-cast ground plane — matches
# ``src.stages.hmr_world._FOOT_PLANE_Z`` (see that constant's docstring
# for why it's a few cm above z=0, not exactly on the turf). This is the
# height the ANKLE ray-cast assumes; it is NOT the pin's output z (see
# ``_PIN_TARGET_Z`` below — the two are independent).
_FOOT_PLANE_Z = 0.05

# Forced z of every span pin, overriding whatever z the (noisy) toe
# position estimate happened to carry. Matches
# ``refined_poses.ground_snap_target_z`` / ``foot_lock.target_foot_z``
# (config/default.yaml) — the height downstream consumers
# (``src.utils.foot_lock.solve_root_with_pins``, the contact-aware
# ground snap) actually want a planted foot pinned at, not the ray-cast
# ankle-plane height.
_PIN_TARGET_Z = 0.02

# FK lower-foot gate margin (metres, plan step 5): a foot only counts as
# a stance candidate if its world-z is no more than this far above the
# other foot's.
_FK_LOWER_FOOT_MARGIN_M = 0.05

# Quality assigned to every frame inside a derive_contacts_from_fk span.
# There is no image-confidence or px-noise signal in the FK-only path
# (it exists precisely because that evidence is unavailable), so every
# accepted frame gets the same fixed quality.
_FK_FALLBACK_QUALITY = 1.0


@dataclass(frozen=True)
class ContactSpan:
    """One contiguous run of a single foot being planted on the ground.

    ``start``/``end`` are frame-array positions (half-open: frames
    ``[start, end)``), aligned with whatever per-frame array
    (``FootContacts.in_contact`` today, an ``SmplWorldTrack``/
    ``RefinedPose``'s ``frames`` later) the enclosing ``FootContacts``
    was computed from. ``pin`` is the robust (median) world-frame pitch
    position (metres, z-up) the foot should be held at for the whole
    span — ``pin[2]`` is the pitch-plane ankle/foot-plane height
    constant used by the detector, not a per-span measurement.
    """

    side: int          # 0 = left, 1 = right
    start: int
    end: int
    pin: np.ndarray     # (3,) pitch-world metres

    def to_json(self) -> dict:
        return {
            "side": int(self.side),
            "start": int(self.start),
            "end": int(self.end),
            "pin": [float(x) for x in np.asarray(self.pin, dtype=float)],
        }

    @classmethod
    def from_json(cls, d: dict) -> "ContactSpan":
        return cls(
            side=int(d["side"]),
            start=int(d["start"]),
            end=int(d["end"]),
            pin=np.array(d["pin"], dtype=float),
        )


@dataclass(frozen=True)
class FootContacts:
    """Per-foot contact state for one player track.

    ``in_contact``/``quality`` are dense per-frame-position arrays
    (shape ``(n_frames, 2)``, column order ``[L, R]``) aligned 1:1 with
    the track's row index — NOT resampled by :meth:`shifted`, see there.
    ``spans`` is the derived contiguous-run view of ``in_contact`` with
    a robust pin position attached to each run; downstream consumers
    that only need "is this frame's foot planted, and where" can use
    ``in_contact`` directly, while consumers that need a stable pin
    target (root-solve, foot-lock IK) use ``spans``.
    """

    n_frames: int
    in_contact: np.ndarray          # (F, 2) bool  [L, R]
    quality: np.ndarray             # (F, 2) float in [0, 1]
    spans: tuple[ContactSpan, ...]

    def to_json(self) -> dict:
        return {
            "n_frames": int(self.n_frames),
            "in_contact": np.asarray(self.in_contact, dtype=bool).tolist(),
            "quality": np.asarray(self.quality, dtype=float).tolist(),
            "spans": [s.to_json() for s in self.spans],
        }

    @classmethod
    def from_json(cls, d: dict) -> "FootContacts":
        return cls(
            n_frames=int(d["n_frames"]),
            in_contact=np.array(d["in_contact"], dtype=bool),
            quality=np.array(d["quality"], dtype=float),
            spans=tuple(ContactSpan.from_json(s) for s in d["spans"]),
        )

    def shifted(self, offset: int) -> "FootContacts":
        """Re-base span frame indices by ``offset`` (e.g. a sync_map
        shot->reference-timeline offset, or ``-i_first`` when a caller
        trims its own per-frame arrays and needs the spans to stay
        aligned with the trimmed row indices).

        Only the span ``start``/``end`` (frame-position labels) move.
        ``in_contact``/``quality``/``n_frames`` are copied unchanged —
        they are dense arrays already indexed 0..n_frames-1 by array
        position, and a pure relabelling of what "position 0" means
        doesn't itself resample them. A caller that also needs to crop
        the dense arrays to a sub-range (e.g. after slicing its own
        track to an anchored span) does that slicing itself and then
        calls ``shifted`` for the span bookkeeping.
        """
        return FootContacts(
            n_frames=self.n_frames,
            in_contact=np.array(self.in_contact, copy=True),
            quality=np.array(self.quality, copy=True),
            spans=tuple(
                ContactSpan(
                    side=s.side,
                    start=s.start + offset,
                    end=s.end + offset,
                    pin=np.array(s.pin, copy=True),
                )
                for s in self.spans
            ),
        )


# ---------------------------------------------------------------------------
# Shared hysteresis + span machinery (module-private; used by both
# detect_contacts and derive_contacts_from_fk).
# ---------------------------------------------------------------------------


def _nanmedian_filter3(x: np.ndarray) -> np.ndarray:
    """NaN-aware centred 3-frame median filter along axis 0.

    Edge frames use whatever fewer neighbours are available (a 2-frame
    window) rather than reaching outside the track. A window that is
    entirely NaN filters to NaN (propagates, does not fabricate data).
    """
    n = int(x.shape[0])
    out = np.full_like(np.asarray(x, dtype=float), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(n):
            lo = max(0, i - 1)
            hi = min(n, i + 2)
            out[i] = np.nanmedian(x[lo:hi], axis=0)
    return out


def _central_diff_speed(pos: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference speed ``|pos[f+1] - pos[f-1]| * fps / 2``.

    NaN at the two track boundaries (no both-sided neighbour) and
    anywhere either neighbour is NaN — the hysteresis state machine
    below treats NaN speed as "exit immediately", per the plan.
    """
    n = int(pos.shape[0])
    v = np.full(n, np.nan, dtype=float)
    if n < 3:
        return v
    delta = np.asarray(pos[2:], dtype=float) - np.asarray(pos[:-2], dtype=float)
    v[1:-1] = np.linalg.norm(delta, axis=-1) * (float(fps) / 2.0)
    return v


def _hysteresis_mask(v: np.ndarray, v_enter: np.ndarray, v_exit: np.ndarray) -> np.ndarray:
    """Per-frame stance state via hysteresis over speed ``v``.

    Enters stance when ``v`` drops below ``v_enter``, holds until ``v``
    rises above ``v_exit``, and exits immediately whenever ``v`` is
    NaN/non-finite (an ambiguous frame can't confirm continued stance).
    ``v_enter``/``v_exit`` may vary per frame (the px-noise-adaptive
    floor in ``detect_contacts``) or be constant (``derive_contacts_from_fk``).
    """
    n = int(v.shape[0])
    out = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        vi = v[i]
        if not np.isfinite(vi):
            state = False
        elif state:
            if vi > v_exit[i]:
                state = False
        elif vi < v_enter[i]:
            state = True
        out[i] = state
    return out


def _runs_at_least(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous True runs of ``mask`` at least ``min_len`` long, as
    half-open ``[start, end)`` index pairs. Shorter runs are dropped
    (this is the ``min_span_frames`` gate — it filters kick-like
    momentary decelerations that briefly dip below the enter speed)."""
    n = int(mask.shape[0])
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _build_spans(
    gated: np.ndarray,
    pos: np.ndarray,
    min_span_frames: int,
    max_pin_spread_m: float | None,
    force_pin_z: float | None,
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """Shared span-extraction step for one foot: contiguous ``gated``
    runs at least ``min_span_frames`` long become spans with a robust
    (nanmedian) pin over ``pos``, rejected if ``max_pin_spread_m`` is
    given and the p90 in-span spread around that pin exceeds it (the
    kick/false-stance defence).

    Returns ``(accepted, spans)``: ``accepted`` is a fresh boolean mask
    with rejected/too-short runs zeroed out (the real per-frame
    ``in_contact`` column), and ``spans`` a list of ``(start, end,
    pin)`` — callers attach ``side`` when wrapping these in
    ``ContactSpan``.
    """
    n = int(gated.shape[0])
    accepted = np.zeros(n, dtype=bool)
    spans: list[tuple[int, int, np.ndarray]] = []
    for start, end in _runs_at_least(gated, min_span_frames):
        seg = pos[start:end]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pin = np.nanmedian(seg, axis=0)
        if not np.all(np.isfinite(pin)):
            continue
        if max_pin_spread_m is not None:
            spread = np.linalg.norm(seg - pin, axis=-1)
            spread = spread[np.isfinite(spread)]
            if spread.size and float(np.percentile(spread, 90)) > max_pin_spread_m:
                continue
        if force_pin_z is not None:
            pin = pin.copy()
            pin[2] = force_pin_z
        accepted[start:end] = True
        spans.append((start, end, pin))
    return accepted, spans


# ---------------------------------------------------------------------------
# detect_contacts — image-faithful ray-cast detector.
# ---------------------------------------------------------------------------


def _ray_cast_ankles(
    kp2d: np.ndarray,
    frame_indices: np.ndarray,
    per_frame_K: dict,
    per_frame_R: dict,
    per_frame_t: dict,
    distortion: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-cast each confident ankle pixel to the ground plane, per foot.

    Returns ``(w, scale, ankle_conf)``: ``w`` is ``(F, 2, 3)`` the raw
    per-frame per-foot pitch-world position (NaN where the keypoint is
    low-confidence, the camera is missing for that frame, or the ray
    is parallel to the ground plane); ``scale`` is ``(F, 2)`` the local
    metres-per-vertical-pixel ground scale at that same ray-cast (NaN
    alongside ``w``, plan step 3's adaptive-floor input); ``ankle_conf``
    is ``(F, 2)`` the raw COCO keypoint confidence (0 where the camera
    is missing for that frame).
    """
    n = int(frame_indices.shape[0])
    w = np.full((n, 2, 3), np.nan, dtype=float)
    scale = np.full((n, 2), np.nan, dtype=float)
    ankle_conf = np.zeros((n, 2), dtype=float)
    for i in range(n):
        fi = int(frame_indices[i])
        K = per_frame_K.get(fi)
        R = per_frame_R.get(fi)
        t = per_frame_t.get(fi)
        if K is None or R is None or t is None:
            continue
        for side, coco_idx in enumerate(_COCO_ANKLE_IDX):
            u, v_px, conf = kp2d[i, coco_idx]
            ankle_conf[i, side] = float(conf)
            if conf < _ANKLE_CONF_MIN:
                continue
            try:
                w[i, side] = ankle_ray_to_pitch(
                    (float(u), float(v_px)), K=K, R=R, t=t,
                    plane_z=_FOOT_PLANE_Z, distortion=distortion,
                )
                w_up = ankle_ray_to_pitch(
                    (float(u), float(v_px) + 1.0), K=K, R=R, t=t,
                    plane_z=_FOOT_PLANE_Z, distortion=distortion,
                )
                scale[i, side] = float(np.linalg.norm(w_up - w[i, side]))
            except ValueError:
                # Ray parallel to the ground plane — leave NaN.
                w[i, side] = np.nan
    return w, scale, ankle_conf


def _adaptive_speed_floors(
    scale: np.ndarray,
    fps: float,
    px_noise: float,
    speed_enter_m_s: float,
    speed_exit_m_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-noise-adaptive enter/exit speed floors (plan step 3).

    Raises the fixed config thresholds by the expected pure-pixel-jitter
    speed at each ray-cast's local ground scale, so a genuinely
    stationary far/small player (large metres-per-pixel scale) doesn't
    fail to register stance just because its ray-cast jitters. Falls
    back to the plain config threshold wherever ``scale`` itself is
    unavailable (no successful ray-cast at that frame) rather than
    propagating NaN into the hysteresis comparisons.
    """
    scale = np.asarray(scale, dtype=float)
    enter = np.maximum(speed_enter_m_s, 0.5 * px_noise * scale * fps)
    exit_ = np.maximum(speed_exit_m_s, 1.0 * px_noise * scale * fps)
    enter = np.where(np.isnan(enter), speed_enter_m_s, enter)
    exit_ = np.where(np.isnan(exit_), speed_exit_m_s, exit_)
    return enter, exit_


def _fk_lower_foot_gate_from_canon(canon: np.ndarray, root_R: np.ndarray) -> np.ndarray:
    """Per-frame, per-side gate (plan step 5): True where this foot's
    world-z is no more than ``_FK_LOWER_FOOT_MARGIN_M`` above the other
    foot's.

    Takes a precomputed ``canon`` (``compute_canonical_joints_batch``'s
    output) rather than ``thetas``/``rest_joints`` directly so
    :func:`detect_contacts` can share the one canonical-FK pass with the
    ankle->toe offset step below it, instead of recomputing it twice.

    Only ``root_R`` (orientation) is applied, not ``root_t`` — the
    shared pelvis translation cancels out of the z *comparison* between
    the two feet, so this gate works before the root translation this
    whole detector feeds into is solved (see
    ``compute_canonical_joints_batch``'s docstring).
    """
    foot_canon = canon[:, _SMPL_FOOT_IDX, :]                      # (F, 2, 3)
    foot_rotated = np.einsum("fba,fsa->fsb", root_R, foot_canon)
    fk_z = foot_rotated[:, :, 2]                                  # (F, 2)
    return np.stack(
        [fk_z[:, side] <= fk_z[:, 1 - side] + _FK_LOWER_FOOT_MARGIN_M for side in (0, 1)],
        axis=1,
    )


def _ankle_to_toe_world_offset(canon: np.ndarray, root_R: np.ndarray) -> np.ndarray:
    """Per-frame, per-side world-frame vector from the ankle (SMPL 7/8)
    to the foot/toe (SMPL 10/11), given the current pose.

    This is the rigid FK offset the toe-pin estimate in
    :func:`detect_contacts` adds to each ray-cast ankle position: since
    only the toe is exactly stationary during real stance (see
    ``_SMPL_FOOT_IDX``'s docstring), augmenting the ray-cast (which can
    only ever measure where the ANKLE keypoint projects to) with this
    offset turns it into an estimate of the toe instead. Rotation-only
    (like :func:`_fk_lower_foot_gate_from_canon`) — this is a vector
    between two joints, not a position, so no ``root_t`` translation
    applies.
    """
    ankle_canon = canon[:, _SMPL_ANKLE_IDX, :]  # (F, 2, 3)
    foot_canon = canon[:, _SMPL_FOOT_IDX, :]    # (F, 2, 3)
    offset_canon = foot_canon - ankle_canon
    return np.einsum("fba,fsa->fsb", root_R, offset_canon)


def detect_contacts(
    *,
    kp2d: np.ndarray,
    frame_indices: np.ndarray,
    per_frame_K: dict,
    per_frame_R: dict,
    per_frame_t: dict,
    distortion: tuple[float, float],
    thetas: np.ndarray,
    root_R: np.ndarray,
    betas: np.ndarray,
    fps: float,
    cfg: dict,
) -> FootContacts:
    """Per-foot ground-contact detection from ray-cast ViTPose ankles.

    See the module docstring and
    ``docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md``
    §2[B]/§3 for the full algorithm. Summary: ray-cast each confident
    ankle pixel to the ``z=0.05`` pitch plane every frame
    (:func:`_ray_cast_ankles`); NaN-aware 3-frame median filter, then
    central-difference world speed (from the raw ankle ray-cast — this
    SPEED signal is unaffected by the toe augmentation below); a
    pixel-noise-adaptive hysteresis state machine
    (:func:`_adaptive_speed_floors`, :func:`_hysteresis_mask`) turns
    speed into per-frame stance flags; an FK gate
    (:func:`_fk_lower_foot_gate_from_canon`) requires the candidate foot
    to actually be the lower one; each ray-cast ankle position is then
    augmented with the current pose's rigid ankle->toe offset
    (:func:`_ankle_to_toe_world_offset`) to turn it into a per-frame TOE
    position estimate — the toe, not the ankle, is what's exactly
    stationary during real stance (see ``_SMPL_FOOT_IDX``'s docstring for
    the Wave-4 root-cause diagnosis this fixes) — and :func:`_build_spans`
    turns the gated mask into ``min_span_frames``+``max_pin_spread_m``-
    filtered spans with a robust pin over those toe estimates, forced to
    ``_PIN_TARGET_Z`` (0.02 m, matching where downstream consumers want a
    planted foot, not the 0.05 m ankle ray-cast plane).

    Args:
        kp2d: ``(F, 17, 3)`` COCO-17 keypoints (u, v, conf), aligned
            1:1 with ``frame_indices`` (i.e. row ``i`` is frame
            ``frame_indices[i]``) — the same alignment
            ``src.stages.hmr_world`` uses for GVHMR's internal ViTPose
            output.
        frame_indices: ``(F,)`` absolute frame numbers for this track.
        per_frame_K/per_frame_R/per_frame_t: ``{frame_number: array}``
            camera intrinsics/rotation/translation, as built by
            ``src.stages.hmr_world`` from the camera track. Frames
            missing from these dicts are treated as camera-absent
            (NaN ray-cast for that frame).
        distortion: ``(k1, k2)`` radial distortion, as in
            ``ankle_ray_to_pitch``.
        thetas: ``(F, 24, 3)`` axis-angle pose, aligned with
            ``frame_indices`` (``thetas[:, 0]`` ignored, per the SMPL
            FK convention — see CLAUDE.md).
        root_R: ``(F, 3, 3)`` per-frame root world orientation, aligned
            with ``frame_indices``.
        betas: ``(10,)`` player shape, used to beta-adjust the rest
            joint table for the FK lower-foot gate.
        fps: clip frame rate, drives the central-difference speed and
            the adaptive floor's px->m/s conversion.
        cfg: mapping with (at least) ``speed_enter_m_s``,
            ``speed_exit_m_s``, ``min_span_frames``, ``max_pin_spread_m``,
            ``px_noise`` — see ``config/default.yaml``'s
            ``hmr_world.contact`` block, whose keys and defaults this
            function's ``cfg.get`` calls mirror exactly.
    """
    frame_indices = np.asarray(frame_indices)
    n = int(frame_indices.shape[0])
    kp2d = np.asarray(kp2d, dtype=float)

    speed_enter_m_s = float(cfg.get("speed_enter_m_s", 0.6))
    speed_exit_m_s = float(cfg.get("speed_exit_m_s", 1.2))
    min_span_frames = int(cfg.get("min_span_frames", 4))
    max_pin_spread_m = float(cfg.get("max_pin_spread_m", 0.25))
    px_noise = float(cfg.get("px_noise", 2.0))

    if n == 0:
        empty_bool = np.zeros((0, 2), dtype=bool)
        return FootContacts(
            n_frames=0, in_contact=empty_bool, quality=np.zeros((0, 2)), spans=(),
        )

    # 1. Ray-cast each confident ankle pixel, per foot.
    w, scale, ankle_conf = _ray_cast_ankles(
        kp2d, frame_indices, per_frame_K, per_frame_R, per_frame_t, distortion,
    )

    # 2. NaN-aware 3-frame median filter, then central-difference speed.
    w_s = np.stack([_nanmedian_filter3(w[:, side, :]) for side in (0, 1)], axis=1)
    v = np.stack(
        [_central_diff_speed(w_s[:, side, :], fps) for side in (0, 1)], axis=1,
    )

    # 3. Pixel-noise-adaptive enter/exit floors.
    v_enter_eff = np.empty((n, 2), dtype=float)
    v_exit_eff = np.empty((n, 2), dtype=float)
    for side in (0, 1):
        v_enter_eff[:, side], v_exit_eff[:, side] = _adaptive_speed_floors(
            scale[:, side], fps, px_noise, speed_enter_m_s, speed_exit_m_s,
        )

    # 4. Hysteresis, per foot.
    hyst = np.stack(
        [_hysteresis_mask(v[:, side], v_enter_eff[:, side], v_exit_eff[:, side])
         for side in (0, 1)],
        axis=1,
    )

    # 5. FK lower-foot gate + per-frame ankle->toe world offset (shares
    #    the one canonical-FK pass — see _fk_lower_foot_gate_from_canon's
    #    docstring for why this isn't computed twice).
    rest_joints = beta_adjusted_rest_joints(betas, load_smpl_neutral_model())
    canon = compute_canonical_joints_batch(thetas, rest_joints)  # (F, 24, 3)
    fk_gate = _fk_lower_foot_gate_from_canon(canon, root_R)
    gated = hyst & fk_gate

    # Augment each ray-cast ANKLE position with the rigid ankle->toe
    # offset implied by the current pose, so the per-span pin below
    # estimates the (truly stationary-in-stance) TOE instead of the
    # ankle. Only XY is augmented — the pin's z is forced to
    # _PIN_TARGET_Z regardless (see _build_spans's force_pin_z), so the
    # offset's z component would be discarded anyway; leaving w_s's own
    # z (the ray-cast ankle-plane height) in the spread-gate input keeps
    # that gate's z contribution consistent with what it measured.
    ankle_to_toe = _ankle_to_toe_world_offset(canon, root_R)  # (F, 2, 3)
    w_toe = w_s.copy()
    w_toe[:, :, :2] = w_s[:, :, :2] + ankle_to_toe[:, :, :2]

    # 6. Spans: min-length + pin-spread gates (over the TOE estimate);
    #    quality inside accepted spans (still keyed on the raw ray-cast
    #    speed signal, unaffected by the toe augmentation).
    in_contact = np.zeros((n, 2), dtype=bool)
    quality = np.zeros((n, 2), dtype=float)
    spans: list[ContactSpan] = []
    for side in (0, 1):
        accepted, side_spans = _build_spans(
            gated[:, side], w_toe[:, side, :], min_span_frames,
            max_pin_spread_m, force_pin_z=_PIN_TARGET_Z,
        )
        in_contact[:, side] = accepted
        v_exit_safe = np.where(v_exit_eff[:, side] > 0, v_exit_eff[:, side], 1.0)
        q = np.minimum(ankle_conf[:, side], 1.0 - v[:, side] / v_exit_safe)
        quality[:, side] = np.where(accepted, np.clip(q, 0.0, 1.0), 0.0)
        for start, end, pin in side_spans:
            spans.append(ContactSpan(side=side, start=start, end=end, pin=pin))

    spans.sort(key=lambda s: (s.start, s.side))
    return FootContacts(
        n_frames=n, in_contact=in_contact, quality=quality, spans=tuple(spans),
    )


# ---------------------------------------------------------------------------
# derive_contacts_from_fk — torch-free FK-only fallback (refined_poses).
# ---------------------------------------------------------------------------


def derive_contacts_from_fk(
    *,
    thetas: np.ndarray,
    root_R: np.ndarray,
    root_t: np.ndarray,
    betas: np.ndarray,
    fps: float,
    speed_enter: float = 0.6,
    speed_exit: float = 1.2,
    max_height: float = 0.12,
    min_span_frames: int = 4,
) -> FootContacts:
    """FK-only ground-contact fallback, for when no ``detect_contacts``
    sidecar (no kp2d/camera evidence) is available — e.g.
    ``refined_poses`` on a track that predates the sidecar, or a track
    whose contacts sidecar failed to load.

    Same hysteresis+span machinery as :func:`detect_contacts`
    (:func:`_hysteresis_mask`, :func:`_build_spans`), but the signal is
    the already-solved FK *foot* joint (SMPL ``l_foot``/``r_foot``,
    index 10/11 — not the ankle) world track (needs ``root_t``, unlike
    ``detect_contacts``'s pre-root-solve gate) and a height gate
    (``z < max_height``) stands in for the pixel-noise-adaptive floor —
    there is no kp2d to derive a floor from. The foot joint, not the
    ankle, is what's invariant during true stance: the two-bone leg IK
    this feeds (``src.utils.foot_lock.lock_feet_ik``) pins the foot
    joint directly, and a planted foot holds the *foot* position fixed
    while the hip/knee angles keep readjusting underneath it to track
    the moving pelvis — which leaves the ankle (a different linear
    combination of those same angles) drifting by a few mm/frame even
    in genuine stance. No pin-spread gate: unlike a ray-cast, this
    signal has no pixel noise to produce a spuriously wide "stance"
    cluster, so ``min_span_frames`` alone suffices.

    Args:
        thetas/root_R/root_t/betas: the solved SMPL track, standard
            conventions (``thetas[:, 0]`` ignored).
        fps: clip frame rate.
        speed_enter/speed_exit: fixed (non-adaptive) hysteresis speed
            thresholds, m/s.
        max_height: foot-joint world-z (m) above which a frame can
            never be in contact, regardless of speed.
        min_span_frames: minimum contiguous run length to trust.
    """
    thetas = np.asarray(thetas, dtype=float)
    n = int(thetas.shape[0])
    if n == 0:
        empty_bool = np.zeros((0, 2), dtype=bool)
        return FootContacts(
            n_frames=0, in_contact=empty_bool, quality=np.zeros((0, 2)), spans=(),
        )

    rest_joints = beta_adjusted_rest_joints(betas, load_smpl_neutral_model())
    world = compute_all_joint_worlds_batch(thetas, root_R, root_t, rest_joints)
    foot_pos = world[:, _SMPL_FOOT_IDX, :]  # (F, 2, 3)

    v = np.stack(
        [_central_diff_speed(foot_pos[:, side, :], fps) for side in (0, 1)], axis=1,
    )
    v_enter_arr = np.full((n, 2), float(speed_enter))
    v_exit_arr = np.full((n, 2), float(speed_exit))
    hyst = np.stack(
        [_hysteresis_mask(v[:, side], v_enter_arr[:, side], v_exit_arr[:, side])
         for side in (0, 1)],
        axis=1,
    )
    height_gate = foot_pos[:, :, 2] < float(max_height)
    gated = hyst & height_gate

    in_contact = np.zeros((n, 2), dtype=bool)
    quality = np.zeros((n, 2), dtype=float)
    spans: list[ContactSpan] = []
    for side in (0, 1):
        accepted, side_spans = _build_spans(
            gated[:, side], foot_pos[:, side, :], int(min_span_frames),
            max_pin_spread_m=None, force_pin_z=None,
        )
        in_contact[:, side] = accepted
        quality[:, side] = np.where(accepted, _FK_FALLBACK_QUALITY, 0.0)
        for start, end, pin in side_spans:
            spans.append(ContactSpan(side=side, start=start, end=end, pin=pin))

    spans.sort(key=lambda s: (s.start, s.side))
    return FootContacts(
        n_frames=n, in_contact=in_contact, quality=quality, spans=tuple(spans),
    )

"""Procedural end-effector motion for the SMPL 24-joint body.

GVHMR leaves the SMPL hand joints (22, 23) frozen at zero for entire
clips and the toe joints (10, 11) near-rigid — COCO-17 evidence ends at
wrists and ankles, so nothing downstream ever animates them. The result
reads as mannequin stubs: no toe roll at push-off, no follow-through.

This pass adds bounded, deterministic secondary motion:

- a static relaxed hand curl (replacing the flat zero pose),
- hand follow-through — a lagged, decaying echo of the wrist's angular
  velocity, so fast arm swings whip through the hand,
- toe roll around detected contact-span edges — plantarflex easing out
  after lift-off (gated on toe height so it cannot dig into the pitch)
  and dorsiflex easing in before touchdown.

Frames inside stance spans keep flat feet; body joints are untouched.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.smpl_skeleton import compute_all_joint_worlds_batch

_HANDS = ((22, 20), (23, 21))   # (hand joint, parent wrist joint)
_FEET = (10, 11)


def animate_end_effectors(*, thetas, root_R, root_t, contacts, fps,
                          rest_joints=None, betas=None, cfg=None):
    """Return ``(new_thetas, stats)``; only joints 10, 11, 22, 23 change.

    ``contacts`` is a FootContacts whose spans are positional on the
    same frame axis as ``thetas`` (the resolved/effective set), or None.
    ``rest_joints`` may be omitted when ``betas`` is given.
    """
    cfg = cfg or {}
    if rest_joints is None:
        from src.utils.foot_lock import _resolve_rest_joints
        rest_joints = _resolve_rest_joints(betas, None)
    th = np.asarray(thetas, dtype=float).copy()
    n = len(th)
    if n == 0:
        return th, {"toe_roll_frames": 0, "hand_frames": 0}

    # --- hands: relaxed curl + follow-through ------------------------
    curl = np.deg2rad(float(cfg.get("hand_curl_deg", 20.0)))
    decay = float(cfg.get("follow_through_decay", 0.75))
    gain = float(cfg.get("follow_through_gain", 0.6))
    max_ft = np.deg2rad(float(cfg.get("follow_through_max_deg", 25.0)))
    for hand, wrist in _HANDS:
        wr = Rotation.from_rotvec(th[:, wrist])
        omega = np.zeros((n, 3))
        omega[1:] = (wr[:-1].inv() * wr[1:]).as_rotvec()
        follow = np.zeros((n, 3))
        for t in range(1, n):
            # The lag opposes the parent's motion, then relaxes — a whip.
            follow[t] = decay * follow[t - 1] - gain * omega[t]
        norm = np.linalg.norm(follow, axis=1, keepdims=True)
        follow *= np.minimum(1.0, max_ft / np.maximum(norm, 1e-9))
        base = Rotation.from_rotvec(np.tile([curl, 0.0, 0.0], (n, 1)))
        th[:, hand] = (base * Rotation.from_rotvec(follow)).as_rotvec()

    # --- toes: roll around contact-span edges ------------------------
    roll = np.zeros((n, 2))
    toe_frames = 0
    if contacts is not None and getattr(contacts, "spans", None):
        fw = compute_all_joint_worlds_batch(th, np.asarray(root_R), np.asarray(root_t), rest_joints)
        toe_z = fw[:, list(_FEET), 2]
        in_stance = np.zeros((n, 2), dtype=bool)
        for span in contacts.spans:
            in_stance[span.start:span.end, span.side] = True
        rf = max(1, int(cfg.get("toe_roll_frames", 4)))
        down = np.deg2rad(float(cfg.get("toe_roll_deg", 22.0)))
        up = np.deg2rad(float(cfg.get("toe_up_deg", 12.0)))
        min_clear = float(cfg.get("min_toe_clearance_m", 0.06))
        for span in contacts.spans:
            side = span.side
            for k in range(rf):           # push-off: toes trail downward
                i = span.end + k
                if i >= n or in_stance[i, side]:
                    break
                if toe_z[i, side] >= min_clear:
                    roll[i, side] = max(roll[i, side], down * (1 - k / rf))
            for k in range(1, rf + 1):    # approach: toes lift for landing
                i = span.start - k
                if i < 0 or in_stance[i, side]:
                    break
                roll[i, side] = min(roll[i, side], -up * (1 - (k - 1) / rf))
        touched = roll != 0
        toe_frames = int(touched.sum())
        for s, joint in enumerate(_FEET):
            idx = np.flatnonzero(touched[:, s])
            if len(idx):
                extra = np.zeros((len(idx), 3))
                extra[:, 0] = roll[idx, s]
                th[idx, joint] = (Rotation.from_rotvec(th[idx, joint])
                                  * Rotation.from_rotvec(extra)).as_rotvec()

    return th, {"toe_roll_frames": toe_frames, "hand_frames": n,
                "max_follow_through_deg": float(np.degrees(max_ft))}

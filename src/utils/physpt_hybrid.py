"""Gated PhysPT takeover: splice physics-refined motion into flagged spans.

The evidence-anchored animation stays authoritative everywhere except
spans a detector flags as unrealistic (acceleration/rotation spikes,
occlusion). There the PhysPT version takes over, re-anchored so its
root trajectory meets the surrounding animation at both span edges —
PhysPT's integrated-XY drift cannot survive that pinning — with an
eased rotation blend so neither splice point pops.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from src.utils.pose_temporal import frame_runs


@dataclass(frozen=True)
class TakeoverConfig:
    """Detector + splice + verification settings for the gated takeover.

    Defaults are the gberch-validated values (docs/physpt-experiment-
    results.md, "Gated takeover"); re-sweep before trusting them on a
    clip with a very different camera or player scale.
    """
    acc_hi: float = 35.0          # root acceleration spike, m/s^2
    step_hi: float = 15.0         # root rotation step spike, deg/frame
    conf_lo: float = 0.25         # mean keypoint confidence occlusion cutoff
    dilate: int = 4
    merge_gap: int = 6
    min_len: int = 3
    ease: int = 6                 # rotation blend ramp, frames
    smooth_window: int = 9        # savgol over the spliced translation (odd; <=2 disables)
    verify_slack: float = 1.05    # per-span acceptance factor vs the current span peak

    @classmethod
    def from_mapping(cls, raw: dict) -> "TakeoverConfig":
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**known)


def flag_spans(bad, *, dilate=4, merge_gap=6, min_len=3):
    """Half-open (a, b) index spans from a per-frame boolean mask.

    Each flagged frame is widened by ``dilate`` on both sides; spans
    closer than ``merge_gap`` merge; spans shorter than ``min_len``
    are dropped. Operates on one contiguous run of frames.
    """
    bad = np.asarray(bad, dtype=bool)
    if not bad.any():
        return []
    idx = np.flatnonzero(bad)
    spans = []
    for i in idx:
        a, b = max(0, i - dilate), min(len(bad), i + dilate + 1)
        if spans and a - spans[-1][1] <= merge_gap:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return [(a, b) for a, b in spans if b - a >= min_len]


def blend_weight(length, ease):
    """Raised-cosine takeover weight for a span: 0→1→0 over ``ease``
    frames at each edge, 1 in the interior (a short span never
    exceeds its edge ramps)."""
    w = np.ones(length)
    e = min(int(ease), (length + 1) // 2)
    if e > 0:
        ramp = (1 - np.cos(np.pi * (np.arange(1, e + 1) / (e + 1)))) / 2
        w[:e] = ramp
        w[length - e:] = ramp[::-1]
    return w


def reanchor_translation(cur, phys, a, b):
    """PhysPT translation over [a, b) offset to equal the current
    animation at both span edges (linear offset interpolation; a span
    touching the run edge uses the inner edge's constant offset)."""
    cur = np.asarray(cur, dtype=float)
    phys = np.asarray(phys, dtype=float)
    lo, hi = a, b - 1
    d0 = cur[lo] - phys[lo]
    d1 = cur[hi] - phys[hi]
    if a == 0 and b < len(cur):
        d0 = d1
    elif b == len(cur) and a > 0:
        d1 = d0
    frac = np.linspace(0.0, 1.0, b - a)[:, None]
    return phys[a:b] + (1 - frac) * d0 + frac * d1


def slerp_blend(R_cur, R_phys, w):
    """Geodesic partial rotation blend: weight 0 returns ``R_cur``,
    weight 1 returns ``R_phys``."""
    rc = Rotation.from_matrix(np.asarray(R_cur, dtype=float))
    rp = Rotation.from_matrix(np.asarray(R_phys, dtype=float))
    key = (rc.inv() * rp).as_rotvec()
    return (rc * Rotation.from_rotvec(key * np.asarray(w)[:, None])).as_matrix()


def keypoint_confidence(kp2d_path: Path, frames, offset: int = 0):
    """Mean COCO-17 keypoint confidence per track frame (0 where the
    dashboard sidecar has no entry — an undetected/occluded frame).
    ``offset`` maps reference-timeline frames to the sidecar's
    shot-local frames (``shot_frame = ref_frame - offset``)."""
    conf = np.zeros(len(frames))
    if not kp2d_path.exists():
        return conf
    by_frame = {f['frame']: f['keypoints']
                for f in json.loads(kp2d_path.read_text())['frames']}
    for i, f in enumerate(frames):
        kp = by_frame.get(int(f) - int(offset))
        if kp is not None:
            conf[i] = float(np.mean(np.asarray(kp)[:, 2]))
    return conf


def detect_bad(track, conf, a, b, fps, cfg: TakeoverConfig):
    """Per-frame badness over one contiguous run [a, b)."""
    t = track.root_t[a:b]
    n = b - a
    acc = np.zeros(n)
    if n > 2:
        acc[1:-1] = np.linalg.norm(np.diff(t, 2, axis=0), axis=1) * fps * fps
    step = np.zeros(n)
    if n > 1:
        r = Rotation.from_matrix(track.root_R[a:b])
        step[1:] = np.degrees(np.linalg.norm((r[:-1].inv() * r[1:]).as_rotvec(), axis=1))
    reasons = {'acc': acc > cfg.acc_hi, 'step': step > cfg.step_hi,
               'occlusion': conf[a:b] < cfg.conf_lo}
    return np.any(list(reasons.values()), axis=0), reasons


def _peak_acc(t, fps):
    return float(np.linalg.norm(np.diff(t, 2, axis=0), axis=1).max() * fps * fps) if len(t) > 2 else 0.


def _peak_step(R):
    if len(R) < 2:
        return 0.
    r = Rotation.from_matrix(R)
    return float(np.degrees(np.linalg.norm((r[:-1].inv() * r[1:]).as_rotvec(), axis=1)).max())


def build_hybrid(cur, phys, conf, fps, cfg: TakeoverConfig):
    """Splice PhysPT into flagged spans, but verify each span per
    channel: a takeover that worsens the span's own peak acceleration
    (translation) or peak rotation step (rotations) is rejected and
    that channel keeps the current animation — reject, don't mangle.

    ``cur``/``phys`` are frame-aligned RefinedPose tracks; frames
    outside accepted spans stay bit-identical to ``cur``. Returns
    ``(hybrid_track, spans_meta)``.
    """
    theta = cur.thetas.astype(float).copy()
    rr = cur.root_R.astype(float).copy()
    rt = cur.root_t.astype(float).copy()
    spans_meta = []
    for a, b in frame_runs(cur.frames):
        bad, reasons = detect_bad(cur, conf, a, b, fps, cfg)
        for sa, sb in flag_spans(bad, dilate=cfg.dilate, merge_gap=cfg.merge_gap, min_len=cfg.min_len):
            ga, gb = a + sa, a + sb
            w = blend_weight(gb - ga, cfg.ease)
            new_t = reanchor_translation(cur.root_t[a:b], phys.root_t[a:b], sa, sb)
            # PhysPT's 20 fps height/trajectory prediction is locally
            # jittery at 30 fps; smooth the spliced span so the takeover
            # cannot import acceleration spikes worse than what it fixes.
            if cfg.smooth_window > 2 and len(new_t) > cfg.smooth_window:
                keep0, keep1 = new_t[0].copy(), new_t[-1].copy()
                new_t = savgol_filter(new_t, cfg.smooth_window, 2, axis=0)
                new_t[0], new_t[-1] = keep0, keep1  # edge pins survive smoothing
            cand_t = (1 - w[:, None]) * cur.root_t[ga:gb] + w[:, None] * new_t
            # Verify with a margin so splice-edge coupling counts too.
            lo, hi = max(a, ga - 2), min(b, gb + 2)
            trial = cur.root_t[lo:hi].copy()
            trial[ga - lo:gb - lo] = cand_t
            t_ok = _peak_acc(trial, fps) <= cfg.verify_slack * _peak_acc(cur.root_t[lo:hi], fps)
            if t_ok:
                rt[ga:gb] = cand_t
            cand_R = slerp_blend(cur.root_R[ga:gb], phys.root_R[ga:gb], w)
            trialR = cur.root_R[lo:hi].copy()
            trialR[ga - lo:gb - lo] = cand_R
            r_ok = _peak_step(trialR) <= cfg.verify_slack * _peak_step(cur.root_R[lo:hi])
            if r_ok:
                rr[ga:gb] = cand_R
                for j in range(1, 24):
                    blended = slerp_blend(Rotation.from_rotvec(cur.thetas[ga:gb, j]).as_matrix(),
                                          Rotation.from_rotvec(phys.thetas[ga:gb, j]).as_matrix(), w)
                    theta[ga:gb, j] = Rotation.from_matrix(blended).as_rotvec()
            spans_meta.append({'start': int(cur.frames[ga]), 'end': int(cur.frames[gb - 1]),
                'frames': int(gb - ga),
                'translation': 'accepted' if t_ok else 'rejected',
                'rotation': 'accepted' if r_ok else 'rejected',
                'triggers': {k: int(v[sa:sb].sum()) for k, v in reasons.items() if v[sa:sb].any()}})
    hybrid = replace(cur, thetas=theta.astype(np.float32), root_R=rr.astype(np.float32),
                     root_t=rt.astype(np.float32))
    return hybrid, spans_meta

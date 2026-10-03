"""Direction-consistency gate for detector observations (design D6.2).

On the gberch finish the detector locked onto a false object (keeper glove /
ad board) from frame 387 to 393 that moved the OPPOSITE way to the ball.
Reprojection residual alone can't always reject such a run (a short run can
drag a free-depth arc towards it), but its image-space *velocity* is flatly
inconsistent with the flight: it reverses against the fitted ball.

``filter_reversed_observations`` takes the fitted flight (a first-pass
solve), and drops runs of >= ``min_run`` consecutive detections whose
frame-to-frame pixel displacement points against the fitted ball's
displacement over the same frames AND which sit more than ``residual_px``
from the fitted position. The detection immediately before such a run (the
false track's first point, which shows up as a jump rather than a reversal)
is dropped too when it is also off the fit.

Pure numpy; knows nothing about anchors -- operator anchors never enter the
observation list, and ``protect_frames`` additionally shields any frame that
carries operator evidence.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

DEFAULT_CFG: dict[str, Any] = {
    "enabled": True,
    "min_run": 2,          # consecutive reversed detections to drop a run
    "min_px": 3.0,         # ignore displacements smaller than this (jitter)
    "max_gap_frames": 3,   # displacement pairs further apart are not compared
    "residual_px": 15.0,   # a reversed detection must also be this far off
}


def direction_gate_cfg(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(DEFAULT_CFG)
    if cfg:
        out.update(cfg)
    return out


def filter_reversed_observations(
    observations: Sequence[Any],
    fitted_uv: Callable[[int], np.ndarray | None],
    *,
    protect_frames: Iterable[int] = (),
    cfg: Mapping[str, Any] | None = None,
) -> tuple[list[Any], list[int]]:
    """Return ``(kept_observations, dropped_frames)``.

    ``observations`` need ``.frame`` and ``.uv``; ``fitted_uv(frame)`` is
    the fitted flight's projected pixel at that frame (``None`` where the
    fit isn't a flight / has no answer -- those frames are never judged).
    """
    c = direction_gate_cfg(cfg)
    if not c["enabled"] or len(observations) < 2:
        return list(observations), []
    protect = set(int(f) for f in protect_frames)
    obs = sorted(observations, key=lambda o: o.frame)

    pred: list[np.ndarray | None] = [fitted_uv(o.frame) for o in obs]
    resid: list[float] = []
    for o, p in zip(obs, pred):
        resid.append(float("nan") if p is None else
                     float(np.hypot(p[0] - o.uv[0], p[1] - o.uv[1])))

    flagged = [False] * len(obs)
    for i in range(1, len(obs)):
        if pred[i] is None or pred[i - 1] is None:
            continue
        if obs[i].frame - obs[i - 1].frame > int(c["max_gap_frames"]):
            continue
        d_obs = np.asarray(obs[i].uv, float) - np.asarray(obs[i - 1].uv, float)
        d_fit = np.asarray(pred[i], float) - np.asarray(pred[i - 1], float)
        if np.linalg.norm(d_obs) < c["min_px"] or np.linalg.norm(d_fit) < c["min_px"]:
            continue
        if float(d_obs @ d_fit) < 0.0 and resid[i] > c["residual_px"]:
            flagged[i] = True

    drop: set[int] = set()
    i = 0
    while i < len(obs):
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(obs) and flagged[j + 1]:
            j += 1
        if j - i + 1 >= int(c["min_run"]):
            drop.update(range(i, j + 1))
            k = i - 1  # the false track's first point: a jump, not a reversal
            if (k >= 0 and resid[k] == resid[k]  # not NaN
                    and resid[k] > c["residual_px"]
                    and obs[i].frame - obs[k].frame <= int(c["max_gap_frames"])):
                drop.add(k)
        i = j + 1

    kept, dropped = [], []
    for idx, o in enumerate(obs):
        if idx in drop and o.frame not in protect:
            dropped.append(int(o.frame))
        else:
            kept.append(o)
    return kept, dropped

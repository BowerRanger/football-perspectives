"""Score a camera track against hand-clicked anchor landmarks.

The clicked pixels are operator ground truth wherever they exist, so
reprojection residuals against them are the honest accuracy measure for
a camera solve (PnLCalib on MPS is nondeterministic across runs, so
exact track comparison is meaningless — metrics with tolerances are the
regression currency). Shared by ``scripts/eval_anchor_clicks.py`` and
the camera regression gate so the two scoring paths cannot drift.
"""

from __future__ import annotations

import numpy as np

from src.utils.camera_projection import project_world_to_image


def score_track(anchors: dict, track: dict) -> dict:
    """Reprojection metrics for ``track`` against ``anchors`` (plain
    JSON-loaded dicts in the on-disk sidecar schemas).

    Returns a metrics dict with overall click residual percentiles
    (``None`` when no anchor frame is covered by the track), per-anchor
    breakdown, anchor coverage, and track-level frame count / mean
    confidence.
    """
    cams = {f["frame"]: f for f in track["frames"]}
    dist = tuple(track.get("distortion", (0.0, 0.0))[:2])

    all_res: list[float] = []
    per_anchor: list[dict] = []
    covered = 0
    anchor_list = anchors.get("anchors", [])
    for anchor in anchor_list:
        frame = anchor["frame"]
        cam = cams.get(frame)
        if cam is None:
            continue
        covered += 1
        residuals = _anchor_residuals(anchor, cam, dist)
        if residuals:
            all_res.extend(residuals)
            per_anchor.append({
                "frame": frame,
                "clicks": len(residuals),
                "med_px": float(np.median(residuals)),
                "max_px": float(max(residuals)),
            })

    confidences = [f["confidence"] for f in track["frames"]
                   if f.get("confidence") is not None]
    return {
        "clicks": len(all_res),
        "med_px": float(np.median(all_res)) if all_res else None,
        "p90_px": float(np.percentile(all_res, 90)) if all_res else None,
        "max_px": float(max(all_res)) if all_res else None,
        "per_anchor": per_anchor,
        "anchor_frames_total": len(anchor_list),
        "anchor_frames_covered": covered,
        "track_frames": len(track["frames"]),
        "mean_confidence": (float(np.mean(confidences))
                            if confidences else None),
    }


def _anchor_residuals(anchor: dict, cam: dict,
                      dist: tuple[float, float]) -> list[float]:
    K = np.array(cam["K"])
    R = np.array(cam["R"])
    t = np.array(cam["t"])
    residuals = []
    for lm in anchor.get("landmarks", []):
        world = np.array([lm["world_xyz"]], dtype=float)
        projected = project_world_to_image(K, R, t, dist, world)[0]
        clicked = np.array(lm["image_xy"], dtype=float)
        residuals.append(float(np.linalg.norm(projected - clicked)))
    return residuals

"""Camera regression gate logic: baseline comparison + fixture discovery.

Fixtures live at ``tests/regression/camera/<clip_id>/`` — one dir per
golden clip, each holding:

- ``anchors.json``  — frozen manual anchor set (the pseudo-ground-truth
  clicks; ``output/`` is gitignored so the copy in the fixture is the
  committed source of truth)
- ``baseline.json`` — captured metrics + run-to-run spread + tolerances
  (written by ``scripts/capture_camera_regression_baseline.py``)
- ``shot.json``     — the shot's manifest entry, so the regression test
  can rebuild a minimal single-shot output dir

Tolerances are metric-relative because PnLCalib on MPS is
nondeterministic across runs: each gate allows
``baseline + max(relative slack, observed multi-run spread)``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

_DEFAULT_TOLERANCES = {"med_rel": 0.15, "p90_rel": 0.20,
                       "confidence_abs": 0.05}


def compare_to_baseline(metrics: dict, baseline: dict) -> list[str]:
    """Gate ``metrics`` (a ``score_track`` result) against ``baseline``.

    Returns a list of human-readable failure messages; empty means the
    track has not regressed. Single-click ``max_px`` is deliberately
    not gated — one noisy worst click should not fail the build.
    """
    base = baseline["metrics"]
    spread = baseline.get("spread", {})
    tol = {**_DEFAULT_TOLERANCES, **baseline.get("tolerances", {})}
    failures: list[str] = []

    if metrics["med_px"] is None:
        return [f"no clicks scored (baseline had {base['clicks']})"]

    for key, rel_key in (("med_px", "med_rel"), ("p90_px", "p90_rel")):
        slack = max(base[key] * tol[rel_key], spread.get(key, 0.0))
        limit = base[key] + slack
        if metrics[key] > limit:
            failures.append(
                f"{key} regressed: {metrics[key]:.2f}px > "
                f"{limit:.2f}px (baseline {base[key]:.2f}px "
                f"+ slack {slack:.2f}px)")

    for key in ("anchor_frames_covered", "track_frames"):
        if metrics[key] < base[key]:
            failures.append(
                f"{key} dropped: {metrics[key]} < baseline {base[key]}")

    if base.get("mean_confidence") is not None:
        slack = max(tol["confidence_abs"],
                    spread.get("mean_confidence", 0.0))
        floor = base["mean_confidence"] - slack
        if (metrics["mean_confidence"] or 0.0) < floor:
            failures.append(
                f"mean_confidence dropped: "
                f"{metrics['mean_confidence']:.3f} < {floor:.3f} "
                f"(baseline {base['mean_confidence']:.3f} "
                f"- slack {slack:.3f})")

    return failures


def build_solve_dir(work_dir: Path, shot_fixture: dict, anchors: dict,
                    clip_path: Path) -> None:
    """Lay out a minimal single-shot output dir the camera stage can run
    against: shots manifest + clip copy + anchors sidecar.

    ``shot_fixture`` is the fixture's ``shot.json``: ``{"fps": ...,
    "shot": <manifest entry>}`` as captured from the source output dir.
    """
    shot = shot_fixture["shot"]
    shots_dir = work_dir / "shots"
    cam_dir = work_dir / "camera"
    shots_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_file": "",
        "fps": shot_fixture["fps"],
        "total_frames": shot["end_frame"] - shot["start_frame"] + 1,
        "shots": [shot],
        "groups": [],
        "match": None,
    }
    (shots_dir / "shots_manifest.json").write_text(
        json.dumps(manifest, indent=1))
    shutil.copyfile(clip_path, work_dir / shot["clip_file"])
    (cam_dir / f"{shot['id']}_anchors.json").write_text(
        json.dumps(anchors, indent=1))


def aggregate_runs(runs: list[dict]) -> tuple[dict, dict]:
    """Fold N ``score_track`` results into (baseline metrics, spread).

    Residual percentiles and confidence take the across-run median;
    count-like fields take the minimum so a once-flaky run can't bake
    in a floor later runs can't meet. Spread is max-min, feeding the
    nondeterminism allowance in ``compare_to_baseline``.
    """
    spread_keys = ("med_px", "p90_px", "mean_confidence")
    metrics = dict(runs[0])
    for key in ("med_px", "p90_px", "max_px", "mean_confidence"):
        metrics[key] = float(np.median([r[key] for r in runs]))
    for key in ("clicks", "anchor_frames_covered", "track_frames"):
        metrics[key] = min(r[key] for r in runs)
    spread = {
        key: float(max(r[key] for r in runs) - min(r[key] for r in runs))
        for key in spread_keys
    }
    return metrics, spread


def discover_clips(fixture_root: Path) -> list[str]:
    """Clip ids with a complete fixture dir under ``fixture_root``."""
    if not fixture_root.is_dir():
        return []
    return sorted(
        d.name for d in fixture_root.iterdir()
        if d.is_dir()
        and (d / "baseline.json").exists()
        and (d / "anchors.json").exists()
    )

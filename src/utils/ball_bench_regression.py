"""Ball-stage regression gate logic: baseline comparison + fixture
discovery + run aggregation. Mirrors ``src/utils/camera_regression.py``'s
pattern for the camera gate.

Fixtures live at ``tests/regression/ball/<clip_id>/`` — one dir per golden
clip, each holding:

- ``anchors.json``         — frozen manual anchor set (``BallAnchorSet``
  JSON) the ``truth_mismatch.json``/``synth_obs_mismatch.json`` fixtures
  were built from — pinned so a later edit to the clip's LIVE anchors in
  ``$M/output*/ball/`` cannot silently change what the gate grades against.
- ``shot.json``             — shot id/fps + the clip video's relative path
  and sha256, so the gate can find + verify the real clip media in ``$M``
  and skip cleanly if it's missing or has changed.
- ``truth_mismatch.json``   — frozen synthetic 3-D ground truth (the
  ``mismatch`` scenario; see ``ball_bench_truth.build_truth``).
- ``synth_obs_mismatch.json`` — frozen synthetic detector stream derived
  from the truth above (see ``ball_bench_synth.make_synth_run``).
- ``baseline.json``         — aggregated metrics + run-to-run spread +
  tolerances, written by ``scripts/capture_ball_regression_baseline.py``.

Tolerances are metric-relative, with an allowance for measured run-to-run
spread: unlike the camera gate (PnLCalib is nondeterministic on MPS), the
ball stage is near-deterministic once its detection cache is warm, so the
measured spread across repeated runs should be small — but it's measured,
not assumed, and folded into the allowed slack the same way the camera
gate does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# key -> (baseline key, "higher_is_better" | "lower_is_better", rel_tol_key)
_SYNTH_GATE_KEYS: tuple[tuple[str, str, str], ...] = (
    ("pct_le_20cm", "higher_is_better", "pct_le_20cm_abs"),
    ("p95", "lower_is_better", "p95_rel"),
    ("ground_float_sink", "lower_is_better", "ground_float_sink_rel"),
    ("naturalness_violations_minus_truth", "lower_is_better",
     "naturalness_violations_abs"),
)
_REAL_GATE_KEYS: tuple[tuple[str, str, str], ...] = (
    ("p50", "lower_is_better", "real_p50_rel"),
)

_DEFAULT_TOLERANCES = {
    "pct_le_20cm_abs": 0.05,       # +/- 5 percentage points
    "p95_rel": 0.20,
    "ground_float_sink_rel": 0.20,
    "naturalness_violations_abs": 2.0,
    "real_p50_rel": 0.20,
}


def _slack(key: str, base_val: float, spread_val: float, tol: dict,
           tol_key: str, kind: str) -> float:
    if kind == "abs" or tol_key.endswith("_abs"):
        return max(tol.get(tol_key, 0.0), spread_val)
    return max(abs(base_val) * tol.get(tol_key, 0.0), spread_val)


def compare_to_baseline(synth_metrics: dict, real_metrics: dict,
                         baseline: dict) -> list[str]:
    """Gate ``(synth_metrics, real_metrics)`` — flat metric dicts for the
    ``mismatch`` scenario / real 2-fold held-out evaluation, in the same
    shape :func:`aggregate_runs` produces — against ``baseline``. Returns
    a list of human-readable failure messages; empty means no regression.
    """
    tol = {**_DEFAULT_TOLERANCES, **baseline.get("tolerances", {})}
    failures: list[str] = []

    base_synth = baseline["synth_metrics"]
    spread_synth = baseline.get("synth_spread", {})
    for key, direction, tol_key in _SYNTH_GATE_KEYS:
        cur = synth_metrics.get(key)
        base = base_synth.get(key)
        if cur is None or base is None:
            failures.append(f"synth.{key}: missing value "
                             f"(current={cur!r}, baseline={base!r})")
            continue
        kind = "abs" if tol_key.endswith("_abs") else "rel"
        slack = _slack(key, base, spread_synth.get(key, 0.0), tol, tol_key, kind)
        if direction == "higher_is_better" and cur < base - slack:
            failures.append(
                f"synth.{key} regressed: {cur:.4f} < {base:.4f} - "
                f"{slack:.4f} (baseline {base:.4f}, slack {slack:.4f})")
        if direction == "lower_is_better" and cur > base + slack:
            failures.append(
                f"synth.{key} regressed: {cur:.4f} > {base:.4f} + "
                f"{slack:.4f} (baseline {base:.4f}, slack {slack:.4f})")

    base_real = baseline.get("real_metrics")
    spread_real = baseline.get("real_spread", {})
    if base_real is not None:
        for key, direction, tol_key in _REAL_GATE_KEYS:
            cur = real_metrics.get(key)
            base = base_real.get(key)
            if cur is None or base is None:
                failures.append(f"real.{key}: missing value "
                                 f"(current={cur!r}, baseline={base!r})")
                continue
            kind = "abs" if tol_key.endswith("_abs") else "rel"
            slack = _slack(key, base, spread_real.get(key, 0.0), tol,
                            tol_key, kind)
            if direction == "lower_is_better" and cur > base + slack:
                failures.append(
                    f"real.{key} regressed: {cur:.4f} > {base:.4f} + "
                    f"{slack:.4f} (baseline {base:.4f}, slack {slack:.4f})")

    return failures


def aggregate_runs(dicts: list[dict], keys: tuple[str, ...]) -> tuple[dict, dict]:
    """Fold N flat metric dicts into ``(median metrics, spread)`` over
    ``keys``. A key missing/``None`` in every run stays ``None`` with
    spread ``0.0``; a key missing in SOME runs only uses the runs where
    it's present (so an occasional zero-event scenario doesn't nuke the
    whole metric to ``None``)."""
    metrics: dict[str, Any] = {}
    spread: dict[str, float] = {}
    for key in keys:
        vals = [d.get(key) for d in dicts if d.get(key) is not None]
        if not vals:
            metrics[key] = None
            spread[key] = 0.0
            continue
        metrics[key] = float(np.median(vals))
        spread[key] = float(max(vals) - min(vals))
    return metrics, spread


def discover_clips(fixture_root: Path) -> list[str]:
    """Clip ids with a complete fixture dir under ``fixture_root``."""
    if not fixture_root.is_dir():
        return []
    required = ("baseline.json", "anchors.json", "shot.json",
                "truth_mismatch.json", "synth_obs_mismatch.json")
    return sorted(
        d.name for d in fixture_root.iterdir()
        if d.is_dir() and all((d / f).exists() for f in required)
    )


__all__ = ["compare_to_baseline", "aggregate_runs", "discover_clips"]

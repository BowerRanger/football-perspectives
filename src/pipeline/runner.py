import contextlib
import json
import re
import sys
import time
from pathlib import Path

from src.pipeline.base import BaseStage
from src.pipeline.quality_report import write_quality_report

# Stages are imported lazily inside _stage_class() so deleting a not-yet-
# rebuilt stage doesn't break other tooling that imports the runner.

_STAGE_NAMES: list[str] = [
    "prepare_shots",
    "tracking",
    "camera",
    "hmr_world",
    # ``refined_poses`` runs before ``ball`` so the ball stage's
    # player_touch anchors resolve to bone positions on the cleaned
    # (outlier-rejected, lean-corrected, smoothed) poses — touches
    # land where the actual player limb is, not on raw HMR jitter.
    "refined_poses",
    "ball",
    "export",
]


def _stage_class(name: str) -> type[BaseStage] | None:
    """Lazy import so partially-implemented pipelines still load."""
    if name == "prepare_shots":
        from src.stages.prepare_shots import PrepareShotsStage
        return PrepareShotsStage
    if name == "tracking":
        from src.stages.tracking import PlayerTrackingStage
        return PlayerTrackingStage
    if name == "camera":
        from src.stages.camera import CameraStage
        return CameraStage
    if name == "hmr_world":
        from src.stages.hmr_world import HmrWorldStage
        return HmrWorldStage
    if name == "ball":
        from src.stages.ball import BallStage
        return BallStage
    if name == "refined_poses":
        from src.stages.refined_poses import RefinedPosesStage
        return RefinedPosesStage
    if name == "export":
        from src.stages.export import ExportStage
        return ExportStage
    raise ValueError(f"Unknown stage: {name!r}")


def resolve_stages(stages: str, from_stage: str | None) -> list[str]:
    if stages == "all":
        selected = list(_STAGE_NAMES)
    else:
        selected = []
        for token in stages.split(","):
            name = token.strip()
            if name not in _STAGE_NAMES:
                raise ValueError(f"Unknown stage: {name!r}")
            selected.append(name)
    if from_stage:
        if from_stage not in _STAGE_NAMES:
            raise ValueError(f"Unknown stage: {from_stage!r}")
        idx = _STAGE_NAMES.index(from_stage)
        selected = [n for n in selected if _STAGE_NAMES.index(n) >= idx]
    return selected


# hmr_world prints one "(i/total) <shot>__<player> done in ..." line per
# (shot, player) pair as GVHMR finishes fitting each one (see
# src/stages/hmr_world.py:284-337). The runner taps that existing stdout
# signal to recover a per-shot timing breakdown without touching stage
# code — only the "<shot>__<player>" label and the print's own arrival
# time matter; the human-formatted duration embedded in the message
# itself is not reparsed.
_HMR_WORLD_DONE_RE = re.compile(r"^\[hmr_world\] \(\d+/\d+\) (?P<label>\S+) done in ")


class _TeeStdout:
    """Write-through stdout wrapper that timestamps each completed line.

    Every write still reaches the real stdout immediately — console/log
    streaming is unaffected — while lines are buffered with an arrival
    timestamp so the caller can recover when each unit of work finished.
    """

    def __init__(self, real_stdout) -> None:
        self._real = real_stdout
        self.lines: list[tuple[float, str]] = []
        self._buf = ""

    def write(self, text: str) -> int:
        self._real.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.lines.append((time.perf_counter(), line))
        return len(text)

    def flush(self) -> None:
        self._real.flush()


def _hmr_world_per_shot_seconds(
    lines: list[tuple[float, str]], stage_start: float,
) -> dict[str, float]:
    """Aggregate hmr_world's per-(shot, player) "done in" prints into
    per-shot wall seconds.

    Each player's cost is the gap between its "done in" line arriving
    and the previous one (or ``stage_start`` for the first) — measured
    by the runner's own clock, not by reparsing the human-formatted
    duration inside the message.
    """
    per_shot: dict[str, float] = {}
    prev_t = stage_start
    for t, line in lines:
        m = _HMR_WORLD_DONE_RE.match(line)
        if m is None:
            continue
        shot_id, _, _player_id = m.group("label").rpartition("__")
        per_shot[shot_id] = per_shot.get(shot_id, 0.0) + (t - prev_t)
        prev_t = t
    return per_shot


def _load_timings(output_dir: Path) -> dict:
    """Load ``output/timings.json`` if present so a partial re-run
    (``--from-stage`` / a single ``--stages`` name) only overwrites the
    entries for the stages it actually ran, leaving earlier stages'
    recorded timings in place."""
    path = output_dir / "timings.json"
    if not path.exists():
        return {"stages": {}}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {"stages": {}}
    if not isinstance(data.get("stages"), dict):
        data["stages"] = {}
    return data


def _write_timings(output_dir: Path, timings: dict) -> None:
    timings["total_seconds"] = sum(
        float(s.get("seconds", 0.0)) for s in timings["stages"].values()
    )
    (output_dir / "timings.json").write_text(json.dumps(timings, indent=2))


def run_pipeline(
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config: dict,
    shot_filter: str | None = None,
    player_filter: str | None = None,
    **stage_kwargs,
) -> None:
    """Run pipeline stages.

    ``shot_filter`` (optional): when set, every stage that iterates the
    shots manifest will only process the named shot. Stages that don't
    use the manifest ignore it. Used by the dashboard's
    /api/run-shot endpoint to re-run a single stage for a single shot
    without re-running everything.

    ``player_filter`` (optional): when set, hmr_world will only fit the
    named ``player_id`` (paired with ``shot_filter`` to disambiguate
    when the same player_id appears in multiple shots). Stages that
    don't iterate per-player ignore it. Used by the dashboard's
    /api/run-shot-player endpoint to iterate quickly on one player.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    active = resolve_stages(stages, from_stage)
    timings = _load_timings(output_dir)
    for name in _STAGE_NAMES:
        if name not in active:
            continue
        StageClass = _stage_class(name)
        if StageClass is None:
            print(f"  [SKIP] {name} (not implemented)")
            continue
        stage = StageClass(config=config, output_dir=output_dir, **stage_kwargs)
        if shot_filter is not None:
            stage.shot_filter = shot_filter
        if player_filter is not None:
            stage.player_filter = player_filter
        # Filtered runs (shot or player) always re-enter the stage —
        # is_complete() reflects the unfiltered state and would short-
        # circuit a per-shot or per-player retry otherwise.
        filtered = shot_filter is not None or player_filter is not None
        if stage.is_complete() and from_stage != name and not filtered:
            print(f"  [SKIP] {name} (cached)")
            continue
        print(f"  [RUN]  {name}")
        stage_start = time.perf_counter()
        if name == "hmr_world":
            # Only hmr_world currently exposes per-shot progress on
            # stdout (its per-player "done in" lines); tap it for the
            # per-shot breakdown without touching stage code.
            tee = _TeeStdout(sys.stdout)
            with contextlib.redirect_stdout(tee):
                stage.run()
            per_shot = _hmr_world_per_shot_seconds(tee.lines, stage_start)
        else:
            stage.run()
            per_shot = {}
        timings["stages"][name] = {
            "seconds": time.perf_counter() - stage_start,
            "per_shot": per_shot,
        }

    _write_timings(output_dir, timings)

    # Aggregate per-stage diagnostics into output/quality_report.json.
    # This always runs (each section is independent of stage activation).
    try:
        write_quality_report(output_dir)
    except Exception as exc:  # noqa: BLE001 — diagnostics must never fail the run
        print(f"  [WARN] quality_report aggregation failed: {exc}")

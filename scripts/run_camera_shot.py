"""Re-run the camera stage for ONE shot with optional config overrides,
then print the eval dashboard for it.

Usage:
  .venv/bin/python scripts/run_camera_shot.py OUTPUT_DIR SHOT_ID [KEY=VALUE ...]
e.g.
  .venv/bin/python scripts/run_camera_shot.py output-origi origi01 \
      camera.line_extraction_propagate_circle=true
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging  # noqa: E402

_DEBUG = "--debug" in sys.argv
if _DEBUG:
    sys.argv.remove("--debug")
logging.basicConfig(
    level=logging.DEBUG if _DEBUG else logging.INFO,
    format="%(levelname)s %(message)s")
if _DEBUG:
    # only our stage at DEBUG; third-party loggers stay at INFO
    for name in list(logging.root.manager.loggerDict):
        if not name.startswith("src."):
            logging.getLogger(name).setLevel(logging.INFO)

from src.pipeline.config import load_config  # noqa: E402
from src.stages.camera import CameraStage  # noqa: E402


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def main() -> None:
    out_dir, shot_id = sys.argv[1], sys.argv[2]
    overrides = sys.argv[3:]
    # --scratch=DIR: run in an isolated copy (symlinked shots, copied anchors)
    # so parallel experiments don't fight over the live output files.
    scratch = [a for a in overrides if a.startswith("--scratch=")]
    if scratch:
        overrides = [a for a in overrides if not a.startswith("--scratch=")]
        import shutil
        src = Path(out_dir).resolve()
        dst = Path(scratch[0].split("=", 1)[1])
        (dst / "camera").mkdir(parents=True, exist_ok=True)
        if not (dst / "shots").exists():
            (dst / "shots").symlink_to(src / "shots")
        for f in (src / "camera").glob(f"{shot_id}_anchors*.json"):
            shutil.copy(f, dst / "camera" / f.name)
        out_dir = str(dst)
        print(f"scratch run in {out_dir}")

    cfg = load_config()
    for kv in overrides:
        key, _, val = kv.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
        print(f"override: {key} = {node[parts[-1]]!r}")

    stage = CameraStage(output_dir=Path(out_dir), config=cfg)
    stage.shot_filter = shot_id
    t0 = time.time()
    stage.run()
    print(f"\ncamera stage done in {time.time() - t0:.0f}s\n")

    from scripts._clip_eval import ev
    n_by_shot = {"gberch": 429, "kroupi01": 156, "origi01": 506, "origi02": 334}
    ev(f"{out_dir}/camera/{shot_id}", shot_id, n_by_shot.get(shot_id, 0))


if __name__ == "__main__":
    main()

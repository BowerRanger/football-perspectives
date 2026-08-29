"""cProfile a single pipeline stage against an existing output directory.

Loads the same stage class the runner would (see
``src.pipeline.runner._stage_class``), restricts it to one shot/player when
asked, profiles ``stage.run()`` with cProfile, and prints the top-N
functions by cumulative (or another pstats sort key) time. Complements
``output/timings.json`` (wall time per stage, written by the runner on every
`recon.py run`) with a breakdown of *where* the time goes inside one stage.

Requires the stage's upstream inputs to already exist in --output (shots,
camera track, refined_poses/hmr_world, etc, depending on the stage) — this
does not run earlier stages for you.

Usage:
  .venv311/bin/python scripts/profile_stage.py --output output --stage ball
  .venv311/bin/python scripts/profile_stage.py --output output --stage camera \
      --shot gberch -n 40 --sort tottime
  .venv311/bin/python scripts/profile_stage.py --output output --stage hmr_world \
      --shot gberch --player P001 --dump /tmp/hmr_world.prof
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_config  # noqa: E402
from src.pipeline.runner import _stage_class  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path,
                     help="existing pipeline output dir (its stage inputs "
                          "must already be present)")
    ap.add_argument("--stage", required=True,
                     help="stage name, e.g. prepare_shots, tracking, camera, "
                          "hmr_world, refined_poses, ball, export")
    ap.add_argument("--shot", default=None,
                     help="restrict to one shot (sets stage.shot_filter)")
    ap.add_argument("--player", default=None,
                     help="restrict to one player_id (hmr_world only; "
                          "sets stage.player_filter, pair with --shot)")
    ap.add_argument("--config", type=Path, default=None,
                     help="optional YAML override merged with defaults")
    ap.add_argument("-n", "--top", type=int, default=30,
                     help="number of pstats rows to print (default 30)")
    ap.add_argument("--sort", default="cumulative",
                     choices=sorted(pstats.Stats.sort_arg_dict_default),
                     help="pstats sort key (default cumulative)")
    ap.add_argument("--dump", type=Path, default=None,
                     help="also write the raw cProfile stats here "
                          "(loadable with pstats.Stats or snakeviz)")
    args = ap.parse_args()

    try:
        StageClass = _stage_class(args.stage)
    except ValueError as exc:
        ap.error(str(exc))  # exits with status 2, does not return

    cfg = load_config(args.config)
    stage = StageClass(config=cfg, output_dir=args.output)
    if args.shot is not None:
        stage.shot_filter = args.shot
    if args.player is not None:
        stage.player_filter = args.player

    print(f"profiling stage={args.stage!r} output={args.output} "
          f"shot={args.shot!r} player={args.player!r}")

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        stage.run()
    finally:
        profiler.disable()

    if args.dump is not None:
        profiler.dump_stats(str(args.dump))
        print(f"raw stats written to {args.dump}")

    stats = pstats.Stats(profiler).sort_stats(args.sort)
    stats.print_stats(args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())

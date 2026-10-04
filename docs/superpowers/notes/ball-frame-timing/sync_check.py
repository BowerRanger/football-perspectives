"""Fresh-pick sync check: origi01 exact ball pixels vs origi02 exact pixels at offsets -144..-140.
Uses the solver directly (no server)."""
import sys
from pathlib import Path

sys.path.insert(0, "/Users/joebower/workplace/football-perspectives/.claude/worktrees/gberch-shorts")
from src.utils import ball_truth_solver as S  # noqa: E402
from src.web.ball_studio import StudioData  # noqa: E402

out = Path("/Users/joebower/workplace/football-perspectives/output-origi-shorts")
data = StudioData(out, None)
c1, c2 = data.shot_cams("origi01"), data.shot_cams("origi02")
O1 = {414: (717.5, 488.5), 416: (648.5, 483.5), 417: (617.5, 480.0)}
O2 = {272: (850.0, 475.5), 273: (817.5, 477.0), 274: (785.0, 480.5), 275: (755.0, 484.0), 276: (727.5, 489.0)}
for r, uv1 in O1.items():
    row = []
    for f2, uv2 in O2.items():
        res = S.triangulate([(c1.cam(r), uv1), (c2.cam(f2), uv2)])
        row.append(f"off {f2 - r:+d}: {max(res.residual_px):5.2f}px gap {res.skew_gap_cm:5.1f}cm z {res.xyz[2]:+.2f}")
    print(f"ref {r}:\n  " + "\n  ".join(row))

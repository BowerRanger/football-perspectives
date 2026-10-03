#!/usr/bin/env python
"""Score the appearance stage against operator ground truth.

Compares ``<out>/appearance/kits.json`` with a hand palette (default: the
frozen gberch palette) by CIEDE2000 per role/part, and
``players_suggested.json`` with the operator ``players.json`` ``kit_role``
labels (role accuracy).

``--run`` first runs the stage into a scratch directory that symlinks the
inputs of ``--output`` (read-only against the real dir):

    .venv311/bin/python scripts/eval_appearance.py --output output-shorts \
        --run --config config/clips/gberch.yaml
    # "no hand edits" mode: ignore the clip's appearance.kits, give the stage
    # only match metadata
    ... --run --no-clip-kits --match "Liverpool,Chelsea,2025-09-14"

Exit status is non-zero when any role misses ``--max-de`` or role accuracy is
below ``--min-role-acc``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.utils.kit_palette import delta_e_hex  # noqa: E402
from src.utils.player_names import _kit_roles_from  # noqa: E402

DEFAULT_PALETTE = Path(__file__).resolve().parents[1] / "tests/fixtures/appearance/gberch_hand_palette.json"
_PARTS = ("shirt", "shorts", "socks")
_LINK_DIRS = ("shots", "hmr_world", "refined_poses", "camera", "tracks")


def kit_errors(auto: dict, truth: dict) -> dict[str, dict[str, float]]:
    """``{role: {part: ΔE}}`` for roles present on both sides."""
    out: dict[str, dict[str, float]] = {}
    for role, hand in truth.items():
        got = auto.get(role)
        if not got:
            continue
        out[role] = {p: round(delta_e_hex(got[p], hand[p]), 2) for p in _PARTS if p in got and p in hand}
    return out


def role_accuracy(suggested: dict[str, str], operator: dict[str, str]) -> tuple[int, int, list[str]]:
    wrong = [f"{pid}: suggested {suggested.get(pid)!r} vs operator {role!r}"
             for pid, role in sorted(operator.items()) if suggested.get(pid) != role]
    return len(operator) - len(wrong), len(operator), wrong


def _run_scratch(output: Path, config: dict) -> Path:
    from src.stages.appearance import AppearanceStage

    scratch = Path(tempfile.mkdtemp(prefix="eval_appearance_"))
    for name in _LINK_DIRS:
        if (output / name).exists():
            (scratch / name).symlink_to((output / name).resolve())
    AppearanceStage(config, scratch).run()
    return scratch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    ap.add_argument("--players", type=Path, help="operator players.json (default <output>/players.json)")
    ap.add_argument("--run", action="store_true", help="run the stage into a scratch dir first")
    ap.add_argument("--config", type=Path, help="clip config merged for --run")
    ap.add_argument("--no-clip-kits", action="store_true", help="drop appearance.kits before --run")
    ap.add_argument("--match", help="home,away,date metadata for --run (e.g. 'Liverpool,Chelsea,2025-09-14')")
    ap.add_argument("--max-de", type=float, default=10.0)
    ap.add_argument("--min-role-acc", type=float, default=1.0)
    args = ap.parse_args(argv)

    out = args.output
    if args.run:
        cfg = yaml.safe_load(args.config.read_text()) if args.config else {}
        cfg = dict(cfg or {})
        acfg = dict(cfg.get("appearance") or {})
        if args.no_clip_kits:
            acfg.pop("kits", None)
        if args.match:
            home, away, date = (args.match.split(",") + ["", ""])[:3]
            acfg["match"] = {"home_team": home, "away_team": away, "date": date}
        cfg["appearance"] = acfg
        out = _run_scratch(args.output, cfg)
        print(f"[eval_appearance] scratch run in {out}")

    kits_path = out / "appearance" / "kits.json"
    if not kits_path.exists():
        print(f"[eval_appearance] {kits_path} missing")
        return 2
    kits = json.loads(kits_path.read_text())["kits"]
    truth = json.loads(args.palette.read_text())
    errors = kit_errors(kits, truth)
    failed = False
    for role, parts in errors.items():
        worst = max(parts.values()) if parts else float("inf")
        flag = "ok " if worst < args.max_de else "BAD"
        failed |= worst >= args.max_de
        src = kits[role].get("ref") or kits[role].get("source")
        print(f"  [{flag}] {role:8s} {parts}  via {src}")
    for role in truth:
        if role not in errors:
            print(f"  [BAD] {role:8s} no auto kit")
            failed = True

    op_players = args.players or (args.output / "players.json")
    operator = {}
    if op_players.exists():
        operator = _kit_roles_from(op_players)
    suggested = {pid: v["kit_role"] for pid, v in
                 json.loads((out / "appearance" / "players_suggested.json").read_text()).items()}
    right, total, wrong = role_accuracy(suggested, operator)
    if total:
        acc = right / total
        print(f"  roles: {right}/{total} correct ({acc:.0%})")
        for w in wrong:
            print("    ", w)
        failed |= acc < args.min_role_acc
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

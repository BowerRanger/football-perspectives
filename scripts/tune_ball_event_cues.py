"""Anti-overfit tuning round for the audio/blur cue thresholds and the
fusion policy (2026-09-25).

Protocol: two folds over the four clips.
    Fold A: TUNE on {gberch, origi01},   TEST (held out) on {kroupi01, s013}
    Fold B: TUNE on {kroupi01, s013},    TEST (held out) on {gberch, origi01}

Within a fold, using ONLY the tuning clips:
  1. Grid-search a few audio ``CueCfg`` candidates (band-limit on/off,
     k_mad, min_rise_ratio) against pooled raw-audio-cue F1; keep the
     best.
  2. Grid-search a few blur ``CueCfg`` candidates (angle_change_deg,
     min_streak_px, speed_ratio_threshold) against pooled raw-blur-cue
     F1; keep the best.
  3. Derive per-cue ``CueReliability`` weights from the chosen cfg's
     pooled tuning-fold PRECISION per raw cue (audio/net/blur), sweep the
     weighted-policy threshold to maximize pooled tuning fused F1.
  4. Score all three fusion policies (any2, net_blur_combo, weighted) on
     the tuning fold; the policy with the best tuning F1 is "chosen" for
     this fold.

The frozen result (audio cfg + blur cfg + chosen policy/weights/threshold)
is then applied UNCHANGED to the held-out test clips, and ALL THREE
fusion policies (not just the chosen one) are scored there too -- so a
policy that wins on the tuning fold but collapses on the held-out fold is
visible, not hidden.

The net cue is NOT tuned here (only audio/blur thresholds + fusion
policy, per the task brief) -- its cached ``cues_net.json`` from
``eval_ball_event_cues.py``'s prior full run is reused as-is; run that
script first if it's missing for a clip.

Usage:
    .venv311/bin/python scripts/tune_ball_event_cues.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.ball_cue_audio import compute_audio_cues  # noqa: E402
from src.utils.ball_cue_blur import compute_blur_cues  # noqa: E402
from src.utils.ball_cue_common import (  # noqa: E402
    iter_video_frames,
    load_auto_event_frames,
    load_camera_frames,
    load_manual_events,
    load_real_observations,
    probe_speed_factor,
)
from src.utils.ball_cue_config import CueCfg  # noqa: E402
from src.utils.ball_cue_fusion import (  # noqa: E402
    CueReliability,
    fuse_cues,
    fuse_cues_combo,
    fuse_cues_weighted,
)
from src.utils.ball_hybrid_types import CueEvidence  # noqa: E402
from scripts.eval_ball_event_cues import CLIPS, _match, _prf  # noqa: E402

CACHE_DIR = Path("/Users/joebower/workplace/football-perspectives/output-ball-poc")
TOL = 3
FRAME_TOL = 2

FOLD_A = {"tune": ["gberch", "origi01"], "test": ["kroupi01", "s013"]}
FOLD_B = {"tune": ["kroupi01", "s013"], "test": ["gberch", "origi01"]}

# --- audio candidate cfgs ---
AUDIO_CANDIDATES = {
    "baseline_fullband": dataclasses.replace(CueCfg(), audio_band_limit_enabled=False),
    "band_k3_r04": dataclasses.replace(
        CueCfg(), audio_band_limit_enabled=True, audio_k_mad=3.0, audio_min_rise_ratio=0.4),
    "band_k4_r06": dataclasses.replace(
        CueCfg(), audio_band_limit_enabled=True, audio_k_mad=4.0, audio_min_rise_ratio=0.6),
}

# --- blur candidate cfgs ---
BLUR_CANDIDATES = {
    "baseline_angle25": dataclasses.replace(
        CueCfg(), blur_angle_change_deg=25.0, blur_min_streak_px=4.0,
        blur_speed_ratio_threshold=float("inf")),
    "angle40_streak8_ratio1.6": dataclasses.replace(
        CueCfg(), blur_angle_change_deg=40.0, blur_min_streak_px=8.0,
        blur_speed_ratio_threshold=1.6),
    "angle50_streak10_ratio2.0": dataclasses.replace(
        CueCfg(), blur_angle_change_deg=50.0, blur_min_streak_px=10.0,
        blur_speed_ratio_threshold=2.0),
}


# ---------------------------------------------------------------- loading

def _clip_context(clip_id: str) -> dict:
    output_dir_s, shot_id = CLIPS[clip_id]
    output_dir = Path(output_dir_s)
    cam = load_camera_frames(output_dir, shot_id)
    manual_events = load_manual_events(output_dir, shot_id)
    manual_frames = [f for f, _s in manual_events]
    auto_frames = load_auto_event_frames(output_dir, shot_id)
    auto_missed = [f for f in manual_frames
                   if not any(abs(f - af) <= TOL for af in auto_frames)]
    speed_factor = probe_speed_factor(output_dir, shot_id)
    video_path = output_dir / "shots" / f"{shot_id}.mp4"

    net_path = CACHE_DIR / clip_id / "cues_net.json"
    net_data = json.loads(net_path.read_text())
    net_events = [CueEvidence(**e) for e in net_data["events"]]

    observations = load_real_observations(output_dir, shot_id)
    detections = [(f, uv) for f, uv, _conf, _src in observations]
    needed = {f for f, _uv in detections}
    frame_cache: dict[int, np.ndarray] = {}
    for idx, frame_bgr in iter_video_frames(video_path):
        if idx in needed:
            frame_cache[idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    from src.utils.ball_cue_audio import decode_audio_mono
    samples = None if abs(speed_factor - 1.0) > 1e-6 else decode_audio_mono(video_path)

    return {
        "clip_id": clip_id, "fps": cam.fps, "manual_frames": manual_frames,
        "auto_frames": auto_frames, "auto_missed": auto_missed,
        "speed_factor": speed_factor, "video_path": video_path,
        "net_events": net_events, "detections": detections,
        "frame_cache": frame_cache, "samples": samples,
    }


# ------------------------------------------------------------- cue sweep

def _pooled_score(events_per_clip: dict[str, list[int]], ctxs: dict[str, dict]) -> dict:
    tp = fp = fn = 0
    for clip_id, frames in events_per_clip.items():
        t, f, n = _match(frames, ctxs[clip_id]["manual_frames"], tol=TOL)
        tp += t
        fp += f
        fn += n
    p, r, f1 = _prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def _audio_events_for_clip(ctx: dict, cfg: CueCfg) -> list[CueEvidence]:
    if ctx["samples"] is None:
        return []
    from src.utils.ball_cue_audio import calibrate_latency
    if cfg.audio_band_limit_enabled:
        from src.utils.ball_cue_audio import suppressed_onsets
        onsets = suppressed_onsets(ctx["samples"], 22050, cfg)
    else:
        from src.utils.ball_cue_audio import spectral_flux_onsets
        onsets = spectral_flux_onsets(ctx["samples"], 22050)
    calib = calibrate_latency(
        onsets, ctx["manual_frames"], ctx["fps"],
        search_window_ms=cfg.audio_search_window_ms, min_matches=cfg.audio_min_matches,
        max_std_ms=cfg.audio_max_std_ms)
    if not calib.enabled:
        return []
    events = []
    for o in onsets:
        corrected_t = o.time_s - (calib.latency_ms or 0.0) / 1000.0
        frame = int(round(corrected_t * ctx["fps"]))
        conf = float(np.clip(0.3 + 0.5 * min(1.0, o.strength / 3.0), 0.0, 0.95))
        events.append(CueEvidence(frame=frame, kind="contact", cue="audio_onset",
                                   conf=conf, xyz=None, uv=None))
    return events


def _blur_events_for_clip(ctx: dict, cfg: CueCfg) -> list[CueEvidence]:
    return compute_blur_cues(
        ctx["detections"], ctx["frame_cache"].get, fps=ctx["fps"],
        min_streak_px=cfg.blur_min_streak_px, angle_change_deg=cfg.blur_angle_change_deg,
        max_frame_gap=cfg.blur_max_frame_gap,
        speed_ratio_threshold=cfg.blur_speed_ratio_threshold)


def _select_best(candidates: dict[str, CueCfg], tune_ctxs: dict[str, dict],
                  compute_fn) -> tuple[str, CueCfg, dict]:
    best_name, best_cfg, best_score = None, None, None
    for name, cfg in candidates.items():
        events_per_clip = {cid: [e.frame for e in compute_fn(ctx, cfg)]
                            for cid, ctx in tune_ctxs.items()}
        score = _pooled_score(events_per_clip, tune_ctxs)
        if best_score is None or score["f1"] > best_score["f1"]:
            best_name, best_cfg, best_score = name, cfg, score
    return best_name, best_cfg, best_score


# --------------------------------------------------------- fusion sweep

def _fused_pooled_score(fused_per_clip: dict[str, list[int]], ctxs: dict[str, dict]) -> dict:
    return _pooled_score(fused_per_clip, ctxs)


def _auto_only_score(ctxs: dict[str, dict]) -> dict:
    events_per_clip = {cid: ctx["auto_frames"] for cid, ctx in ctxs.items()}
    return _pooled_score(events_per_clip, ctxs)


def run_fold(fold_name: str, tune_ids: list[str], test_ids: list[str],
             all_ctxs: dict[str, dict], all_evidence_cache: dict) -> dict:
    tune_ctxs = {cid: all_ctxs[cid] for cid in tune_ids}
    test_ctxs = {cid: all_ctxs[cid] for cid in test_ids}

    audio_name, audio_cfg, audio_tune_score = _select_best(
        AUDIO_CANDIDATES, tune_ctxs, _audio_events_for_clip)
    blur_name, blur_cfg, blur_tune_score = _select_best(
        BLUR_CANDIDATES, tune_ctxs, _blur_events_for_clip)

    # Raw-cue generalization check (independent of fusion-policy choice):
    # apply the tuning-fold-chosen audio/blur cfg to the HELD-OUT clips
    # and report pooled P/R/F1 there too.
    audio_test_score = _pooled_score(
        {cid: [e.frame for e in _audio_events_for_clip(all_ctxs[cid], audio_cfg)]
         for cid in test_ids}, test_ctxs)
    blur_test_score = _pooled_score(
        {cid: [e.frame for e in _blur_events_for_clip(all_ctxs[cid], blur_cfg)]
         for cid in test_ids}, test_ctxs)

    chosen_cfg = dataclasses.replace(
        CueCfg(),
        audio_band_limit_enabled=audio_cfg.audio_band_limit_enabled,
        audio_k_mad=audio_cfg.audio_k_mad, audio_min_rise_ratio=audio_cfg.audio_min_rise_ratio,
        blur_angle_change_deg=blur_cfg.blur_angle_change_deg,
        blur_min_streak_px=blur_cfg.blur_min_streak_px,
        blur_speed_ratio_threshold=blur_cfg.blur_speed_ratio_threshold,
    )

    # Compute audio/blur evidence for every clip in this fold (tune+test)
    # under the frozen chosen cfg; net evidence is reused unchanged.
    def evidence_for(cid: str) -> dict[str, list[CueEvidence]]:
        key = (cid, audio_name, blur_name)
        if key not in all_evidence_cache:
            ctx = all_ctxs[cid]
            all_evidence_cache[key] = {
                "audio": _audio_events_for_clip(ctx, chosen_cfg),
                "net": ctx["net_events"],
                "blur": _blur_events_for_clip(ctx, chosen_cfg),
            }
        return all_evidence_cache[key]

    # -- reliability weights from tuning-fold raw-cue precision --
    weights = {}
    for cue_key, cue_name in (("audio", "audio_onset"), ("net", "net_energy"),
                               ("blur", "blur_direction_change")):
        events_per_clip = {cid: [e.frame for e in evidence_for(cid)[cue_key]]
                            for cid in tune_ids}
        score = _pooled_score(events_per_clip, tune_ctxs)
        weights[cue_name] = round(score["precision"], 3)
    auto_weight = 1.0

    # -- sweep weighted-policy threshold on the tuning fold --
    max_score = sum(weights.values()) + auto_weight
    thresholds = sorted({round(x, 3) for x in np.arange(0.05, max_score + 0.05, 0.05)})
    best_threshold, best_weighted_tune_f1 = thresholds[0], -1.0
    for thr in thresholds:
        reliability = CueReliability(weights=weights, auto_weight=auto_weight, threshold=thr)
        fused_per_clip = {
            cid: [f.frame for f in fuse_cues_weighted(
                evidence_for(cid), tune_ctxs[cid]["auto_frames"], reliability,
                frame_tol=FRAME_TOL)]
            for cid in tune_ids
        }
        f1 = _fused_pooled_score(fused_per_clip, tune_ctxs)["f1"]
        if f1 > best_weighted_tune_f1:
            best_threshold, best_weighted_tune_f1 = thr, f1
    reliability = CueReliability(weights=weights, auto_weight=auto_weight,
                                  threshold=best_threshold)

    # -- score all 3 policies on tuning fold, pick the winner --
    def policy_fn(policy: str, cid: str):
        ev = evidence_for(cid)
        auto = all_ctxs[cid]["auto_frames"]
        if policy == "any2":
            return fuse_cues(ev, auto, frame_tol=FRAME_TOL)
        if policy == "net_blur_combo":
            return fuse_cues_combo(ev, auto, frame_tol=FRAME_TOL)
        if policy == "weighted":
            return fuse_cues_weighted(ev, auto, reliability, frame_tol=FRAME_TOL)
        raise ValueError(policy)

    policies = ["any2", "net_blur_combo", "weighted"]
    tune_scores = {}
    for policy in policies:
        fused_per_clip = {cid: [f.frame for f in policy_fn(policy, cid)] for cid in tune_ids}
        tune_scores[policy] = _fused_pooled_score(fused_per_clip, tune_ctxs)
    chosen_policy = max(policies, key=lambda p: tune_scores[p]["f1"])

    # -- held-out: score ALL 3 policies on the test fold, untouched --
    test_scores = {}
    for policy in policies:
        fused_per_clip = {cid: [f.frame for f in policy_fn(policy, cid)] for cid in test_ids}
        test_scores[policy] = _fused_pooled_score(fused_per_clip, test_ctxs)
    auto_only_test = _auto_only_score(test_ctxs)

    return {
        "fold": fold_name, "tune_clips": tune_ids, "test_clips": test_ids,
        "audio_cfg": audio_name, "audio_tune_score": audio_tune_score,
        "audio_test_score": audio_test_score,
        "blur_cfg": blur_name, "blur_tune_score": blur_tune_score,
        "blur_test_score": blur_test_score,
        "reliability_weights": weights, "reliability_threshold": best_threshold,
        "chosen_policy": chosen_policy,
        "tune_scores": tune_scores, "test_scores": test_scores,
        "auto_only_test_score": auto_only_test,
    }


def _print_fold(result: dict) -> None:
    print(f"\n=== fold {result['fold']}: tune={result['tune_clips']} "
          f"test(held-out)={result['test_clips']} ===")
    at, ah = result["audio_tune_score"], result["audio_test_score"]
    bt, bh = result["blur_tune_score"], result["blur_test_score"]
    print(f"  audio cfg: {result['audio_cfg']}")
    print(f"    raw-audio  tune F1={at['f1']:.2f} P={at['precision']:.2f} R={at['recall']:.2f}"
          f"   held-out F1={ah['f1']:.2f} P={ah['precision']:.2f} R={ah['recall']:.2f}")
    print(f"  blur cfg:  {result['blur_cfg']}")
    print(f"    raw-blur   tune F1={bt['f1']:.2f} P={bt['precision']:.2f} R={bt['recall']:.2f}"
          f"   held-out F1={bh['f1']:.2f} P={bh['precision']:.2f} R={bh['recall']:.2f}")
    print(f"  reliability weights: {result['reliability_weights']} "
          f"threshold={result['reliability_threshold']:.2f}")
    print(f"  chosen policy (by tuning F1): {result['chosen_policy']}")
    print(f"  {'policy':16s} {'tune F1':>8s} {'tune P':>7s} {'tune R':>7s}  "
          f"{'test F1':>8s} {'test P':>7s} {'test R':>7s}")
    for policy in ("any2", "net_blur_combo", "weighted"):
        ts, hs = result["tune_scores"][policy], result["test_scores"][policy]
        marker = " *" if policy == result["chosen_policy"] else "  "
        print(f"  {policy:16s} {ts['f1']:8.2f} {ts['precision']:7.2f} {ts['recall']:7.2f}  "
              f"{hs['f1']:8.2f} {hs['precision']:7.2f} {hs['recall']:7.2f}{marker}")
    ao = result["auto_only_test_score"]
    print(f"  auto-only baseline on held-out test clips: "
          f"F1={ao['f1']:.2f} P={ao['precision']:.2f} R={ao['recall']:.2f}")


def main() -> int:
    all_ctxs = {cid: _clip_context(cid) for cid in CLIPS}
    all_evidence_cache: dict = {}

    results = []
    for fold_name, spec in (("A", FOLD_A), ("B", FOLD_B)):
        result = run_fold(fold_name, spec["tune"], spec["test"], all_ctxs, all_evidence_cache)
        results.append(result)
        _print_fold(result)

    out_path = CACHE_DIR / "tune_ball_event_cues_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

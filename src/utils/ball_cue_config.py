"""Frozen tunable configuration for the single-camera event-cue modules
(``ball_cue_audio``, ``ball_cue_blur``, ``ball_cue_fusion``).

``DEFAULT_CUE_CFG`` reflects the EVIDENCE from the 2026-09-25 anti-overfit
tuning round -- two folds, tune on {gberch, origi01} / held-out on
{kroupi01, s013}, and the reverse -- not just whatever won tuning-fold
F1. See ``scripts/tune_ball_event_cues.py`` and the IC-D wiring note for
the full numbers; summary of what changed from the naive
tuning-fold-winner pick and why:

- ``blur_*``: VALIDATED. Both fold directions independently converged on
  the SAME stricter thresholds (angle_change_deg=50, min_streak_px=10,
  speed_ratio_threshold=2.0), and raw-blur pooled F1 stayed in a similar
  0.29-0.34 band on both that cfg's own tuning fold AND its held-out
  fold -- a genuine, cross-validated improvement over the pre-tuning
  defaults (angle_change_deg=25, min_streak_px=4, speed path disabled).
  Adopted as the default.
- ``audio_*``: NOT adopted at the tuning-fold-winning strict setting
  (k_mad=4.0, min_rise_ratio=0.6). That setting scored well on ITS OWN
  tuning fold (pooled F1 0.47) but collapsed to F1=0.00 (zero raw onsets
  survived calibration at all) on the held-out fold -- textbook
  overfitting, most likely because the stricter thresholds starve an
  already-marginal clip's onset count below the >=4-matched-anchors
  calibration floor rather than degrading gracefully. The reverse fold
  couldn't cross-check this at all: its tuning clips (kroupi01, s013)
  have almost no audio signal to learn from (kroupi01 has only 4 total
  contact anchors -- borderline for calibration under ANY setting; s013
  is retimed, audio structurally disabled) so it picked the untouched
  full-band baseline by a tie-break, not by evidence. Net: audio
  band-limiting/crowd-floor/sharpness gating is implemented and unit
  tested (see ``ball_cue_audio.suppressed_onsets``) but this round did
  NOT produce a held-out-validated threshold set, so ``DEFAULT_CUE_CFG``
  keeps ``audio_band_limit_enabled=True`` (band-limiting to the impact
  band is physically motivated on its own) at the MILDER of the two
  band-limited candidates tried (k_mad=3.0, min_rise_ratio=0.4) as a
  conservative middle ground, NOT a validated pick -- re-tune audio
  specifically once more audio-eligible (real-time, >=4-contact) clips
  are available.
- ``fusion_policy``: "any2" (the original IC-D-brief policy), NOT
  "weighted" despite "weighted" winning tuning-fold F1 in BOTH folds --
  it was the WORST or near-worst policy on BOTH held-out folds (fold A
  held-out F1 0.31 vs any2's 0.36 and net_blur_combo's 0.40; fold B
  held-out F1 0.20 vs any2's 0.33 and net_blur_combo's 0.28). This is
  the clearest overfitting signal in the whole round: picking a fusion
  policy by tuning-fold F1 alone would have shipped the worst-generalizing
  option. "any2" was the most consistent across both held-out folds
  (never worst, F1 0.36 and 0.33) though it did not robustly clear
  "held-out F1 >= auto-only F1 + 0.05" in both folds either (see the
  wiring note) -- fusion stays opt-in.

IC-A should map ``ball.hybrid.cues.*`` YAML onto this via
``CueCfg.from_dict(raw_cfg.get("ball", {}).get("hybrid", {}).get("cues", {}))``;
any key a clip's config omits falls back to the field's default here.
Suggested YAML shape (nested exactly as ``from_dict`` expects):

    ball:
      hybrid:
        cues:
          audio:
            freq_lo_hz: 1500.0
            freq_hi_hz: 6000.0
            floor_window_s: 1.0
            k_mad: 3.0
            min_rise_ratio: 0.4
            band_limit_enabled: true
            search_window_ms: 150.0
            min_matches: 4
            max_std_ms: 60.0
          blur:
            angle_change_deg: 50.0
            min_streak_px: 10.0
            speed_ratio_threshold: 2.0
            max_frame_gap: 3
          fusion:
            policy: any2   # any2 | net_blur_combo | weighted
            frame_tol: 2
            weights: {audio_onset: 0.12, net_energy: 0.32, blur_direction_change: 0.16}
            auto_weight: 1.0
            threshold: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CueCfg:
    # -- audio -- (see module docstring: NOT held-out validated at this
    # exact setting; the milder of the two band-limited candidates tried,
    # kept as a conservative default pending a dedicated audio re-tune)
    audio_freq_lo_hz: float = 1500.0
    audio_freq_hi_hz: float = 6000.0
    audio_floor_window_s: float = 1.0
    audio_k_mad: float = 3.0
    audio_min_rise_ratio: float = 0.4
    audio_band_limit_enabled: bool = True
    audio_search_window_ms: float = 150.0
    audio_min_matches: int = 4
    audio_max_std_ms: float = 60.0

    # -- blur -- (cross-fold VALIDATED, see module docstring)
    blur_angle_change_deg: float = 50.0
    blur_min_streak_px: float = 10.0
    # speed_ratio_threshold = float("inf") disables the speed-ratio path
    # (angle-only, the pre-tuning behaviour); a finite value requires
    # EITHER the angle OR the speed-ratio gate to fire.
    blur_speed_ratio_threshold: float = 2.0
    blur_max_frame_gap: int = 3

    # -- fusion -- ("any2" kept as default: "weighted" won tuning-fold F1
    # in both folds but was the worst/near-worst held-out performer in
    # both -- see module docstring. weights/auto_weight/threshold below
    # are illustrative only; a real "weighted" deployment needs its own
    # fresh per-fold tuning pass, not these numbers.)
    fusion_policy: str = "any2"  # "any2" | "net_blur_combo" | "weighted"
    fusion_frame_tol: int = 2
    fusion_weights: dict = field(default_factory=lambda: {
        "audio_onset": 0.12, "net_energy": 0.32, "blur_direction_change": 0.16,
    })
    fusion_auto_weight: float = 1.0
    fusion_threshold: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict | None) -> "CueCfg":
        """Build a ``CueCfg`` from a ``ball.hybrid.cues`` YAML sub-dict
        (see the module docstring for the expected shape). Missing keys
        -- including a missing/empty ``raw`` -- fall back to this
        dataclass's defaults."""
        raw = raw or {}
        audio = raw.get("audio", {}) or {}
        blur = raw.get("blur", {}) or {}
        fusion = raw.get("fusion", {}) or {}
        defaults = cls()
        return cls(
            audio_freq_lo_hz=float(audio.get("freq_lo_hz", defaults.audio_freq_lo_hz)),
            audio_freq_hi_hz=float(audio.get("freq_hi_hz", defaults.audio_freq_hi_hz)),
            audio_floor_window_s=float(
                audio.get("floor_window_s", defaults.audio_floor_window_s)),
            audio_k_mad=float(audio.get("k_mad", defaults.audio_k_mad)),
            audio_min_rise_ratio=float(
                audio.get("min_rise_ratio", defaults.audio_min_rise_ratio)),
            audio_band_limit_enabled=bool(
                audio.get("band_limit_enabled", defaults.audio_band_limit_enabled)),
            audio_search_window_ms=float(
                audio.get("search_window_ms", defaults.audio_search_window_ms)),
            audio_min_matches=int(audio.get("min_matches", defaults.audio_min_matches)),
            audio_max_std_ms=float(audio.get("max_std_ms", defaults.audio_max_std_ms)),
            blur_angle_change_deg=float(
                blur.get("angle_change_deg", defaults.blur_angle_change_deg)),
            blur_min_streak_px=float(
                blur.get("min_streak_px", defaults.blur_min_streak_px)),
            blur_speed_ratio_threshold=float(
                blur.get("speed_ratio_threshold", defaults.blur_speed_ratio_threshold)),
            blur_max_frame_gap=int(blur.get("max_frame_gap", defaults.blur_max_frame_gap)),
            fusion_policy=str(fusion.get("policy", defaults.fusion_policy)),
            fusion_frame_tol=int(fusion.get("frame_tol", defaults.fusion_frame_tol)),
            fusion_weights=dict(fusion.get("weights", defaults.fusion_weights)),
            fusion_auto_weight=float(
                fusion.get("auto_weight", defaults.fusion_auto_weight)),
            fusion_threshold=float(fusion.get("threshold", defaults.fusion_threshold)),
        )


DEFAULT_CUE_CFG = CueCfg()

__all__ = ["CueCfg", "DEFAULT_CUE_CFG"]

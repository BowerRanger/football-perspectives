"""Tests for CueCfg: sane defaults and the ``from_dict`` YAML-mapping
classmethod (missing keys/sections fall back to defaults)."""

from __future__ import annotations

from src.utils.ball_cue_config import CueCfg, DEFAULT_CUE_CFG


def test_default_cfg_matches_module_constant():
    assert DEFAULT_CUE_CFG == CueCfg()


def test_default_cfg_sane_values():
    cfg = CueCfg()
    assert cfg.audio_band_limit_enabled is True
    assert cfg.audio_freq_lo_hz < cfg.audio_freq_hi_hz
    assert cfg.fusion_policy in ("any2", "net_blur_combo", "weighted")
    assert cfg.blur_speed_ratio_threshold > 1.0


def test_from_dict_none_and_empty_fall_back_to_defaults():
    assert CueCfg.from_dict(None) == CueCfg()
    assert CueCfg.from_dict({}) == CueCfg()


def test_from_dict_overrides_only_specified_fields():
    raw = {
        "audio": {"k_mad": 5.0},
        "blur": {"angle_change_deg": 55.0, "speed_ratio_threshold": 2.5},
        "fusion": {"policy": "weighted", "threshold": 3.0},
    }
    cfg = CueCfg.from_dict(raw)
    defaults = CueCfg()

    assert cfg.audio_k_mad == 5.0
    assert cfg.audio_freq_lo_hz == defaults.audio_freq_lo_hz  # unspecified -> default
    assert cfg.audio_min_matches == defaults.audio_min_matches

    assert cfg.blur_angle_change_deg == 55.0
    assert cfg.blur_speed_ratio_threshold == 2.5
    assert cfg.blur_min_streak_px == defaults.blur_min_streak_px

    assert cfg.fusion_policy == "weighted"
    assert cfg.fusion_threshold == 3.0
    assert cfg.fusion_weights == defaults.fusion_weights


def test_from_dict_weights_override_replaces_whole_dict():
    raw = {"fusion": {"weights": {"audio_onset": 0.9}}}
    cfg = CueCfg.from_dict(raw)
    assert cfg.fusion_weights == {"audio_onset": 0.9}

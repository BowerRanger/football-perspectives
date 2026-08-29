"""Decode-equality tests for ``gvhmr_estimator._read_video_frames``.

Task 3 of the runtime-performance pass reimplements ``_read_video_frames``
on top of ``src.utils.video_reader`` (Task 2's sequential-friendly reader)
instead of doing a ``cap.set(CAP_PROP_POS_FRAMES)`` seek per requested frame
per player. A 20-player shot previously re-decoded the clip ~20 times with
a per-frame seek each; the new version reuses the shared reader's
grab()-skip-for-small-gaps / seek-for-large-gaps strategy.

These tests pin the *contract* down against a local copy of the old
per-frame-seek logic (kept here only as a baseline, not in the module) so
the refactor is provably decode-identical:

- return order == the order of the requested ``frame_indices`` (not sorted,
  not deduped)
- frames past EOF / otherwise unreadable come back as a black frame shaped
  like the last successfully decoded frame (or 720x1280x3 if none decoded
  yet)
- dtype/shape of every returned frame
- RuntimeError when the video can't be opened at all

Covers: contiguous ranges, sparse indices (small + large gaps), indices
past EOF (black-frame padding), single-frame requests, and duplicate
indices in the request list.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.gvhmr_estimator import _read_video_frames

FRAME_SIZE = (64, 48)  # (width, height)
N_FRAMES = 50
FPS = 25.0


def _make_synthetic_video(path: Path, n_frames: int = N_FRAMES) -> Path:
    """Writes a video whose frames are each uniquely identifiable (distinct
    solid-ish color per index + burned-in frame number), so decoded frames
    can be compared with np.array_equal without ambiguity."""
    w, h = FRAME_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (w, h))
    assert writer.isOpened(), "failed to open synthetic video writer"
    try:
        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 7) % 256
            frame[:, :, 1] = (i * 17) % 256
            frame[:, :, 2] = (i * 31) % 256
            cv2.putText(
                frame, str(i), (2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def _old_read_video_frames(
    video_path: Path, frame_indices: list[int]
) -> list[np.ndarray]:
    """Reference copy of the pre-Task-3 implementation: cap.set + cap.read
    per requested (deduped, sorted) index. Kept ONLY as a test baseline —
    the module implementation is what's under test, this is not imported
    from it and must not be reintroduced there."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    try:
        wanted = sorted(set(frame_indices))
        cache: dict[int, np.ndarray] = {}
        last_shape: tuple[int, int, int] | None = None
        for fi in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_shape is None:
                    last_shape = (720, 1280, 3)
                cache[fi] = np.zeros(last_shape, dtype=np.uint8)
            else:
                last_shape = frame.shape
                cache[fi] = frame
        return [cache[fi] for fi in frame_indices]
    finally:
        cap.release()


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    return _make_synthetic_video(tmp_path / "fixture.mp4")


def _assert_matches_baseline(video_path: Path, indices: list[int]) -> None:
    expected = _old_read_video_frames(video_path, indices)
    actual = _read_video_frames(video_path, indices)

    assert len(actual) == len(expected) == len(indices)
    for i, (exp, act) in enumerate(zip(expected, actual)):
        assert act.dtype == exp.dtype, i
        assert act.shape == exp.shape, i
        assert np.array_equal(act, exp), i


@pytest.mark.unit
def test_contiguous_range_matches_old_implementation(video_path: Path):
    _assert_matches_baseline(video_path, list(range(5, 25)))


@pytest.mark.unit
def test_sparse_indices_matches_old_implementation(video_path: Path):
    # Includes both small gaps (grab-skip path) and gaps well beyond the
    # reader's default skip threshold (explicit-seek path).
    _assert_matches_baseline(video_path, [0, 1, 2, 4, 45, 46, 49])


@pytest.mark.unit
def test_single_frame_request_matches_old_implementation(video_path: Path):
    _assert_matches_baseline(video_path, [17])
    _assert_matches_baseline(video_path, [0])
    _assert_matches_baseline(video_path, [N_FRAMES - 1])


@pytest.mark.unit
def test_indices_past_eof_padded_black_matches_old_implementation(
    video_path: Path,
):
    # Mix of in-range and past-EOF indices: the past-EOF ones must come
    # back as black frames shaped like the last successfully decoded
    # frame, in both implementations.
    indices = [N_FRAMES - 3, N_FRAMES - 1, N_FRAMES, N_FRAMES + 25]
    _assert_matches_baseline(video_path, indices)

    # Direct black-frame assertion, not just "matches old" (guards against
    # both implementations being wrong the same way).
    actual = _read_video_frames(video_path, indices)
    last_real = actual[1]  # N_FRAMES - 1, in range
    for pad in (actual[2], actual[3]):  # N_FRAMES, N_FRAMES + 25
        assert pad.shape == last_real.shape
        assert np.array_equal(pad, np.zeros_like(pad))


@pytest.mark.unit
def test_all_indices_past_eof_uses_default_shape(video_path: Path):
    # No frame ever decodes successfully -> falls back to the hardcoded
    # (720, 1280, 3) default shape rather than crashing on "no last_shape".
    indices = [N_FRAMES + 5, N_FRAMES + 10]
    _assert_matches_baseline(video_path, indices)

    actual = _read_video_frames(video_path, indices)
    for frame in actual:
        assert frame.shape == (720, 1280, 3)
        assert np.array_equal(frame, np.zeros((720, 1280, 3), dtype=np.uint8))


@pytest.mark.unit
def test_duplicate_and_out_of_order_indices_matches_old_implementation(
    video_path: Path,
):
    _assert_matches_baseline(video_path, [10, 3, 10, 3, 20, 3])


@pytest.mark.unit
def test_return_order_follows_request_order_not_sorted(video_path: Path):
    indices = [30, 5, 15]
    actual = _read_video_frames(video_path, indices)
    # Frame 30 should NOT equal frame 5's or frame 15's content, and the
    # returned list must line up positionally with `indices`, not sorted
    # order -- burned-in frame numbers make this checkable directly.
    naive = _old_read_video_frames(video_path, indices)
    for act, exp in zip(actual, naive):
        assert np.array_equal(act, exp)


@pytest.mark.unit
def test_raises_on_unopenable_video(tmp_path: Path):
    bogus = tmp_path / "does_not_exist.mp4"
    with pytest.raises(RuntimeError):
        _read_video_frames(bogus, [0, 1])

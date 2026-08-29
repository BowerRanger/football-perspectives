"""Tests for src.utils.video_reader: sequential video-frame reading that
replaces the per-frame ``cap.set(CAP_PROP_POS_FRAMES)`` seek pattern.

Correctness is proven frame-for-frame against the naive baseline (seek +
read for every requested index) on a synthetic fixture video generated
in-process. A handful of tests additionally assert the *mechanism*
(fewer ``set()`` calls than a naive loop) via a counting capture wrapper,
since the whole point of this module is avoiding per-frame seeks.

The synthetic fixture above is written with ``cv2.VideoWriter`` + the
``mp4v`` fourcc, which produces an all-intra stream with none of a real
broadcast clip's B-frame reordering or ~1s GOP structure. Real-codec
coverage (``TestRealH264Fixture`` below) closes that gap: it encodes a
genuine ``libx264`` stream (B-frames, GOP=29, matching the camera stage's
actual origi01/gberch/origi02 test clips) and checks the same
seek-vs-grab-skip equivalence, including the exact non-monotonic
down-then-up neighbour-sweep access pattern the camera stage's
propagate-coverage loop uses. This was written to investigate a real,
deterministic origi01 camera-track divergence between the naive
per-frame-seek code this module replaced and the grab-skip
``VideoFrameReader`` it was replaced with; the investigation (direct
decode-content probing on origi01's actual shot video, and on this
fixture, across thousands of forward-gap and non-monotonic access
patterns) found zero content mismatches in either direction -- i.e. no
evidence that CAP_PROP_POS_FRAMES seeking is imprecise on real H.264
streams with B-frames for this codebase's access patterns. The origi01
divergence's root cause was not resolved by that probing; see
video_reader.py's module docstring for the current, empirically-qualified
correctness claim.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.video_reader import VideoFrameReader, read_frames

FRAME_SIZE = (64, 48)  # (width, height)
N_FRAMES = 80
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
            frame[:, :, 0] = (i * 5) % 256
            frame[:, :, 1] = (i * 13) % 256
            frame[:, :, 2] = (i * 29) % 256
            cv2.putText(
                frame, str(i), (2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def _naive_read(path: Path, indices) -> dict[int, np.ndarray]:
    """The pattern being replaced: one seek + read per requested index."""
    cap = cv2.VideoCapture(str(path))
    out: dict[int, np.ndarray] = {}
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                out[idx] = frame
    finally:
        cap.release()
    return out


class _CountingCap:
    """Wraps a real cv2.VideoCapture and counts set()/grab()/read() calls,
    so tests can assert the seek-avoidance mechanism, not just the output.
    Duck-types the same surface VideoFrameReader expects (read/set/grab),
    so it is accepted as a "capture-like" object rather than a path."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self.set_calls = 0
        self.grab_calls = 0
        self.read_calls = 0

    def set(self, prop, value):
        self.set_calls += 1
        return self._cap.set(prop, value)

    def get(self, prop):
        return self._cap.get(prop)

    def read(self):
        self.read_calls += 1
        return self._cap.read()

    def grab(self):
        self.grab_calls += 1
        return self._cap.grab()

    def release(self):
        self._cap.release()


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    return _make_synthetic_video(tmp_path / "fixture.mp4")


# --------------------------------------------------------------------------
# read_frames: batch API, correctness vs. the naive per-frame-seek baseline
# --------------------------------------------------------------------------


def test_read_frames_contiguous_matches_naive(video_path: Path):
    indices = list(range(10, 30))
    expected = _naive_read(video_path, indices)
    actual = read_frames(video_path, indices)

    assert set(actual) == set(expected)
    for idx in indices:
        assert np.array_equal(actual[idx], expected[idx]), idx


def test_read_frames_sparse_matches_naive(video_path: Path):
    # Includes gaps both smaller and (well) larger than the default
    # skip-threshold, so both the grab-skip and explicit-seek paths run.
    indices = [0, 1, 2, 5, 40, 41, 75]
    expected = _naive_read(video_path, indices)
    actual = read_frames(video_path, indices)

    assert set(actual) == set(expected)
    for idx in indices:
        assert np.array_equal(actual[idx], expected[idx]), idx


def test_read_frames_duplicate_indices_deduped(video_path: Path):
    indices = [5, 5, 10, 10, 10, 5, 20]
    expected = _naive_read(video_path, sorted(set(indices)))
    actual = read_frames(video_path, indices)

    assert set(actual) == set(expected)
    for idx in set(indices):
        assert np.array_equal(actual[idx], expected[idx]), idx


def test_read_frames_out_of_order_input_matches_naive(video_path: Path):
    shuffled = [30, 5, 70, 0, 20, 5, 65, 12]
    expected = _naive_read(video_path, sorted(set(shuffled)))
    actual = read_frames(video_path, shuffled)

    assert set(actual) == set(expected)
    for idx in shuffled:
        assert np.array_equal(actual[idx], expected[idx]), idx


def test_read_frames_past_eof_returns_no_entry(video_path: Path):
    indices = [N_FRAMES - 2, N_FRAMES - 1, N_FRAMES, N_FRAMES + 10]
    expected = _naive_read(video_path, indices)
    actual = read_frames(video_path, indices)

    # The naive baseline itself should show the past-EOF indices missing;
    # this pins down the "matches naive" contract for EOF handling too.
    assert N_FRAMES not in expected
    assert N_FRAMES + 10 not in expected
    assert set(actual) == set(expected)
    for idx in expected:
        assert np.array_equal(actual[idx], expected[idx]), idx


def test_read_frames_empty_indices_returns_empty_dict(video_path: Path):
    assert read_frames(video_path, []) == {}


def test_read_frames_reuses_passed_capture_without_releasing(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        out = read_frames(cap, [0, 1, 2])
        assert set(out) == {0, 1, 2}
        # Caller-owned capture must still be usable afterwards.
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, _ = cap.read()
        assert ok
    finally:
        cap.release()


# --------------------------------------------------------------------------
# VideoFrameReader: incremental/cached form, and the seek-avoidance mechanism
# --------------------------------------------------------------------------


def test_video_frame_reader_matches_naive_for_random_access_pattern(
    video_path: Path,
):
    # Mimics the down-then-up neighbour sweep in the camera stage's
    # propagate-coverage loop: repeated, non-monotonic, overlapping access.
    pattern = [10, 11, 12, 11, 13, 12, 14, 5, 6, 7, 6, 60, 61, 60, 0, 79]
    expected = _naive_read(video_path, pattern)

    cap = cv2.VideoCapture(str(video_path))
    try:
        reader = VideoFrameReader(cap)
        for idx in pattern:
            got = reader.read(idx)
            exp = expected.get(idx)
            if exp is None:
                assert got is None, idx
            else:
                assert got is not None and np.array_equal(got, exp), idx
    finally:
        cap.release()


def test_video_frame_reader_caches_repeat_reads(video_path: Path):
    raw_cap = cv2.VideoCapture(str(video_path))
    try:
        cap = _CountingCap(raw_cap)
        reader = VideoFrameReader(cap)
        first = reader.read(15)
        reads_after_first = cap.read_calls
        second = reader.read(15)
        assert cap.read_calls == reads_after_first, (
            "repeat read of a cached index must not touch the capture again"
        )
        assert first is second or np.array_equal(first, second)
    finally:
        raw_cap.release()


def test_video_frame_reader_sequential_reads_use_grab_not_seek(video_path: Path):
    raw_cap = cv2.VideoCapture(str(video_path))
    try:
        cap = _CountingCap(raw_cap)
        reader = VideoFrameReader(cap)
        for idx in range(N_FRAMES):
            reader.read(idx)
        # One seek to establish position, then pure sequential decode —
        # far fewer than one set() per frame.
        assert cap.set_calls <= 2, cap.set_calls
        assert cap.read_calls == N_FRAMES
    finally:
        raw_cap.release()


def test_video_frame_reader_large_gap_forces_seek(video_path: Path):
    raw_cap = cv2.VideoCapture(str(video_path))
    try:
        cap = _CountingCap(raw_cap)
        reader = VideoFrameReader(cap, skip_threshold=4)
        reader.read(0)
        calls_after_first = cap.set_calls
        # Gap of 20 frames, well beyond the threshold of 4 -> must reseek
        # rather than grab-skip 19 frames.
        reader.read(20)
        assert cap.set_calls == calls_after_first + 1
    finally:
        raw_cap.release()


def test_video_frame_reader_small_gap_uses_grab_skip(video_path: Path):
    raw_cap = cv2.VideoCapture(str(video_path))
    try:
        cap = _CountingCap(raw_cap)
        reader = VideoFrameReader(cap, skip_threshold=10)
        reader.read(0)
        calls_after_first = cap.set_calls
        reader.read(5)  # gap of 4, within threshold
        assert cap.set_calls == calls_after_first
        assert cap.grab_calls >= 4
    finally:
        raw_cap.release()


def test_video_frame_reader_backward_jump_forces_seek(video_path: Path):
    raw_cap = cv2.VideoCapture(str(video_path))
    try:
        cap = _CountingCap(raw_cap)
        reader = VideoFrameReader(cap)
        reader.read(20)
        calls_after_first = cap.set_calls
        reader.read(5)
        assert cap.set_calls == calls_after_first + 1
    finally:
        raw_cap.release()


def test_video_frame_reader_past_eof_returns_none(video_path: Path):
    raw_cap = cv2.VideoCapture(str(video_path))
    try:
        reader = VideoFrameReader(raw_cap)
        assert reader.read(N_FRAMES + 50) is None
        # Reader must recover cleanly for a subsequent in-range read.
        got = reader.read(0)
        expected = _naive_read(video_path, [0])[0]
        assert np.array_equal(got, expected)
    finally:
        raw_cap.release()


def test_video_frame_reader_as_context_manager_opens_and_closes_path(
    video_path: Path,
):
    with VideoFrameReader(video_path) as reader:
        frame = reader.read(3)
        expected = _naive_read(video_path, [3])[3]
        assert np.array_equal(frame, expected)


def test_video_frame_reader_survives_external_capture_movement(
    video_path: Path,
):
    # The camera stage shares one capture between several readers and a few
    # legacy per-frame ``cap.set``+``cap.read`` sites. A reader whose grab-skip
    # gap were counted from private bookkeeping alone would, after someone
    # else moved the decoder, silently return frames from the wrong index
    # (the origi01 camera-track divergence). The gap must therefore follow
    # the capture's actual reported position.
    raw_cap = cv2.VideoCapture(str(video_path))
    try:
        reader = VideoFrameReader(raw_cap)
        reader.read(5)
        # External consumer moves the shared decoder (legacy seek pattern).
        raw_cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
        raw_cap.read()
        # A small forward gap from the reader's view (8 - 5 = 3 <= threshold):
        # stale bookkeeping would grab-skip from frame 41 and hand back ~43.
        got = reader.read(8)
        expected = _naive_read(video_path, [8])[8]
        assert np.array_equal(got, expected)
        # And interleave the other direction: reader ahead of the capture.
        raw_cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
        raw_cap.read()
        got = reader.read(12)
        expected = _naive_read(video_path, [12])[12]
        assert np.array_equal(got, expected)
    finally:
        raw_cap.release()


def test_two_readers_sharing_one_capture_both_stay_correct(video_path: Path):
    # Two persistent readers over one capture, reads interleaved — the
    # camera stage's actual topology (_prop/_bd/_lens/_resolve/_relock
    # readers all wrap the same cap).
    raw_cap = cv2.VideoCapture(str(video_path))
    try:
        a = VideoFrameReader(raw_cap)
        b = VideoFrameReader(raw_cap)
        expected = _naive_read(video_path, [3, 6, 9, 30, 33, 36])
        for idx_a, idx_b in ((3, 30), (6, 33), (9, 36)):
            assert np.array_equal(a.read(idx_a), expected[idx_a])
            assert np.array_equal(b.read(idx_b), expected[idx_b])
    finally:
        raw_cap.release()


# --------------------------------------------------------------------------
# Real-codec (libx264, B-frames, ~1s GOP) fixture -- see module docstring.
# --------------------------------------------------------------------------

H264_N_FRAMES = 80
H264_FPS = 25
H264_GOP = 29  # matches origi01/origi02/gberch's actual GOP size


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_h264_video(path: Path, n_frames: int = H264_N_FRAMES) -> Path:
    """Encodes a genuine libx264 stream (B-frames, ``H264_GOP``-frame GOP,
    yuv420p -- the same profile ffprobe reports for the camera stage's real
    test clips) by piping uniquely-identifiable raw BGR frames (same
    per-index color + burned-in frame number scheme as
    ``_make_synthetic_video`` above) into ffmpeg. Unlike ``cv2.VideoWriter``
    with ``mp4v``, this produces the B-frame reordering and multi-frame GOP
    structure that makes ``CAP_PROP_POS_FRAMES`` seeking nontrivial."""
    w, h = FRAME_SIZE
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", str(H264_FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-g", str(H264_GOP), "-bf", "2", "-x264-params", "scenecut=0",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 5) % 256
        frame[:, :, 1] = (i * 13) % 256
        frame[:, :, 2] = (i * 29) % 256
        cv2.putText(
            frame, str(i), (2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (255, 255, 255), 1,
        )
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    assert ret == 0, "ffmpeg failed to encode the H.264 fixture"
    return path


@pytest.fixture(scope="module")
def h264_video_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not on PATH -- required for the real-codec fixture")
    path = tmp_path_factory.mktemp("h264_fixture") / "real.mp4"
    return _make_h264_video(path)


def _ground_truth_decode(path: Path) -> dict[int, np.ndarray]:
    """Continuous decode from frame 0, no seeks at all -- cannot possibly
    land on the wrong frame, so this is the reference every other read
    strategy is checked against (in addition to the naive-seek baseline,
    which is itself the pattern the old per-frame-seek camera.py code
    used)."""
    cap = cv2.VideoCapture(str(path))
    out: dict[int, np.ndarray] = {}
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out[i] = frame
            i += 1
    finally:
        cap.release()
    return out


class TestRealH264Fixture:
    """VideoFrameReader / read_frames correctness on a real H.264 stream
    with B-frames, checked against both the naive per-frame-seek baseline
    (what camera.py did before the seek-avoidance refactor) and a
    continuous no-seek decode (ground truth)."""

    def test_contiguous_matches_naive_and_ground_truth(
        self, h264_video_path: Path
    ):
        indices = list(range(5, 40))
        truth = _ground_truth_decode(h264_video_path)
        expected = _naive_read(h264_video_path, indices)
        actual = read_frames(h264_video_path, indices)
        for idx in indices:
            assert np.array_equal(actual[idx], truth[idx]), idx
            assert np.array_equal(expected[idx], truth[idx]), idx

    def test_sparse_forward_gaps_match_naive_and_ground_truth(
        self, h264_video_path: Path
    ):
        # Gaps spanning well under and well over the default skip
        # threshold (32), and crossing GOP (29-frame) boundaries, so both
        # the grab-skip and explicit-seek paths run against real B-frame
        # content.
        indices = [0, 1, 2, 5, 10, 20, 28, 29, 30, 40, 41, 55, 60, 75, 79]
        truth = _ground_truth_decode(h264_video_path)
        expected = _naive_read(h264_video_path, indices)
        actual = read_frames(h264_video_path, indices)
        for idx in indices:
            assert np.array_equal(actual[idx], truth[idx]), idx
            assert np.array_equal(expected[idx], truth[idx]), idx

    def test_nonmonotonic_bounce_matches_naive_and_ground_truth(
        self, h264_video_path: Path
    ):
        """Replicates the camera stage's propagate-coverage access pattern
        (``_run_propagation`` in src/stages/camera.py): a full high->low
        sweep, then a full low->high sweep, repeated, with occasional
        multi-frame bridge jumps -- on a real H.264/B-frame/GOP stream."""
        truth = _ground_truth_decode(h264_video_path)
        pattern: list[int] = []
        for _round in range(3):
            pattern.extend(range(H264_N_FRAMES - 1, 0, -3))
            pattern.extend(range(0, H264_N_FRAMES - 1, 2))
            pattern.extend(range(H264_N_FRAMES - 1, 0, -5))
            pattern.extend(range(0, H264_N_FRAMES - 1, 4))
        expected = _naive_read(h264_video_path, pattern)

        cap = cv2.VideoCapture(str(h264_video_path))
        try:
            reader = VideoFrameReader(cap)
            for idx in pattern:
                got = reader.read(idx)
                assert got is not None and np.array_equal(got, truth[idx]), idx
                assert np.array_equal(expected[idx], truth[idx]), idx
        finally:
            cap.release()

    def test_forward_gap_grab_skip_matches_ground_truth_across_cursor_range(
        self, h264_video_path: Path
    ):
        """Directly exercises VideoFrameReader's grab()-skip branch (not
        just its explicit-seek branch) at every gap from 1 to just past the
        default skip threshold, from cursor positions spanning the whole
        clip -- the specific case ('does N grab()s from a real mid-GOP
        position land on the same frame a fresh seek would') that a
        same-index full sequential sweep can't distinguish from a
        trivial gap=0 walk."""
        truth = _ground_truth_decode(h264_video_path)
        cap = cv2.VideoCapture(str(h264_video_path))
        try:
            for c in range(0, H264_N_FRAMES - 33, 3):
                for g in range(1, 33):
                    target = c + g
                    reader = VideoFrameReader(cap)
                    reader.read(c)  # establish cursor at c via a fresh seek
                    got = reader.read(target)
                    assert got is not None
                    assert np.array_equal(got, truth[target]), (c, g, target)
        finally:
            cap.release()

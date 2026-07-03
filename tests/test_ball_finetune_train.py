"""Fine-tune harness: XML parsing, run building, dataset parity, loss."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.utils.ball_finetune_train import (
    FinetuneDataset,
    build_runs,
    parse_labels_xml,
    wbce_loss,
)
from src.utils.ball_weak_labels import labels_to_cvat_xml
from src.utils.wasb_ball_detector import _get_affine_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import finetune_wasb  # noqa: E402


def _mini_corpus(tmp_path: Path, labels: dict[int, tuple[float, float]],
                 n_frames: int = 8, size=(64, 48)) -> Path:
    corpus = tmp_path / "corpus"
    fdir = corpus / "frames" / "clipA"
    fdir.mkdir(parents=True)
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
        if i in labels:
            u, v = labels[i]
            cv2.circle(img, (int(u), int(v)), 2, (255, 255, 255), -1)
        cv2.imwrite(str(fdir / f"{i:05d}.png"), img)
    (corpus / "annos").mkdir(parents=True)
    (corpus / "annos" / "clipA.xml").write_text(
        labels_to_cvat_xml("clipA", labels))
    return corpus


def _mini_corpus_partial_frames(
    tmp_path: Path, labels: dict[int, tuple[float, float]],
    n_frames: int, size=(64, 48),
) -> Path:
    """Like ``_mini_corpus`` but only extracts frames ``0..n_frames-1``,
    even though ``labels`` may reference frames beyond that (mirrors the
    real-corpus decoder off-by-one: labels can reference frames the
    extractor never produced)."""
    corpus = tmp_path / "corpus"
    fdir = corpus / "frames" / "clipA"
    fdir.mkdir(parents=True)
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
        if i in labels:
            u, v = labels[i]
            cv2.circle(img, (int(u), int(v)), 2, (255, 255, 255), -1)
        cv2.imwrite(str(fdir / f"{i:05d}.png"), img)
    (corpus / "annos").mkdir(parents=True)
    (corpus / "annos" / "clipA.xml").write_text(
        labels_to_cvat_xml("clipA", labels))
    return corpus


def test_dataset_drops_samples_referencing_missing_frames(tmp_path: Path):
    # Labels at 2..6 (consecutive) but only frames 0..4 are on disk, i.e.
    # the run [4,5,6] references frames 5 and 6 which were never extracted
    # (decoder-off-by-one) while [2,3,4] and [3,4,5]... only [2,3,4] is
    # fully on disk since frame 5 is missing.
    labels = {
        2: (10.0, 10.0), 3: (11.0, 10.0), 4: (12.0, 10.0),
        5: (13.0, 10.0), 6: (14.0, 10.0),
    }
    corpus = _mini_corpus_partial_frames(tmp_path, labels, n_frames=5)
    ds = FinetuneDataset(corpus, ["clipA"], input_size=(128, 72))
    # build_runs(labels) -> [[2,3,4],[3,4,5],[4,5,6]]; only [2,3,4] has all
    # three frames (0..4) present on disk.
    assert len(ds) == 1
    # Must not raise despite the missing-frame runs having been dropped.
    x, y = ds[0]
    assert x.shape[0] == 9 and y.shape[0] == 3


def test_parse_labels_roundtrip(tmp_path: Path):
    labels = {3: (10.0, 20.0), 4: (11.0, 21.0)}
    corpus = _mini_corpus(tmp_path, labels)
    parsed = parse_labels_xml(corpus / "annos" / "clipA.xml")
    assert parsed == {3: (10.0, 20.0), 4: (11.0, 21.0)}


def test_build_runs_consecutive_only():
    labels = {4: (0, 0), 5: (0, 0), 6: (0, 0), 7: (0, 0), 20: (0, 0)}
    assert build_runs(labels) == [[4, 5, 6], [5, 6, 7]]
    assert build_runs({1: (0, 0), 3: (0, 0)}) == []


def test_dataset_shapes_and_label_mapping(tmp_path: Path):
    labels = {2: (40.0, 24.0), 3: (41.0, 24.0), 4: (42.0, 24.0)}
    corpus = _mini_corpus(tmp_path, labels)
    ds = FinetuneDataset(corpus, ["clipA"], input_size=(128, 72), sigma=2.5)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (9, 72, 128) and x.dtype == torch.float32
    assert y.shape == (3, 72, 128) and y.dtype == torch.float32
    # The target peak must sit where the SAME forward affine maps the label.
    trans = _get_affine_transform((64 / 2, 48 / 2), 64.0, (128, 72), inv=False)
    lbl = np.array([40.0, 24.0, 1.0])
    exp = trans @ lbl
    peak = np.unravel_index(int(torch.argmax(y[0])), y[0].shape)
    assert abs(peak[1] - exp[0]) <= 3 and abs(peak[0] - exp[1]) <= 3
    assert float(y.max()) == 1.0 and float(y.min()) == 0.0


def test_wbce_loss_decreases_toward_target():
    torch.manual_seed(0)
    target = torch.zeros(1, 3, 8, 8)
    target[0, :, 4, 4] = 1.0
    good = torch.full((1, 3, 8, 8), -6.0)
    good[0, :, 4, 4] = 6.0
    bad = torch.full((1, 3, 8, 8), 6.0)
    bad[0, :, 4, 4] = -6.0
    assert wbce_loss(good, target) < wbce_loss(bad, target)
    assert wbce_loss(good, target).item() >= 0.0


def _mini_finetune_corpus(tmp_path: Path) -> Path:
    """A tiny corpus with a ``manifest.json``: one ``train`` clip and one
    ``holdout`` clip (mirrors ``_mini_corpus`` but builds two distinctly
    named clips, since the CLI splits clips by manifest membership)."""
    corpus = tmp_path / "corpus"
    labels = {2: (30.0, 20.0), 3: (31.0, 20.0), 4: (32.0, 20.0)}

    for clip_id in ("train_clip", "holdout_clip"):
        fdir = corpus / "frames" / clip_id
        fdir.mkdir(parents=True)
        for i in range(8):
            img = np.full((48, 64, 3), 30, dtype=np.uint8)
            if i in labels:
                u, v = labels[i]
                cv2.circle(img, (int(u), int(v)), 2, (255, 255, 255), -1)
            cv2.imwrite(str(fdir / f"{i:05d}.png"), img)
        (corpus / "annos").mkdir(parents=True, exist_ok=True)
        (corpus / "annos" / f"{clip_id}.xml").write_text(
            labels_to_cvat_xml(clip_id, labels))

    (corpus / "manifest.json").write_text(json.dumps({
        "train": ["train_clip"],
        "holdout": ["holdout_clip"],
    }))
    return corpus


class _TinyStandInModel(torch.nn.Module):
    """WASB-shaped (9ch in, 3ch out) stand-in — avoids needing the real
    HRNet checkpoint / vendored weights just to exercise the CLI's save
    and history-writing behaviour."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_cli_saves_last_and_best_and_records_best_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A 2-epoch run must leave both best.pth.tar and last.pth.tar on
    disk, and history.json must record which epoch was best — this is
    the harness-gap fix: a small noisy holdout picking an early epoch as
    "best" must not lose the later epoch's weights, and a killed run
    must still have progress on disk (exercised here via the per-epoch
    write, not an actual kill)."""
    corpus = _mini_finetune_corpus(tmp_path)
    run_dir = tmp_path / "run"

    stand_in = _TinyStandInModel()
    monkeypatch.setattr(
        finetune_wasb, "load_wasb_model",
        lambda init, device: (stand_in, "cpu"),
    )

    rc = finetune_wasb.main([
        "--corpus-root", str(corpus),
        "--run-dir", str(run_dir),
        "--epochs", "2",
        "--batch", "1",
        "--val-frac", "0",
        "--device", "cpu",
    ])
    assert rc == 0

    assert (run_dir / "best.pth.tar").exists()
    assert (run_dir / "last.pth.tar").exists()

    history = json.loads((run_dir / "history.json").read_text())
    assert "best_epoch" in history
    assert len(history["epochs"]) == 2
    if history["best_metric"] > -1.0:
        assert history["best_epoch"] in (0, 1)


def test_one_training_step_runs_on_cpu(tmp_path: Path):
    labels = {2: (30.0, 20.0), 3: (31.0, 20.0), 4: (32.0, 20.0)}
    corpus = _mini_corpus(tmp_path, labels)
    ds = FinetuneDataset(corpus, ["clipA"], input_size=(128, 72))
    x, y = ds[0]
    # Tiny stand-in model with the WASB io contract (9ch in, 3ch out).
    model = torch.nn.Conv2d(9, 3, kernel_size=3, padding=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = wbce_loss(model(x.unsqueeze(0)), y.unsqueeze(0))
    for _ in range(20):
        opt.zero_grad()
        loss = wbce_loss(model(x.unsqueeze(0)), y.unsqueeze(0))
        loss.backward()
        opt.step()
    after = wbce_loss(model(x.unsqueeze(0)), y.unsqueeze(0))
    assert after < before

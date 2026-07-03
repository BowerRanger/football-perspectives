"""Fine-tune harness: XML parsing, run building, dataset parity, loss."""

from __future__ import annotations

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

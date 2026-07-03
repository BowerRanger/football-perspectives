"""Repo-side WASB fine-tune harness (spec §4.3 step 3).

The vendored WASB trainer (``third_party/wasb_sbdt/src/train.py``) is
hard-coded to CUDA and expects a Hydra-style config tree, so — mirroring
``src.utils.wasb_ball_detector`` — this module bypasses it entirely: a
plain :class:`torch.utils.data.Dataset` reads the Task-2 corpus layout
(``frames/{clip}/{fid:05d}.png`` + ``annos/{clip}.xml`` +
``manifest.json``) and preprocesses frames with the EXACT inference
pipeline (``_get_affine_transform`` + ImageNet normalisation from
``src.utils.wasb_ball_detector``), so a fine-tuned checkpoint can never
silently diverge from what the detector will feed it at inference time.

Targets are the same binary fixed-size heatmaps WASB trained on
(sigma 2.5), generated via the vendored ``gen_binary_map`` — imported
directly rather than reimplemented, per house rule (never edit or fork
vendored source; import pure pieces only).
"""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.wasb_ball_detector import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _get_affine_transform,
    _WASB_SRC,
)


def _load_vendored_heatmap_module():
    """Import ``third_party/wasb_sbdt/src/utils/heatmap.py`` directly.

    ``wasb_ball_detector.py`` imports vendored submodules via a
    ``sys.path`` insert + dotted import (e.g. ``from models.hrnet import
    HRNet``). That mechanic is unsafe for THIS module specifically:
    ``utils`` is a painfully generic package name — this project also has
    its own top-level ``src.utils`` package, and a sibling module
    (``src/utils/pnlcalib_pitch_map.py``) does its own temporary
    ``sys.path`` swap to import a *different* vendored ``utils`` package
    (PnLCalib's). Registering WASB's ``utils`` in ``sys.modules['utils']``
    would leak across those imports depending on import order.

    So: load ``heatmap.py`` directly from its file path via
    ``importlib``, under a private module name, without ever touching
    ``sys.path`` or ``sys.modules['utils']``. ``heatmap.py``'s own
    imports (``PIL``, ``numpy``, ``cv2``) are standalone — it doesn't
    import anything else from the vendored ``utils`` package.
    """
    heatmap_path = _WASB_SRC / "utils" / "heatmap.py"
    spec = importlib.util.spec_from_file_location(
        "_wasb_vendored_heatmap", heatmap_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen_binary_map = _load_vendored_heatmap_module().gen_binary_map

_SIGMA = 2.5


def parse_labels_xml(path: Path) -> dict[int, tuple[float, float]]:
    """Parse the CVAT-dialect label XML into ``{frame: (x, y)}``.

    Mirrors ``third_party/wasb_sbdt/src/datasets/soccer.py``'s
    ``load_xml``: only ``<points>`` with ``outside=="0"`` and a child
    ``<attribute name="used_in_game">1</attribute>`` count as labels.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    labels: dict[int, tuple[float, float]] = {}
    for track in root:
        if track.tag != "track":
            continue
        for points in track:
            if points.tag != "points":
                continue
            if points.attrib.get("outside") != "0":
                continue
            used_in_game = None
            for attr in points:
                if attr.attrib.get("name") == "used_in_game":
                    used_in_game = attr.text
            if used_in_game != "1":
                continue
            frame = int(points.attrib["frame"])
            x_str, y_str = points.attrib["points"].split(",")
            labels[frame] = (float(x_str), float(y_str))
    return labels


def build_runs(
    labels: Mapping[int, tuple[float, float]], frames_in: int = 3,
) -> list[list[int]]:
    """All windows of ``frames_in`` CONSECUTIVE labelled frames (stride 1).

    E.g. labels ``{4,5,6,7}`` -> ``[[4,5,6],[5,6,7]]``; sparse labels
    (no run of ``frames_in`` consecutive integers) yield ``[]``.
    """
    frames = sorted(labels)
    runs: list[list[int]] = []
    for i in range(len(frames) - frames_in + 1):
        window = frames[i:i + frames_in]
        if window[-1] - window[0] == frames_in - 1:
            runs.append(window)
    return runs


class FinetuneDataset(Dataset):
    """Torch dataset over the fine-tune corpus with inference-parity preprocessing.

    Each sample is a run of ``frames_in`` consecutive labelled frames from
    one clip. ``__getitem__`` returns ``(x, y)``:

    - ``x``: ``float32 (3 * frames_in, inp_h, inp_w)`` — the same
      letterbox-affine + ImageNet-normalise + transpose + concat pipeline
      as ``WASBBallDetector._preprocess_buffer``, applied per frame using
      that frame's own ``(w, h)`` to derive ``center``/``scale``.
    - ``y``: ``float32 (frames_in, inp_h, inp_w)`` — binary sigma-2.5
      heatmaps, one per frame, with the label pixel mapped through the
      SAME forward affine used for that frame's image warp. A label that
      maps outside the model canvas produces an all-zero target for that
      frame.
    """

    def __init__(
        self,
        corpus_root: Path,
        clips: list[str],
        input_size: tuple[int, int] = (512, 288),
        sigma: float = 2.5,
        frames_in: int = 3,
    ) -> None:
        self._corpus_root = Path(corpus_root)
        self._inp_w, self._inp_h = int(input_size[0]), int(input_size[1])
        self._sigma = float(sigma)
        self._frames_in = int(frames_in)

        self._samples: list[tuple[str, list[int]]] = []
        for clip in clips:
            xml_path = self._corpus_root / "annos" / f"{clip}.xml"
            labels = parse_labels_xml(xml_path)
            for run in build_runs(labels, frames_in=self._frames_in):
                self._samples.append((clip, run))
        self._labels_by_clip: dict[str, dict[int, tuple[float, float]]] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def _labels_for(self, clip: str) -> dict[int, tuple[float, float]]:
        cached = self._labels_by_clip.get(clip)
        if cached is None:
            cached = parse_labels_xml(self._corpus_root / "annos" / f"{clip}.xml")
            self._labels_by_clip[clip] = cached
        return cached

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip, run = self._samples[idx]
        labels = self._labels_for(clip)

        x_channels: list[np.ndarray] = []
        y_channels: list[np.ndarray] = []
        for fid in run:
            frame_path = self._corpus_root / "frames" / clip / f"{fid:05d}.png"
            img = cv2.imread(str(frame_path))
            assert img is not None, f"failed to read frame {frame_path}"
            h, w = img.shape[:2]
            center = (w / 2.0, h / 2.0)
            scale = float(max(h, w))
            trans = _get_affine_transform(
                center, scale, (self._inp_w, self._inp_h), inv=False,
            )

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            warped = cv2.warpAffine(
                rgb, trans, (self._inp_w, self._inp_h), flags=cv2.INTER_LINEAR,
            )
            frame_x = warped.astype(np.float32) / 255.0
            frame_x = (frame_x - _IMAGENET_MEAN) / _IMAGENET_STD
            frame_x = frame_x.transpose(2, 0, 1)  # (3, H, W)
            x_channels.append(frame_x)

            u, v = labels[fid]
            mapped = trans @ np.array([u, v, 1.0], dtype=np.float32)
            mx, my = float(mapped[0]), float(mapped[1])
            if 0 <= mx < self._inp_w and 0 <= my < self._inp_h:
                hm = gen_binary_map(
                    (self._inp_w, self._inp_h), (mx, my), self._sigma,
                    data_type=np.float32,
                )
            else:
                hm = np.zeros((self._inp_h, self._inp_w), dtype=np.float32)
            y_channels.append(hm)

        x = np.concatenate(x_channels, axis=0)  # (3*frames_in, H, W)
        y = np.stack(y_channels, axis=0)  # (frames_in, H, W)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def wbce_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TrackNetV2 weighted BCE (focal-gamma=2 form) on sigmoid-clamped predictions.

    ``y_hat = clamp(sigmoid(pred_logits), 1e-4, 1-1e-4)``::

        mean( -( (1-y_hat)**2 * y * log(y_hat)
                 + y_hat**2 * (1-y) * log(1-y_hat) ) )
    """
    y_hat = torch.clamp(torch.sigmoid(pred_logits), 1e-4, 1.0 - 1e-4)
    loss = -(
        (1 - y_hat) ** 2 * target * torch.log(y_hat)
        + y_hat ** 2 * (1 - target) * torch.log(1 - y_hat)
    )
    return loss.mean()


def evaluate_hit_rate(
    model: torch.nn.Module,
    dataset: FinetuneDataset,
    device: str,
    *,
    tol_px: float = 5.0,
    max_samples: int = 200,
) -> float:
    """Fraction of samples whose LAST output frame's argmax peak lies
    within ``tol_px`` (model space) of the label."""
    if len(dataset) == 0:
        return 0.0
    model.eval()
    n = min(len(dataset), max_samples)
    hits = 0
    with torch.no_grad():
        for i in range(n):
            x, y = dataset[i]
            out = model(x.unsqueeze(0).to(device))
            # HRNet returns {scale: tensor} (out_scales=[0]); plain
            # nn.Module stand-ins (e.g. in unit tests) return a tensor.
            pred = out[0] if isinstance(out, dict) else out
            pred = torch.sigmoid(pred)[0, -1].cpu().numpy()
            target = y[-1].numpy()

            py, px = np.unravel_index(int(np.argmax(pred)), pred.shape)
            ty, tx = np.unravel_index(int(np.argmax(target)), target.shape)
            if target.max() <= 0:
                continue
            dist = float(np.hypot(px - tx, py - ty))
            if dist <= tol_px:
                hits += 1
    return hits / n

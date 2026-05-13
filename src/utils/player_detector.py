from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str  # "player" | "goalkeeper" | "referee" | "ball"


def _iter_tiles(
    frame: np.ndarray,
    tile_size: int,
    overlap_ratio: float,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(x_offset, y_offset, tile_view)`` for each tile covering
    the frame. Tile starts are spaced by ``tile_size * (1 - overlap)``;
    the final tile in each row/col is anchored to the right/bottom
    edge so the entire frame is covered even when the frame isn't an
    integer multiple of the step."""
    H, W = frame.shape[:2]
    step = max(1, int(tile_size * (1.0 - overlap_ratio)))

    def starts(dim_len: int) -> list[int]:
        if dim_len <= tile_size:
            return [0]
        xs = list(range(0, dim_len - tile_size + 1, step))
        last_start = dim_len - tile_size
        if xs[-1] != last_start:
            xs.append(last_start)
        return xs

    for y0 in starts(H):
        for x0 in starts(W):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            yield x0, y0, frame[y0:y1, x0:x1]


def _is_edge_clipped(
    bbox: tuple[float, float, float, float],
    *,
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    frame_w: int,
    frame_h: int,
    margin: int = 2,
) -> bool:
    """A box is considered "clipped by a tile edge" when it touches a
    tile boundary that isn't also a frame boundary — the same body is
    presumably visible in full inside the neighbouring tile. Lets us
    drop partial detections cheaply before NMS-merge."""
    x1, y1, x2, y2 = bbox
    touches_tile_left = (x1 - tile_x) < margin and tile_x > 0
    touches_tile_right = (
        (tile_x + tile_w) - x2 < margin and (tile_x + tile_w) < frame_w
    )
    touches_tile_top = (y1 - tile_y) < margin and tile_y > 0
    touches_tile_bottom = (
        (tile_y + tile_h) - y2 < margin and (tile_y + tile_h) < frame_h
    )
    return (
        touches_tile_left
        or touches_tile_right
        or touches_tile_top
        or touches_tile_bottom
    )


def _nms_merge_per_class(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """Greedy per-class NMS over a list of full-frame detections. Used
    after SAHI tile-merging to suppress duplicate detections of the
    same player from neighbouring tiles."""
    if not detections:
        return detections
    out: list[Detection] = []
    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_name, []).append(d)
    for class_dets in by_class.values():
        boxes = np.asarray([d.bbox for d in class_dets], dtype=np.float64)
        confs = np.asarray([d.confidence for d in class_dets], dtype=np.float64)
        order = np.argsort(-confs)
        suppressed = np.zeros(len(class_dets), dtype=bool)
        for idx_i in order:
            if suppressed[idx_i]:
                continue
            out.append(class_dets[idx_i])
            for idx_j in order:
                if idx_j == idx_i or suppressed[idx_j]:
                    continue
                xa, ya, xb, yb = boxes[idx_i]
                xc, yc, xd, yd = boxes[idx_j]
                inter_w = max(0.0, min(xb, xd) - max(xa, xc))
                inter_h = max(0.0, min(yb, yd) - max(ya, yc))
                inter = inter_w * inter_h
                if inter == 0.0:
                    continue
                area_a = (xb - xa) * (yb - ya)
                area_b = (xd - xc) * (yd - yc)
                iou = inter / max(area_a + area_b - inter, 1e-9)
                if iou >= iou_threshold:
                    suppressed[idx_j] = True
    return out


class PlayerDetector(ABC):
    """Detects players (and optionally the ball) in a single frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Returns a list of detections found in the frame."""
        ...


class YOLOPlayerDetector(PlayerDetector):
    """Player detector backed by a YOLOv8 model fine-tuned on football data."""

    # Class IDs for a football-fine-tuned model: 0=player, 1=goalkeeper, 2=referee, 3=ball
    _CLASS_NAMES: dict[int, str] = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        confidence: float = 0.3,
        iou_threshold: float = 0.85,
        imgsz: int = 1280,
        sahi_enabled: bool = False,
        sahi_tile_size: int = 960,
        sahi_overlap_ratio: float = 0.25,
        sahi_nms_iou_threshold: float = 0.5,
    ) -> None:
        """Wrap a YOLOv8 model.

        Parameters
        ----------
        confidence:
            Minimum class-confidence to keep a detection.
        iou_threshold:
            NMS IoU threshold — boxes overlapping by *more* than this
            relative to a higher-confidence box are suppressed. The
            ultralytics default is 0.7; we raise it because two
            footballers running close together overlap at IoU 0.5-0.8
            and at the default a second valid detection gets silently
            dropped during the merge.
        imgsz:
            Inference resolution (longest-side). When ``sahi_enabled``
            is False this applies to the full frame; when True it
            applies per tile.
        sahi_enabled:
            Run SAHI-style tiled inference. The frame is split into
            ``sahi_tile_size``-square tiles with ``sahi_overlap_ratio``
            overlap, YOLO runs on each, and detections are NMS-merged
            back to full-frame coords. Effective when a single-frame
            pass collapses close players into one bbox even at high
            ``imgsz`` — each tile gives a fresh anchor lattice over
            its slice of the frame.
        sahi_tile_size, sahi_overlap_ratio:
            Geometry of the tiling. Defaults (960, 0.25) yield a 3x2
            grid on 1920x1080 with 240 px overlap — comfortably wider
            than a typical player, so anyone straddling a tile seam is
            fully detected in at least one neighbouring tile.
        sahi_nms_iou_threshold:
            IoU above which two detections from different tiles are
            considered duplicates and the lower-confidence one is
            dropped during the merge step.
        """
        from ultralytics import YOLO  # lazy import — model download on first use

        self._model = YOLO(model_name)
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._imgsz = imgsz
        self._sahi_enabled = sahi_enabled
        self._sahi_tile_size = sahi_tile_size
        self._sahi_overlap_ratio = sahi_overlap_ratio
        self._sahi_nms_iou_threshold = sahi_nms_iou_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._sahi_enabled:
            return self._detect_tiled(frame)
        return self._detect_single(frame, x_off=0, y_off=0)

    def _detect_single(
        self,
        frame: np.ndarray,
        x_off: int,
        y_off: int,
    ) -> list[Detection]:
        results = self._model(
            frame,
            verbose=False,
            iou=self._iou_threshold,
            imgsz=self._imgsz,
        )[0]
        detections: list[Detection] = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf < self._confidence:
                continue
            cls_id = int(box.cls)
            class_name = self._CLASS_NAMES.get(cls_id, "player")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                bbox=(x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off),
                confidence=conf,
                class_name=class_name,
            ))
        return detections

    def _detect_tiled(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        all_detections: list[Detection] = []
        for x_off, y_off, tile in _iter_tiles(
            frame, self._sahi_tile_size, self._sahi_overlap_ratio
        ):
            tile_dets = self._detect_single(tile, x_off=x_off, y_off=y_off)
            # Drop boxes touching a tile edge that is NOT also a frame
            # edge — those are clipped partials; the neighbouring tile
            # will contain the full body.
            th, tw = tile.shape[:2]
            kept = []
            for d in tile_dets:
                if _is_edge_clipped(
                    d.bbox,
                    tile_x=x_off, tile_y=y_off,
                    tile_w=tw, tile_h=th,
                    frame_w=W, frame_h=H,
                ):
                    continue
                kept.append(d)
            all_detections.extend(kept)
        return _nms_merge_per_class(all_detections, self._sahi_nms_iou_threshold)


class FakePlayerDetector(PlayerDetector):
    """Deterministic detector for tests — cycles through a pre-supplied sequence."""

    def __init__(self, detections_sequence: list[list[Detection]]) -> None:
        self._seq = detections_sequence
        self._idx = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets = self._seq[self._idx % len(self._seq)]
        self._idx += 1
        return list(dets)

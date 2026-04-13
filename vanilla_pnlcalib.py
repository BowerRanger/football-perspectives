#!/usr/bin/env python
"""One-off experiment: run raw PnLCalib on every frame of a shot.

No static-camera fuser, no plausibility filter, no manual anchors,
no VP/ICL/Hampel/Savgol refinement.  Just PnLCalib's per-frame
output written straight into a ``CalibrationResult`` for the
dashboard's calibration video preview to pick up.

Usage::

    .venv311/bin/python vanilla_pnlcalib.py origi01 origi02

The script writes one ``CameraFrame`` per video frame to
``output/calibration/<shot>_calibration.json``.  Back up any
existing calibration before running if you want to restore the
post-processed version afterwards (the previous invocation's
output was saved to ``/tmp/calibration_backup/``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.utils.neural_calibrator import NeuralCalibration, PnLCalibrator


def _to_camera_frame(
    calib: NeuralCalibration, frame_idx: int,
) -> CameraFrame:
    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=np.asarray(calib.K, dtype=np.float64).tolist(),
        rotation_vector=np.asarray(calib.rvec, dtype=np.float64).reshape(3).tolist(),
        translation_vector=np.asarray(calib.tvec, dtype=np.float64).reshape(3).tolist(),
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
        tracked_landmark_types=[],
    )


@click.command()
@click.argument("shot_ids", nargs=-1, required=True)
@click.option("--output", type=click.Path(), default="output")
@click.option("--sample-every", type=int, default=1,
              help="Run PnLCalib every Nth frame (1 = every frame).")
@click.option("--device", default="auto",
              help="PyTorch device: auto|cpu|cuda|mps")
@click.option("--kp-threshold", type=float, default=0.3434,
              help="PnLCalib keypoint confidence threshold (default: 0.3434)")
@click.option("--line-threshold", type=float, default=0.7867,
              help="PnLCalib line confidence threshold (default: 0.7867)")
def main(shot_ids, output, sample_every, device, kp_threshold, line_threshold):
    out = Path(output)
    cal_dir = out / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"loading PnLCalib on device={device} "
               f"kp_thr={kp_threshold} line_thr={line_threshold}…")
    calibrator = PnLCalibrator(
        device=device,
        kp_threshold=kp_threshold,
        line_threshold=line_threshold,
    )

    for shot_id in shot_ids:
        clip_path = out / "shots" / f"{shot_id}.mp4"
        if not clip_path.exists():
            click.echo(f"  - {shot_id}: no clip at {clip_path}, skipping")
            continue

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            click.echo(f"  - {shot_id}: failed to open clip")
            continue

        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        click.echo(f"  - {shot_id}: {n_total} frames, sampling every {sample_every}")

        frames: list[CameraFrame] = []
        n_succeeded = 0
        n_failed = 0
        t0 = time.perf_counter()
        try:
            for fi in range(0, n_total, sample_every):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                result = calibrator.calibrate(frame)
                if result is None:
                    n_failed += 1
                    continue
                frames.append(_to_camera_frame(result, fi))
                n_succeeded += 1
                if n_succeeded > 0 and n_succeeded % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    per_frame = elapsed / n_succeeded
                    remaining = (n_total - fi) / sample_every * per_frame
                    click.echo(
                        f"     {n_succeeded} frames ok, {n_failed} failed, "
                        f"{per_frame*1000:.0f} ms/frame, ~{remaining:.0f}s remaining"
                    )
        finally:
            cap.release()

        cal = CalibrationResult(
            shot_id=shot_id, camera_type="static", frames=frames,
        )
        cal.save(cal_dir / f"{shot_id}_calibration.json")
        elapsed = time.perf_counter() - t0
        click.echo(
            f"  -> {shot_id}: {n_succeeded}/{n_total} frames calibrated "
            f"({n_failed} PnLCalib failures) in {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()

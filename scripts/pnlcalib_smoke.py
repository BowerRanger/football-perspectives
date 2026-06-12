"""Register ONE real frame and print mapped (image -> world) correspondences
in OUR pitch frame. Manual validation gate for the bridge + provider.

  python scripts/pnlcalib_smoke.py --video CLIP --frame 0
Eyeball that goal-post keypoints (12-19) land on the posts and the four
crossbar tops (12,15,16,19) report z = +2.44.
"""

from __future__ import annotations

import argparse

import cv2

from src.utils.neural_calibrator import PnLCalibrator
from src.utils.pnlcalib_pitch_map import keypoint_world_xyz_ours


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("could not read frame")
        return

    calibrator = PnLCalibrator(device=args.device)
    pixels = calibrator.extract_keypoints_pixels(frame)
    print(f"frame {args.frame}: {len(pixels)} keypoints")
    for kp_id in sorted(pixels):
        px, py = pixels[kp_id]
        world = keypoint_world_xyz_ours(kp_id)
        if world is None:
            continue
        wx, wy, wz = world
        print(f"  kp{kp_id:>2}  px=({px:7.1f},{py:7.1f})  "
              f"world=({wx:6.2f},{wy:6.2f},{wz:4.2f})")


if __name__ == "__main__":
    main()

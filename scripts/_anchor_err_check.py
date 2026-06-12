"""One-off: anchor-click reprojection error (px and metres) under the saved track."""
import json
import sys

sys.path.insert(0, '.')

import numpy as np

from src.utils.camera_projection import project_world_to_image


def analyse(clip_id: str, anchors_path: str, track_path: str) -> list[dict]:
    anchors = json.load(open(anchors_path))
    track = json.load(open(track_path))
    distortion = tuple(track['distortion'])
    frames_by_idx = {f['frame']: f for f in track['frames']}

    rows = []
    for anchor in anchors['anchors']:
        fidx = anchor['frame']
        cam = frames_by_idx.get(fidx)
        if cam is None:
            print(f'{clip_id} frame {fidx}: not covered by track, skipped')
            continue
        K = np.array(cam['K'], dtype=np.float64)
        R = np.array(cam['R'], dtype=np.float64)
        t = np.array(cam['t'], dtype=np.float64)
        fx = K[0][0]
        for lm in anchor.get('landmarks', []):
            world = np.array(lm['world_xyz'], dtype=np.float64)
            clicked = np.array(lm['image_xy'], dtype=np.float64)
            proj = project_world_to_image(K, R, t, distortion, world.reshape(1, 3))[0]
            px_err = float(np.linalg.norm(proj - clicked))
            cam_pt = R @ world + t
            Z = float(cam_pt[2])
            m_err = px_err * Z / fx
            rows.append({
                'clip': clip_id,
                'frame': fidx,
                'name': lm['name'],
                'px': px_err,
                'metres': m_err,
                'Z': Z,
                'du': float(proj[0] - clicked[0]),
                'dv': float(proj[1] - clicked[1]),
                'world': lm['world_xyz'],
            })
    return rows


def main() -> None:
    all_rows = []
    for clip in ('origi01', 'origi02'):
        all_rows += analyse(
            clip,
            f'output-origi/camera/{clip}_anchors__manual.json',
            f'output-origi/camera/{clip}_camera_track.json',
        )

    # Per anchor frame: worst landmark + median metres
    print('\n=== per anchor frame ===')
    keys = sorted({(r['clip'], r['frame']) for r in all_rows})
    for clip, fidx in keys:
        frame_rows = [r for r in all_rows if r['clip'] == clip and r['frame'] == fidx]
        worst = max(frame_rows, key=lambda r: r['metres'])
        med = float(np.median([r['metres'] for r in frame_rows]))
        print(
            f"{clip} f{fidx:4d} n={len(frame_rows):2d} "
            f"worst={worst['name']} px={worst['px']:.2f} m={worst['metres']:.3f} "
            f"(du={worst['du']:+.1f}, dv={worst['dv']:+.1f}, world={worst['world']}) "
            f"median_m={med:.3f}"
        )

    print('\n=== landmarks > 0.30 m (sorted desc) ===')
    bad = sorted((r for r in all_rows if r['metres'] > 0.30), key=lambda r: -r['metres'])
    for r in bad:
        print(
            f"{r['clip']} f{r['frame']:4d} {r['name']}: px={r['px']:.2f} "
            f"m={r['metres']:.3f} du={r['du']:+.1f} dv={r['dv']:+.1f} "
            f"world={r['world']} Z={r['Z']:.1f}"
        )
    print(f'\ntotal landmarks={len(all_rows)} over 0.30m={len(bad)}')


if __name__ == '__main__':
    main()

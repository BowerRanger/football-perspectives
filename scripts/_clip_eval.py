"""Per-clip camera-quality dashboard.

Reports several INDEPENDENT signals, because line-RMS alone is misleading:
it is self-referential (the solve minimises the very residual it measures),
blind to under-determined frames (2 lines fit at ~0px with a wrong camera) and
to anything not detected (circle, far lines), and coverage-blind. So we also
print:

  * coverage       — how much of the clip is covered / line-solved vs interp.
  * jitter         — temporal smoothness (frame-to-frame rotation).
  * circle misfit  — HELD-OUT: project the catalogue centre circle and measure
                     it against the painted ring detected directly in the image
                     (independent of the line solve — this is what catches lens
                     / wide-field error the line-RMS can't see).
  * vs-manual      — geometric Δ (rotation, centre) from the hand-anchored
                     track where one exists — true camera accuracy, not line-fit.

Usage: python scripts/_clip_eval.py BASE LABEL NCLIP
  BASE e.g. output-origi/camera/origi02  (video inferred at output-origi/shots/origi02.mp4)
"""
import json
import os
import sys

import numpy as np

from src.utils.camera_projection import project_world_to_image


def _geo(a, b):
    c = (np.trace(np.array(a).T @ np.array(b)) - 1) / 2
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def _line_rms(cam, dl, dist):
    res, perf = [], {}
    for fk, fv in dl["frames"].items():
        f = int(fk)
        if f not in cam:
            continue
        K = np.array(cam[f]["K"]); R = np.array(cam[f]["R"]); t = np.array(cam[f]["t"])
        rr = []
        for ln in fv["lines"]:
            if "circle" in ln["name"]:
                continue
            proj = project_world_to_image(K, R, t, dist, np.array(ln["world_segment"]))
            pa, pb = proj[0], proj[1]
            d = pb - pa
            nn = np.array([-d[1], d[0]])
            if np.linalg.norm(nn) < 1e-6:
                continue
            nn = nn / np.linalg.norm(nn)
            for ip in ln["image_segment"]:
                rr.append(abs(np.dot(np.array(ip) - pa, nn)))
        res += rr
        if rr:
            perf[f] = float(np.mean(rr))
    return (np.array(res) if res else np.array([np.nan])), perf


def _circle_misfit(base, cam, dist):
    """HELD-OUT: detect the painted centre circle directly and measure how far
    the projected catalogue circle is from it. Independent of the line solve."""
    vid = base.replace("/camera/", "/shots/") + ".mp4"
    if not os.path.exists(vid):
        return None
    try:
        import cv2
        from src.utils.ellipse_detector import detect_circle_ellipse
        from src.utils.line_detector import DetectorConfig
        from src.utils.static_line_solver import _ellipse_residuals_distorted
    except Exception:
        return None
    cap = cv2.VideoCapture(vid)
    cov = sorted(cam)
    mis = []
    for fid in cov[:: max(1, len(cov) // 25)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, img = cap.read()
        if not ok:
            continue
        K = np.array(cam[fid]["K"]); R = np.array(cam[fid]["R"]); t = np.array(cam[fid]["t"])
        ed = detect_circle_ellipse(img, K, R, t, dist, DetectorConfig(), band_px=50)
        if ed is None:
            continue
        rv, _ = cv2.Rodrigues(R)
        r = _ellipse_residuals_distorted(ed.ellipse, K, rv.reshape(3), t, dist)
        nz = r[r != 0]
        if nz.size:
            mis.append(float(np.median(np.abs(nz))))
    cap.release()
    return (len(mis), float(np.median(mis))) if mis else (0, None)


def _vs_manual(base, cam):
    mpath = base + "_track__manual.json"
    if not os.path.exists(mpath):
        return None
    m = {f["frame"]: f for f in json.load(open(mpath))["frames"]}
    rot, cen = [], []
    for f in cam:
        if f not in m:
            continue
        Ra, Rm = np.array(cam[f]["R"]), np.array(m[f]["R"])
        rot.append(_geo(Ra, Rm))
        Ca = -Ra.T @ np.array(cam[f]["t"])
        Cm = -Rm.T @ np.array(m[f]["t"])
        cen.append(float(np.linalg.norm(Ca - Cm)))
    if not rot:
        return None
    return float(np.median(rot)), float(np.median(cen)), len(rot)


def ev(base, label, clip):
    if not os.path.exists(f"{base}_camera_track.json"):
        print(f"### {label}: NO TRACK")
        return
    tr = json.load(open(f"{base}_camera_track.json"))
    cam = {f["frame"]: f for f in tr["frames"]}
    fr = sorted(cam)
    dl = (
        json.load(open(f"{base}_detected_lines.json"))
        if os.path.exists(f"{base}_detected_lines.json") else {"frames": {}}
    )
    dist = tuple(tr.get("distortion", (0, 0))[:2])
    dr = [_geo(cam[a]["R"], cam[b]["R"]) for a, b in zip(fr, fr[1:]) if b - a == 1]
    res, perf = _line_rms(cam, dl, dist)
    span = range(fr[0], fr[-1] + 1)
    zero = sum(1 for f in span if len(dl["frames"].get(str(f), {}).get("lines", [])) == 0)
    line_solved = sum(
        1 for v in dl["frames"].values()
        if any("circle" not in ln["name"] for ln in v["lines"])
    )

    print(f"### {label} ({clip}f): {len(fr)}f span {fr[0]}..{fr[-1]}  ({len(fr)/clip:.0%} coverage)")
    print(f"    coverage : line-solved {line_solved} | interp/0-line {zero}")
    print(f"    line-RMS : mean {np.nanmean(res):.2f} med {np.nanmedian(res):.2f} "
          f"<1px {np.mean(res < 1):.0%} | out>5px {sum(1 for v in perf.values() if v > 5)}"
          f"   [self-referential — line crispness only]")
    print(f"    jitter   : max {max(dr):.2f} p95 {np.percentile(dr, 95):.2f} deg")
    cm = _circle_misfit(base, cam, dist)
    if cm is None:
        print("    circle   : (video unavailable)")
    elif cm[1] is None:
        print(f"    circle   : not detected on sampled frames")
    else:
        print(f"    circle   : HELD-OUT misfit med {cm[1]:.1f} px on {cm[0]} frames "
              f"[catches lens/wide-field error line-RMS can't]")
    vm = _vs_manual(base, cam)
    if vm is not None:
        print(f"    vs-manual: rotation med {vm[0]:.2f} deg | centre med {vm[1]:.2f} m "
              f"({vm[2]} common frames)")


if __name__ == "__main__":
    ev(sys.argv[1], sys.argv[2], int(sys.argv[3]))

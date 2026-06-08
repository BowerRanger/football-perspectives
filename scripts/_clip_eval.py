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


def _video_frames(base):
    vid = base.replace("/camera/", "/shots/") + ".mp4"
    if not os.path.exists(vid):
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(vid)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n or None
    except Exception:
        return None


def _safe(x):
    """NaN/inf -> None so the dict round-trips as valid JSON."""
    return None if x is None or not np.isfinite(x) else float(x)


def compute_camera_metrics(base, clip_frames=None):
    """Compute the full per-clip camera dashboard as a dict (the values the
    CLI prints and the web shot-summary renders). Returns None if no track."""
    if not os.path.exists(f"{base}_camera_track.json"):
        return None
    tr = json.load(open(f"{base}_camera_track.json"))
    cam = {f["frame"]: f for f in tr["frames"]}
    fr = sorted(cam)
    if not fr:
        return None
    dl = (
        json.load(open(f"{base}_detected_lines.json"))
        if os.path.exists(f"{base}_detected_lines.json") else {"frames": {}}
    )
    dist = tuple(tr.get("distortion", (0, 0))[:2])
    dr = [_geo(cam[a]["R"], cam[b]["R"]) for a, b in zip(fr, fr[1:]) if b - a == 1]
    res, perf = _line_rms(cam, dl, dist)
    zero = sum(
        1 for f in range(fr[0], fr[-1] + 1)
        if len(dl["frames"].get(str(f), {}).get("lines", [])) == 0
    )
    line_solved = sum(
        1 for v in dl["frames"].values()
        if any("circle" not in ln["name"] for ln in v["lines"])
    )
    if clip_frames is None:
        clip_frames = _video_frames(base) or (fr[-1] + 1)
    cm = _circle_misfit(base, cam, dist)
    vm = _vs_manual(base, cam)
    return {
        "covered": len(fr), "clip_frames": int(clip_frames),
        "span": [fr[0], fr[-1]], "line_solved": line_solved, "interp": zero,
        "line_rms_mean": _safe(np.nanmean(res)),
        "line_rms_med": _safe(np.nanmedian(res)),
        "line_under1px": _safe(np.mean(res < 1)),
        "line_out5": sum(1 for v in perf.values() if v > 5),
        "jitter_max": _safe(max(dr)) if dr else None,
        "jitter_p95": _safe(np.percentile(dr, 95)) if dr else None,
        "circle": None if cm is None else {"misfit": _safe(cm[1]), "frames": cm[0]},
        "vs_manual": None if vm is None else {
            "rotation": _safe(vm[0]), "centre": _safe(vm[1]), "frames": vm[2]},
    }


def ev(base, label, clip):
    m = compute_camera_metrics(base, clip_frames=clip)
    if m is None:
        print(f"### {label}: NO TRACK")
        return
    print(f"### {label} ({clip}f): {m['covered']}f span {m['span'][0]}..{m['span'][1]}"
          f"  ({m['covered'] / clip:.0%} coverage)")
    print(f"    coverage : line-solved {m['line_solved']} | interp/0-line {m['interp']}")
    print(f"    line-RMS : mean {m['line_rms_mean']:.2f} med {m['line_rms_med']:.2f} "
          f"<1px {m['line_under1px']:.0%} | out>5px {m['line_out5']}"
          f"   [self-referential — line crispness only]")
    if m["jitter_max"] is not None:
        print(f"    jitter   : max {m['jitter_max']:.2f} p95 {m['jitter_p95']:.2f} deg")
    cm = m["circle"]
    if cm is None:
        print("    circle   : (video unavailable)")
    elif cm["misfit"] is None:
        print("    circle   : not detected on sampled frames")
    else:
        print(f"    circle   : HELD-OUT misfit med {cm['misfit']:.1f} px on "
              f"{cm['frames']} frames [catches lens/wide-field error line-RMS can't]")
    vm = m["vs_manual"]
    if vm is not None:
        print(f"    vs-manual: rotation med {vm['rotation']:.2f} deg | "
              f"centre med {vm['centre']:.2f} m ({vm['frames']} common frames)")


if __name__ == "__main__":
    ev(sys.argv[1], sys.argv[2], int(sys.argv[3]))

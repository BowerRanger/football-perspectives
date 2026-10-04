"""PROTOTYPE (gberch shorts, gap G16) — Two-knot arc with Magnus curl (spin about the vertical axis): strike knot
(371) -> point on the frame-394 anchor ray. Unknowns: x_end (depth on the
ray) and curl c (m/s^2, accel = c * up x v_hat). Gravity + drag."""
import json
import sys

import numpy as np
from scipy.optimize import fsolve, minimize

out = sys.argv[1]
cam = json.load(open(f"{out}/camera/gberch_camera_track.json"))
cf = {f["frame"]: f for f in cam["frames"]}
FPS, F0, F1 = 30.0, 371, 394
G = np.array([0, 0, -9.81])
UP = np.array([0, 0, 1.0])
K_DRAG = 0.5 * 1.2 * 0.25 * np.pi * 0.11**2 / 0.43
P0 = np.array([18.0, 24.37, 0.11])
raw = json.load(open(f"{out}/ball/gberch_ball_observations.json"))
obs = raw if isinstance(raw, list) else (raw.get("observations") or raw.get("frames"))
pix = {r["frame"]: r["uv"] for r in obs
       if 374 <= r["frame"] <= 386 and r["source"] in ("detector", "anchor", "foot_guided", "second_pass")}
pix.update({388: (851, 304), 390: (800, 296), 391: (773, 295)})


def cam_of(f):
    c = cf[f]
    return np.array(c["K"]), np.array(c["R"]), np.array(c["t"])


def project(f, p):
    K, R, t = cam_of(f)
    q = K @ (R @ p + t)
    return q[:2] / q[2]


def ray(f, uv):
    K, R, t = cam_of(f)
    C = -R.T @ t
    d = R.T @ np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    return C, d / np.linalg.norm(d)


def simulate(v0, curl, last, sub=10):
    p, v = P0.copy(), np.array(v0, float)
    dt = 1.0 / FPS / sub
    res = {F0: p.copy()}
    for f in range(F0 + 1, last + 1):
        for _ in range(sub):
            sp = np.linalg.norm(v)
            mag = curl * np.cross(UP, v / sp)
            v = v + (G + mag - K_DRAG * sp * v) * dt
            p = p + v * dt
        res[f] = p.copy()
    return res


C394, D394 = ray(394, (729.1, 297.9))


def end_point(x_end):
    return C394 + (x_end - C394[0]) / D394[0] * D394


def solve(x_end, curl):
    end = end_point(x_end)
    T = (F1 - F0) / FPS
    guess = (end - P0) / T - 0.5 * G * T
    v0 = fsolve(lambda v: simulate(v, curl, F1)[F1] - end, guess)
    return v0


def errs(params):
    x_end, curl = params
    v0 = solve(x_end, curl)
    tr = simulate(v0, curl, F1)
    return np.array([np.linalg.norm(project(f, tr[f]) - np.array(pix[f])) for f in sorted(pix)]), v0


def cost(params):
    e, _ = errs(params)
    return float(np.mean(np.minimum(e, 25.0) ** 2))


best = None
for x0 in (2.0, 0.0, -1.0):
    for c0 in (-6.0, 0.0, 6.0):
        r = minimize(cost, [x0, c0], method="Nelder-Mead",
                     options={"xatol": 0.01, "fatol": 0.01, "maxiter": 300})
        if best is None or r.fun < best.fun:
            best = r
x_end, curl = best.x
e, v0 = errs(best.x)
print("x_end", round(x_end, 2), "end", np.round(end_point(x_end), 2), "curl m/s2", round(curl, 2),
      "speed km/h", round(float(np.linalg.norm(v0)) * 3.6, 1), "v0", np.round(v0, 2))
print("median px", round(float(np.median(e)), 1), dict(zip(sorted(pix), np.round(e, 1).tolist())))
tr = simulate(v0, curl, 412)
prev = None
for f in range(F0, 413):
    if prev is not None and prev[0] > 0 >= tr[f][0]:
        print("LINE CROSS frame", f, np.round(tr[f], 2))
    prev = tr[f]
for f in range(F0, 412, 3):
    print(f, np.round(tr[f], 2))
json.dump({"v0": v0.tolist(), "curl": curl, "x_end": x_end,
           "frames": {int(f): tr[f].tolist() for f in tr}}, open(sys.argv[2], "w"))

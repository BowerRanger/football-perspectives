import { describe, expect, it } from "vitest"

import {
  closestBetweenRays,
  epipolarPolyline,
  makeFrameCam,
  project,
  rayAt,
  refToShot,
  shotToRef,
  snapToEpipolar,
  unproject,
  type FrameCam,
} from "./camera-model"
import type { Vec3 } from "./types"

/** Look-at camera: world->camera (OpenCV: x right, y down, z forward). */
function lookAt(C: Vec3, target: Vec3, distortion: [number, number] = [0, 0]): FrameCam {
  const f = [target[0] - C[0], target[1] - C[1], target[2] - C[2]]
  const fn = Math.hypot(...f)
  const z = f.map((v) => v / fn)
  // right = z × up(0,0,1) normalised ; down = z × right
  let r = [z[1] * 1 - z[2] * 0, z[2] * 0 - z[0] * 1, 0]
  const rn = Math.hypot(...r)
  r = r.map((v) => v / rn)
  const d = [z[1] * r[2] - z[2] * r[1], z[2] * r[0] - z[0] * r[2], z[0] * r[1] - z[1] * r[0]]
  const R = [...r, ...d, ...z]
  const t = [
    -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
    -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
    -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
  ]
  return makeFrameCam([1800, 1800, 960, 540], R, t, distortion)
}

const camA = lookAt([52.5, -30, 22], [40, 34, 0], [-0.05, 0.01])
const camB = lookAt([-20, 34, 15], [20, 34, 0], [0.03, 0])

describe("frame mapping", () => {
  it("r = shot_frame - offset and back", () => {
    expect(refToShot(440, -142)).toBe(298)
    expect(shotToRef(298, -142)).toBe(440)
  })
})

describe("camera model", () => {
  it("camera centre is recovered from R,t", () => {
    expect(camA.C[0]).toBeCloseTo(52.5, 6)
    expect(camA.C[1]).toBeCloseTo(-30, 6)
    expect(camA.C[2]).toBeCloseTo(22, 6)
  })

  it("project(unproject) is the identity through a distorted lens", () => {
    for (const uv of [[960, 540], [100, 80], [1800, 1000], [1500, 200]] as const) {
      const ray = unproject(camA, uv[0], uv[1])
      const p = project(camA, rayAt(ray, 37))
      expect(p).not.toBeNull()
      expect(p!.uv[0]).toBeCloseTo(uv[0], 3)
      expect(p!.uv[1]).toBeCloseTo(uv[1], 3)
    }
  })

  it("returns null behind the camera", () => {
    expect(project(camA, [52.5, -80, 22])).toBeNull()
  })

  it("two rays through the projections of a point meet at that point", () => {
    const X: Vec3 = [30, 31, 1.4]
    const pa = project(camA, X)!
    const pb = project(camB, X)!
    const pair = closestBetweenRays(unproject(camA, pa.uv[0], pa.uv[1]), unproject(camB, pb.uv[0], pb.uv[1]))!
    expect(pair.gap_m).toBeLessThan(1e-4)
    expect(pair.point[0]).toBeCloseTo(X[0], 3)
    expect(pair.point[1]).toBeCloseTo(X[1], 3)
    expect(pair.point[2]).toBeCloseTo(X[2], 3)
    expect(pair.angleDeg).toBeGreaterThan(10)
  })

  it("parallel rays have no intersection", () => {
    const ray = { origin: [0, 0, 1] as Vec3, dir: [1, 0, 0] as Vec3 }
    expect(closestBetweenRays(ray, { origin: [0, 3, 1], dir: [1, 0, 0] })).toBeNull()
  })
})

describe("epipolar polyline", () => {
  const X: Vec3 = [30, 31, 1.4]
  const pa = project(camA, X)!
  const ray = unproject(camA, pa.uv[0], pa.uv[1])
  const epi = epipolarPolyline(ray, camB)

  it("passes through the other view's pick of the same point", () => {
    const pb = project(camB, X)!
    const snap = snapToEpipolar(epi, pb.uv)!
    expect(snap.dist).toBeLessThan(0.6)
    // the snapped sample is the true height, which is what the operator reads off the ticks
    expect(snap.z).toBeCloseTo(1.4, 1)
  })

  it("carries height ticks inside the sampled range", () => {
    const labels = epi.ticks.map((t) => t.label)
    expect(labels).toContain("ground")
    expect(labels).toContain("2 m")
    const t2 = epi.ticks.find((t) => t.z === 2)!
    const p = project(camB, rayAt(ray, t2.s))!
    expect(t2.uv[0]).toBeCloseTo(p.uv[0], 6)
  })

  it("snap reports the perpendicular distance for an off-line click", () => {
    const pb = project(camB, X)!
    const snap = snapToEpipolar(epi, [pb.uv[0], pb.uv[1] + 6])!
    expect(snap.dist).toBeGreaterThan(0)
    expect(snap.dist).toBeLessThanOrEqual(6.01)
  })
})

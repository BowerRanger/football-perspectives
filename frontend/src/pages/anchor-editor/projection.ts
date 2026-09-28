// Pinhole projection with the same conventions as src/utils/anchor_solver.py.
import type { CameraFrame, CameraTrack, Vec3 } from "./types"

function distortionFoldR2(k1: number, k2: number): number {
  // Largest r^2 where r*(1 + k1 r^2 + k2 r^4) is still increasing; beyond it
  // the polynomial folds back and pulls far-outside points INSIDE the image.
  if (Math.abs(k2) < 1e-12) return k1 >= 0 ? Infinity : -1 / (3 * k1)
  const a = 5 * k2
  const b = 3 * k1
  const disc = b * b - 4 * a
  if (disc < 0) return Infinity
  const sq = Math.sqrt(disc)
  const roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)].filter((r) => r > 0)
  return roots.length ? Math.min(...roots) : Infinity
}

function applyRadialDistortion(
  uv: [number, number],
  distortion: readonly number[],
  K: number[][],
): [number, number] | null {
  if (distortion[0] === 0 && distortion[1] === 0) return uv
  const fx = K[0][0]
  const fy = K[1][1]
  const cx = K[0][2]
  const cy = K[1][2]
  const x = (uv[0] - cx) / fx
  const y = (uv[1] - cy) / fy
  const r2 = x * x + y * y
  const [k1, k2] = distortion
  if (r2 > 0.9 * distortionFoldR2(k1, k2)) return null
  const factor = 1 + k1 * r2 + k2 * r2 * r2
  return [cx + fx * x * factor, cy + fy * y * factor]
}

/** World point -> image pixel, or null when behind / beyond the distortion fold. */
export function projectPoint(
  p: Vec3 | readonly number[],
  K: number[][],
  R: number[][],
  t: readonly number[],
  distortion: readonly number[],
): [number, number] | null {
  // cam = R @ P + t (OpenCV extrinsics: t is the world origin in camera frame).
  const cx = R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0]
  const cy = R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1]
  const cz = R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2]
  if (cz <= 0.05) return null
  const u = (K[0][0] * cx + K[0][1] * cy + K[0][2] * cz) / cz
  const v = (K[1][0] * cx + K[1][1] * cy + K[1][2] * cz) / cz
  const distorted = applyRadialDistortion([u, v], distortion, K)
  if (!distorted) return null
  // Guard the FINAL post-distortion coordinate: grazing-incidence points get
  // amplified to millions of pixels and would slash a line across the frame.
  if (
    !Number.isFinite(distorted[0]) ||
    !Number.isFinite(distorted[1]) ||
    Math.abs(distorted[0]) > 1e5 ||
    Math.abs(distorted[1]) > 1e5
  ) {
    return null
  }
  return distorted
}

export interface FrameCamera {
  K: number[][]
  R: number[][]
  t: number[]
  distortion: number[]
}

/** Resolve the camera for a frame; per-frame t wins over the clip-shared t_world. */
export function cameraForFrame(track: CameraTrack | null, frame: number): FrameCamera | null {
  if (!track || track.frames.length === 0) return null
  const cf: CameraFrame | undefined =
    track.frames.find((f) => f.frame === frame) ?? track.frames[Math.min(track.frames.length - 1, frame)]
  if (!cf) return null
  const t = cf.t ?? track.t_world
  if (!cf.K || !cf.R || !t) return null
  return { K: cf.K, R: cf.R, t, distortion: track.distortion ?? [0, 0] }
}

function densifyPolyline(poly: Vec3[], stepM: number): Vec3[] {
  // A long line often has both endpoints off-frame; sampling every metre lets
  // the visible run draw and horizon-crossing points simply break the line.
  const out: Vec3[] = []
  for (let i = 0; i < poly.length - 1; i++) {
    const a = poly[i]
    const b = poly[i + 1]
    const dx = b[0] - a[0]
    const dy = b[1] - a[1]
    const dz = b[2] - a[2]
    const segs = Math.max(1, Math.round(Math.hypot(dx, dy, dz) / stepM))
    for (let k = 0; k < segs; k++) {
      const s = k / segs
      out.push([a[0] + dx * s, a[1] + dy * s, a[2] + dz * s])
    }
  }
  out.push(poly[poly.length - 1])
  return out
}

function buildPitchPolylines(): Vec3[][] {
  const L = 105
  const W = 68
  const polys: Vec3[][] = [
    [[0, 0, 0], [L, 0, 0]],
    [[0, W, 0], [L, W, 0]],
    [[0, 0, 0], [0, W, 0]],
    [[L, 0, 0], [L, W, 0]],
    [[L / 2, 0, 0], [L / 2, W, 0]],
    [[0, 13.84, 0], [16.5, 13.84, 0], [16.5, 54.16, 0], [0, 54.16, 0]],
    [[L, 13.84, 0], [L - 16.5, 13.84, 0], [L - 16.5, 54.16, 0], [L, 54.16, 0]],
  ]
  const circle: Vec3[] = []
  for (let a = 0; a <= Math.PI * 2 + 0.01; a += Math.PI / 30) {
    circle.push([L / 2 + 9.15 * Math.cos(a), W / 2 + 9.15 * Math.sin(a), 0])
  }
  polys.push(circle)
  return polys.map((p) => densifyPolyline(p, 1))
}

/** FIFA pitch outline subset — enough to verify camera registration. */
export const PITCH_POLYLINES: Vec3[][] = buildPitchPolylines()

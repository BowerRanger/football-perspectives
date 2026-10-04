// Client-side camera maths for previews (ghost points, epipolar polylines,
// snap). The server solver stays authoritative; this mirrors the contract's
// "Camera model" section: Xc = R·X + t, radial k1/k2 distortion, K rows are
// [fx, fy, cx, cy].
import type { SceneShot, Vec2, Vec3 } from "./types"

export const BALL_RADIUS_M = 0.11

export interface FrameCam {
  fx: number
  fy: number
  cx: number
  cy: number
  /** Row-major 3x3 world->camera rotation. */
  R: readonly number[]
  t: readonly number[]
  k1: number
  k2: number
  /** Camera centre in world coordinates, C = -Rᵀ t. */
  C: Vec3
}

export interface Projection {
  uv: Vec2
  /** Depth along the optical axis (Xc.z). */
  depth: number
}

export interface Ray {
  origin: Vec3
  dir: Vec3
}

export function makeFrameCam(
  K: readonly number[],
  R: readonly number[],
  t: readonly number[],
  distortion: readonly number[],
): FrameCam {
  const C: Vec3 = [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ]
  return { fx: K[0], fy: K[1], cx: K[2], cy: K[3], R, t, k1: distortion[0] ?? 0, k2: distortion[1] ?? 0, C }
}

/** Per-shot lookup of the solved camera by shot-local frame. */
export class ShotCameras {
  readonly shotId: string
  readonly frameOffset: number
  readonly imageSize: Vec2
  readonly centre: Vec3
  private readonly rowByFrame = new Map<number, number>()
  private readonly cache = new Map<number, FrameCam>()
  private readonly shot: SceneShot

  constructor(shot: SceneShot) {
    this.shot = shot
    this.shotId = shot.shot_id
    this.frameOffset = shot.frame_offset
    this.imageSize = shot.image_size
    this.centre = shot.camera_centre
    shot.frames.forEach((f, i) => this.rowByFrame.set(f, i))
  }

  /** Solved camera at a shot-local frame, or null when the camera track has none. */
  at(shotFrame: number): FrameCam | null {
    const row = this.rowByFrame.get(shotFrame)
    if (row === undefined) return null
    let cam = this.cache.get(row)
    if (!cam) {
      cam = makeFrameCam(this.shot.K[row], this.shot.R[row], this.shot.t[row], this.shot.distortion)
      this.cache.set(row, cam)
    }
    return cam
  }

  /** Camera at a reference-timeline frame. */
  atRef(refFrame: number): FrameCam | null {
    return this.at(refToShot(refFrame, this.frameOffset))
  }

  get firstFrame(): number {
    return this.shot.frames[0] ?? 0
  }

  get lastFrame(): number {
    return this.shot.frames[this.shot.frames.length - 1] ?? 0
  }
}

// ---- frame mapping --------------------------------------------------------

/** shot_frame = r + frame_offset (contract: "Reference timeline"). */
export const refToShot = (ref: number, frameOffset: number): number => ref + frameOffset
/** r = shot_frame - frame_offset. */
export const shotToRef = (shotFrame: number, frameOffset: number): number => shotFrame - frameOffset

/** Frames that have video: from 0 to the later of the clip length and the last camera frame. */
export function shotVideoRange(shot: { n_frames: number; frame_range?: readonly [number, number] }): [number, number] {
  return [0, Math.max(shot.n_frames - 1, shot.frame_range?.[1] ?? 0)]
}

// ---- projection -----------------------------------------------------------

export function toCamera(cam: FrameCam, X: Vec3): Vec3 {
  const { R, t } = cam
  return [
    R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0],
    R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1],
    R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2],
  ]
}

/** Project a world point; null when it is behind (or on) the image plane. */
export function project(cam: FrameCam, X: Vec3): Projection | null {
  const c = toCamera(cam, X)
  if (c[2] <= 1e-6) return null
  const x = c[0] / c[2]
  const y = c[1] / c[2]
  const r2 = x * x + y * y
  const s = 1 + cam.k1 * r2 + cam.k2 * r2 * r2
  // Far outside the image the polynomial folds back on itself; those points are not real projections.
  if (s < 0.3) return null
  return { uv: [cam.fx * x * s + cam.cx, cam.fy * y * s + cam.cy], depth: c[2] }
}

/** Invert the radial model by fixed-point iteration. */
export function undistort(cam: FrameCam, u: number, v: number): Vec2 {
  const xd = (u - cam.cx) / cam.fx
  const yd = (v - cam.cy) / cam.fy
  let x = xd
  let y = yd
  for (let i = 0; i < 12; i++) {
    const r2 = x * x + y * y
    const s = 1 + cam.k1 * r2 + cam.k2 * r2 * r2
    x = xd / s
    y = yd / s
  }
  return [x, y]
}

/** World-space ray through a pixel. */
export function unproject(cam: FrameCam, u: number, v: number): Ray {
  const [x, y] = undistort(cam, u, v)
  const { R } = cam
  // d = Rᵀ [x, y, 1]
  const d: [number, number, number] = [
    R[0] * x + R[3] * y + R[6],
    R[1] * x + R[4] * y + R[7],
    R[2] * x + R[5] * y + R[8],
  ]
  const n = Math.hypot(d[0], d[1], d[2]) || 1
  return { origin: cam.C, dir: [d[0] / n, d[1] / n, d[2] / n] }
}

export const rayAt = (ray: Ray, s: number): Vec3 => [
  ray.origin[0] + ray.dir[0] * s,
  ray.origin[1] + ray.dir[1] * s,
  ray.origin[2] + ray.dir[2] * s,
]

// ---- constraint intersections (preview only) ------------------------------

/** Ray parameter where the ray meets the plane `axis = value`, or null (parallel / behind). */
export function rayPlaneParam(ray: Ray, axis: 0 | 1 | 2, value: number): number | null {
  const d = ray.dir[axis]
  if (Math.abs(d) < 1e-9) return null
  const s = (value - ray.origin[axis]) / d
  return s > 0 ? s : null
}

/** Parameter along the ray closest to a world point (used for the player-joint depth). */
export function rayParamClosestTo(ray: Ray, P: Vec3): number {
  return (P[0] - ray.origin[0]) * ray.dir[0] + (P[1] - ray.origin[1]) * ray.dir[1] + (P[2] - ray.origin[2]) * ray.dir[2]
}

export interface RayPair {
  point: Vec3
  gap_m: number
  angleDeg: number
  s1: number
  s2: number
}

/** Midpoint of the shortest segment between two rays (null when near-parallel). */
export function closestBetweenRays(a: Ray, b: Ray): RayPair | null {
  const w: [number, number, number] = [a.origin[0] - b.origin[0], a.origin[1] - b.origin[1], a.origin[2] - b.origin[2]]
  const dot = (p: Vec3, q: Vec3) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
  const A = dot(a.dir, a.dir)
  const B = dot(a.dir, b.dir)
  const C = dot(b.dir, b.dir)
  const D = dot(a.dir, w)
  const E = dot(b.dir, w)
  const den = A * C - B * B
  if (den < 1e-10) return null
  const s1 = (B * E - C * D) / den
  const s2 = (A * E - B * D) / den
  const p1 = rayAt(a, s1)
  const p2 = rayAt(b, s2)
  const point: Vec3 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2]
  const angleDeg = (Math.acos(Math.max(-1, Math.min(1, B / Math.sqrt(A * C)))) * 180) / Math.PI
  return { point, gap_m: Math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]), angleDeg, s1, s2 }
}

// ---- epipolar polylines ---------------------------------------------------

export interface EpipolarSample {
  uv: Vec2
  /** Parameter along the source ray (metres from its origin). */
  s: number
  /** World height of the sample. */
  z: number
}

export interface EpipolarTick {
  uv: Vec2
  z: number
  s: number
  label: string
}

export interface Epipolar {
  /** Visible runs (camera-front samples), each a polyline in pixels. */
  runs: EpipolarSample[][]
  ticks: EpipolarTick[]
}

export const HEIGHT_TICKS_M: readonly number[] = [BALL_RADIUS_M, 1, 2, 5, 10, 20]
const EPI_NEAR_M = 2
const EPI_FAR_M = 250
const EPI_SAMPLES = 160

function tickLabel(z: number): string {
  return z < 0.5 ? "ground" : `${z} m`
}

/**
 * Project `ray` (from another view) into `cam`, sampling geometrically from
 * 2 m out to 250 m (or until it falls below the pitch), through the full
 * distortion model so it is a polyline, not an assumed-straight line.
 */
export function epipolarPolyline(ray: Ray, cam: FrameCam): Epipolar {
  const runs: EpipolarSample[][] = []
  let run: EpipolarSample[] = []
  const ratio = Math.pow(EPI_FAR_M / EPI_NEAR_M, 1 / (EPI_SAMPLES - 1))
  let sFar = EPI_FAR_M
  if (ray.dir[2] < -1e-9) {
    const sGround = (-1 - ray.origin[2]) / ray.dir[2]
    if (sGround > EPI_NEAR_M) sFar = Math.min(sFar, sGround)
  }
  for (let i = 0; i < EPI_SAMPLES; i++) {
    const s = EPI_NEAR_M * Math.pow(ratio, i)
    if (s > sFar) break
    const X = rayAt(ray, s)
    const p = project(cam, X)
    if (!p) {
      if (run.length) runs.push(run)
      run = []
      continue
    }
    run.push({ uv: p.uv, s, z: X[2] })
  }
  if (run.length) runs.push(run)

  const ticks: EpipolarTick[] = []
  for (const h of HEIGHT_TICKS_M) {
    const s = rayPlaneParam(ray, 2, h)
    if (s === null || s > sFar + 1e-6) continue
    const p = project(cam, rayAt(ray, s))
    if (p) ticks.push({ uv: p.uv, z: h, s, label: tickLabel(h) })
  }
  return { runs, ticks }
}

export interface Snap {
  uv: Vec2
  dist: number
  s: number
  z: number
}

/** Nearest point on the polyline runs to `uv` (perpendicular projection per segment). */
export function snapToEpipolar(epi: Epipolar, uv: Vec2): Snap | null {
  let best: Snap | null = null
  for (const run of epi.runs) {
    for (let i = 0; i < run.length - 1; i++) {
      const a = run[i]
      const b = run[i + 1]
      const dx = b.uv[0] - a.uv[0]
      const dy = b.uv[1] - a.uv[1]
      const len2 = dx * dx + dy * dy
      const tt = len2 < 1e-12 ? 0 : Math.max(0, Math.min(1, ((uv[0] - a.uv[0]) * dx + (uv[1] - a.uv[1]) * dy) / len2))
      const px = a.uv[0] + dx * tt
      const py = a.uv[1] + dy * tt
      const dist = Math.hypot(uv[0] - px, uv[1] - py)
      if (!best || dist < best.dist) {
        best = { uv: [px, py], dist, s: a.s + (b.s - a.s) * tt, z: a.z + (b.z - a.z) * tt }
      }
    }
  }
  return best
}

/** True when any part of the polyline lies inside the image rectangle. */
export function epipolarVisible(epi: Epipolar, size: Vec2, margin = 0): boolean {
  return epi.runs.some((run) =>
    run.some((p) => p.uv[0] >= -margin && p.uv[0] <= size[0] + margin && p.uv[1] >= -margin && p.uv[1] <= size[1] + margin),
  )
}

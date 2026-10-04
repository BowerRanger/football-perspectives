// The pending-pick state machine and its local (instant) preview geometry.
// idle -> picked in one view -> picked in several views -> committed (by the
// caller, which then dispatches `clear`). Nothing here touches the truth
// document: a click only ever creates a pending pick.
import {
  BALL_RADIUS_M,
  closestBetweenRays,
  epipolarPolyline,
  project,
  rayAt,
  rayParamClosestTo,
  rayPlaneParam,
  snapToEpipolar,
  unproject,
  type Epipolar,
  type Ray,
  type ShotCameras,
} from "./camera-model"
import type { ConstraintMode, KeySource, TriangulateRequest, Vec2, Vec3 } from "./types"

export type ClickMode = "triangulate" | "constraint" | "observation"

export interface PickParams {
  heightM: number
  /** Index into the scene's goal planes (or -1 for none). */
  planeIndex: number
  depthM: number
  playerId: string | null
  bone: string
}

export interface PickState {
  mode: ClickMode
  constraint: ConstraintMode
  params: PickParams
  /** Reference frame the picks belong to; a different frame starts afresh. */
  frame: number | null
  picks: Readonly<Record<string, Vec2>>
  /** Preferences. */
  snap: boolean
  autoCommit: boolean
}

export const DEFAULT_PARAMS: PickParams = { heightM: 1, planeIndex: 0, depthM: 40, playerId: null, bone: "r_foot" }

export const initialPickState: PickState = {
  mode: "triangulate",
  constraint: "ground",
  params: DEFAULT_PARAMS,
  frame: null,
  picks: {},
  snap: true,
  autoCommit: true,
}

export type PickAction =
  | { type: "pick"; frame: number; shotId: string; uv: Vec2 }
  | { type: "nudge"; shotId: string; dx: number; dy: number }
  | { type: "clear" }
  | { type: "mode"; mode: ClickMode }
  | { type: "constraint"; constraint: ConstraintMode }
  | { type: "params"; params: Partial<PickParams> }
  | { type: "prefs"; snap?: boolean; autoCommit?: boolean }
  | { type: "frame"; frame: number }

export function pickReducer(s: PickState, a: PickAction): PickState {
  switch (a.type) {
    case "pick": {
      const sameInstant = s.frame === a.frame
      const picks = { ...(sameInstant ? s.picks : {}), [a.shotId]: a.uv }
      return { ...s, frame: a.frame, picks }
    }
    case "nudge": {
      const cur = s.picks[a.shotId]
      if (!cur) return s
      return { ...s, picks: { ...s.picks, [a.shotId]: [cur[0] + a.dx, cur[1] + a.dy] } }
    }
    case "clear":
      return s.frame === null && !Object.keys(s.picks).length ? s : { ...s, frame: null, picks: {} }
    case "mode":
      return { ...s, mode: a.mode }
    case "constraint":
      return { ...s, mode: "constraint", constraint: a.constraint }
    case "params":
      return { ...s, params: { ...s.params, ...a.params } }
    case "prefs":
      return { ...s, snap: a.snap ?? s.snap, autoCommit: a.autoCommit ?? s.autoCommit }
    case "frame":
      // Picks are bound to the instant they were made at: scrubbing drops them.
      return s.frame !== null && s.frame !== a.frame ? { ...s, frame: null, picks: {} } : s
    default:
      return s
  }
}

export const pickedShots = (s: PickState): string[] => Object.keys(s.picks)

export type CommitKind = "triangulated" | "constraint" | "observation" | "none"

/** What Enter would do right now. */
export function naturalCommit(s: PickState): CommitKind {
  const n = pickedShots(s).length
  if (n === 0) return "none"
  if (s.mode === "observation") return "observation"
  if (n >= 2) return "triangulated"
  return s.mode === "constraint" ? "constraint" : "none"
}

export function constraintSource(mode: ConstraintMode): KeySource {
  switch (mode) {
    case "ground":
      return "ray_ground"
    case "height":
      return "ray_height"
    case "plane":
      return "ray_plane"
    case "depth":
      return "ray_depth"
    case "player":
      return "player"
  }
}

export interface GoalPlaneRef {
  id: string
  axis: "x" | "y" | "z"
  value: number
}

/** Constraint body for /triangulate and for the committed key. */
export function constraintPayload(s: PickState, planes: readonly GoalPlaneRef[]): TriangulateRequest["constraint"] {
  if (s.mode !== "constraint") return null
  switch (s.constraint) {
    case "ground":
      return { mode: "ground" }
    case "height":
      return { mode: "height", height_m: s.params.heightM }
    case "plane": {
      const p = planes[s.params.planeIndex] ?? planes[0]
      return p ? { mode: "plane", plane: { axis: p.axis, value: p.value } } : null
    }
    case "depth":
      return { mode: "depth", depth_m: s.params.depthM }
    case "player":
      return s.params.playerId ? { mode: "player", player_id: s.params.playerId, bone: s.params.bone } : null
  }
}

// ---- local preview --------------------------------------------------------

export interface PreviewRay {
  shotId: string
  ray: Ray
}

export interface Preview {
  rays: PreviewRay[]
  /** Epipolar line of each pick, drawn in every other view (key = target shot). */
  epipolar: Record<string, { from: string; epi: Epipolar }[]>
  ghost: Vec3 | null
  ghostSource: KeySource | null
  /** Shortest connecting segment of two skew rays (3-D view "gap"). */
  skew: { a: Vec3; b: Vec3; gap_m: number } | null
  rayAngleDeg: number | null
}

export interface PreviewInput {
  frame: number
  picks: Readonly<Record<string, Vec2>>
  cams: readonly ShotCameras[]
  state: PickState
  planes: readonly GoalPlaneRef[]
  joint: (playerId: string, bone: string, refFrame: number) => Vec3 | null
}

export const EMPTY_PREVIEW: Preview = { rays: [], epipolar: {}, ghost: null, ghostSource: null, skew: null, rayAngleDeg: null }

/** Instant geometry for the current picks (the server's /triangulate result replaces it when it lands). */
export function computePreview(inp: PreviewInput): Preview {
  const { frame, picks, cams, state } = inp
  const rays: PreviewRay[] = []
  for (const cam of cams) {
    const uv = picks[cam.shotId]
    if (!uv) continue
    const fc = cam.atRef(frame)
    if (fc) rays.push({ shotId: cam.shotId, ray: unproject(fc, uv[0], uv[1]) })
  }
  if (!rays.length) return EMPTY_PREVIEW

  const epipolar: Preview["epipolar"] = {}
  for (const pr of rays) {
    for (const cam of cams) {
      if (cam.shotId === pr.shotId) continue
      const fc = cam.atRef(frame)
      if (!fc) continue
      const epi = epipolarPolyline(pr.ray, fc)
      ;(epipolar[cam.shotId] ??= []).push({ from: pr.shotId, epi })
    }
  }

  let ghost: Vec3 | null = null
  let ghostSource: KeySource | null = null
  let skew: Preview["skew"] = null
  let angle: number | null = null
  if (rays.length >= 2) {
    const pair = closestBetweenRays(rays[0].ray, rays[1].ray)
    if (pair) {
      ghost = pair.point
      ghostSource = "triangulated"
      angle = pair.angleDeg
      skew = { a: rayAt(rays[0].ray, pair.s1), b: rayAt(rays[1].ray, pair.s2), gap_m: pair.gap_m }
    }
  } else if (state.mode === "constraint") {
    ghost = constrainRay(rays[0].ray, state, inp.planes, frame, inp.joint)
    ghostSource = ghost ? constraintSource(state.constraint) : null
  }
  return { rays, epipolar, ghost, ghostSource, skew, rayAngleDeg: angle }
}

/** Intersect one ray with the active constraint (client preview of constrain_ray). */
export function constrainRay(
  ray: Ray,
  s: PickState,
  planes: readonly GoalPlaneRef[],
  frame: number,
  joint: PreviewInput["joint"],
): Vec3 | null {
  switch (s.constraint) {
    case "ground": {
      const t = rayPlaneParam(ray, 2, BALL_RADIUS_M)
      return t === null ? null : rayAt(ray, t)
    }
    case "height": {
      const t = rayPlaneParam(ray, 2, s.params.heightM)
      return t === null ? null : rayAt(ray, t)
    }
    case "plane": {
      const p = planes[s.params.planeIndex] ?? planes[0]
      if (!p) return null
      const t = rayPlaneParam(ray, p.axis === "x" ? 0 : p.axis === "y" ? 1 : 2, p.value)
      return t === null ? null : rayAt(ray, t)
    }
    case "depth":
      return rayAt(ray, s.params.depthM)
    case "player": {
      if (!s.params.playerId) return null
      const j = joint(s.params.playerId, s.params.bone, frame)
      return j ? rayAt(ray, Math.max(0.5, rayParamClosestTo(ray, j))) : null
    }
  }
}

// ---- snapping -------------------------------------------------------------

export interface SnapResult {
  uv: Vec2
  snapped: boolean
  dist: number
  /** Height of the snapped epipolar sample, when snapped. */
  z: number | null
}

/**
 * Snap a click to the nearest epipolar polyline of another view's pick when
 * within `radiusNative` pixels. `alt` disables snapping for a free click.
 */
export function snapClick(
  uv: Vec2,
  lines: readonly { from: string; epi: Epipolar }[] | undefined,
  enabled: boolean,
  radiusNative: number,
  alt: boolean,
): SnapResult {
  if (!enabled || alt || !lines?.length) return { uv, snapped: false, dist: Infinity, z: null }
  let best: { uv: Vec2; dist: number; z: number } | null = null
  for (const l of lines) {
    const s = snapToEpipolar(l.epi, uv)
    if (s && (!best || s.dist < best.dist)) best = { uv: s.uv, dist: s.dist, z: s.z }
  }
  if (best && best.dist <= radiusNative) return { uv: best.uv, snapped: true, dist: best.dist, z: best.z }
  return { uv, snapped: false, dist: best?.dist ?? Infinity, z: null }
}

/** Project a world point into a view at the reference frame (null when behind the camera). */
export function projectInto(cam: ShotCameras, frame: number, X: Vec3): Vec2 | null {
  const fc = cam.atRef(frame)
  if (!fc) return null
  return project(fc, X)?.uv ?? null
}

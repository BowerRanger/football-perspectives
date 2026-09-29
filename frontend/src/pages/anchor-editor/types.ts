// Shapes of the anchor-editor API payloads (src/web/server.py) and local state.

export type Vec2 = readonly [number, number]
export type Vec3 = readonly [number, number, number]

export interface Landmark {
  name: string
  world_xyz: Vec3
}

export interface PitchLine {
  name: string
  world_segment?: readonly [Vec3, Vec3] | null
  world_direction?: Vec3 | null
  category?: string
}

export interface Stadium {
  id: string
  display_name?: string
}

export interface PointObs {
  name: string
  image_xy: Vec2
  world_xyz: Vec3
}

export interface LineObs {
  name: string
  image_segment: readonly [Vec2, Vec2]
  world_segment: readonly [Vec3, Vec3] | null
  world_direction: Vec3 | null
}

export interface AnchorFrame {
  points: readonly PointObs[]
  lines: readonly LineObs[]
}

/** frame index -> anchor. Always replaced, never mutated. */
export type AnchorMap = ReadonlyMap<number, AnchorFrame>

export interface AnchorsResponse {
  clip_id?: string
  image_size?: Vec2
  stadium?: string | null
  anchors?: {
    frame: number
    landmarks?: PointObs[]
    lines?: LineObs[]
  }[]
}

export interface CameraFrame {
  frame: number
  K: number[][]
  R: number[][]
  t?: number[] | null
  confidence?: number
  is_anchor?: boolean
}

export interface CameraTrack {
  clip_id?: string
  fps?: number
  t_world?: number[]
  distortion?: number[]
  frames: CameraFrame[]
}

export interface DetectedLine {
  image_segment: readonly [Vec2, Vec2]
}

export type DetectedLinesByFrame = Record<string, { lines?: DetectedLine[] }>

export interface SnapResult {
  xy: Vec2
  snapped: boolean
  mode_used: string
  confidence: number
}

export type PaletteMode = "points" | "lines"

export interface ViewOptions {
  snap: boolean
  pitch: boolean
  detected: boolean
  labels: boolean
  anchors: boolean
}

export const DEFAULT_VIEW: ViewOptions = {
  snap: true,
  pitch: true,
  detected: false,
  labels: false,
  anchors: true,
}

export const EMPTY_ANCHOR: AnchorFrame = { points: [], lines: [] }

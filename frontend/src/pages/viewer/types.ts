// Typed payloads for the 3D viewer's endpoints (src/web/server.py).

export type Vec3 = readonly [number, number, number]
export type Mat3 = readonly (readonly number[])[]

export interface MatchKits {
  home_primary?: string | null
  away_primary?: string | null
  referee?: string | null
}

export interface MatchInfo {
  home_team?: string | null
  away_team?: string | null
  home_score?: number | null
  away_score?: number | null
  venue?: string | null
  moment?: { minute?: number | null; added_time?: number | null } | null
  kits?: MatchKits | null
}

export interface SceneMetadata {
  match?: MatchInfo | null
}

export interface CameraFrameRaw {
  frame: number
  K?: Mat3 | null
  R?: Mat3 | null
  t?: Vec3 | null
}

export interface CameraTrackRaw {
  clip_id?: string
  fps?: number
  t_world?: Vec3
  image_size?: [number, number]
  frames?: CameraFrameRaw[]
}

export interface TrackedPose {
  K: Mat3
  R: Mat3
  t: Vec3
}

export interface PlayerRow {
  player_id: string
  player_name?: string | null
  shot_id?: string
  contributing_shots?: string[]
}

export interface PlayerPreview {
  player_id: string
  team?: string | null
  betas?: number[] | null
  frames?: number[]
  root_t?: (Vec3 | null)[]
  root_R?: (Mat3 | null)[]
  thetas?: ((readonly number[])[] | null)[]
  confidence?: number[]
}

export interface BallFrameRaw {
  frame: number
  world_xyz?: Vec3 | null
  state?: string
}

/** One player as consumed by the engine (frame lookup pre-indexed). */
export interface PlayerTrack {
  id: string
  name: string
  team: string
  /** Data colour: kit colour for known teams, palette colour otherwise. */
  colour: number
  betas: number[]
  frames: number[]
  frameIndex: Map<number, number>
  rootT: (Vec3 | null)[]
  rootR: (Mat3 | null)[]
  thetas: ((readonly number[])[] | null)[]
}

export interface SmplModel {
  vTemplate: Float32Array
  faces: Uint32Array
  skinIndex: Uint16Array
  skinWeight: Float32Array
  jointPositions: Float32Array
  parents: number[]
  nVerts: number
  nBetas: number
  shapedirs?: Float32Array
  jointShapedirs?: Float32Array
}

export interface KitColours {
  A: number
  B: number
  referee: number
  unknown: number
}

export interface SceneData {
  fps: number
  totalFrames: number
  players: PlayerTrack[]
  ball: Map<number, Vec3>
  hasBall: boolean
  ballRadius: number
  track: Map<number, TrackedPose> | null
  trackClipId: string
  trackImageHeight: number
  smpl: SmplModel | null
  match: MatchInfo | null
  colours: KitColours
  playerSource: "refined_poses" | "hmr_world"
}

export type CameraMode = "broadcast" | "tactical" | "behind-goal" | "tracked"

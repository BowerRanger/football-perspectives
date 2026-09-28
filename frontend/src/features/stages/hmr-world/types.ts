// Payload shapes for the hmr_world / refined_poses preview endpoints.

export interface PlayerRef {
  shot_id?: string
  player_id: string
  player_name?: string | null
}

/** /hmr_world/preview and /refined_poses/preview (extra fields ignored). */
export interface PosePreview {
  player_id: string
  shot_id?: string
  frames: number[]
  root_t: number[][]
  confidence: number[]
  contributing_shots?: string[]
}

export interface Kp2dFrame {
  frame: number
  keypoints: number[][]
}

export interface Kp2dPreview {
  player_id: string
  shot_id?: string
  frames: Kp2dFrame[]
}

export interface PlayersResponse<T extends PlayerRef = PlayerRef> {
  players: T[]
}

/** A player plus the fixed palette colour used everywhere for them. */
export type Coloured<T> = T & { colour: string }

export interface TrajectoryPlayer {
  pid: string
  label: string
  colour: string
  byFrame: Map<number, { pos: number[]; conf: number }>
  firstFrame: number
  lastFrame: number
}

export interface CameraSample {
  pos: [number, number]
  z: number
  fwd: [number, number]
}

export interface CameraTrackResponse {
  fps?: number
  t_world?: number[]
  frames?: { frame: number; R?: number[][]; t?: number[] }[]
}

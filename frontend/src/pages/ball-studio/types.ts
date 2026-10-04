// Ball Studio payloads. Coded against docs/superpowers/specs/2026-10-04-ball-studio-api.md
// (plus the additive fields the implemented router returns: goal_planes,
// frame_range, skew_gap_cm, reprojected_uv, polyline_uv, offsets, meta.status).

export type Vec3 = readonly [number, number, number]
export type Vec2 = readonly [number, number]

// ---- Truth document -------------------------------------------------------

export type Outcome = "goal" | "no_goal" | "unknown"
export type KeySource =
  | "triangulated"
  | "ray_ground"
  | "ray_height"
  | "ray_plane"
  | "ray_depth"
  | "player"
  | "manual"
export type SegmentKind = "flight" | "roll" | "carried" | "linear" | "static"
export type EventKind = "touch" | "bounce" | "post" | "crossbar" | "net" | "line_cross" | "out" | "keeper_save"
export type TruthStatus = "draft" | "reviewed"

export interface Observation {
  shot_id: string
  shot_frame: number
  uv: Vec2
}

export interface PlaneConstraint {
  axis: "x" | "y" | "z"
  value: number
}

export interface KeyConstraint {
  height_m: number | null
  plane: PlaneConstraint | null
  depth_m: number | null
  player_id: string | null
  bone: string | null
  offset: Vec3 | null
}

export interface TruthKey {
  id: string
  frame: number
  xyz: Vec3
  source: KeySource
  constraint: KeyConstraint
  observations: Observation[]
  residual_px: Record<string, number>
  note: string
}

export interface SegmentParams {
  drag: boolean
  cd: number | null
  magnus: "auto" | "off"
  player_id: string | null
  bone: string | null
}

export interface TruthSegment {
  from: string
  to: string
  kind: SegmentKind
  params: SegmentParams
}

export interface TruthEvent {
  frame: number
  kind: EventKind
  player_id: string | null
  bone: string | null
  note: string
}

export interface TruthMeta {
  authored_by: string
  updated_at: string | null
  notes: string
  status: TruthStatus
}

export interface TruthDoc {
  version: 1
  group_id: string
  reference_shot: string
  fps: number
  shots: { shot_id: string; frame_offset: number }[]
  outcome: Outcome
  keys: TruthKey[]
  segments: TruthSegment[]
  observations: Observation[]
  events: TruthEvent[]
  meta: TruthMeta
}

export interface TruthResponse {
  exists: boolean
  truth: TruthDoc
  dense: unknown
}

export interface PutTruthResponse {
  ok: boolean
  updated_at: string
  history_file: string | null
  solve_ok: boolean
  n_flags: number
}

// ---- Groups / scene -------------------------------------------------------

export interface GroupShot {
  shot_id: string
  frame_offset: number
  n_frames: number
  fps: number
  image_size: Vec2
  frame_range?: Vec2
  excluded: boolean
  video_url: string
  frame_url: string
}

export interface GroupInfo {
  group_id: string
  label: string
  reference_shot: string
  fps: number
  ref_frame_range: Vec2
  shots: GroupShot[]
  has_truth: boolean
  truth_updated_at: string | null
  n_keys: number
  outcome: Outcome
  status?: TruthStatus
}

export interface SceneShot extends GroupShot {
  distortion: Vec2
  camera_centre: Vec3
  /** Shot-local frame index of row i. */
  frames: number[]
  K: number[][]
  R: number[][]
  t: number[][]
  confidence: number[]
}

export interface GoalPlane {
  id: string
  axis: "x" | "y" | "z"
  value: number
}

export interface SceneGoals {
  goal_line_x_near: number
  goal_line_x_far: number
  post_y_left: number
  post_y_right: number
  crossbar_z: number
  net_depth: number
  pitch: { length_m: number; width_m: number }
  goal_planes?: GoalPlane[]
}

export interface ScenePlayer {
  player_id: string
  /** Reference frames. */
  frames: number[]
  root: number[][]
  joints: Record<string, number[][]>
  confidence: number[]
}

export interface PipelineTrack {
  shot_id: string
  frames: number[]
  shot_frames: number[]
  xyz: (number[] | null)[]
  state: string[]
  confidence: number[]
}

export interface PipelineAnchor {
  shot_id: string
  ref_frame: number
  shot_frame: number
  kind: string
  uv: Vec2
  source: string
}

export interface Scene {
  group_id: string
  reference_shot: string
  fps: number
  ref_frame_range: Vec2
  shots: SceneShot[]
  goals: SceneGoals
  bones: string[]
  players: ScenePlayer[]
  pipeline_tracks: PipelineTrack[]
  pipeline_anchors: PipelineAnchor[]
}

// ---- Solve / triangulate --------------------------------------------------

export type Level = "ok" | "warn" | "error"

export interface SolveFlag {
  level: "warn" | "error"
  code: string
  frame?: number
  ref?: { segment?: number; key?: string }
  message: string
}

export interface SolvedKey {
  id: string
  frame: number
  xyz: Vec3
  source: KeySource
  residual_px: Record<string, number>
  ray_angle_deg?: number | null
  status: Level
  messages: string[]
}

export interface SolvedSegment {
  index: number
  from: string
  to: string
  kind: SegmentKind
  auto: boolean
  frame_range: Vec2
  params: Record<string, unknown>
  rms_obs_px: number | null
  n_soft_obs: number
  max_speed_m_s: number | null
  status: Level
}

export interface SolveDense {
  frames: number[]
  xyz: number[][]
  segment: number[]
  kind: SegmentKind[]
  speed_m_s: number[]
}

export interface SolveProjection {
  frames: number[]
  shot_frames: number[]
  uv: (number[] | null)[]
  depth_m: (number | null)[]
}

export interface SolveObservation {
  kind: "key" | "soft"
  key_id?: string
  index?: number
  shot_id: string
  shot_frame: number
  uv: Vec2
  projected_uv: Vec2 | null
  residual_px: number | null
}

export interface SolveResult {
  ok: boolean
  keys: SolvedKey[]
  segments: SolvedSegment[]
  dense: SolveDense
  projections: Record<string, SolveProjection>
  observations: SolveObservation[]
  flags: SolveFlag[]
  stats: { n_keys: number; n_segments: number; n_dense: number; max_key_residual_px?: number | null; max_speed_m_s?: number | null }
}

export type ConstraintMode = "ground" | "height" | "plane" | "depth" | "player"

export interface TriangulateRequest {
  frame: number
  observations: Observation[]
  constraint: (Partial<KeyConstraint> & { mode: ConstraintMode }) | null
  offsets?: Record<string, number>
}

export interface TriRay {
  shot_id: string
  origin: Vec3
  direction: Vec3
}

export interface TriEpipolar {
  shot_id: string
  shot_frame: number
  segment_uv: [Vec2, Vec2] | null
  polyline_uv?: Vec2[]
}

export interface TriangulateResult {
  ok: boolean
  reason?: string
  source?: KeySource | "ray"
  xyz: Vec3 | null
  residual_px: Record<string, number>
  max_residual_px?: number | null
  ray_angle_deg?: number
  skew_gap_cm?: number
  reprojected_uv?: Record<string, Vec2>
  rays?: TriRay[]
  epipolar?: TriEpipolar[]
  flags: { level: "warn" | "error"; code: string; message: string }[]
  offsets_used?: Record<string, number>
}

// ---- Editor-local ---------------------------------------------------------

export type Selection =
  | { type: "key"; id: string }
  | { type: "event"; index: number }
  | { type: "observation"; index: number }
  | { type: "segment"; from: string }
  | null

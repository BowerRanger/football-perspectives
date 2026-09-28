// Types + normalisers for the ball-anchor sidecar contract
// (`BallAnchorPayload` in src/web/server.py). Every field the server
// persists is carried through load -> edit -> save so a save can never
// erase operator data (chains, dismissals, end frames, landmarks, spin...).

export interface BallAnchor {
  frame: number
  image_xy: [number, number] | null
  state: string
  player_id: string | null
  bone: string | null
  goal_element: string | null
  touch_type: string | null
  spin: string | null
  confidence: number
  end_frame: number | null
  landmark: string | null
}

export interface DismissedAuto {
  frame: number
  state: string
  player_id: string | null
  bone: string | null
}

export interface AutoAnchor {
  frame: number
  image_xy: [number, number] | null
  state: string
  player_id: string | null
  bone: string | null
  goal_element: string | null
  touch_type: string | null
  confidence: number | null
}

export interface BallAnchorPayload {
  clip_id: string
  image_size: [number, number]
  anchors: BallAnchor[]
  shot_chains: number[][]
  dismissed_auto: DismissedAuto[]
}

export interface BallQualityFrame {
  frame: number
  confidence?: number | null
  gap_fill?: boolean
}

export interface AnnotateNextItem {
  start: number
  end: number
  reason: string
}

export interface BallQuality {
  n_frames?: number
  frames?: BallQualityFrame[]
  events?: { frame: number }[]
  annotate_next?: AnnotateNextItem[]
}

export interface CameraFrame {
  frame: number
  R?: number[][]
  K?: number[][]
  t?: number[] | null
}

export interface CameraTrackData {
  fps?: number
  image_size?: number[]
  t_world?: number[] | null
  frames?: CameraFrame[]
}

export interface PlayerOption {
  player_id: string
  player_name?: string | null
}

export interface PreviewFrame {
  frame: number
  state: string
  world_xyz?: number[] | null
}

export interface ChainWarning {
  frames: number[]
  warnings: { detail: string }[]
}

export interface PreviewResult {
  frames: PreviewFrame[]
  shot_chain_warnings?: ChainWarning[]
}

type Rec = Record<string, unknown>

const asRec = (v: unknown): Rec => (v && typeof v === "object" ? (v as Rec) : {})
const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null)
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null)

function xy(v: unknown): [number, number] | null {
  if (!Array.isArray(v) || v.length < 2) return null
  const a = Number(v[0])
  const b = Number(v[1])
  return Number.isFinite(a) && Number.isFinite(b) ? [a, b] : null
}

export function normalizeAnchor(raw: unknown): BallAnchor {
  const r = asRec(raw)
  return {
    frame: Number(r.frame ?? 0),
    image_xy: xy(r.image_xy),
    state: String(r.state ?? ""),
    player_id: str(r.player_id),
    bone: str(r.bone),
    goal_element: str(r.goal_element),
    touch_type: str(r.touch_type),
    spin: str(r.spin),
    confidence: num(r.confidence) ?? 1,
    end_frame: num(r.end_frame),
    landmark: str(r.landmark),
  }
}

export function normalizeAuto(raw: unknown): AutoAnchor {
  const r = asRec(raw)
  return {
    frame: Number(r.frame ?? 0),
    image_xy: xy(r.image_xy),
    state: String(r.state ?? ""),
    player_id: str(r.player_id),
    bone: str(r.bone),
    goal_element: str(r.goal_element),
    touch_type: str(r.touch_type),
    confidence: num(r.confidence),
  }
}

export function normalizeDismissed(raw: unknown): DismissedAuto {
  const r = asRec(raw)
  return {
    frame: Number(r.frame ?? 0),
    state: String(r.state ?? ""),
    player_id: str(r.player_id),
    bone: str(r.bone),
  }
}

export const normalizeChain = (raw: unknown): number[] =>
  Array.isArray(raw) ? raw.map(Number).filter((n) => Number.isFinite(n)) : []

export function dismissKey(a: { frame: number; state: string; player_id: string | null; bone: string | null }): string {
  return `${a.frame}|${a.state}|${a.player_id ?? ""}|${a.bone ?? ""}`
}

/** Project a pitch-space point through a camera frame (pipeline R/t/K, OpenCV). */
export function projectWorld(
  xyz: number[],
  cf: CameraFrame | undefined,
  tFallback: number[] | null | undefined,
): [number, number] | null {
  const R = cf?.R
  const K = cf?.K
  const t = cf?.t ?? tFallback
  if (!R || !K || !t) return null
  const [x, y, z] = xyz
  const cx = R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0]
  const cy = R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1]
  const cz = R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2]
  if (cz <= 0) return null
  return [(K[0][0] * cx + K[0][1] * cy + K[0][2] * cz) / cz, (K[1][0] * cx + K[1][1] * cy + K[1][2] * cz) / cz]
}

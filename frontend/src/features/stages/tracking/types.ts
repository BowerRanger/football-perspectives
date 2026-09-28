// Payload shapes for /tracking/* (see src/web/server.py) plus the derived
// "player group" the editor renders one row for.

export interface TrackSummary {
  track_id: string
  class_name: string
  team?: string | null
  player_id?: string | null
  player_name?: string | null
  frame_count: number
  frame_range: number[]
  mean_confidence?: number
}

export interface TrackingPreview {
  shot_id: string
  tracks: TrackSummary[]
}

export interface FrameBox {
  track_id: string
  player_id?: string | null
  player_name?: string | null
  class_name: string
  team?: string | null
  bbox: number[]
  confidence?: number | null
}

export interface FrameEntry {
  frame: number
  boxes: FrameBox[]
}

export interface TrackingFrames {
  shot_id: string
  frames: FrameEntry[]
  fps?: number | null
}

export interface RosterEntry {
  team: string
  name: string
}

export interface MatchInfo {
  roster?: RosterEntry[]
}

/** One list row = one player (tracks merged under the same player_id). */
export interface PlayerGroup {
  key: string
  tracks: TrackSummary[]
  frameCount: number
  frameRange: [number, number]
  name: string
  team: string
}

export const IGNORE_NAME = "ignore"

/** Team colours are data (bbox strokes / badges), not chrome. */
export const TEAM_COLORS: Record<string, string> = {
  A: "#3b82f6",
  B: "#ef4444",
  referee: "#f59e0b",
  unknown: "#94a3b8",
}

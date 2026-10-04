// Payload shapes for the shots manifest, classifier sidecar and sync map,
// plus the joined "model" the board and sync editor render from.

export interface Shot {
  id: string
  start_frame: number
  end_frame: number
  start_time: number
  end_time: number
  clip_file: string
  speed_factor: number
  kind: string
  excluded: boolean
  exclude_reason: string
  group_id: string
  source_start_s: number
  source_end_s: number
  /** True once the clip was re-encoded to real time (native clip kept). */
  retimed?: boolean
  /** Frames in the native (pre-retime) clip; 0 when never retimed. */
  native_frames?: number
}

export interface ShotGroup {
  id: string
  label: string
  shot_ids: string[]
  boundary_rule: string
  boundary_confidence: number
}

export interface Manifest {
  shots: Shot[]
  groups?: ShotGroup[]
}

export interface ShotFeatures {
  scale?: string
  pitch_ratio_median?: number
  speed_factor?: number
}

export type FeaturesMap = Record<string, ShotFeatures>

export interface SyncAlignment {
  shot_id: string
  frame_offset: number
  method: string
  confidence: number
  /** Reference frames per shot frame (1 = real time, 0.34 = slow motion). */
  playback_rate?: number
}

export interface GroupSync {
  group_id: string
  reference_shot: string
  alignments: SyncAlignment[]
}

export interface SyncMap {
  groups: GroupSync[]
}

export type ReplaySyncDecision =
  | "applied"
  | "applied_retimed"
  | "kept_manual"
  | "low_confidence"
  | "ramp_not_applied"
  | "no_camera"
  | "no_tracks"

export interface ReplaySyncEstimate {
  rate: number
  offset: number
  confidence: number
  cost_m?: number
  coverage?: number
  ramp?: boolean
  rate_first?: number
  rate_second?: number
  /** Live frames the geometric match could use. */
  live_window_frames?: number
  /** Relative 1 sigma of the rate, e.g. 0.08 = +-8 %. */
  rate_uncertainty?: number
}

export interface ReplaySyncMember {
  shot_id: string
  against?: string
  estimate: ReplaySyncEstimate | null
  decision: ReplaySyncDecision
  reason: string
  /** Rate is only as precise as a short live window allows: confirm with marked moments. */
  approximate?: boolean
}

/** `GET /api/replay-sync` (shots/replay_sync.json). */
export interface ReplaySyncReport {
  version: number
  groups: { group_id: string; reference_shot: string; members: ReplaySyncMember[] }[]
}

export interface ShotView extends Shot {
  features: ShotFeatures | null
}

export interface GroupView {
  id: string
  label: string
  boundary_rule: string
  boundary_confidence: number
  members: ShotView[]
  sync: GroupSync | null
}

export interface ShotModel {
  groups: GroupView[]
  ungrouped: ShotView[]
  ungroupedSync: GroupSync | null
  dropped: ShotView[]
  activeIds: string[]
  groupIds: string[]
}

export interface ShotUpdate {
  shot_id: string
  group_id?: string
  excluded?: boolean
  exclude_reason?: string
}

/** Joins manifest shots, the features sidecar and the sync map. */
export function buildShotModel(manifest: Manifest, features: FeaturesMap, sync: SyncMap): ShotModel {
  const byId = new Map<string, ShotView>()
  for (const s of manifest.shots) byId.set(s.id, { ...s, features: features[s.id] ?? null })
  const syncByGroup = new Map<string, GroupSync>()
  for (const g of sync.groups ?? []) syncByGroup.set(g.group_id, g)

  const groups: GroupView[] = (manifest.groups ?? []).map((g) => ({
    id: g.id,
    label: g.label,
    boundary_rule: g.boundary_rule,
    boundary_confidence: g.boundary_confidence,
    members: g.shot_ids.map((sid) => byId.get(sid)).filter((s): s is ShotView => !!s && !s.excluded),
    sync: syncByGroup.get(g.id) ?? null,
  }))
  const pick = (pred: (s: Shot) => boolean) =>
    manifest.shots.filter(pred).map((s) => byId.get(s.id)).filter((s): s is ShotView => !!s)
  return {
    groups,
    ungrouped: pick((s) => !s.excluded && !s.group_id),
    ungroupedSync: syncByGroup.get("") ?? null,
    dropped: pick((s) => s.excluded),
    activeIds: manifest.shots.filter((s) => !s.excluded).map((s) => s.id),
    groupIds: (manifest.groups ?? []).map((g) => g.id),
  }
}

export function nextFreeGroupId(groupIds: string[]): string {
  const taken = new Set(groupIds)
  for (let i = 1; i < 1000; i++) {
    const gid = `g${String(i).padStart(2, "0")}`
    if (!taken.has(gid)) return gid
  }
  return `g${Date.now() % 100000}`
}

export function fmtClock(seconds: number): string {
  if (!(seconds >= 0)) return "?"
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, "0")}`
}

/** Identity colour per group so tile ribbons match their card. Data colours. */
const GROUP_PALETTE = [
  "#6366f1", "#22c55e", "#f59e0b", "#06b6d4", "#ec4899",
  "#84cc16", "#a855f7", "#f97316", "#14b8a6", "#e11d48",
] as const

export function groupColor(groupId: string, groupIds: string[]): string {
  if (!groupId) return "#64748b"
  const idx = groupIds.indexOf(groupId)
  return GROUP_PALETTE[(idx >= 0 ? idx : groupIds.length) % GROUP_PALETTE.length]
}

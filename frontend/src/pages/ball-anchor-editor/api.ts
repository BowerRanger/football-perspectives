// Endpoint calls for the ball editor. Paths/params are identical to the
// legacy ball_anchor_editor.html. Every read uses getJson: the server answers
// 200 with an empty payload when a stage hasn't produced output, so a
// rejection is always a real failure. Callers decide whether that failure is
// blocking (saved anchors, camera track) or an optional overlay that only
// earns a muted notice (auto anchors, kp2d players, quality strip, suggestions).

import { errorMessage, getJson, getJsonOrNull, postJson, qs } from "@/lib/api"
import {
  normalizeAnchor,
  normalizeAuto,
  normalizeChain,
  normalizeDismissed,
  type AutoAnchor,
  type BallAnchor,
  type BallAnchorPayload,
  type BallQuality,
  type CameraTrackData,
  type DismissedAuto,
  type PlayerOption,
  type PreviewResult,
} from "./types"

const enc = encodeURIComponent

export interface LoadedShot {
  fps: number | null
  imageSize: [number, number]
  camera: CameraTrackData | null
  anchors: BallAnchor[]
  shotChains: number[][]
  dismissedAuto: DismissedAuto[]
  autoAnchors: AutoAnchor[]
  players: PlayerOption[]
  /** Optional overlays that failed to load (muted notice, never blocking). */
  notices: string[]
}

export const videoUrl = (shot: string) => `/api/video/${enc(shot)}`

function pickImageSize(cam: CameraTrackData | null, saved: unknown): [number, number] {
  const c = cam?.image_size
  if (c && c.length >= 2 && c[0] > 0) return [Number(c[0]), Number(c[1])]
  if (Array.isArray(saved) && Number(saved[0]) > 0) return [Number(saved[0]), Number(saved[1])]
  return [1280, 720]
}

interface RawAnchorSet {
  image_size?: unknown
  anchors?: unknown[]
  shot_chains?: unknown[]
  dismissed_auto?: unknown[]
}

/** Wrap a blocking read so its error says which payload failed. */
async function required<T>(label: string, p: Promise<T>): Promise<T> {
  try {
    return await p
  } catch (err) {
    throw new Error(`${label}: ${errorMessage(err)}`)
  }
}

/** Optional read: resolves to null and records a notice on failure. */
async function optional<T>(label: string, p: Promise<T>, notices: string[]): Promise<T | null> {
  try {
    return await p
  } catch (err) {
    notices.push(`${label} unavailable (${errorMessage(err)}).`)
    return null
  }
}

export async function loadShot(shot: string): Promise<LoadedShot> {
  const notices: string[] = []
  const [camera, saved, auto, players] = await Promise.all([
    required("camera track", getJson<CameraTrackData>(`/camera/track${qs({ shot })}`)),
    required("saved anchors", getJson<RawAnchorSet>(`/ball-anchors/${enc(shot)}`)),
    optional("Auto-detected anchors", getJson<{ anchors?: unknown[] }>(`/ball-anchors/${enc(shot)}/auto`), notices),
    optional(
      "Player list for touch authoring",
      getJson<{ players?: PlayerOption[] }>(`/hmr_world/kp2d_players${qs({ shot })}`),
      notices,
    ),
  ])
  // An empty track (camera stage not run) is valid: fps/size fall back to the saved set.
  const cam = camera?.frames?.length ? camera : null
  return {
    fps: cam?.fps ?? null,
    imageSize: pickImageSize(cam, saved?.image_size),
    camera: cam,
    anchors: (saved?.anchors ?? []).map(normalizeAnchor),
    shotChains: (saved?.shot_chains ?? []).map(normalizeChain),
    dismissedAuto: (saved?.dismissed_auto ?? []).map(normalizeDismissed),
    autoAnchors: (auto?.anchors ?? []).map(normalizeAuto),
    players: players?.players ?? [],
    notices,
  }
}

/** Quality strip data. Throws on failure; the caller shows a notice. */
export const loadQuality = (shot: string) => getJson<BallQuality>(`/ball-quality/${enc(shot)}`)

export const saveAnchors = (shot: string, payload: BallAnchorPayload) =>
  postJson<{ saved: boolean; count: number; labels_recorded?: number }>(`/ball-anchors/${enc(shot)}`, payload)

export const previewAnchors = (shot: string, payload: BallAnchorPayload) =>
  postJson<PreviewResult>(`/ball-anchors/${enc(shot)}/preview`, payload)

export interface JointHit {
  player_id: string
  bone: string
}

// The three suggestion helpers throw on failure (the server answers 200 with
// an empty list when it has nothing to suggest); placement.ts turns a failure
// into an operator-visible message and never blocks manual choices.
export async function jointsNear(shot: string, frame: number, u: number, v: number): Promise<JointHit[]> {
  const res = await getJson<{ joints?: JointHit[] }>(`/joints-near${qs({ shot, frame, u, v, r: 40 })}`)
  return res?.joints ?? []
}

export async function goalElementSuggest(shot: string, frame: number, u: number, v: number): Promise<string[]> {
  const res = await getJson<{ candidates?: { element: string }[] }>(`/goal-element-suggest${qs({ shot, frame, u, v })}`)
  return (res?.candidates ?? []).map((c) => c.element)
}

export interface PitchFixSuggestion {
  name: string
  distance_m: number
}

export async function pitchFixSuggest(shot: string, frame: number, u: number, v: number): Promise<PitchFixSuggestion[]> {
  const res = await getJson<{ suggestions?: PitchFixSuggestion[] }>(`/pitch-fix-suggest${qs({ shot, frame, u, v })}`)
  return res?.suggestions ?? []
}

export interface ShotOption {
  id: string
  anchorCount: number | null
}

/**
 * Shot ids from the manifest-aware picker endpoint, with saved anchor counts.
 * The per-shot count is decorative (a dropdown suffix), so a failed count
 * reads as "unknown" (null) rather than blocking the picker.
 */
export async function loadShotOptions(): Promise<ShotOption[]> {
  const list = await getJson<{ shots?: string[] }>("/api/output/shots")
  const ids = list.shots ?? []
  return Promise.all(
    ids.map(async (id) => {
      const set = await getJsonOrNull<{ anchors?: unknown[] }>(`/ball-anchors/${enc(id)}`)
      return { id, anchorCount: set ? (set.anchors?.length ?? 0) : null }
    }),
  )
}

export interface BallPreviewTrack {
  clip_id?: string
  fps?: number
  frames?: { frame: number; world_xyz: number[] | null; state: string; confidence?: number | null }[]
  flight_segments?: {
    id: number
    frame_range?: number[]
    fit_residual_px?: number | null
    parabola?: {
      spin_omega_rad_s?: number | null
      spin_axis_world?: number[] | null
      spin_confidence?: number | null
    } | null
  }[]
}

/** Predicted ball track. Throws on failure (200 + empty frames = stage not run). */
export const loadBallTrack = (shot: string) => getJson<BallPreviewTrack>(`/ball/preview${qs({ shot })}`)

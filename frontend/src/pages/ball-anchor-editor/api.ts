// Endpoint calls for the ball editor. Paths/params are identical to the
// legacy ball_anchor_editor.html; suggestion helpers are best-effort (they
// resolve to empty results instead of throwing, as the server never 500s).

import { getJson, getJsonOrNull, postJson, qs } from "@/lib/api"
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

export async function loadShot(shot: string): Promise<LoadedShot> {
  const [camera, saved, auto, players] = await Promise.all([
    getJsonOrNull<CameraTrackData>(`/camera/track${qs({ shot })}`),
    getJsonOrNull<RawAnchorSet>(`/ball-anchors/${enc(shot)}`),
    getJsonOrNull<{ anchors?: unknown[] }>(`/ball-anchors/${enc(shot)}/auto`),
    getJsonOrNull<{ players?: PlayerOption[] }>(`/hmr_world/kp2d_players${qs({ shot })}`),
  ])
  return {
    fps: camera?.fps ?? null,
    imageSize: pickImageSize(camera, saved?.image_size),
    camera,
    anchors: (saved?.anchors ?? []).map(normalizeAnchor),
    shotChains: (saved?.shot_chains ?? []).map(normalizeChain),
    dismissedAuto: (saved?.dismissed_auto ?? []).map(normalizeDismissed),
    autoAnchors: (auto?.anchors ?? []).map(normalizeAuto),
    players: players?.players ?? [],
  }
}

export const loadQuality = (shot: string) => getJsonOrNull<BallQuality>(`/ball-quality/${enc(shot)}`)

export const saveAnchors = (shot: string, payload: BallAnchorPayload) =>
  postJson<{ saved: boolean; count: number; labels_recorded?: number }>(`/ball-anchors/${enc(shot)}`, payload)

export const previewAnchors = (shot: string, payload: BallAnchorPayload) =>
  postJson<PreviewResult>(`/ball-anchors/${enc(shot)}/preview`, payload)

export interface JointHit {
  player_id: string
  bone: string
}

export async function jointsNear(shot: string, frame: number, u: number, v: number): Promise<JointHit[]> {
  const res = await getJsonOrNull<{ joints?: JointHit[] }>(`/joints-near${qs({ shot, frame, u, v, r: 40 })}`)
  return res?.joints ?? []
}

export async function goalElementSuggest(shot: string, frame: number, u: number, v: number): Promise<string[]> {
  const res = await getJsonOrNull<{ candidates?: { element: string }[] }>(
    `/goal-element-suggest${qs({ shot, frame, u, v })}`,
  )
  return (res?.candidates ?? []).map((c) => c.element)
}

export interface PitchFixSuggestion {
  name: string
  distance_m: number
}

export async function pitchFixSuggest(shot: string, frame: number, u: number, v: number): Promise<PitchFixSuggestion[]> {
  const res = await getJsonOrNull<{ suggestions?: PitchFixSuggestion[] }>(
    `/pitch-fix-suggest${qs({ shot, frame, u, v })}`,
  )
  return res?.suggestions ?? []
}

export interface ShotOption {
  id: string
  anchorCount: number | null
}

/** Shot ids from the manifest-aware picker endpoint, with saved anchor counts. */
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

export const loadBallTrack = (shot: string) => getJsonOrNull<BallPreviewTrack>(`/ball/preview${qs({ shot })}`)

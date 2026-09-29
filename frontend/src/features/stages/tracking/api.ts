import { ApiError, deleteJson, getJson, getJsonOrNull, postJson, qs } from "@/lib/api"
import type { MatchInfo, TrackingFrames, TrackingPreview } from "./types"

// Every path/payload below is identical to the legacy dashboard.

const enc = encodeURIComponent

export const listTrackedShots = () => getJson<{ shots: string[] }>("/tracking/shots")
export const fetchPreview = (shot: string) => getJson<TrackingPreview>(`/tracking/preview${qs({ shot_id: shot })}`)
export const fetchFrames = (shot: string) => getJson<TrackingFrames>(`/tracking/frames${qs({ shot_id: shot })}`)
export const fetchMatch = () => getJsonOrNull<MatchInfo>("/api/match")
export const fetchCameraFps = async () => (await getJsonOrNull<{ fps?: number }>("/camera/track"))?.fps ?? null

export async function renameTrack(shot: string, trackId: string, playerName: string): Promise<void> {
  const url = `/api/tracks/${enc(shot)}/${enc(trackId)}`
  const res = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_name: playerName }),
  })
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => null), url)
}

/** Every destructive edit returns the snapshot id that /api/tracks/undo restores (null = nothing changed). */
export interface Undoable {
  undo_id?: string | null
}

export interface UndoEntry {
  undo_id: string
  label: string
  created_at: string
  files: string[]
}

export const listUndo = () => getJson<UndoEntry[]>("/api/tracks/undo")

export const undoEdit = (undoId?: string) =>
  postJson<{ undone: string; undo_id: string; files: string[] }>("/api/tracks/undo", undoId ? { undo_id: undoId } : {})

export const deleteTrack = (shot: string, trackId: string) =>
  deleteJson<Undoable>(`/api/tracks/${enc(shot)}/${enc(trackId)}`)

export const deleteTracksBulk = (shot: string, trackIds: string[]) =>
  postJson<Undoable>(`/api/tracks/${enc(shot)}/delete-bulk`, { track_ids: trackIds })

export const splitTrack = (shot: string, trackId: string, splitFrame: number) =>
  postJson<Undoable>("/api/tracks/split", { shot_id: shot, track_id: trackId, split_frame: splitFrame })

export const mergeTracks = (shot: string, trackIds: string[]) =>
  postJson<Undoable & { merged_into: string; frame_collisions?: number }>("/api/tracks/merge", {
    shot_id: shot,
    track_ids: trackIds,
  })

export const mergeByName = () =>
  postJson<Undoable & { tracks_removed: number; merged_groups: number; frame_collisions?: number }>("/api/tracks/merge-by-name")

export const ignoreUnknown = (shot: string) =>
  postJson<Undoable & { count: number }>(`/api/tracks/ignore-unknown/${enc(shot)}`)

export const deleteIgnored = () => postJson<Undoable & { deleted: number }>("/api/tracks/delete-ignored")

export const interpolateGaps = (shot: string, trackIds: string[]) =>
  postJson<Undoable & { total_frames_added: number; results: unknown[] }>(`/api/tracks/${enc(shot)}/interpolate-gaps`, {
    track_ids: trackIds,
  })

export const videoUrl = (shot: string) => `/api/video/${enc(shot)}`

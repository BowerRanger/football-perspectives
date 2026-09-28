import * as React from "react"

import { errorMessage } from "@/lib/api"
import { fetchCameraFps, fetchFrames, fetchMatch, fetchPreview } from "./api"
import type { FrameBox, RosterEntry, TrackSummary } from "./types"

const DEFAULT_FPS = 30

export interface TrackEditorData {
  tracks: TrackSummary[]
  boxesByFrame: Map<number, FrameBox[]>
  fps: number
  roster: RosterEntry[]
}

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: TrackEditorData }

async function loadAll(shot: string): Promise<TrackEditorData> {
  const [preview, frames, match] = await Promise.all([
    fetchPreview(shot),
    fetchFrames(shot).catch(() => null),
    fetchMatch(),
  ])
  // Prefer the shot's own fps (delivered with /tracking/frames); the camera
  // track only exists once the camera stage has run and 30 misplaces boxes
  // on 25 fps clips.
  const fps = frames?.fps || (await fetchCameraFps()) || DEFAULT_FPS
  const boxesByFrame = new Map<number, FrameBox[]>()
  for (const f of frames?.frames ?? []) boxesByFrame.set(f.frame, f.boxes ?? [])
  return { tracks: preview.tracks, boxesByFrame, fps, roster: match?.roster ?? [] }
}

/** Loads preview + frames + roster for a shot; `reload` refetches in place. */
export function useTrackEditorData(shot: string) {
  const [state, setState] = React.useState<LoadState>({ status: "loading" })

  const reload = React.useCallback(async () => {
    try {
      setState({ status: "ready", data: await loadAll(shot) })
    } catch (err) {
      setState({ status: "error", message: errorMessage(err) })
    }
  }, [shot])

  React.useEffect(() => {
    setState({ status: "loading" })
    void reload()
  }, [reload])

  /** Rename tracks in memory so the overlay updates without a round-trip. */
  const applyName = React.useCallback((trackIds: ReadonlySet<string>, name: string) => {
    setState((s) =>
      s.status !== "ready"
        ? s
        : {
            status: "ready",
            data: {
              ...s.data,
              tracks: s.data.tracks.map((t) => (trackIds.has(t.track_id) ? { ...t, player_name: name } : t)),
            },
          },
    )
  }, [])

  return { state, reload, applyName }
}

import * as React from "react"

import { errorMessage, getJson, getJsonOrNull, qs } from "@/lib/api"
import { anchorFromResponse } from "./anchor-ops"
import type {
  AnchorMap,
  AnchorsResponse,
  CameraTrack,
  DetectedLinesByFrame,
  Landmark,
  PitchLine,
  Stadium,
  Vec2,
} from "./types"

interface ShotList {
  shots: string[]
  /** Shot referenced by the legacy global anchors.json, when it exists. */
  defaultShot: string | null
  defaultStadium: string
  loaded: boolean
  error: string | null
}

/** Shot ids + the stadium/clip the global /anchors file points at. */
export function useShotList(): ShotList {
  const [state, setState] = React.useState<ShotList>({
    shots: [],
    defaultShot: null,
    defaultStadium: "",
    loaded: false,
    error: null,
  })
  React.useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const [shots, global] = await Promise.all([
          getJson<{ shots?: string[] }>("/api/output/shots"),
          getJsonOrNull<AnchorsResponse>("/anchors"),
        ])
        if (cancelled) return
        const ids = shots.shots ?? []
        setState({
          shots: ids,
          defaultShot: global?.clip_id && ids.includes(global.clip_id) ? global.clip_id : null,
          defaultStadium: global?.stadium ?? "",
          loaded: true,
          error: null,
        })
      } catch (err) {
        if (cancelled) return
        setState((s) => ({ ...s, loaded: true, error: err instanceof Error ? err.message : String(err) }))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])
  return state
}

interface Catalogues {
  landmarks: Landmark[]
  stadiums: Stadium[]
  loaded: boolean
}

export function useCatalogues(): Catalogues {
  const [state, setState] = React.useState<Catalogues>({ landmarks: [], stadiums: [], loaded: false })
  React.useEffect(() => {
    let cancelled = false
    void (async () => {
      const [lms, stadiums] = await Promise.all([
        getJsonOrNull<{ landmarks?: Landmark[] }>("/landmarks"),
        getJsonOrNull<{ stadiums?: Stadium[] }>("/stadiums"),
      ])
      if (!cancelled) {
        setState({ landmarks: lms?.landmarks ?? [], stadiums: stadiums?.stadiums ?? [], loaded: true })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])
  return state
}

/** Line catalogue; re-fetched with the stadium's mow stripes merged in. */
export function usePitchLines(stadium: string): PitchLine[] {
  const [lines, setLines] = React.useState<PitchLine[]>([])
  React.useEffect(() => {
    let cancelled = false
    void getJsonOrNull<{ lines?: PitchLine[] }>(`/pitch_lines${qs({ stadium })}`).then((res) => {
      if (!cancelled) setLines(res?.lines ?? [])
    })
    return () => {
      cancelled = true
    }
  }, [stadium])
  return lines
}

interface ShotData {
  anchors: AnchorMap
  anchorImageSize: Vec2 | null
  /** Stadium stored in the shot's anchors.json (null when none). */
  savedStadium: string | null
  loading: boolean
  /** Set when the saved anchors could not be read. Saving must stay disabled. */
  loadError: string | null
  retry: () => void
}

const EMPTY: AnchorMap = new Map()

/**
 * Loads the shot's saved anchors. A missing file is a valid empty response
 * from the server; any other failure is surfaced as `loadError` so the caller
 * can block saving (an unread file must never be overwritten).
 */
export function useShotAnchors(shot: string): ShotData {
  const [data, setData] = React.useState<Omit<ShotData, "retry">>({
    anchors: EMPTY,
    anchorImageSize: null,
    savedStadium: null,
    loading: false,
    loadError: null,
  })
  const [attempt, setAttempt] = React.useState(0)
  const retry = React.useCallback(() => setAttempt((n) => n + 1), [])
  React.useEffect(() => {
    if (!shot) return
    let cancelled = false
    setData({ anchors: EMPTY, anchorImageSize: null, savedStadium: null, loading: true, loadError: null })
    getJson<AnchorsResponse>(`/anchors/${encodeURIComponent(shot)}`)
      .then((res) => {
        if (cancelled) return
        if (!res || res.clip_id !== shot) throw new Error(`the server returned anchors for "${res?.clip_id ?? "unknown"}"`)
        setData({
          anchors: anchorFromResponse(res),
          anchorImageSize: res.image_size && res.image_size[0] > 0 ? res.image_size : null,
          savedStadium: res.stadium ? res.stadium : null,
          loading: false,
          loadError: null,
        })
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setData({ anchors: EMPTY, anchorImageSize: null, savedStadium: null, loading: false, loadError: errorMessage(err) })
      })
    return () => {
      cancelled = true
    }
  }, [shot, attempt])
  return { ...data, retry }
}

interface CameraData {
  track: CameraTrack | null
  detected: DetectedLinesByFrame
}

/** Camera track + detected-lines debug output; reloads when a job finishes. */
export function useCameraData(shot: string, outputVersion: number): CameraData {
  const [state, setState] = React.useState<CameraData>({ track: null, detected: {} })
  React.useEffect(() => {
    if (!shot) return
    let cancelled = false
    const q = qs({ shot })
    void (async () => {
      const [track, detected] = await Promise.all([
        getJsonOrNull<CameraTrack>(`/camera/track${q}`),
        getJsonOrNull<{ frames?: DetectedLinesByFrame }>(`/camera/detected-lines${q}`),
      ])
      if (cancelled) return
      // Per-shot fetch: a foreign clip's track must never overlay this shot.
      const usable = track && track.clip_id === shot && (track.frames ?? []).length > 0
      setState({ track: usable ? track : null, detected: detected?.frames ?? {} })
    })()
    return () => {
      cancelled = true
    }
  }, [shot, outputVersion])
  return state
}

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
  retry: () => void
}

/**
 * Shot ids + the stadium/clip the global /anchors file points at. The shot
 * list is the page's main payload (server always answers 200); the legacy
 * global /anchors lookup only picks a default, so it stays best-effort.
 */
export function useShotList(): ShotList {
  const [state, setState] = React.useState<Omit<ShotList, "retry">>({
    shots: [],
    defaultShot: null,
    defaultStadium: "",
    loaded: false,
    error: null,
  })
  const [attempt, setAttempt] = React.useState(0)
  const retry = React.useCallback(() => setAttempt((n) => n + 1), [])
  React.useEffect(() => {
    let cancelled = false
    setState((s) => ({ ...s, loaded: false, error: null }))
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
        setState((s) => ({ ...s, loaded: true, error: errorMessage(err) }))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [attempt])
  return { ...state, retry }
}

interface Catalogues {
  landmarks: Landmark[]
  stadiums: Stadium[]
  loaded: boolean
  /** Landmark catalogue failed: nothing can be placed (main payload). */
  error: string | null
  /** Stadium registry failed: only the mow-stripe dropdown is affected (optional). */
  stadiumsError: string | null
  retry: () => void
}

/**
 * The landmark catalogue is the palette's main payload (the server always
 * answers 200, so any failure is real). Stadiums only feed a dropdown, so
 * their failure is reported separately as a muted notice.
 */
export function useCatalogues(): Catalogues {
  const [state, setState] = React.useState<Omit<Catalogues, "retry">>({
    landmarks: [],
    stadiums: [],
    loaded: false,
    error: null,
    stadiumsError: null,
  })
  const [attempt, setAttempt] = React.useState(0)
  const retry = React.useCallback(() => setAttempt((n) => n + 1), [])
  React.useEffect(() => {
    let cancelled = false
    setState((s) => ({ ...s, loaded: false, error: null, stadiumsError: null }))
    void (async () => {
      const [lms, stadiums] = await Promise.allSettled([
        getJson<{ landmarks?: Landmark[] }>("/landmarks"),
        getJson<{ stadiums?: Stadium[] }>("/stadiums"),
      ])
      if (cancelled) return
      setState({
        landmarks: lms.status === "fulfilled" ? (lms.value.landmarks ?? []) : [],
        stadiums: stadiums.status === "fulfilled" ? (stadiums.value.stadiums ?? []) : [],
        loaded: true,
        error: lms.status === "rejected" ? errorMessage(lms.reason) : null,
        stadiumsError: stadiums.status === "rejected" ? errorMessage(stadiums.reason) : null,
      })
    })()
    return () => {
      cancelled = true
    }
  }, [attempt])
  return { ...state, retry }
}

interface PitchLines {
  lines: PitchLine[]
  /** Optional: the Lines palette and snap targets fall back to empty. */
  error: string | null
}

/** Line catalogue; re-fetched with the stadium's mow stripes merged in. */
export function usePitchLines(stadium: string): PitchLines {
  const [state, setState] = React.useState<PitchLines>({ lines: [], error: null })
  React.useEffect(() => {
    let cancelled = false
    getJson<{ lines?: PitchLine[] }>(`/pitch_lines${qs({ stadium })}`)
      .then((res) => {
        if (!cancelled) setState({ lines: res.lines ?? [], error: null })
      })
      .catch((err: unknown) => {
        if (!cancelled) setState({ lines: [], error: errorMessage(err) })
      })
    return () => {
      cancelled = true
    }
  }, [stadium])
  return state
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
  /** Camera track failed to load (main payload; an empty track is not an error). */
  trackError: string | null
  /** Detected-line debug overlay failed (optional). */
  detectedError: string | null
  retry: () => void
}

/**
 * Camera track + detected-lines debug output; reloads when a job finishes.
 * Both endpoints answer 200 with an empty payload when the stage hasn't run,
 * so any rejection here is a real failure, not "not run yet".
 */
export function useCameraData(shot: string, outputVersion: number): CameraData {
  const [state, setState] = React.useState<Omit<CameraData, "retry">>({
    track: null,
    detected: {},
    trackError: null,
    detectedError: null,
  })
  const [attempt, setAttempt] = React.useState(0)
  const retry = React.useCallback(() => setAttempt((n) => n + 1), [])
  React.useEffect(() => {
    if (!shot) return
    let cancelled = false
    const q = qs({ shot })
    void (async () => {
      const [track, detected] = await Promise.allSettled([
        getJson<CameraTrack>(`/camera/track${q}`),
        getJson<{ frames?: DetectedLinesByFrame }>(`/camera/detected-lines${q}`),
      ])
      if (cancelled) return
      const t = track.status === "fulfilled" ? track.value : null
      // Per-shot fetch: a foreign clip's track must never overlay this shot.
      const usable = t && t.clip_id === shot && (t.frames ?? []).length > 0
      setState({
        track: usable ? t : null,
        detected: detected.status === "fulfilled" ? (detected.value.frames ?? {}) : {},
        trackError: track.status === "rejected" ? errorMessage(track.reason) : null,
        detectedError: detected.status === "rejected" ? errorMessage(detected.reason) : null,
      })
    })()
    return () => {
      cancelled = true
    }
  }, [shot, outputVersion, attempt])
  return { ...state, retry }
}

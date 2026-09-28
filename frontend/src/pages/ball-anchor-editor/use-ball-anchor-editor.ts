// Controller for the ball anchor editor: loading, authoring state, click
// placement, save and Solve & preview. Used by the standalone page and the
// Ball stage panel so there is exactly one implementation (and one save
// payload that always round-trips chains, dismissals, end frames, landmarks).

import * as React from "react"
import { toast } from "sonner"

import { errorMessage } from "@/lib/api"
import { loadQuality, loadShot, previewAnchors, saveAnchors } from "./api"
import { DEFAULT_AUTHORING, placeAnchor, type Authoring } from "./placement"
import { useAnchorDoc, type AnchorDocApi } from "./use-anchor-doc"
import { useFramePlayer, type FramePlayer } from "./use-frame-player"
import {
  projectWorld,
  type AutoAnchor,
  type BallAnchorPayload,
  type BallQuality,
  type CameraTrackData,
  type PlayerOption,
  type PreviewFrame,
  type PreviewResult,
} from "./types"
import type { PitchFixSuggestion } from "./api"

export interface Layers {
  anchors: boolean
  predicted: boolean
  preview: boolean
}

export interface PredictedPoint {
  uv: [number, number]
  state: string
  z: number
}

export interface EditorController {
  shot: string
  loading: boolean
  loadError: string | null
  /** Re-attempt the saved-anchor load after a failure. */
  retryLoad: () => void
  player: FramePlayer
  docApi: AnchorDocApi
  autoAnchors: AutoAnchor[]
  quality: BallQuality | null
  players: PlayerOption[]
  selectedTag: string
  setSelectedTag: (id: string) => void
  authoring: Authoring
  patchAuthoring: (patch: Partial<Authoring>) => void
  touchSuggestion: { player: string; bone: string } | null
  pitchFixes: { frame: number; items: PitchFixSuggestion[] } | null
  layers: Layers
  patchLayers: (patch: Partial<Layers>) => void
  predictedByFrame: Map<number, PredictedPoint>
  previewByFrame: Map<number, [number, number]>
  previewResult: PreviewResult | null
  saving: boolean
  solving: boolean
  save: () => Promise<boolean>
  solve: () => Promise<void>
  placeAt: (u: number, v: number) => Promise<void>
  removeAt: (u: number, v: number) => void
  setEndFrameHere: (anchorFrame: number) => void
}

interface Options {
  shot: string
  /** Predicted ball frames (from /ball/preview) for the optional path layer. */
  predicted?: PreviewFrame[]
  onFrameChange?: (frame: number) => void
}

function buildProjection(
  camera: CameraTrackData | null,
  frames: PreviewFrame[] | undefined,
): Map<number, PredictedPoint> {
  const out = new Map<number, PredictedPoint>()
  if (!camera?.frames || !frames) return out
  const byFrame = new Map(camera.frames.map((f) => [f.frame, f]))
  for (const f of frames) {
    if (!f.world_xyz) continue
    const uv = projectWorld(f.world_xyz, byFrame.get(f.frame), camera.t_world)
    if (uv) out.set(f.frame, { uv, state: f.state, z: f.world_xyz[2] ?? 0 })
  }
  return out
}

export function useBallAnchorEditor({ shot, predicted, onFrameChange }: Options): EditorController {
  const player = useFramePlayer(onFrameChange)
  const docApi = useAnchorDoc()
  const [loading, setLoading] = React.useState(Boolean(shot))
  const [loadError, setLoadError] = React.useState<string | null>(null)
  const [camera, setCamera] = React.useState<CameraTrackData | null>(null)
  const [imageSize, setImageSize] = React.useState<[number, number]>([1280, 720])
  const [autoAnchors, setAutoAnchors] = React.useState<AutoAnchor[]>([])
  const [players, setPlayers] = React.useState<PlayerOption[]>([])
  const [quality, setQuality] = React.useState<BallQuality | null>(null)
  const [selectedTag, setSelectedTag] = React.useState("grounded")
  const [authoring, setAuthoring] = React.useState<Authoring>(DEFAULT_AUTHORING)
  const [touchSuggestion, setTouchSuggestion] = React.useState<{ player: string; bone: string } | null>(null)
  const [pitchFixes, setPitchFixes] = React.useState<{ frame: number; items: PitchFixSuggestion[] } | null>(null)
  const [layers, setLayers] = React.useState<Layers>({ anchors: true, predicted: true, preview: true })
  const [previewResult, setPreviewResult] = React.useState<PreviewResult | null>(null)
  const [saving, setSaving] = React.useState(false)
  const [solving, setSolving] = React.useState(false)
  const [attempt, setAttempt] = React.useState(0)
  const retryLoad = React.useCallback(() => setAttempt((n) => n + 1), [])
  const { reset } = docApi
  const { setFps } = player

  React.useEffect(() => {
    if (!shot) return
    let cancelled = false
    setLoading(true)
    setLoadError(null)
    setPreviewResult(null)
    setPitchFixes(null)
    setTouchSuggestion(null)
    setQuality(null)
    loadShot(shot)
      .then((s) => {
        if (cancelled) return
        if (s.fps) setFps(s.fps)
        setCamera(s.camera)
        setImageSize(s.imageSize)
        setAutoAnchors(s.autoAnchors)
        setPlayers(s.players)
        reset({ anchors: s.anchors, shotChains: s.shotChains, dismissedAuto: s.dismissedAuto })
        setLoading(false)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setLoadError(errorMessage(err))
        setLoading(false)
      })
    void loadQuality(shot).then((q) => !cancelled && setQuality(q))
    return () => {
      cancelled = true
    }
  }, [shot, attempt, reset, setFps])

  const predictedByFrame = React.useMemo(() => buildProjection(camera, predicted), [camera, predicted])
  const previewByFrame = React.useMemo(() => {
    const map = new Map<number, [number, number]>()
    for (const [k, v] of buildProjection(camera, previewResult?.frames)) map.set(k, v.uv)
    return map
  }, [camera, previewResult])

  const payload = React.useCallback(
    (): BallAnchorPayload => ({
      clip_id: shot,
      image_size: imageSize,
      anchors: docApi.doc.anchors,
      shot_chains: docApi.doc.shotChains,
      dismissed_auto: docApi.doc.dismissedAuto,
    }),
    [shot, imageSize, docApi.doc],
  )

  const save = React.useCallback(async () => {
    // A failed load means the saved set is unknown: never overwrite it.
    if (loading || loadError) return false
    setSaving(true)
    try {
      const res = await saveAnchors(shot, payload())
      docApi.markSaved()
      toast.success(`Saved ${res.count} anchors`, { description: "Re-run the Ball stage to apply them." })
      void loadQuality(shot).then(setQuality)
      return true
    } catch (err) {
      toast.error("Could not save ball anchors", { description: errorMessage(err) })
      return false
    } finally {
      setSaving(false)
    }
  }, [shot, payload, docApi, loading, loadError])

  const solve = React.useCallback(async () => {
    setSolving(true)
    try {
      setPreviewResult(await previewAnchors(shot, payload()))
    } catch (err) {
      toast.error("Solve & preview failed", { description: errorMessage(err) })
    } finally {
      setSolving(false)
    }
  }, [shot, payload])

  const placeAt = React.useCallback(
    async (u: number, v: number) => {
      const fi = player.currentFrame()
      const res = await placeAnchor(selectedTag, shot, fi, [u, v], authoring)
      if (!res.ok) {
        toast.warning(res.message)
        return
      }
      docApi.addAnchor(res.anchor)
      if (res.touchSuggestion) setTouchSuggestion(res.touchSuggestion)
      if (res.pitchFixes) setPitchFixes({ frame: fi, items: res.pitchFixes })
      if (res.message) toast.message(res.message)
    },
    [selectedTag, shot, authoring, player, docApi],
  )

  const removeAt = React.useCallback(
    (u: number, v: number) => {
      docApi.removeNear(player.currentFrame(), u, v, 14)
    },
    [docApi, player],
  )

  const setEndFrameHere = React.useCallback(
    (anchorFrame: number) => {
      const fi = player.currentFrame()
      if (fi <= anchorFrame) {
        toast.warning("End frame must be after the anchor frame", {
          description: "Scrub the video past the touch, then set the end frame.",
        })
        return
      }
      docApi.setEndFrame(anchorFrame, fi)
    },
    [player, docApi],
  )

  return {
    shot,
    loading,
    loadError,
    retryLoad,
    player,
    docApi,
    autoAnchors,
    quality,
    players,
    selectedTag,
    setSelectedTag,
    authoring,
    patchAuthoring: (patch) => setAuthoring((a) => ({ ...a, ...patch })),
    touchSuggestion,
    pitchFixes,
    layers,
    patchLayers: (patch) => setLayers((l) => ({ ...l, ...patch })),
    predictedByFrame,
    previewByFrame,
    previewResult,
    saving,
    solving,
    save,
    solve,
    placeAt,
    removeAt,
    setEndFrameHere,
  }
}

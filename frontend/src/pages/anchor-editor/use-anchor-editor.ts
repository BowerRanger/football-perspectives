import * as React from "react"
import { toast } from "sonner"

import { errorMessage, postJson } from "@/lib/api"
import { useConfirm } from "@/hooks/use-dialogs"
import { usePipeline } from "@/hooks/use-pipeline"
import {
  addEmptyAnchor,
  removeAnchor,
  removeLine,
  removePoint,
  serialiseAnchors,
} from "./anchor-ops"
import { useCameraData, useCatalogues, usePitchLines, useShotAnchors, useShotList } from "./use-anchor-data"
import { useEditorKeys } from "./use-editor-keys"
import { useFramePlayer } from "./use-frame-player"
import { useScopedKeyboard } from "./use-scoped-keyboard"
import { usePlacement, type Flash } from "./use-placement"
import { DEFAULT_VIEW } from "./types"
import type { AnchorMap, CameraTrack, Vec2, ViewOptions } from "./types"
import type { VideoMeta } from "./stage-canvas"
import { useUnsavedGuard } from "@/hooks/use-unsaved-guard"

interface Options {
  /** Controlled shot id (embedded panel / `?shot=`); undefined = self-managed. */
  shot?: string
  onShotChange?: (shot: string) => void
  embedded: boolean
}

export interface LoadIssue {
  title: string
  message: string
  retry: () => void
}

const DEFAULT_FPS = 30
const NO_ANCHORS: AnchorMap = new Map()

function deriveTiming(track: CameraTrack | null, duration: number) {
  const fps = track?.fps && track.fps > 0 ? track.fps : DEFAULT_FPS
  const fromDuration = Number.isFinite(duration) ? Math.round(duration * fps) : 0
  const totalFrames = Math.max(1, fromDuration, track ? track.frames.length : 0)
  return { fps, totalFrames }
}

export function useAnchorEditor({ shot: shotProp, onShotChange, embedded }: Options) {
  const confirm = useConfirm()
  const pipeline = usePipeline()
  const list = useShotList()
  const catalogues = useCatalogues()

  const [ownShot, setOwnShot] = React.useState("")
  const shot = shotProp ?? ownShot
  const [stadium, setStadium] = React.useState("")
  const { lines: pitchLines, error: pitchLinesError } = usePitchLines(stadium)
  const saved = useShotAnchors(shot)
  const camera = useCameraData(shot, pipeline.outputVersion)
  const track = camera.track?.clip_id === shot ? camera.track : null

  const [anchors, setAnchors] = React.useState<AnchorMap>(NO_ANCHORS)
  const [dirty, setDirty] = React.useState(false)
  const [view, setView] = React.useState<ViewOptions>(DEFAULT_VIEW)
  const [meta, setMeta] = React.useState<VideoMeta | null>(null)
  const [flashMsg, setFlash] = React.useState<Flash | null>(null)
  const [saving, setSaving] = React.useState(false)
  const [rerunPhase, setRerunPhase] = React.useState<"idle" | "saving" | "starting">("idle")
  const [viewerShot, setViewerShot] = React.useState<string | null>(null)
  const videoRef = React.useRef<HTMLVideoElement>(null)
  const rootRef = React.useRef<HTMLDivElement>(null)

  const { fps, totalFrames } = deriveTiming(track, meta?.duration ?? 0)
  const imageSize: Vec2 = saved.anchorImageSize ?? [meta?.width ?? 0, meta?.height ?? 0]
  const player = useFramePlayer(videoRef, fps, totalFrames)
  const { reset: resetPlayer } = player

  // Resolve the default shot once the list arrives.
  React.useEffect(() => {
    if (shotProp || ownShot || !list.loaded || list.shots.length === 0) return
    const first = list.defaultShot ?? list.shots[0]
    setOwnShot(first)
    onShotChange?.(first)
  }, [shotProp, ownShot, list, onShotChange])

  // Fresh shot: adopt its saved anchors/stadium and start from frame 0.
  React.useEffect(() => {
    setAnchors(saved.anchors)
    setDirty(false)
  }, [saved.anchors])
  React.useEffect(() => {
    if (saved.savedStadium) setStadium(saved.savedStadium)
  }, [saved.savedStadium])
  React.useEffect(() => {
    setStadium((s) => s || list.defaultStadium)
  }, [list.defaultStadium])
  React.useEffect(() => {
    setMeta(null)
    setFlash(null)
    setViewerShot(null)
    resetPlayer()
  }, [shot, resetPlayer])

  useUnsavedGuard(dirty, { what: "anchor edits" })

  const edit = React.useCallback((fn: (m: AnchorMap) => AnchorMap) => {
    setAnchors((prev) => fn(prev))
    setDirty(true)
  }, [])

  const placement = usePlacement({
    shot,
    frame: player.frame,
    snapEnabled: view.snap,
    disabled: saved.loading || Boolean(saved.loadError) || !shot,
    landmarks: catalogues.landmarks,
    pitchLines,
    edit,
    flash: setFlash,
  })

  const requestShot = React.useCallback(
    async (next: string) => {
      if (next === shot) return
      if (dirty) {
        const ok = await confirm({
          title: "Discard unsaved anchor changes?",
          description: `Switching shots drops your unsaved edits to ${shot}. Save first if you want to keep them.`,
          confirmLabel: "Discard and switch",
          destructive: true,
        })
        if (!ok) return
      }
      setOwnShot(next)
      onShotChange?.(next)
    },
    [shot, dirty, confirm, onShotChange],
  )

  const changeStadium = React.useCallback((next: string) => {
    setStadium(next)
    setDirty(true)
  }, [])

  const save = React.useCallback(
    async (quiet = false): Promise<boolean> => {
      if (!shot || saved.loading || saved.loadError) return false
      setSaving(true)
      setFlash({ text: "Saving…", tone: "warning" })
      try {
        const res = await postJson<{ count?: number }>(`/anchors/${encodeURIComponent(shot)}`, {
          clip_id: shot,
          image_size: imageSize,
          stadium: stadium || null,
          anchors: serialiseAnchors(anchors),
        })
        setDirty(false)
        const count = res?.count ?? anchors.size
        setFlash({ text: `Saved ${count} anchor${count === 1 ? "" : "s"}`, tone: "muted" })
        if (!quiet) toast.success("Anchors saved", { description: `${count} anchor frame${count === 1 ? "" : "s"} for ${shot}` })
        return true
      } catch (err) {
        setFlash({ text: "Save failed", tone: "destructive" })
        toast.error("Could not save anchors", { description: errorMessage(err) })
        return false
      } finally {
        setSaving(false)
      }
    },
    [shot, imageSize, stadium, anchors, saved.loading, saved.loadError],
  )

  const rerun = React.useCallback(async () => {
    if (!shot || saved.loadError || saved.loading) return
    const ok = await confirm({
      title: `Rerun camera tracking for ${shot}?`,
      description:
        "Your anchors are saved first, then this shot's camera output (track and debug files) is deleted and re-solved from them. Anchors are never overwritten.",
      confirmLabel: "Save and rerun",
      destructive: true,
    })
    if (!ok) return
    setRerunPhase("saving")
    try {
      if (!(await save(true))) return
      setRerunPhase("starting")
      const { job_id } = await postJson<{ job_id: string }>("/api/run-shot", { stage: "camera", shot_id: shot })
      setFlash({ text: `Camera stage running for ${shot}`, tone: "warning" })
      pipeline.attachToJob(job_id, "camera", (status) => {
        if (status === "done") {
          setViewerShot(shot)
          setFlash({ text: `Camera stage finished for ${shot}`, tone: "muted" })
        }
      })
    } catch (err) {
      setFlash({ text: "Camera rerun failed to start", tone: "destructive" })
      toast.error("Could not start camera tracking", { description: errorMessage(err) })
    } finally {
      setRerunPhase("idle")
    }
  }, [shot, confirm, save, pipeline, saved.loading, saved.loadError])

  const deleteAnchorFrame = React.useCallback(
    async (frame: number) => {
      const cur = anchors.get(frame)
      if (!cur) {
        setFlash({ text: `No anchor at frame ${frame} to delete`, tone: "muted" })
        return
      }
      const n = cur.points.length + cur.lines.length
      if (n > 0) {
        const ok = await confirm({
          title: `Delete the anchor at frame ${frame}?`,
          description: `This removes ${n} placed point${n === 1 ? "" : "s"} and line${n === 1 ? "" : "s"} from the editor. The saved file changes only when you save.`,
          confirmLabel: "Delete anchor",
          destructive: true,
        })
        if (!ok) return
      }
      edit((m) => removeAnchor(m, frame))
      setFlash({ text: `Deleted anchor at frame ${frame}`, tone: "muted" })
    },
    [anchors, confirm, edit],
  )

  const toggleView = React.useCallback((key: keyof ViewOptions) => {
    setView((v) => ({ ...v, [key]: !v[key] }))
  }, [])

  const keyboard = useScopedKeyboard(rootRef, embedded)

  useEditorKeys({
    rootRef,
    embedded,
    onEscape: placement.clear,
    onSave: () => void save(),
    onToggleView: toggleView,
  })

  const here = anchors.get(player.frame)
  const placedHere = new Set<string>([
    ...(here?.points.map((p) => p.name) ?? []),
    ...(here?.lines.map((l) => l.name) ?? []),
  ])

  const baseline = meta
    ? `${shot} · ${meta.width}×${meta.height} · ${Number(fps.toFixed(2))} fps`
    : shot
      ? `Loading ${shot}`
      : "No shot selected"

  // Main payloads fail loudly (PanelError + Retry); optional overlays get a muted notice.
  const loadErrors: LoadIssue[] = []
  if (list.error) loadErrors.push({ title: "Could not list shots", message: list.error, retry: list.retry })
  if (catalogues.error) {
    loadErrors.push({ title: "Could not load the landmark catalogue", message: catalogues.error, retry: catalogues.retry })
  }
  if (camera.trackError) {
    loadErrors.push({ title: `Could not load the camera track for ${shot}`, message: camera.trackError, retry: camera.retry })
  }
  const notices: string[] = []
  if (catalogues.stadiumsError) notices.push(`Stadium list unavailable (${catalogues.stadiumsError}); mow-stripe lines can't be added.`)
  if (pitchLinesError) notices.push(`Pitch line catalogue unavailable (${pitchLinesError}); the Lines palette is empty.`)
  if (camera.detectedError) notices.push(`Detected-line overlay unavailable (${camera.detectedError}).`)

  return {
    rootRef,
    videoRef,
    loadErrors,
    notices,
    list,
    catalogues,
    pitchLines,
    shot,
    requestShot,
    stadium,
    changeStadium,
    anchors,
    loadingAnchors: saved.loading,
    anchorLoadError: saved.loadError,
    retryAnchorLoad: saved.retry,
    track,
    detected: camera.detected,
    dirty,
    saving,
    save: () => void save(),
    rerun: () => void rerun(),
    rerunPhase,
    rerunBlockedReason: saved.loadError
      ? "Saved anchors could not be loaded, so they can't be saved first."
      : pipeline.isRunning
        ? `${pipeline.runningLabel} is running. Wait for it to finish.`
        : null,
    viewerHref: viewerShot ? `/viewer?shot=${encodeURIComponent(viewerShot)}` : null,
    view,
    toggleView,
    player,
    totalFrames,
    fps,
    keyboard,
    imageSize,
    setMeta,
    placement,
    placedHere,
    hasAnchorHere: anchors.has(player.frame),
    addAnchorHere: () => edit((m) => addEmptyAnchor(m, player.frame)),
    deleteAnchorFrame: (f: number) => void deleteAnchorFrame(f),
    deletePoint: (name: string) => edit((m) => removePoint(m, player.frame, name)),
    deleteLine: (i: number) => edit((m) => removeLine(m, player.frame, i)),
    status: flashMsg ?? { text: baseline, tone: "muted" as const },
    statusIsFlash: flashMsg !== null,
  }
}

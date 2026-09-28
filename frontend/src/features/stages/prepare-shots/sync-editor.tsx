import * as React from "react"
import { ChevronLeftIcon, ChevronRightIcon, LockIcon, PauseIcon, PlayIcon, SaveIcon } from "lucide-react"
import { toast } from "sonner"

import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { errorMessage, postJson } from "@/lib/api"

import { IconButton } from "./icon-button"
import { MethodBadge, OffsetInput, SyncOffsetRows } from "./sync-offsets"
import { SyncTimeline, type AlignMethod } from "./sync-timeline"
import { SyncVideoColumn } from "./sync-video-column"
import type { GroupView } from "./types"
import { useUnsavedGuard } from "@/hooks/use-unsaved-guard"

interface EditorProps {
  group: GroupView
  /** Called after a successful save so tile badges can refresh (no reseed). */
  onSaved: () => void
}

const DRIFT_S = 0.08

function seedState(group: GroupView) {
  const ids = group.members.map((m) => m.id)
  const framesByShot: Record<string, number> = {}
  const fpsByShot: Record<string, number> = {}
  for (const m of group.members) {
    const frames = Math.max(1, m.end_frame - m.start_frame + 1)
    framesByShot[m.id] = frames
    const dur = m.end_time - m.start_time
    fpsByShot[m.id] = dur > 0 ? frames / dur : 25
  }
  const offsets: Record<string, number> = {}
  const methods: Record<string, AlignMethod> = {}
  for (const a of group.sync?.alignments ?? []) {
    offsets[a.shot_id] = a.frame_offset
    methods[a.shot_id] = { method: a.method, confidence: a.confidence }
  }
  for (const id of ids) {
    offsets[id] ??= 0
    methods[id] ??= { method: "manual", confidence: 1 }
  }
  const saved = group.sync?.reference_shot
  const reference = saved && ids.includes(saved) ? saved : ids[0]
  const active = ids.find((s) => s !== reference) ?? ids[0]
  return { ids, framesByShot, fpsByShot, offsets, methods, reference, active }
}

/**
 * Two-video scrub + draggable timeline for one highlight group.
 * Sign convention (sync_map.py): frame_offset = frame_in_active - frame_in_reference.
 * Any edit becomes `manual` and survives re-alignment.
 */
export function SyncEditor({ group, onSaved }: EditorProps) {
  // Seeded once per mount; the parent re-keys the editor to reseed.
  const [seed] = React.useState(() => seedState(group))
  const { ids, framesByShot, fpsByShot } = seed
  const [referenceShot, setReferenceShot] = React.useState(seed.reference)
  const [activeShot, setActiveShot] = React.useState(seed.active)
  const [offsets, setOffsets] = React.useState(seed.offsets)
  const [methods, setMethods] = React.useState(seed.methods)
  const [playing, setPlaying] = React.useState(false)
  const [cursorFrame, setCursorFrame] = React.useState(0)
  const [dirty, setDirty] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const refVideo = React.useRef<HTMLVideoElement>(null)
  const actVideo = React.useRef<HTMLVideoElement>(null)

  const fps = (id: string) => fpsByShot[id] || 25
  const offsetOf = (id: string) => offsets[id] ?? 0
  const latest = React.useRef({ referenceShot, activeShot, offsets })
  latest.current = { referenceShot, activeShot, offsets }

  useUnsavedGuard(dirty, { what: "sync offsets" })

  /** Move the active video to the instant matching the reference. */
  const syncActive = React.useCallback((tolerance = 0.05) => {
    const rv = refVideo.current
    const av = actVideo.current
    if (!rv || !av) return
    const { referenceShot: r, activeShot: a, offsets: o } = latest.current
    const target = Math.max(0, (rv.currentTime * fps(r) + (o[a] ?? 0)) / fps(a))
    if (Math.abs(av.currentTime - target) > tolerance) {
      try {
        av.currentTime = target
      } catch {
        /* metadata not loaded yet; loadedmetadata re-syncs */
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fpsByShot])

  const readCursor = React.useCallback(() => {
    const rv = refVideo.current
    if (rv) setCursorFrame(rv.currentTime * fps(latest.current.referenceShot))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fpsByShot])

  React.useEffect(() => {
    syncActive()
  }, [referenceShot, activeShot, offsets, syncActive])

  React.useEffect(() => {
    if (!playing) return
    const rv = refVideo.current
    const av = actVideo.current
    syncActive()
    rv?.play().catch(() => undefined)
    av?.play().catch(() => undefined)
    let raf = 0
    const tick = () => {
      readCursor()
      syncActive(DRIFT_S)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      rv?.pause()
      av?.pause()
    }
  }, [playing, syncActive, readCursor])

  const setOffset = (shotId: string, value: number) => {
    if (shotId === referenceShot) return // reference is pinned to 0
    setOffsets((prev) => ({ ...prev, [shotId]: Math.round(value) }))
    setMethods((prev) => ({ ...prev, [shotId]: { method: "manual", confidence: 1 } }))
    setDirty(true)
  }

  const pickActive = (shotId: string) => {
    if (shotId !== referenceShot) setActiveShot(shotId)
  }

  const changeReference = (shotId: string) => {
    setReferenceShot(shotId)
    setOffsets((prev) => ({ ...prev, [shotId]: 0 }))
    setDirty(true)
  }

  const lock = () => {
    const rv = refVideo.current
    const av = actVideo.current
    if (!rv || !av) return
    const refFrame = Math.round(rv.currentTime * fps(referenceShot))
    const actFrame = Math.round(av.currentTime * fps(activeShot))
    setOffset(activeShot, actFrame - refFrame)
  }

  const scrub = (globalFrame: number) => {
    const rv = refVideo.current
    if (!rv) return
    try {
      rv.currentTime = Math.max(0, globalFrame / fps(referenceShot))
    } catch {
      /* not seekable yet */
    }
  }

  const save = async () => {
    const alignments = ids.map((sid) => {
      const isRef = sid === referenceShot
      const m = methods[sid] ?? { method: "manual", confidence: 1 }
      return {
        shot_id: sid,
        frame_offset: isRef ? 0 : offsetOf(sid),
        method: isRef ? "manual" : m.method,
        confidence: isRef ? 1 : m.confidence,
      }
    })
    setSaving(true)
    try {
      const body = await postJson<{ count: number }>("/api/sync", {
        group_id: group.id,
        reference_shot: referenceShot,
        alignments,
      })
      setDirty(false)
      toast.success(`Saved ${group.label} sync (${body.count} shots)`)
      onSaved()
    } catch (err) {
      toast.error("Could not save sync offsets", { description: errorMessage(err) })
    } finally {
      setSaving(false)
    }
  }

  const maxFrames = Math.max(...ids.map((id) => framesByShot[id] ?? 0))

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 md:grid-cols-2">
        <SyncVideoColumn
          role="reference"
          title="Reference (offset 0)"
          shotIds={ids}
          value={referenceShot}
          disabledId={activeShot}
          fps={fps(referenceShot)}
          videoRef={refVideo}
          onChange={changeReference}
          onSeeked={() => {
            syncActive()
            readCursor()
          }}
          onTimeUpdate={readCursor}
          onLoadedMetadata={readCursor}
          onEnded={() => setPlaying(false)}
        />
        <SyncVideoColumn
          role="active"
          title="Clip to sync"
          shotIds={ids}
          value={activeShot}
          disabledId={referenceShot}
          fps={fps(activeShot)}
          videoRef={actVideo}
          onChange={pickActive}
          onLoadedMetadata={() => syncActive()}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="default" size="sm" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
          {playing ? <PauseIcon data-icon="inline-start" /> : <PlayIcon data-icon="inline-start" />}
          {playing ? "Pause" : "Play both"}
        </Button>
        <Button
          variant="outline"
          size="sm"
          title="Set the active clip's offset to (active frame − reference frame) from both videos' current positions."
          onClick={lock}
        >
          <LockIcon data-icon="inline-start" />
          Lock offset to current frames
        </Button>
        <span className="text-sm text-muted-foreground">Active offset</span>
        <OffsetInput label="Active clip offset in frames" value={offsetOf(activeShot)} onCommit={(v) => setOffset(activeShot, v)} />
        <IconButton label="Nudge active clip 1 frame earlier (offset −1)" onClick={() => setOffset(activeShot, offsetOf(activeShot) - 1)}>
          <ChevronLeftIcon />
        </IconButton>
        <IconButton label="Nudge active clip 1 frame later (offset +1)" onClick={() => setOffset(activeShot, offsetOf(activeShot) + 1)}>
          <ChevronRightIcon />
        </IconButton>
        <MethodBadge method={methods[activeShot]} />
        <div className="ml-auto flex items-center gap-2">
          {dirty ? <ToneBadge tone="warning">Unsaved changes</ToneBadge> : null}
          <Button size="sm" disabled={saving} onClick={() => void save()}>
            <SaveIcon data-icon="inline-start" />
            Save group
          </Button>
        </div>
      </div>

      <SyncTimeline
        shotIds={ids}
        framesByShot={framesByShot}
        fpsByShot={fpsByShot}
        referenceShot={referenceShot}
        activeShot={activeShot}
        offsets={offsets}
        methods={methods}
        cursorFrame={cursorFrame}
        onCommitOffset={setOffset}
        onPick={pickActive}
        onScrub={scrub}
      />

      <SyncOffsetRows
        shotIds={ids}
        referenceShot={referenceShot}
        activeShot={activeShot}
        offsets={offsets}
        methods={methods}
        maxFrames={maxFrames}
        onSetOffset={setOffset}
        onPick={pickActive}
      />
    </div>
  )
}

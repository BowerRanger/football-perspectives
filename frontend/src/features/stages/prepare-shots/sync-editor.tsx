import { frameAtTime, frameTime } from "@/lib/frame-time"
import * as React from "react"
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  CrosshairIcon,
  LockIcon,
  PauseIcon,
  PencilIcon,
  PlayIcon,
  SaveIcon,
  StepBackIcon,
  StepForwardIcon,
} from "lucide-react"
import { toast } from "sonner"

import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { useConfirm, usePrompt } from "@/hooks/use-dialogs"
import { useIsMobile } from "@/hooks/use-mobile"
import { usePipeline } from "@/hooks/use-pipeline"
import { useUnsavedGuard } from "@/hooks/use-unsaved-guard"
import { errorMessage, postJson } from "@/lib/api"

import { IconButton } from "./icon-button"
import { MomentsTray } from "./moments-tray"
import {
  deriveSpeedState,
  fmtRate,
  isCallToAction,
  momentsProblem,
  REAL_TIME_TOLERANCE,
  refFrameForShot,
  type SpeedState,
} from "./replay-speed"
import { RetimeButtons } from "./retime-actions"
import { retimeGateReason, type RetimeGateInputs } from "./retime-gate"
import { SpeedBadge } from "./speed-badge"
import { MethodBadge, OffsetInput, SyncOffsetRows } from "./sync-offsets"
import { SyncTimeline, type AlignMethod } from "./sync-timeline"
import { SyncVideoColumn } from "./sync-video-column"
import type { GroupView, SyncAlignment } from "./types"
import { useMomentKeys } from "./use-moment-keys"
import { useMoments } from "./use-moments"
import { useRetimeActions } from "./use-retime-actions"
import { usePlaybackSync } from "./use-sync-playback"
import type { ReplaySyncLookup } from "./use-replay-sync"

interface EditorProps {
  group: GroupView
  /** Called after a successful save so tile badges can refresh (no reseed). */
  onSaved: () => void
  /** Full reload of manifest + sync (reseeds this editor); used after retime / restore. */
  onReload: () => Promise<void>
  replaySync: ReplaySyncLookup
  /**
   * Gate inputs, owned by GroupSync so they survive the re-key that follows a
   * retime: an Undo toast made by the old editor must see the new state.
   */
  gateRef: React.MutableRefObject<RetimeGateInputs>
}

const MIN_PLAYBACK = 0.0625
const MAX_PLAYBACK = 16
/** Shown wherever a control is locked because marked pairs drive the active clip. */
const PAIRS_LOCK = "Your marked pairs set this clip's offset and rate — save or clear them."
const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

function seedState(group: GroupView) {
  const ids = group.members.map((m) => m.id)
  const framesByShot: Record<string, number> = {}
  const fpsByShot: Record<string, number> = {}
  for (const m of group.members) {
    const clipFrames = Math.max(1, m.end_frame - m.start_frame + 1)
    const dur = m.end_time - m.start_time
    fpsByShot[m.id] = dur > 0 ? clipFrames / dur : 25
    // A retimed clip has native_frames / speed_factor frames (= native * rate).
    framesByShot[m.id] =
      m.retimed && (m.native_frames ?? 0) > 0 && m.speed_factor > 0
        ? Math.max(1, Math.round((m.native_frames ?? 0) / m.speed_factor))
        : clipFrames
  }
  const offsets: Record<string, number> = {}
  const rates: Record<string, number> = {}
  const methods: Record<string, AlignMethod> = {}
  for (const a of group.sync?.alignments ?? []) {
    offsets[a.shot_id] = a.frame_offset
    rates[a.shot_id] = a.playback_rate && a.playback_rate > 0 ? a.playback_rate : 1
    methods[a.shot_id] = { method: a.method, confidence: a.confidence }
  }
  for (const id of ids) {
    offsets[id] ??= 0
    rates[id] ??= 1
    methods[id] ??= { method: "manual", confidence: 1 }
  }
  const saved = group.sync?.reference_shot
  const reference = saved && ids.includes(saved) ? saved : ids[0]
  const active = ids.find((s) => s !== reference) ?? ids[0]
  return { ids, framesByShot, fpsByShot, offsets, rates, methods, reference, active }
}

/**
 * Two-video scrub + draggable timeline for one highlight group.
 * Time map (sync_map.py): reference_frame = rate * shot_frame - frame_offset;
 * at rate 1 that is frame_offset = frame_in_active - frame_in_reference.
 * Any edit becomes `manual` and survives re-alignment.
 */
export function SyncEditor({ group, onSaved, onReload, replaySync, gateRef }: EditorProps) {
  // Seeded once per mount; the parent re-keys the editor to reseed.
  const [seed] = React.useState(() => seedState(group))
  const { ids, framesByShot, fpsByShot } = seed
  const isMobile = useIsMobile()
  const confirm = useConfirm()
  const prompt = usePrompt()
  const [referenceShot, setReferenceShot] = React.useState(seed.reference)
  const [activeShot, setActiveShot] = React.useState(seed.active)
  const [offsets, setOffsets] = React.useState(seed.offsets)
  const [rates, setRates] = React.useState(seed.rates)
  const [methods, setMethods] = React.useState(seed.methods)
  const [playing, setPlaying] = React.useState(false)
  const [dirty, setDirty] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [trayOpen, setTrayOpen] = React.useState(false)
  const [savingMoments, setSavingMoments] = React.useState(false)
  const [speedMode, setSpeedMode] = React.useState<"fit" | "raw">("fit")
  const [focusWell, setFocusWell] = React.useState<"reference" | "active">("reference")
  const refVideo = React.useRef<HTMLVideoElement>(null)
  const actVideo = React.useRef<HTMLVideoElement>(null)
  const regionRef = React.useRef<HTMLDivElement>(null)
  const { isRunning, runningLabel } = usePipeline()

  const moments = useMoments(activeShot)
  // Re-read at click time: toast actions outlive the state they were created in.
  React.useEffect(() => {
    gateRef.current = { isRunning, runningLabel, dirty, hasUnsaved: moments.hasUnsaved }
  })
  const retimeGate = React.useCallback((): string => retimeGateReason(gateRef.current), [gateRef])
  const retimeActions = useRetimeActions(onReload, retimeGate)

  // The tray's fit previews on the active clip until it is saved or cleared.
  const preview = moments.fit && moments.pairs.length >= 2 ? moments.fit : null
  const effOffsets = React.useMemo(
    () => (preview ? { ...offsets, [activeShot]: preview.frameOffset } : offsets),
    [offsets, preview, activeShot],
  )
  const effRates = React.useMemo(
    () => (preview ? { ...rates, [activeShot]: preview.rate } : rates),
    [rates, preview, activeShot],
  )

  const effMethods = React.useMemo(
    () => (preview ? { ...methods, [activeShot]: { method: "manual", confidence: 1 } } : methods),
    [methods, preview, activeShot],
  )

  const fps = (id: string) => fpsByShot[id] || 25
  const offsetOf = (id: string) => effOffsets[id] ?? 0
  const rateOf = (id: string) => (id === referenceShot ? 1 : (effRates[id] ?? 1))
  const { syncActive, readCursor, cursorFrame } = usePlaybackSync({
    refVideo,
    actVideo,
    fpsByShot,
    state: { referenceShot, activeShot, offsets: effOffsets, rates: effRates, speedMode, playing },
  })

  useUnsavedGuard(dirty || moments.hasUnsaved, { what: moments.hasUnsaved ? "marked moments" : "sync offsets" })

  const setOffset = (shotId: string, value: number) => {
    if (shotId === referenceShot) return // reference is pinned to 0
    if (preview && shotId === activeShot) return // marked pairs own this clip until saved or cleared
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
    setRates((prev) => ({ ...prev, [shotId]: 1 }))
    setDirty(true)
  }

  const frameOf = (video: HTMLVideoElement | null, shotId: string): number | null =>
    video ? frameAtTime(video.currentTime, fps(shotId), Math.max(0, (framesByShot[shotId] ?? 1) - 1)) : null

  /** Offset so that the two videos' current frames correspond, at the active clip's rate. */
  const lock = () => {
    const refFrame = frameOf(refVideo.current, referenceShot)
    const actFrame = frameOf(actVideo.current, activeShot)
    if (refFrame == null || actFrame == null) return
    setOffset(activeShot, rateOf(activeShot) * actFrame - refFrame)
  }

  const editRate = async () => {
    const current = rates[activeShot] ?? 1
    const raw = await prompt({
      title: `Playback rate for ${activeShot}`,
      description: "Reference frames per replay frame: 1 is real time, 0.34 is slow motion. Saved as a manual alignment.",
      label: "Rate",
      defaultValue: current.toFixed(3),
      confirmLabel: "Set rate",
      validate: (v) => {
        const n = Number(v)
        return Number.isFinite(n) && n >= 0.05 && n <= 4 ? null : "Enter a rate between 0.05 and 4."
      },
    })
    if (raw == null) return
    setRates((prev) => ({ ...prev, [activeShot]: Number(raw) }))
    setMethods((prev) => ({ ...prev, [activeShot]: { method: "manual", confidence: 1 } }))
    setDirty(true)
  }

  const scrub = (globalFrame: number) => {
    const rv = refVideo.current
    if (!rv) return
    try {
      rv.currentTime = Math.max(0, frameTime(globalFrame, fps(referenceShot)))
    } catch {
      /* not seekable yet */
    }
  }

  const stepFocused = (delta: number) => {
    setPlaying(false)
    const isRef = focusWell === "reference"
    const video = isRef ? refVideo.current : actVideo.current
    const id = isRef ? referenceShot : activeShot
    const f = frameOf(video, id)
    if (!video || f == null) return
    const next = clamp(f + delta, 0, Math.max(0, (framesByShot[id] ?? 1) - 1))
    try {
      video.currentTime = frameTime(next, fps(id))
    } catch {
      /* not seekable yet */
    }
  }

  const save = async () => {
    if (preview) {
      toast.error("Can't save the group yet", { description: PAIRS_LOCK })
      return
    }
    const alignments: (SyncAlignment & { playback_rate: number })[] = ids.map((sid) => {
      const isRef = sid === referenceShot
      const m = methods[sid] ?? { method: "manual", confidence: 1 }
      return {
        shot_id: sid,
        frame_offset: isRef ? 0 : (offsets[sid] ?? 0),
        method: isRef ? "manual" : m.method,
        confidence: isRef ? 1 : m.confidence,
        playback_rate: isRef ? 1 : (rates[sid] ?? 1),
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

  const momentsBlocked = !group.id
    ? "Ungrouped shots can't be marked: put them in a highlight group first."
    : referenceShot !== seed.reference
      ? "The reference changed: save the group first."
      : ""

  const frameCount = (id: string) => framesByShot[id] ?? 0

  const saveMoments = async () => {
    const fit = moments.fit
    if (!fit) return
    setSavingMoments(true)
    try {
      const res = await postJson<{
        rate: number
        offset: number
        residual_frames: number
        ramp: boolean
        alignment: SyncAlignment | null
        note: string
      }>(`/api/sync/groups/${encodeURIComponent(group.id)}/moments`, {
        shot_id: activeShot,
        moments: moments.pairs,
        retime: false,
      })
      const al = res.alignment
      if (al) {
        setOffsets((prev) => ({ ...prev, [al.shot_id]: al.frame_offset }))
        setRates((prev) => ({ ...prev, [al.shot_id]: al.playback_rate ?? res.rate }))
        setMethods((prev) => ({ ...prev, [al.shot_id]: { method: al.method, confidence: al.confidence } }))
      }
      const savedShot = activeShot
      const shotFrames = frameCount(savedShot)
      moments.reset(savedShot)
      onSaved()
      const canRetime = !res.ramp && res.rate < 1 - REAL_TIME_TOLERANCE
      toast.success(`Saved ${savedShot}: ${fmtRate(res.rate)}, offset ${(-res.offset).toFixed(1)} (marked by you)`, {
        description: res.ramp ? "The pairs show a speed ramp; the average rate was saved." : undefined,
        action: canRetime
          ? {
              label: "Retime to real time",
              onClick: () => void retimeActions.retime({ shotId: savedShot, rate: res.rate, frames: shotFrames }),
            }
          : undefined,
      })
    } catch (err) {
      toast.error("Could not save the matched moments", { description: errorMessage(err) })
    } finally {
      setSavingMoments(false)
    }
  }

  const clearMoments = async () => {
    if (moments.pairs.length >= 3) {
      const ok = await confirm({
        title: "Clear all marked pairs?",
        description: `${moments.pairs.length} unsaved pairs for ${activeShot} will be removed.`,
        confirmLabel: "Clear pairs",
        destructive: true,
      })
      if (!ok) return
    }
    moments.clear()
  }

  const keys = useMomentKeys({
    trayOpen,
    togglePlay: () => setPlaying((p) => !p),
    step: stepFocused,
    switchWell: () => setFocusWell((w) => (w === "reference" ? "active" : "reference")),
    toggleTray: () => setTrayOpen((o) => !o),
    markRef: () => {
      const f = frameOf(refVideo.current, referenceShot)
      if (f != null) moments.markRef(f)
    },
    markShot: () => {
      const f = frameOf(actVideo.current, activeShot)
      if (f != null) moments.markShot(f)
    },
    nudgeMark: (d) => {
      if (moments.pendingShot == null) return
      const next = clamp(moments.pendingShot + d, 0, Math.max(0, frameCount(activeShot) - 1))
      moments.nudgeShot(next - moments.pendingShot)
      try {
        if (actVideo.current) actVideo.current.currentTime = frameTime(next, fps(activeShot))
      } catch {
        /* not seekable yet */
      }
    },
    add: () => {
      moments.add()
    },
    discard: () => moments.discardPending(),
    closeTray: () => setTrayOpen(false),
    removeLast: () => moments.removeLatest(),
    save: () => {
      if (!preview) {
        void save()
        return
      }
      const why = momentsBlocked || momentsProblem(moments.pairs, moments.fit)
      if (why) toast.error("Can't save the marked pairs", { description: why })
      else void saveMoments()
    },
  })

  // Speed state per member (reference gets none).
  const speedStates = React.useMemo(() => {
    const out: Record<string, SpeedState> = {}
    for (const m of group.members) {
      const previewing = !!preview && m.id === activeShot
      const mt = previewing ? { method: "manual", confidence: 1 } : methods[m.id]
      const state = deriveSpeedState({
        isReference: m.id === referenceShot,
        shot: m,
        alignment: mt ? { method: mt.method, confidence: mt.confidence, playback_rate: effRates[m.id] ?? 1 } : undefined,
        member: replaySync.byShot[m.id],
        detecting: replaySync.detecting,
      })
      out[m.id] = previewing ? { ...state, text: state.text.replace("set by you", "your pairs, not saved") } : state
    }
    return out
  }, [group.members, methods, effRates, referenceShot, replaySync, preview, activeShot])

  const speedNotes = React.useMemo(() => {
    const out: Record<string, string> = {}
    for (const m of group.members) {
      const est = replaySync.byShot[m.id]?.estimate
      if (est && methods[m.id]?.method === "manual") {
        out[m.id] = `Automatic estimate: ${fmtRate(est.rate)} at ${Math.round(est.confidence * 100)} %; your alignment was kept.`
      }
    }
    return out
  }, [group.members, methods, replaySync])

  const rampShots = ids.filter((id) => speedStates[id]?.kind === "ramp")
  const approxShots = ids.filter((id) => speedStates[id]?.kind === "approximate")
  const clipVersion = (id: string) => {
    const s = group.members.find((m) => m.id === id)
    return s ? `${s.retimed ? 1 : 0}-${s.native_frames ?? 0}` : ""
  }
  const activeRate = rateOf(activeShot)
  const playbackRate = clamp(speedMode === "fit" ? 1 / activeRate : 1, MIN_PLAYBACK, MAX_PLAYBACK)
  const activeSpeed = speedStates[activeShot]
  const maxFrames = Math.max(...ids.map((id) => framesByShot[id] ?? 0))

  return (
    <div
      ref={regionRef}
      role="group"
      aria-label="Group sync editor"
      tabIndex={0}
      onKeyDown={isMobile ? undefined : keys}
      className="flex flex-col gap-4 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
    >
      <div className="grid gap-3 md:grid-cols-2">
        <SyncVideoColumn
          role="reference"
          title="Reference (offset 0)"
          shotIds={ids}
          value={referenceShot}
          disabledId={activeShot}
          fps={fps(referenceShot)}
          videoRef={refVideo}
          badge={<ToneBadge tone="info">Reference</ToneBadge>}
          clipVersion={clipVersion(referenceShot)}
          focused={trayOpen && focusWell === "reference"}
          onFocusWell={() => setFocusWell("reference")}
          pendingMark={moments.pendingRef}
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
          badge={
            activeSpeed ? (
              <SpeedBadge
                state={activeSpeed}
                note={speedNotes[activeShot]}
                onClick={isCallToAction(activeSpeed) && !isMobile ? () => setTrayOpen(true) : undefined}
              />
            ) : undefined
          }
          playbackRate={playbackRate}
          clipVersion={clipVersion(activeShot)}
          focused={trayOpen && focusWell === "active"}
          onFocusWell={() => setFocusWell("active")}
          pendingMark={moments.pendingShot}
          equivalent={
            activeRate === 1 ? undefined : (f) => `live frame ${refFrameForShot(f, activeRate, offsetOf(activeShot)).toFixed(1)}`
          }
          onChange={pickActive}
          onLoadedMetadata={() => syncActive()}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button variant={trayOpen ? "outline" : "default"} size="sm" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
          {playing ? <PauseIcon data-icon="inline-start" /> : <PlayIcon data-icon="inline-start" />}
          {playing ? "Pause" : "Play both"}
        </Button>
        <span className="hidden items-center gap-2 md:flex">
          <IconButton label="Step the focused clip back 1 frame ( , )" onClick={() => stepFocused(-1)}>
            <StepBackIcon />
          </IconButton>
          <IconButton label="Step the focused clip forward 1 frame ( . )" onClick={() => stepFocused(1)}>
            <StepForwardIcon />
          </IconButton>
          {activeRate !== 1 ? (
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={speedMode}
              onValueChange={(v) => v && setSpeedMode(v as "fit" | "raw")}
              aria-label="Replay playback speed"
            >
              <ToggleGroupItem value="fit" title={`Plays ${activeShot} at ${(1 / activeRate).toFixed(2)}× so both stay in step`}>
                Fit
              </ToggleGroupItem>
              <ToggleGroupItem value="raw" title="Both clips at their own speed">
                1×
              </ToggleGroupItem>
            </ToggleGroup>
          ) : null}
        </span>
        {isMobile ? (
          <span className="text-xs text-muted-foreground">Marking moments and retiming are desktop-only.</span>
        ) : null}
      </div>

      <div className="hidden flex-wrap items-center gap-2 md:flex">
        <Button variant="outline" size="sm" aria-pressed={trayOpen} onClick={() => setTrayOpen((o) => !o)}>
          <CrosshairIcon data-icon="inline-start" />
          Match moments <Kbd>M</Kbd>
        </Button>
        <Button
          variant="outline"
          size="sm"
          title={preview ? PAIRS_LOCK : "Set the active clip's offset from both videos' current frames, at its current rate."}
          disabled={!!preview}
          onClick={lock}
        >
          <LockIcon data-icon="inline-start" />
          Lock offset to current frames
        </Button>
        <span className="text-sm text-muted-foreground">Active offset</span>
        <OffsetInput
          label="Active clip offset in frames"
          value={offsetOf(activeShot)}
          disabled={!!preview}
          onCommit={(v) => setOffset(activeShot, v)}
        />
        <IconButton label="Nudge active clip 1 frame earlier (offset −1)" disabled={!!preview} onClick={() => setOffset(activeShot, offsetOf(activeShot) - 1)}>
          <ChevronLeftIcon />
        </IconButton>
        <IconButton label="Nudge active clip 1 frame later (offset +1)" disabled={!!preview} onClick={() => setOffset(activeShot, offsetOf(activeShot) + 1)}>
          <ChevronRightIcon />
        </IconButton>
        <span className="text-sm text-muted-foreground">Rate</span>
        <span className="font-mono text-sm tabular-nums" data-testid="active-rate">
          {activeRate.toFixed(3)}×
        </span>
        <IconButton label="Set the playback rate by hand" variant="ghost" size="icon-xs" disabled={!!preview} onClick={() => void editRate()}>
          <PencilIcon />
        </IconButton>
        {preview ? <ToneBadge tone="warning">Unsaved pairs</ToneBadge> : <MethodBadge method={effMethods[activeShot]} />}
        <div className="ml-auto flex items-center gap-2">
          {dirty ? <ToneBadge tone="warning">Unsaved changes</ToneBadge> : null}
          <Button size="sm" variant={trayOpen && moments.pairs.length > 0 ? "outline" : "default"} disabled={saving || !!preview} title={preview ? PAIRS_LOCK : undefined} onClick={() => void save()}>
            <SaveIcon data-icon="inline-start" />
            Save group
          </Button>
        </div>
      </div>

      {preview && !isMobile ? <p className="text-xs text-warning">{PAIRS_LOCK}</p> : null}

      {trayOpen && !isMobile ? (
        <MomentsTray
          referenceShot={referenceShot}
          memberShot={activeShot}
          moments={moments}
          saving={savingMoments}
          blockedReason={momentsBlocked}
          readReference={() => frameOf(refVideo.current, referenceShot)}
          readMember={() => frameOf(actVideo.current, activeShot)}
          onActed={() => regionRef.current?.focus({ preventScroll: true })}
          onSave={() => void saveMoments()}
          onClose={() => setTrayOpen(false)}
          onClear={() => void clearMoments()}
        />
      ) : null}

      <SyncTimeline
        shotIds={ids}
        framesByShot={framesByShot}
        fpsByShot={fpsByShot}
        referenceShot={referenceShot}
        activeShot={activeShot}
        offsets={effOffsets}
        rates={effRates}
        methods={effMethods}
        rampShots={rampShots}
        approxShots={approxShots}
        pairs={moments.pairs}
        previewActive={!!preview}
        lockedShot={preview ? activeShot : null}
        readOnly={isMobile}
        cursorFrame={cursorFrame}
        onCommitOffset={setOffset}
        onPick={pickActive}
        onScrub={scrub}
      />

      <SyncOffsetRows
        shotIds={ids}
        referenceShot={referenceShot}
        activeShot={activeShot}
        offsets={effOffsets}
        methods={effMethods}
        maxFrames={maxFrames}
        onSetOffset={setOffset}
        onPick={pickActive}
        speedStates={speedStates}
        speedNotes={speedNotes}
        readOnly={isMobile}
        lockedShot={preview ? activeShot : null}
        lockedReason={PAIRS_LOCK}
        onOpenMoments={(id) => {
          pickActive(id)
          setTrayOpen(true)
        }}
        renderActions={(id) => {
          const m = group.members.find((x) => x.id === id)
          const st = speedStates[id]
          if (!m || !st) return null
          return (
            <RetimeButtons
              shotId={id}
              state={st}
              frames={frameCount(id)}
              retimed={!!m.retimed}
              dirty={dirty}
              pendingPairs={moments.hasUnsaved}
              wasRate={m.retimed && m.speed_factor > 0 ? 1 / m.speed_factor : undefined}
              actions={retimeActions}
            />
          )
        }}
      />
    </div>
  )
}

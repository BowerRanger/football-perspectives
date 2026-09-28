import * as React from "react"
import { toast } from "sonner"

import { deleteJson, errorMessage, getJson, postJson } from "@/lib/api"
import {
  humanizeStageName,
  STAGE_DEPS,
  type LiveStageState,
  type StageInfo,
  type StageName,
} from "@/lib/stages"

// Pipeline run + log state shared by the sidebar, stage header, log dock and
// any panel that launches a job (uploads, per-shot runs). One job may be in
// flight at a time; every run trigger reads `runningLabel` to lock itself.

export type LogStatus = "idle" | "running" | "done" | "error"

const MAX_LOG_LINES = 5000

interface LogState {
  lines: string[]
  status: LogStatus
  title: string
  startedAt: number | null
  finishedAt: number | null
}

interface PipelineContextValue {
  stages: StageInfo[]
  stagesLoaded: boolean
  refreshStages: () => Promise<void>
  liveState: Partial<Record<StageName, LiveStageState>>
  /** Human label of the in-flight run ("Camera Tracking", "Pipeline"), or null. */
  runningLabel: string | null
  isRunning: boolean
  /** Bumps whenever a job finishes so stage panels re-fetch their output. */
  outputVersion: number
  bumpOutputVersion: () => void
  startRun: (stages: StageName | "all", fromStage?: StageName) => Promise<void>
  rerunStage: (stage: StageName) => Promise<void>
  attachToJob: (jobId: string, targetStage: StageName | string, onDone?: (status: string) => void) => void
  missingDeps: (stage: StageName) => StageName[]
  log: LogState
  logOpen: boolean
  setLogOpen: (open: boolean) => void
  clearLog: () => void
}

const PipelineContext = React.createContext<PipelineContextValue | null>(null)

export function usePipeline(): PipelineContextValue {
  const ctx = React.useContext(PipelineContext)
  if (!ctx) throw new Error("usePipeline must be used within <PipelineProvider>")
  return ctx
}

const EMPTY_LOG: LogState = { lines: [], status: "idle", title: "", startedAt: null, finishedAt: null }

export function PipelineProvider({ children }: { children: React.ReactNode }) {
  const [stages, setStages] = React.useState<StageInfo[]>([])
  const [stagesLoaded, setStagesLoaded] = React.useState(false)
  const [liveState, setLiveState] = React.useState<Partial<Record<StageName, LiveStageState>>>({})
  const [runningLabel, setRunningLabel] = React.useState<string | null>(null)
  const [outputVersion, setOutputVersion] = React.useState(0)
  const [log, setLog] = React.useState<LogState>(EMPTY_LOG)
  const [logOpen, setLogOpen] = React.useState(false)
  const sourceRef = React.useRef<EventSource | null>(null)

  const refreshStages = React.useCallback(async () => {
    try {
      const data = await getJson<StageInfo[]>("/api/stages")
      setStages(data)
    } catch (err) {
      toast.error("Could not load pipeline stages", { description: errorMessage(err) })
    } finally {
      setStagesLoaded(true)
    }
  }, [])

  React.useEffect(() => {
    void refreshStages()
    return () => sourceRef.current?.close()
  }, [refreshStages])

  const bumpOutputVersion = React.useCallback(() => setOutputVersion((v) => v + 1), [])

  const markRunning = React.useCallback((targets: string[], label: string) => {
    setLiveState((prev) => {
      const next = { ...prev }
      for (const t of targets) next[t as StageName] = "running"
      return next
    })
    setRunningLabel(label)
    setLog({ lines: [], status: "running", title: label, startedAt: Date.now(), finishedAt: null })
    setLogOpen(true)
  }, [])

  const streamJob = React.useCallback(
    (jobId: string, targets: string[], label: string, onDone?: (status: string) => void) => {
      sourceRef.current?.close()
      const source = new EventSource(`/api/jobs/${jobId}/logs`)
      sourceRef.current = source
      // Batch log lines per animation frame — a chatty stage emits hundreds
      // of lines a second and one setState per line would thrash React.
      let pending: string[] = []
      let raf = 0
      const flush = () => {
        raf = 0
        if (!pending.length) return
        const chunk = pending
        pending = []
        setLog((prev) => {
          const lines = prev.lines.concat(chunk)
          return { ...prev, lines: lines.length > MAX_LOG_LINES ? lines.slice(-MAX_LOG_LINES) : lines }
        })
      }
      source.addEventListener("log", (e) => {
        try {
          const { line } = JSON.parse((e as MessageEvent).data) as { line: string }
          pending.push(line)
          if (!raf) raf = requestAnimationFrame(flush)
        } catch {
          /* ignore malformed frame */
        }
      })
      source.addEventListener("done", (e) => {
        source.close()
        sourceRef.current = null
        if (raf) cancelAnimationFrame(raf)
        flush()
        let status = "error"
        try {
          status = (JSON.parse((e as MessageEvent).data) as { status: string }).status
        } catch {
          /* keep error */
        }
        const ok = status === "done"
        setLiveState((prev) => {
          const next = { ...prev }
          for (const t of targets) {
            if (ok) delete next[t as StageName]
            else next[t as StageName] = "error"
          }
          return next
        })
        setRunningLabel(null)
        setLog((prev) => ({ ...prev, status: ok ? "done" : "error", finishedAt: Date.now() }))
        if (ok) toast.success(`${label} finished`)
        else toast.error(`${label} failed`, { description: "The log dock has the full traceback." })
        void refreshStages().then(() => {
          setOutputVersion((v) => v + 1)
          onDone?.(status)
        })
      })
      source.onerror = () => {
        // EventSource auto-reconnects; only surface a permanent close.
        if (source.readyState === EventSource.CLOSED && sourceRef.current === source) {
          setLog((prev) => ({ ...prev, lines: prev.lines.concat("[dashboard] log stream disconnected") }))
        }
      }
    },
    [refreshStages],
  )

  const startRun = React.useCallback(
    async (target: StageName | "all", fromStage?: StageName) => {
      const targets = target === "all" ? stages.map((s) => s.name) : [target]
      const label = target === "all" ? "Pipeline" : humanizeStageName(target)
      markRunning(targets, label)
      try {
        const body = fromStage ? { stages: target, from_stage: fromStage } : { stages: target }
        const { job_id } = await postJson<{ job_id: string }>("/api/run", body)
        streamJob(job_id, targets, label)
      } catch (err) {
        setLiveState((prev) => {
          const next = { ...prev }
          for (const t of targets) next[t as StageName] = "error"
          return next
        })
        setRunningLabel(null)
        setLog((prev) => ({
          ...prev,
          status: "error",
          finishedAt: Date.now(),
          lines: prev.lines.concat(`Failed to start: ${errorMessage(err)}`),
        }))
        toast.error(`Could not start ${label}`, { description: errorMessage(err) })
      }
    },
    [stages, markRunning, streamJob],
  )

  const rerunStage = React.useCallback(
    async (stage: StageName) => {
      try {
        await deleteJson(`/api/output/${stage}`)
      } catch (err) {
        toast.error(`Could not clear ${humanizeStageName(stage)} output`, { description: errorMessage(err) })
        return
      }
      await startRun(stage)
    },
    [startRun],
  )

  const attachToJob = React.useCallback(
    (jobId: string, targetStage: StageName | string, onDone?: (status: string) => void) => {
      const label = humanizeStageName(targetStage)
      markRunning([targetStage], label)
      streamJob(jobId, [targetStage], label, onDone)
    },
    [markRunning, streamJob],
  )

  const missingDeps = React.useCallback(
    (stage: StageName) => {
      const done = new Map(stages.map((s) => [s.name, s.complete]))
      return (STAGE_DEPS[stage] ?? []).filter((d) => !done.get(d))
    },
    [stages],
  )

  const clearLog = React.useCallback(() => {
    setLog(EMPTY_LOG)
    setLogOpen(false)
  }, [])

  const value = React.useMemo<PipelineContextValue>(
    () => ({
      stages,
      stagesLoaded,
      refreshStages,
      liveState,
      runningLabel,
      isRunning: runningLabel !== null,
      outputVersion,
      bumpOutputVersion,
      startRun,
      rerunStage,
      attachToJob,
      missingDeps,
      log,
      logOpen,
      setLogOpen,
      clearLog,
    }),
    [
      stages, stagesLoaded, refreshStages, liveState, runningLabel, outputVersion, bumpOutputVersion,
      startRun, rerunStage, attachToJob, missingDeps, log, logOpen, clearLog,
    ],
  )

  return <PipelineContext.Provider value={value}>{children}</PipelineContext.Provider>
}

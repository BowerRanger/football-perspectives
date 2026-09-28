import * as React from "react"
import { toast } from "sonner"

import { ApiError, errorMessage, getJson, postJson } from "@/lib/api"
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
  /** SSE health: "reconnecting" while the dashboard retries a dropped stream. */
  connection: "ok" | "reconnecting"
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

const EMPTY_LOG: LogState = {
  lines: [],
  status: "idle",
  title: "",
  startedAt: null,
  finishedAt: null,
  connection: "ok",
}

const RECONNECT_MAX_MS = 15_000

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

  const markRunning = React.useCallback((targets: string[], label: string, startedAt: number = Date.now()) => {
    setLiveState((prev) => {
      const next = { ...prev }
      for (const t of targets) next[t as StageName] = "running"
      return next
    })
    setRunningLabel(label)
    setLog({ ...EMPTY_LOG, status: "running", title: label, startedAt })
    setLogOpen(true)
  }, [])

  const finishJob = React.useCallback(
    (targets: string[], label: string, status: string, onDone?: (status: string) => void) => {
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
      setLog((prev) => ({ ...prev, status: ok ? "done" : "error", finishedAt: Date.now(), connection: "ok" }))
      if (ok) toast.success(`${label} finished`)
      else toast.error(`${label} failed`, { description: "The log dock shows the traceback." })
      void refreshStages().then(() => {
        setOutputVersion((v) => v + 1)
        onDone?.(status)
      })
    },
    [refreshStages],
  )

  type StreamFn = (
    jobId: string,
    targets: string[],
    label: string,
    onDone?: (status: string) => void,
    attempt?: number,
  ) => void
  // Reconnects re-enter the stream through a ref: a useCallback can't
  // safely capture itself while it is being initialised.
  const streamJobRef = React.useRef<StreamFn | null>(null)

  const streamJob = React.useCallback<StreamFn>(
    (jobId, targets, label, onDone, attempt = 0) => {
      sourceRef.current?.close()
      const source = new EventSource(`/api/jobs/${jobId}/logs`)
      sourceRef.current = source
      // The server replays the whole log on (re)connect, so a reconnect
      // starts from an empty buffer instead of appending duplicates.
      let replaced = attempt === 0
      // Batch log lines per animation frame — a chatty stage emits hundreds
      // of lines a second and one setState per line would thrash React.
      let pending: string[] = []
      let raf = 0
      const flush = () => {
        raf = 0
        if (!pending.length) return
        const chunk = pending
        pending = []
        const reset = !replaced
        replaced = true
        setLog((prev) => {
          const lines = (reset ? [] : prev.lines).concat(chunk)
          return {
            ...prev,
            connection: "ok",
            lines: lines.length > MAX_LOG_LINES ? lines.slice(-MAX_LOG_LINES) : lines,
          }
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
        finishJob(targets, label, status, onDone)
      })
      source.onerror = () => {
        if (sourceRef.current !== source) return
        // Take over from EventSource's own retry (which would replay into
        // the existing buffer): back off, confirm the job still exists,
        // then reopen the stream.
        source.close()
        sourceRef.current = null
        setLog((prev) => ({ ...prev, connection: "reconnecting" }))
        const delay = Math.min(1000 * 2 ** attempt, RECONNECT_MAX_MS)
        window.setTimeout(() => {
          void getJson<{ status: string }>(`/api/jobs/${jobId}/status`)
            .then(() => streamJobRef.current?.(jobId, targets, label, onDone, attempt + 1))
            .catch((err: unknown) => {
              if (err instanceof ApiError && err.status === 404) {
                setLog((prev) => ({
                  ...prev,
                  lines: prev.lines.concat("[dashboard] The server restarted and this job is gone."),
                }))
                finishJob(targets, label, "error", onDone)
              } else {
                streamJobRef.current?.(jobId, targets, label, onDone, attempt + 1)
              }
            })
        }, delay)
      }
    },
    [finishJob],
  )

  React.useEffect(() => {
    streamJobRef.current = streamJob
  }, [streamJob])

  const runJob = React.useCallback(
    async (target: StageName | "all", opts: { fromStage?: StageName; cleanFirst?: boolean }) => {
      const targets = target === "all" ? stages.map((s) => s.name) : [target]
      const label = target === "all" ? "Pipeline" : humanizeStageName(target)
      markRunning(targets, label)
      try {
        const body = {
          stages: target,
          ...(opts.fromStage ? { from_stage: opts.fromStage } : {}),
          // Server clears outputs only once the run is accepted, so a
          // rejected run (409/429) never loses the previous output.
          ...(opts.cleanFirst ? { clean_first: true } : {}),
        }
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

  const startRun = React.useCallback(
    (target: StageName | "all", fromStage?: StageName) => runJob(target, { fromStage }),
    [runJob],
  )

  const rerunStage = React.useCallback((stage: StageName) => runJob(stage, { cleanFirst: true }), [runJob])

  // Reattach to a run that was already in flight when the page loaded
  // (reload, or a second tab): the SSE endpoint replays the whole log.
  const reattachedRef = React.useRef(false)
  React.useEffect(() => {
    if (!stagesLoaded || reattachedRef.current) return
    reattachedRef.current = true
    void getJson<{ job_id: string; stages: string; started_at?: number }[]>("/api/jobs?status=running")
      .then((jobs) => {
        const job = jobs[0]
        if (!job) return
        const targets = job.stages === "all" ? stages.map((s) => s.name) : job.stages.split(",")
        const label = job.stages === "all" ? "Pipeline" : humanizeStageName(targets[0])
        // Server epoch seconds → the log dock's elapsed timer shows the real run time.
        markRunning(targets, label, job.started_at ? job.started_at * 1000 : Date.now())
        streamJob(job.job_id, targets, label)
      })
      .catch(() => {
        /* older server without /api/jobs — nothing to reattach */
      })
  }, [stagesLoaded, stages, markRunning, streamJob])

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

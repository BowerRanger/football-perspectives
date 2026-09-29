import * as React from "react"
import { useVirtualizer } from "@tanstack/react-virtual"
import {
  ArrowDownToLineIcon,
  CopyIcon,
  DownloadIcon,
  MinusIcon,
  SearchXIcon,
  SquareIcon,
  TerminalIcon,
  WifiOffIcon,
  XIcon,
} from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { ToneBadge } from "@/components/status"
import { useConfirm } from "@/hooks/use-dialogs"
import { usePipeline, type LogStatus } from "@/hooks/use-pipeline"
import { logText, type LogLine } from "@/lib/log-buffer"
import { cn } from "@/lib/utils"

const LINE_HEIGHT_PX = 20

function useElapsed(startedAt: number | null, finishedAt: number | null): string {
  const [now, setNow] = React.useState(() => Date.now())
  React.useEffect(() => {
    if (!startedAt || finishedAt) return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [startedAt, finishedAt])
  if (!startedAt) return ""
  const secs = Math.max(0, Math.round(((finishedAt ?? now) - startedAt) / 1000))
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`
}

function IconAction({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon-sm" onClick={onClick} aria-label={label}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

const LEVEL_CLASS: Record<LogLine["level"], string> = {
  info: "",
  warn: "text-warning",
  error: "bg-destructive/15 text-destructive",
}

function StatusChip({ status, cancelling }: { status: LogStatus; cancelling: boolean }) {
  if (status === "running") {
    return (
      <ToneBadge tone="warning">
        <Spinner className="size-3" /> {cancelling ? "Cancelling…" : "Running"}
      </ToneBadge>
    )
  }
  if (status === "done") return <ToneBadge tone="success">Finished</ToneBadge>
  if (status === "cancelled") return <ToneBadge tone="muted">Cancelled</ToneBadge>
  return <ToneBadge tone="destructive">Failed</ToneBadge>
}

/**
 * Run log, docked to the bottom of the content area. Virtualised (a GVHMR
 * run emits tens of thousands of lines), follows the tail until the
 * operator scrolls up, and jumps to the first error on failure.
 */
export function LogDock() {
  const { log, logOpen, setLogOpen, clearLog, cancelRun } = usePipeline()
  const confirm = useConfirm()
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const [follow, setFollow] = React.useState(true)
  const elapsed = useElapsed(log.startedAt, log.finishedAt)
  const count = log.lines.length

  const virtualizer = useVirtualizer({
    count,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => LINE_HEIGHT_PX,
    overscan: 30,
  })

  const jumpToError = React.useCallback(() => {
    if (log.firstError === null) return
    setFollow(false)
    virtualizer.scrollToIndex(log.firstError, { align: "start" })
  }, [log.firstError, virtualizer])

  React.useLayoutEffect(() => {
    if (follow && count) virtualizer.scrollToIndex(count - 1, { align: "end" })
  }, [count, follow, logOpen, virtualizer])

  React.useEffect(() => {
    if (log.status === "running") setFollow(true)
    if (log.status === "error") requestAnimationFrame(jumpToError)
    // jumpToError intentionally not a dependency: only react to the transition.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [log.status])

  if (log.status === "idle" || !logOpen) return null

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24)
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(logText(log))
      toast.success("Log copied")
    } catch {
      toast.error("Clipboard unavailable", { description: "Use Download instead." })
    }
  }

  const download = () => {
    const blob = new Blob([logText(log)], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${(log.title || "run").toLowerCase().replace(/\s+/g, "-")}-log.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const onCancel = async () => {
    const ok = await confirm({
      title: `Cancel ${log.title || "this run"}?`,
      description:
        "The stage in progress stops at its next step and may leave partial output. Stages already finished keep their results; Continue resumes from per-player caches.",
      confirmLabel: "Cancel run",
      cancelLabel: "Keep running",
      destructive: true,
    })
    if (ok) await cancelRun()
  }

  return (
    <section
      aria-label="Run log"
      className={cn(
        "sticky bottom-0 z-20 mx-4 mb-4 flex max-h-[45vh] min-h-40 flex-col overflow-hidden rounded-xl border bg-card shadow-lg shadow-black/20",
        log.status === "error" && "border-destructive/50",
      )}
    >
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <TerminalIcon className="size-4 text-muted-foreground" />
        <h2 className="text-sm font-medium">{log.title || "Run"} log</h2>
        <span aria-live="polite" className="contents">
          <StatusChip status={log.status} cancelling={log.cancelling} />
          {log.connection === "reconnecting" ? (
            <ToneBadge tone="muted">
              <WifiOffIcon /> Connection lost — reconnecting…
            </ToneBadge>
          ) : null}
        </span>
        <span className="font-mono text-xs text-muted-foreground" data-numeric>
          {elapsed}
        </span>
        {log.dropped ? (
          <span className="text-xs text-muted-foreground">
            first <span className="font-mono tabular-nums">{log.dropped.toLocaleString()}</span> lines trimmed — full log
            in <span className="font-mono">output/logs/job_{log.jobId}.log</span>
          </span>
        ) : null}
        <div className="ml-auto flex items-center gap-1">
          {log.status === "running" && log.jobId ? (
            <Button
              variant="destructive"
              size="xs"
              disabled={log.cancelling}
              onClick={() => void onCancel()}
            >
              <SquareIcon /> {log.cancelling ? "Cancelling…" : "Cancel run"}
            </Button>
          ) : null}
          {log.firstError !== null ? (
            <Button variant="ghost" size="xs" className="text-destructive" onClick={jumpToError}>
              <SearchXIcon /> First error
            </Button>
          ) : null}
          {!follow ? (
            <Button variant="ghost" size="xs" onClick={() => setFollow(true)}>
              <ArrowDownToLineIcon /> Follow
            </Button>
          ) : null}
          <IconAction label="Copy log" onClick={() => void copy()}>
            <CopyIcon />
          </IconAction>
          <IconAction label="Download log" onClick={download}>
            <DownloadIcon />
          </IconAction>
          <IconAction label="Minimise (reopen from the sidebar)" onClick={() => setLogOpen(false)}>
            <MinusIcon />
          </IconAction>
          {log.status !== "running" ? (
            <IconAction label="Dismiss log" onClick={clearLog}>
              <XIcon />
            </IconAction>
          ) : null}
        </div>
      </div>
      <div
        ref={scrollRef}
        onScroll={onScroll}
        tabIndex={0}
        role="log"
        aria-label="Log output"
        className="relative min-h-0 flex-1 overflow-auto bg-stage px-3 py-2 font-mono text-xs leading-5 text-stage-foreground/85"
      >
        {count ? (
          <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
            {virtualizer.getVirtualItems().map((item) => {
              const line = log.lines[item.index]
              return (
                <div
                  key={item.key}
                  data-index={item.index}
                  ref={virtualizer.measureElement}
                  className={cn("absolute top-0 left-0 w-full break-all whitespace-pre-wrap", LEVEL_CLASS[line.level])}
                  style={{ transform: `translateY(${item.start}px)` }}
                >
                  {line.text || " "}
                </div>
              )
            })}
          </div>
        ) : (
          "Waiting for output…"
        )}
      </div>
    </section>
  )
}

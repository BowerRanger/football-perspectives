import * as React from "react"
import {
  ArrowDownToLineIcon,
  CopyIcon,
  DownloadIcon,
  MinusIcon,
  SearchXIcon,
  TerminalIcon,
  WifiOffIcon,
  XIcon,
} from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { ToneBadge } from "@/components/status"
import { usePipeline } from "@/hooks/use-pipeline"
import { cn } from "@/lib/utils"

const ERROR_LINE = /(Traceback \(most recent call last\)|\bERROR\b|Error:|Exception:|\[FAIL)/
const WARN_LINE = /(\bWARN(ING)?\b|\[SKIP\])/

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

const LogLines = React.memo(function LogLines({ lines }: { lines: string[] }) {
  return (
    <>
      {lines.map((line, i) => (
        <span
          key={i}
          data-log-error={ERROR_LINE.test(line) || undefined}
          className={cn(
            "block",
            ERROR_LINE.test(line) && "bg-destructive/15 text-destructive",
            !ERROR_LINE.test(line) && WARN_LINE.test(line) && "text-warning",
          )}
        >
          {line || " "}
        </span>
      ))}
    </>
  )
})

/**
 * Run log, docked to the bottom of the content area. It follows the tail
 * until the operator scrolls up, highlights error lines, and jumps to the
 * first one on failure.
 */
export function LogDock() {
  const { log, logOpen, setLogOpen, clearLog } = usePipeline()
  const preRef = React.useRef<HTMLPreElement>(null)
  const [follow, setFollow] = React.useState(true)
  const elapsed = useElapsed(log.startedAt, log.finishedAt)
  const hasError = React.useMemo(() => log.lines.some((l) => ERROR_LINE.test(l)), [log.lines])

  const jumpToError = React.useCallback(() => {
    const el = preRef.current?.querySelector<HTMLElement>("[data-log-error]")
    if (!el || !preRef.current) return
    setFollow(false)
    preRef.current.scrollTop = el.offsetTop - 8
  }, [])

  React.useLayoutEffect(() => {
    const el = preRef.current
    if (el && follow) el.scrollTop = el.scrollHeight
  }, [log.lines, follow, logOpen])

  React.useEffect(() => {
    if (log.status === "running") setFollow(true)
    if (log.status === "error") requestAnimationFrame(jumpToError)
  }, [log.status, jumpToError])

  if (log.status === "idle" || !logOpen) return null

  const onScroll = () => {
    const el = preRef.current
    if (!el) return
    setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24)
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(log.lines.join("\n"))
      toast.success("Log copied")
    } catch {
      toast.error("Clipboard unavailable", { description: "Use Download instead." })
    }
  }

  const download = () => {
    const blob = new Blob([log.lines.join("\n")], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${(log.title || "run").toLowerCase().replace(/\s+/g, "-")}-log.txt`
    a.click()
    URL.revokeObjectURL(url)
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
          {log.status === "running" ? (
            <ToneBadge tone="warning">
              <Spinner className="size-3" /> Running
            </ToneBadge>
          ) : log.status === "done" ? (
            <ToneBadge tone="success">Finished</ToneBadge>
          ) : (
            <ToneBadge tone="destructive">Failed</ToneBadge>
          )}
          {log.connection === "reconnecting" ? (
            <ToneBadge tone="muted">
              <WifiOffIcon /> Connection lost — reconnecting…
            </ToneBadge>
          ) : null}
        </span>
        <span className="font-mono text-xs text-muted-foreground" data-numeric>
          {elapsed}
        </span>
        <div className="ml-auto flex items-center gap-1">
          {hasError ? (
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
      <pre
        ref={preRef}
        onScroll={onScroll}
        tabIndex={0}
        aria-label="Log output"
        className="relative min-h-0 flex-1 overflow-auto bg-stage px-3 py-2 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap text-stage-foreground/85"
      >
        {log.lines.length ? <LogLines lines={log.lines} /> : "Waiting for output…"}
      </pre>
    </section>
  )
}

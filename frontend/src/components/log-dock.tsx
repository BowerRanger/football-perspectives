import * as React from "react"
import { ArrowDownToLineIcon, CopyIcon, MinusIcon, TerminalIcon, XIcon } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { ToneBadge } from "@/components/status"
import { usePipeline } from "@/hooks/use-pipeline"
import { cn } from "@/lib/utils"

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

/**
 * Run log, docked to the bottom of the content area. It follows the tail
 * until the operator scrolls up, and jumps to the traceback on failure.
 */
export function LogDock() {
  const { log, logOpen, setLogOpen, clearLog } = usePipeline()
  const preRef = React.useRef<HTMLPreElement>(null)
  const [follow, setFollow] = React.useState(true)
  const elapsed = useElapsed(log.startedAt, log.finishedAt)

  React.useLayoutEffect(() => {
    const el = preRef.current
    if (el && follow) el.scrollTop = el.scrollHeight
  }, [log.lines, follow, logOpen])

  React.useEffect(() => {
    if (log.status === "running") setFollow(true)
  }, [log.status])

  if (log.status === "idle" || !logOpen) return null

  const onScroll = () => {
    const el = preRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24
    setFollow(atBottom)
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(log.lines.join("\n"))
      toast.success("Log copied")
    } catch {
      toast.error("Clipboard unavailable")
    }
  }

  return (
    <section
      aria-label="Run log"
      className={cn(
        "sticky bottom-0 z-20 mx-4 mb-4 flex max-h-[45vh] min-h-40 flex-col overflow-hidden rounded-xl border bg-card shadow-lg shadow-black/20",
        log.status === "error" && "border-destructive/50",
      )}
    >
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <TerminalIcon className="size-4 text-muted-foreground" />
        <span className="text-sm font-medium">{log.title || "Run"} log</span>
        {log.status === "running" ? (
          <ToneBadge tone="warning">
            <Spinner className="size-3" /> Running
          </ToneBadge>
        ) : log.status === "done" ? (
          <ToneBadge tone="success">Finished</ToneBadge>
        ) : (
          <ToneBadge tone="destructive">Failed — traceback at the end</ToneBadge>
        )}
        <span className="font-mono text-xs text-muted-foreground" data-numeric>
          {elapsed}
        </span>
        <div className="ml-auto flex items-center gap-1">
          {!follow ? (
            <Button variant="ghost" size="xs" onClick={() => setFollow(true)}>
              <ArrowDownToLineIcon /> Follow
            </Button>
          ) : null}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={() => void copy()} aria-label="Copy log">
                <CopyIcon />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Copy log</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={() => setLogOpen(false)} aria-label="Minimise log">
                <MinusIcon />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Minimise (reopen from the sidebar)</TooltipContent>
          </Tooltip>
          {log.status !== "running" ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon-sm" onClick={clearLog} aria-label="Dismiss log">
                  <XIcon />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Dismiss</TooltipContent>
            </Tooltip>
          ) : null}
        </div>
      </div>
      <pre
        ref={preRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-auto bg-stage px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-all text-stage-foreground/80"
      >
        {log.lines.length ? log.lines.join("\n") : "Waiting for output…"}
      </pre>
    </section>
  )
}

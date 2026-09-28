import * as React from "react"
import { useSearchParams } from "react-router"

import { PageHeader } from "@/components/page-header"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { getJsonOrNull } from "@/lib/api"
import { useIsMobile } from "@/hooks/use-mobile"
import { cn } from "@/lib/utils"
import { EmptyOverlay, ErrorOverlay, LoadingOverlay, MatchHeader, PlayerLegend } from "./overlays"
import { Transport } from "./transport"
import { useViewer, type ViewerActions } from "./use-viewer"

const INTERACTIVE = "button,input,select,textarea,[role=combobox],[role=slider],[role=option]"

/** Space play/pause, arrows step (Shift = 10). Ignores keys aimed at other controls. */
function useViewerKeys(actions: ViewerActions, target: HTMLElement | null, embedded: boolean) {
  React.useEffect(() => {
    const el: HTMLElement | Window | null = embedded ? target : window
    if (!el) return
    const onKey = (e: Event) => {
      const ev = e as KeyboardEvent
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return
      const t = ev.target instanceof Element ? ev.target : null
      const inField = !!t?.closest("input,textarea,select,[role=combobox],[role=slider]")
      if (ev.key === " " && !t?.closest(INTERACTIVE)) {
        ev.preventDefault()
        actions.togglePlay()
      } else if ((ev.key === "ArrowLeft" || ev.key === "ArrowRight") && !inField) {
        ev.preventDefault()
        actions.step((ev.key === "ArrowLeft" ? -1 : 1) * (ev.shiftKey ? 10 : 1))
      }
    }
    el.addEventListener("keydown", onKey)
    return () => el.removeEventListener("keydown", onKey)
  }, [actions, target, embedded])
}

/** The reusable 3D viewer. Embedded mode fills its parent and omits page chrome. */
export function Viewer({ embedded = false, shot }: { embedded?: boolean; shot?: string }) {
  const { containerRef, state, actions } = useViewer(shot)
  const isMobile = useIsMobile()
  const rootRef = React.useRef<HTMLDivElement>(null)
  const [rootEl, setRootEl] = React.useState<HTMLDivElement | null>(null)
  useViewerKeys(actions, rootEl, embedded)
  const setRoot = React.useCallback((el: HTMLDivElement | null) => {
    rootRef.current = el
    setRootEl(el)
  }, [])
  const { data, phase } = state
  const ready = phase === "ready" && data !== null

  return (
    <div
      ref={setRoot}
      tabIndex={embedded ? 0 : -1}
      aria-label="3D scene viewer"
      onPointerDown={() => rootRef.current?.focus({ preventScroll: true })}
      className={cn(
        "relative isolate size-full min-h-64 overflow-hidden bg-stage text-stage-foreground outline-none",
        embedded ? "rounded-md focus-visible:ring-[3px] focus-visible:ring-ring/50" : "",
      )}
    >
      <div ref={containerRef} className="absolute inset-0" />
      {ready ? (
        <>
          <div className="pointer-events-none absolute inset-x-2 top-2 z-10 grid grid-cols-[minmax(0,1fr)_auto] items-start gap-2 md:grid-cols-[1fr_auto_1fr]">
            {data.match ? (
              <div className="pointer-events-auto col-start-1 row-start-1 flex min-w-0 md:col-start-2 md:justify-center">
                <MatchHeader match={data.match} />
              </div>
            ) : null}
            <div className="pointer-events-auto col-start-2 row-start-1 flex justify-end md:col-start-3">
              <PlayerLegend
                data={data}
                selectedId={state.selectedId}
                onSelect={actions.selectPlayer}
                defaultOpen={!isMobile && !embedded}
              />
            </div>
          </div>
          <Transport data={data} state={state} actions={actions} className="absolute inset-x-2 bottom-2 z-10" />
        </>
      ) : null}
      {phase === "loading" ? <LoadingOverlay progress={state.progress} /> : null}
      {phase === "empty" ? <EmptyOverlay shot={shot} /> : null}
      {phase === "error" ? <ErrorOverlay message={state.error ?? "Unknown error"} onRetry={actions.reload} /> : null}
    </div>
  )
}

function ShotSelect({ shots, value, onChange }: { shots: string[]; value: string; onChange: (s: string) => void }) {
  return (
    <Select value={value} onValueChange={onChange} disabled={shots.length === 0}>
      <SelectTrigger size="sm" className="w-auto min-w-36" aria-label="Shot" title="Switch which shot's reconstruction is loaded">
        <SelectValue placeholder="No shots" />
      </SelectTrigger>
      <SelectContent>
        {shots.map((id) => (
          <SelectItem key={id} value={id}>
            {id}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export default function ViewerPage() {
  const [params, setParams] = useSearchParams()
  const [shots, setShots] = React.useState<string[] | null>(null)
  const shotParam = params.get("shot") || undefined

  React.useEffect(() => {
    const controller = new AbortController()
    getJsonOrNull<{ shots?: string[] }>("/api/output/shots", { signal: controller.signal }).then((r) => {
      if (!controller.signal.aborted) setShots(r?.shots ?? [])
    })
    return () => controller.abort()
  }, [])

  const shot = shotParam ?? shots?.[0]
  const onShot = React.useCallback(
    (id: string) => {
      const next = new URLSearchParams(params)
      next.set("shot", id)
      setParams(next)
    },
    [params, setParams],
  )

  return (
    <div className="flex h-svh min-h-0 flex-col">
      <PageHeader
        title="3D viewer"
        description="Pitch-registered reconstruction: players, ball and solved broadcast camera."
        actions={<ShotSelect shots={shots ?? []} value={shot ?? ""} onChange={onShot} />}
      />
      <div className="min-h-0 flex-1">
        {shots === null ? <Skeleton className="size-full rounded-none" /> : <Viewer shot={shot} />}
      </div>
    </div>
  )
}

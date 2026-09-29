import * as React from "react"
import { useSearchParams } from "react-router"

import { PageHeader } from "@/components/page-header"
import { PanelError } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { getJson } from "@/lib/api"
import { useIsMobile } from "@/hooks/use-mobile"
import { useResource } from "@/hooks/use-resource"
import { cn } from "@/lib/utils"
import { EmptyOverlay, ErrorOverlay, LoadingOverlay, MatchHeader, PlayerLegend } from "./overlays"
import { Transport } from "./transport"
import { useViewer } from "./use-viewer"
import { useScopedKeyboard } from "@/pages/anchor-editor/use-scoped-keyboard"

/**
 * The reusable 3D viewer. Embedded mode fills its parent and omits page chrome.
 * Playback keys (Space, arrows, Home/End) come from the transport's FramePlayer,
 * which is the single owner of the keyboard while the viewer is mounted.
 */
export function Viewer({ embedded = false, shot }: { embedded?: boolean; shot?: string }) {
  const { containerRef, state, actions } = useViewer(shot)
  const isMobile = useIsMobile()
  const rootRef = React.useRef<HTMLDivElement>(null)
  // Embedded (Export stage) it shares the page with other players: own the keys only while in use.
  const keyboard = useScopedKeyboard(rootRef, embedded)
  const { data, phase } = state
  const ready = phase === "ready" && data !== null

  return (
    <div
      ref={rootRef}
      aria-label="3D scene viewer"
      className={cn(
        "relative isolate size-full min-h-64 overflow-hidden bg-stage text-stage-foreground",
        embedded ? "rounded-md" : "",
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
          {data.warnings.length > 0 ? (
            <ul
              aria-label="Unavailable optional data"
              className="pointer-events-none absolute inset-x-2 bottom-24 z-10 space-y-0.5 text-xs text-stage-foreground/70 sm:bottom-16"
            >
              {data.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          ) : null}
          <Transport data={data} state={state} actions={actions} keyboard={keyboard} className="absolute inset-x-2 bottom-2 z-10" />
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
  const shotParam = params.get("shot") || undefined
  // /api/output/shots answers 200 with an empty list when nothing is prepared, so a rejection is a real failure.
  const shotList = useResource((signal) => getJson<{ shots?: string[] }>("/api/output/shots", { signal }), [])
  const shots = shotList.state.status === "ready" ? (shotList.state.data.shots ?? []) : null

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
        description="Pitch-registered reconstruction: players and ball, viewed through the solved broadcast camera or a free orbit."
        actions={<ShotSelect shots={shots ?? []} value={shot ?? ""} onChange={onShot} />}
      />
      <div className="min-h-0 flex-1">
        {shotList.state.status === "error" ? (
          <div className="p-4">
            <PanelError
              title="Could not list shots"
              message={shotList.state.error}
              action={
                <Button variant="outline" size="sm" className="mt-2" onClick={shotList.retry}>
                  Retry
                </Button>
              }
            />
          </div>
        ) : shots === null ? (
          <Skeleton className="size-full rounded-none" />
        ) : (
          <Viewer shot={shot} />
        )}
      </div>
    </div>
  )
}

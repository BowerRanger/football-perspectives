import * as React from "react"
import { useSearchParams } from "react-router"
import { MonitorIcon } from "lucide-react"

import { PageHeader } from "@/components/page-header"
import { PanelEmpty } from "@/components/panel"
import { Badge } from "@/components/ui/badge"
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useIsMobile } from "@/hooks/use-mobile"
import { AnchorList } from "./anchor-list"
import { CoverageStrip } from "./coverage-strip"
import { PalettePanel } from "./palette-panel"
import { StageCanvas } from "./stage-canvas"
import { Toolbar } from "./toolbar"
import { TransportBar } from "./transport-bar"
import { useAnchorEditor } from "./use-anchor-editor"

type Editor = ReturnType<typeof useAnchorEditor>

function PaneHeader({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="flex h-10 shrink-0 items-center justify-between gap-2 border-b px-3">
      <h2 className="text-sm font-medium">{title}</h2>
      {children}
    </div>
  )
}

function Palette({ ed }: { ed: Editor }) {
  const p = ed.placement
  return (
    <PalettePanel
      mode={p.mode}
      onModeChange={p.setMode}
      landmarks={ed.catalogues.landmarks}
      pitchLines={ed.pitchLines}
      selected={p.selected}
      placed={ed.placedHere}
      onSelect={p.select}
    />
  )
}

function Anchors({ ed }: { ed: Editor }) {
  return ed.loadingAnchors ? (
    <div className="flex flex-col gap-2 p-3">
      <Skeleton className="h-8" />
      <Skeleton className="h-8" />
      <Skeleton className="h-8" />
    </div>
  ) : (
    <AnchorList
      anchors={ed.anchors}
      frame={ed.player.frame}
      onSeek={ed.player.seek}
      onDeleteFrame={ed.deleteAnchorFrame}
      onDeletePoint={ed.deletePoint}
      onDeleteLine={ed.deleteLine}
    />
  )
}

function Stage({ ed }: { ed: Editor }) {
  const { player, placement, anchors, track, detected, view } = ed
  const overlay = React.useMemo(
    () => ({
      frame: player.frame,
      view,
      anchor: anchors.get(player.frame),
      track,
      detected,
      landmarks: ed.catalogues.landmarks,
      pendingLineStart: placement.pendingLineStart,
    }),
    [player.frame, view, anchors, track, detected, ed.catalogues.landmarks, placement.pendingLineStart],
  )
  return (
    <div className="flex h-full min-h-0 flex-col">
      <StageCanvas
        shot={ed.shot}
        videoRef={ed.videoRef}
        imageSize={ed.imageSize}
        overlay={overlay}
        videoHandlers={player.videoHandlers}
        onMeta={ed.setMeta}
        armed={placement.selected !== null}
        hint={placement.hint}
        onPlace={(xy) => void placement.place(xy)}
        onDeleteAnchor={() => ed.deleteAnchorFrame(player.frame)}
      />
      <TransportBar
        player={player}
        totalFrames={ed.totalFrames}
        hasAnchorHere={ed.hasAnchorHere}
        onAddAnchor={ed.addAnchorHere}
      />
      <CoverageStrip track={track} frame={player.frame} totalFrames={ed.totalFrames} onSeek={player.seek} />
    </div>
  )
}

function DesktopBody({ ed }: { ed: Editor }) {
  return (
    <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
      <ResizablePanel defaultSize="22%" minSize="15%" maxSize="38%">
        <div className="flex h-full min-h-0 flex-col">
          <PaneHeader title="Landmark palette" />
          <div className="min-h-0 flex-1">
            <Palette ed={ed} />
          </div>
        </div>
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel defaultSize="56%" minSize="30%">
        <Stage ed={ed} />
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel defaultSize="22%" minSize="15%" maxSize="38%">
        <div className="flex h-full min-h-0 flex-col">
          <PaneHeader title="Anchors">
            <Badge variant="secondary" className="tabular-nums">{ed.anchors.size}</Badge>
          </PaneHeader>
          <div className="min-h-0 flex-1">
            <Anchors ed={ed} />
          </div>
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}

function MobileBody({ ed }: { ed: Editor }) {
  return (
    <div className="flex flex-col">
      <p className="flex items-center gap-2 border-b px-3 py-1.5 text-xs text-muted-foreground">
        <MonitorIcon className="size-3.5 shrink-0" aria-hidden />
        Placing landmarks is easier on a desktop screen.
      </p>
      <div className="h-[52svh] min-h-72">
        <Stage ed={ed} />
      </div>
      <Tabs defaultValue="palette" className="gap-0 border-t">
        <TabsList className="m-2">
          <TabsTrigger value="palette">Palette</TabsTrigger>
          <TabsTrigger value="anchors">Anchors ({ed.anchors.size})</TabsTrigger>
        </TabsList>
        <TabsContent value="palette" className="h-[50svh] min-h-64 flex-none overflow-hidden">
          <Palette ed={ed} />
        </TabsContent>
        <TabsContent value="anchors" className="h-[50svh] min-h-64 flex-none overflow-hidden">
          <Anchors ed={ed} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

interface AnchorEditorProps {
  /** Fill the parent's height and omit page chrome (Camera stage panel). */
  embedded?: boolean
  /** Controlled shot id. Without it the editor manages its own selection. */
  shot?: string
  /** Called when the operator (or the default resolution) picks a shot. */
  onShotChange?: (shot: string) => void
}

export function AnchorEditor({ embedded = false, shot, onShotChange }: AnchorEditorProps) {
  const ed = useAnchorEditor({ shot, onShotChange, embedded })
  const isMobile = useIsMobile()
  const showShotSelect = shot === undefined || onShotChange !== undefined

  return (
    <div
      ref={ed.rootRef}
      tabIndex={-1}
      className="flex h-full min-h-0 flex-col overflow-y-auto bg-background outline-none md:overflow-hidden"
    >
      <Toolbar
        shots={ed.list.shots}
        shot={ed.shot}
        showShotSelect={showShotSelect}
        onShotChange={(s) => void ed.requestShot(s)}
        stadiums={ed.catalogues.stadiums}
        stadium={ed.stadium}
        onStadiumChange={ed.changeStadium}
        view={ed.view}
        onToggleView={ed.toggleView}
        status={ed.status.text}
        statusTone={ed.status.tone}
        dirty={ed.dirty}
        saving={ed.saving}
        onSave={ed.save}
        rerunPhase={ed.rerunPhase}
        rerunBlockedReason={ed.rerunBlockedReason}
        onRerun={ed.rerun}
        viewerHref={ed.viewerHref}
      />
      {ed.list.loaded && ed.list.shots.length === 0 && !ed.shot ? (
        <PanelEmpty
          title="No shots to annotate"
          description="Run the prepare_shots stage first. Anchors are placed on frames of its shot clips."
        />
      ) : isMobile ? (
        <MobileBody ed={ed} />
      ) : (
        <DesktopBody ed={ed} />
      )}
    </div>
  )
}

export default function AnchorEditorPage() {
  const [params, setParams] = useSearchParams()
  const shot = params.get("shot") ?? undefined
  const setShot = React.useCallback(
    (next: string) =>
      setParams(
        (prev) => {
          const out = new URLSearchParams(prev)
          out.set("shot", next)
          return out
        },
        { replace: true },
      ),
    [setParams],
  )
  return (
    <>
      <PageHeader title="Pitch anchors" />
      <div className="flex min-h-0 flex-col md:h-[calc(100svh-3.6rem)] md:min-h-[520px]">
        <AnchorEditor shot={shot} onShotChange={setShot} />
      </div>
    </>
  )
}

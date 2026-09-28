import { PanelError, PanelSkeleton } from "@/components/panel"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Kbd } from "@/components/ui/kbd"
import { cn } from "@/lib/utils"
import { useUnsavedGuard } from "@/hooks/use-unsaved-guard"
import { AuthoringPanel } from "./authoring-panels"
import { DirtyBadge, EditorActions } from "./editor-actions"
import { EventsList } from "./events-list"
import { FrameCanvas } from "./frame-canvas"
import { PreviewSummary } from "./preview-result"
import { QualityStrip } from "./quality-strip"
import { TagPalette } from "./tag-palette"
import { Transport } from "./transport"
import type { EditorController } from "./use-ball-anchor-editor"
import { useEditorShortcuts } from "./use-editor-shortcuts"

interface WorkspaceProps {
  ctrl: EditorController
  /** Embedded (stage panel) shows its own toolbar; the page puts actions in the header. */
  embedded?: boolean
}

/** Three-column editor: tools left, frame + transport centre, events right (stacked on narrow screens). */
export function EditorWorkspace({ ctrl, embedded = false }: WorkspaceProps) {
  useEditorShortcuts(ctrl, true)
  useUnsavedGuard(ctrl.docApi.dirty, { what: "ball anchors" })

  if (ctrl.loadError) {
    return <PanelError title="Could not load ball anchors" message={ctrl.loadError} />
  }
  const railHeight = embedded ? "lg:h-[680px]" : "lg:h-[calc(100svh-9rem)]"
  // Radix wraps ScrollArea content in a display:table div that ignores truncation; force block.
  const rail = "[&_[data-radix-scroll-area-viewport]>div]:!block"
  return (
    <div className="flex flex-col gap-3">
      {embedded ? (
        <div className="flex flex-wrap items-center gap-2">
          <EditorActions ctrl={ctrl} />
          <DirtyBadge ctrl={ctrl} />
          <span className="ml-auto text-xs text-muted-foreground">
            Left-click places · right-click deletes · <Kbd>Space</Kbd> play · <Kbd>←</Kbd>
            <Kbd>→</Kbd> step
          </span>
        </div>
      ) : null}
      <p className="text-xs text-muted-foreground lg:hidden">
        Frame annotation works best on a desktop screen; tools stack below the frame on narrow displays.
      </p>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-y-3 lg:grid-cols-[250px_minmax(0,1fr)_320px] lg:gap-y-0">
        <ScrollArea className={cn("order-2 h-80 border-t pt-3 lg:order-none lg:border-t-0 lg:pt-0 lg:pr-3", rail, railHeight)}>
          <div className="flex flex-col gap-3 p-1 lg:pl-0">
            <TagPalette selected={ctrl.selectedTag} onSelect={ctrl.setSelectedTag} />
            <AuthoringPanel ctrl={ctrl} />
          </div>
        </ScrollArea>
        <div className="order-first flex min-w-0 flex-col gap-3 lg:order-none lg:border-l lg:px-3">
          {ctrl.loading ? <PanelSkeleton rows={2} media /> : null}
          <div className={cn("flex flex-col gap-3", ctrl.loading && "hidden")}>
            <FrameCanvas ctrl={ctrl} />
            <Transport ctrl={ctrl} />
            <QualityStrip ctrl={ctrl} />
            {ctrl.previewResult ? <PreviewSummary result={ctrl.previewResult} /> : null}
          </div>
        </div>
        <ScrollArea className={cn("order-3 h-96 border-t pt-3 lg:order-none lg:border-l lg:border-t-0 lg:pt-0", rail, railHeight)}>
          <div className="p-3 lg:pr-0">
            <EventsList
              anchors={ctrl.docApi.doc.anchors}
              autoAnchors={ctrl.autoAnchors}
              dismissedAuto={ctrl.docApi.doc.dismissedAuto}
              shotChains={ctrl.docApi.doc.shotChains}
              currentFrame={ctrl.player.frame}
              docApi={ctrl.docApi}
              onSeek={ctrl.player.seekTo}
              onSetEnd={ctrl.setEndFrameHere}
            />
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}

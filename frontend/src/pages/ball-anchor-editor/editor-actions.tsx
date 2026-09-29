import { CheckIcon, SaveIcon, WandSparklesIcon } from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { Spinner } from "@/components/ui/spinner"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { EditorController } from "./use-ball-anchor-editor"

/** Unsaved-changes / saved indicator with the anchor count. */
export function DirtyBadge({ ctrl }: { ctrl: EditorController }) {
  const count = ctrl.docApi.doc.anchors.length
  return ctrl.docApi.dirty ? (
    <ToneBadge tone="warning">Unsaved changes · {count} anchors</ToneBadge>
  ) : (
    <ToneBadge tone="muted">
      <CheckIcon /> {count} anchors saved
    </ToneBadge>
  )
}

/** Save + Solve & preview buttons (page header and embedded toolbar). */
export function EditorActions({ ctrl }: { ctrl: EditorController }) {
  const disabled = ctrl.loading || Boolean(ctrl.loadError)
  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button size="sm" onClick={() => void ctrl.save()} disabled={disabled || ctrl.saving}>
            {ctrl.saving ? <Spinner /> : <SaveIcon />} Save
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          Save anchors, chains and dismissals <Kbd>⌘S</Kbd>
        </TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button size="sm" variant="secondary" onClick={() => void ctrl.solve()} disabled={disabled || ctrl.solving}>
            {ctrl.solving ? <Spinner /> : <WandSparklesIcon />} Solve &amp; preview
          </Button>
        </TooltipTrigger>
        <TooltipContent>Run the ball solver on the current (unsaved) anchors without writing output</TooltipContent>
      </Tooltip>
    </>
  )
}

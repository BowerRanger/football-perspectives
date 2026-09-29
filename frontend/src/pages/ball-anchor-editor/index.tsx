import * as React from "react"
import { useSearchParams } from "react-router"

import { PageHeader } from "@/components/page-header"
import { PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { useConfirm } from "@/hooks/use-dialogs"
import { DirtyBadge, EditorActions } from "./editor-actions"
import { ShotSelect, useShotOptions } from "./shot-select"
import { useBallAnchorEditor } from "./use-ball-anchor-editor"
import type { PreviewFrame } from "./types"
import { EditorWorkspace } from "./workspace"

interface BallAnchorEditorProps {
  embedded?: boolean
  shot?: string
  /** Optional predicted ball frames (Ball stage) for the "Predicted ball path" layer. */
  predicted?: PreviewFrame[]
  /** Fires whenever the playhead frame changes (Ball stage syncs its trajectory views). */
  onFrameChange?: (frame: number) => void
  /** Reports unsaved-change state so the host can guard shot switches. */
  onDirtyChange?: (dirty: boolean) => void
}

/** The single ball-anchor editor implementation, used by the Ball stage panel and the standalone page. */
export function BallAnchorEditor({ embedded = false, shot, predicted, onFrameChange, onDirtyChange }: BallAnchorEditorProps) {
  const ctrl = useBallAnchorEditor({ shot: shot ?? "", predicted, onFrameChange })
  const dirty = ctrl.docApi.dirty
  React.useEffect(() => {
    onDirtyChange?.(dirty)
    return () => onDirtyChange?.(false)
  }, [dirty, onDirtyChange])
  if (!shot) {
    return (
      <PanelEmpty
        title="Choose a shot to annotate"
        description="The ball anchor editor works on one shot at a time. Pick a shot to load its video and saved anchors."
      />
    )
  }
  return <EditorWorkspace ctrl={ctrl} embedded={embedded} />
}

export default function BallAnchorEditorPage() {
  const [params, setParams] = useSearchParams()
  const shot = params.get("shot") ?? ""
  const shots = useShotOptions()
  const confirm = useConfirm()
  const ctrl = useBallAnchorEditor({ shot })
  const { dirty } = ctrl.docApi

  const setShot = React.useCallback(
    (next: string, replace: boolean) =>
      setParams(
        (prev) => {
          const p = new URLSearchParams(prev)
          p.set("shot", next)
          return p
        },
        { replace },
      ),
    [setParams],
  )

  // No ?shot= -> default to the first shot instead of a blank canvas.
  React.useEffect(() => {
    if (!shot && shots.options.length) setShot(shots.options[0].id, true)
  }, [shot, shots.options, setShot])

  const onChangeShot = async (next: string) => {
    if (next === shot) return
    if (dirty) {
      const ok = await confirm({
        title: "Discard unsaved anchors?",
        description: `You have unsaved changes on ${shot}. Switching to ${next} will discard them.`,
        confirmLabel: "Discard and switch",
        destructive: true,
      })
      if (!ok) return
    }
    setShot(next, false)
  }

  return (
    <>
      <PageHeader
        title="Ball anchors"
        description="Mark ball position, touches, bounces and goal impacts on the frame; the ball stage solves around them."
        status={shot ? <DirtyBadge ctrl={ctrl} /> : undefined}
        actions={
          <>
            <ShotSelect value={shot} options={shots.options} onChange={(s) => void onChangeShot(s)} />
            {shot ? <EditorActions ctrl={ctrl} /> : null}
          </>
        }
      />
      <div className="min-w-0 p-4">
        {shots.error ? (
          <PanelError title="Could not list shots" message={shots.error} />
        ) : shot ? (
          <EditorWorkspace ctrl={ctrl} />
        ) : shots.loading ? (
          <PanelSkeleton rows={4} media />
        ) : (
          <PanelEmpty
            title="No shots to annotate"
            description="Run the Prepare Shots stage from the dashboard first; the editor lists every active shot."
          />
        )}
      </div>
    </>
  )
}

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"

import type { ShotView } from "./types"

interface ShotPreviewProps {
  shot: ShotView | null
  groupLabel?: string
  onClose: () => void
}

/** Large click-to-expand player; a real Dialog (focus trap, Esc, focus return). */
export function ShotPreview({ shot, groupLabel, onClose }: ShotPreviewProps) {
  return (
    <Dialog open={!!shot} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle className="font-mono">{shot?.id}</DialogTitle>
          <DialogDescription>{groupLabel ? `Highlight group: ${groupLabel}` : "Shot preview"}</DialogDescription>
        </DialogHeader>
        {shot ? (
          <div className="bg-stage">
            <video
              key={shot.id}
              src={`/api/video/${encodeURIComponent(shot.id)}`}
              controls
              autoPlay
              className="mx-auto max-h-[70vh] w-full"
            />
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}

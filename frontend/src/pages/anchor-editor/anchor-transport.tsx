import { CheckIcon, PlusIcon } from "lucide-react"

import { FramePlayer } from "@/components/frame-player"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { FramePlayer as EditorPlayer } from "./use-frame-player"

interface AnchorTransportProps {
  player: EditorPlayer
  totalFrames: number
  fps: number
  /** Whether this player owns Space / arrows (false while another player on the page does). */
  keyboard: boolean
  hasAnchorHere: boolean
  onAddAnchor: () => void
}

/** The shared FramePlayer plus the anchor editor's "Anchor here" action. */
export function AnchorTransport({ player, totalFrames, fps, keyboard, hasAnchorHere, onAddAnchor }: AnchorTransportProps) {
  return (
    <FramePlayer
      keyboard={keyboard}
      className="border-t bg-card px-3 py-2"
      label="Seek frame"
      frame={player.frame}
      max={Math.max(0, totalFrames - 1)}
      fps={fps}
      frameInput
      playing={player.playing}
      onTogglePlay={player.togglePlay}
      onSeek={player.seek}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          {/* span keeps the tooltip working while the button is disabled */}
          <span>
            <Button variant="outline" size="sm" disabled={hasAnchorHere} onClick={onAddAnchor}>
              {hasAnchorHere ? <CheckIcon /> : <PlusIcon />}
              {hasAnchorHere ? "Anchor set" : "Anchor here"}
            </Button>
          </span>
        </TooltipTrigger>
        <TooltipContent>
          {hasAnchorHere ? "This frame is already an anchor frame" : "Mark the current frame as an anchor frame"}
        </TooltipContent>
      </Tooltip>
    </FramePlayer>
  )
}

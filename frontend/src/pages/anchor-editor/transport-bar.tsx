import * as React from "react"
import { CheckIcon, ChevronLeftIcon, ChevronRightIcon, PauseIcon, PlayIcon, PlusIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Kbd } from "@/components/ui/kbd"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { FramePlayer } from "./use-frame-player"

interface TransportBarProps {
  player: FramePlayer
  totalFrames: number
  hasAnchorHere: boolean
  onAddAnchor: () => void
}

function IconTip({ label, keys, children }: { label: string; keys?: string; children: React.ReactElement }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent className="flex items-center gap-2">
        {label}
        {keys ? <Kbd>{keys}</Kbd> : null}
      </TooltipContent>
    </Tooltip>
  )
}

function FrameInput({ frame, max, onCommit }: { frame: number; max: number; onCommit: (f: number) => void }) {
  const [draft, setDraft] = React.useState<string | null>(null)
  const commit = () => {
    if (draft !== null) {
      const n = Number.parseInt(draft, 10)
      if (Number.isFinite(n)) onCommit(n)
    }
    setDraft(null)
  }
  return (
    <div className="flex items-center gap-1.5">
      <Input
        type="number"
        inputMode="numeric"
        min={0}
        max={max}
        aria-label="Frame number"
        value={draft ?? String(frame)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit()
          if (e.key === "Escape") setDraft(null)
        }}
        className="h-7 w-20 px-2 font-mono text-xs tabular-nums"
      />
      <span className="font-mono text-xs text-muted-foreground tabular-nums">/ {max}</span>
    </div>
  )
}

export function TransportBar({ player, totalFrames, hasAnchorHere, onAddAnchor }: TransportBarProps) {
  const max = Math.max(0, totalFrames - 1)
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t bg-card px-3 py-2">
      <div className="flex items-center gap-1">
        <IconTip label="Previous frame" keys="←">
          <Button variant="outline" size="icon-sm" aria-label="Previous frame" onClick={() => player.step(-1)}>
            <ChevronLeftIcon />
          </Button>
        </IconTip>
        <IconTip label={player.playing ? "Pause" : "Play"} keys="Space">
          <Button
            variant="outline"
            size="icon-sm"
            aria-label={player.playing ? "Pause" : "Play"}
            onClick={player.togglePlay}
          >
            {player.playing ? <PauseIcon /> : <PlayIcon />}
          </Button>
        </IconTip>
        <IconTip label="Next frame" keys="→">
          <Button variant="outline" size="icon-sm" aria-label="Next frame" onClick={() => player.step(1)}>
            <ChevronRightIcon />
          </Button>
        </IconTip>
      </div>
      <Slider
        aria-label="Seek frame"
        className="order-last min-w-32 basis-full sm:order-none sm:basis-0 sm:flex-1"
        min={0}
        max={Math.max(1, max)}
        step={1}
        value={[Math.min(player.frame, Math.max(1, max))]}
        onValueChange={([v]) => player.seek(v)}
      />
      <FrameInput frame={player.frame} max={max} onCommit={player.seek} />
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
    </div>
  )
}

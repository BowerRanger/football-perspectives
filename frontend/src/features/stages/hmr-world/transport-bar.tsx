import { PauseIcon, PlayIcon, StepBackIcon, StepForwardIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"

interface TransportBarProps {
  playing: boolean
  onToggle: () => void
  onPrev: () => void
  onNext: () => void
  value: number
  min: number
  max: number
  onSeek: (frame: number) => void
  readout: string
}

/** Play / step / scrub row shared by the kp2d viewer and the trajectory panel. */
export function TransportBar({ playing, onToggle, onPrev, onNext, value, min, max, onSeek, readout }: TransportBarProps) {
  return (
    <div className="flex items-center gap-2">
      <Button size="icon-sm" onClick={onToggle} aria-label={playing ? "Pause" : "Play"}>
        {playing ? <PauseIcon /> : <PlayIcon />}
      </Button>
      <Button size="icon-sm" variant="outline" onClick={onPrev} aria-label="Previous frame">
        <StepBackIcon />
      </Button>
      <Button size="icon-sm" variant="outline" onClick={onNext} aria-label="Next frame">
        <StepForwardIcon />
      </Button>
      <Slider
        className="mx-2 flex-1"
        min={min}
        max={Math.max(min, max)}
        step={1}
        value={[Math.min(Math.max(value, min), Math.max(min, max))]}
        onValueChange={(v) => onSeek(v[0])}
        aria-label="Frame"
      />
      <span className="min-w-24 text-right font-mono text-xs tabular-nums text-muted-foreground">{readout}</span>
    </div>
  )
}

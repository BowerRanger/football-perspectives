import * as React from "react"
import { ChevronLeftIcon, ChevronRightIcon, PauseIcon, PlayIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useFrameKeys } from "@/hooks/use-frame-keys"
import { cn } from "@/lib/utils"

export interface FramePlayerProps {
  frame: number
  /** Last valid frame (inclusive). */
  max: number
  min?: number
  playing?: boolean
  /** Omit for a scrub-only player (no play button, no Space shortcut). */
  onTogglePlay?: () => void
  onSeek: (frame: number) => void
  /** Frames per second — adds a seconds readout. */
  fps?: number
  /** Show an editable frame-number input instead of a static readout. */
  frameInput?: boolean
  /** Replace the default "Frame n / max" readout. */
  readout?: React.ReactNode
  /** Attach Space / ← → / Shift ±10 / Home End shortcuts. Default true. */
  keyboard?: boolean
  /** Show the shortcut legend under the bar. */
  keyHints?: boolean
  /** Accessible name for the scrubber (default "Frame"). */
  label?: string
  /** Extra controls rendered after the readout (speed, layers, "Anchor here"…). */
  children?: React.ReactNode
  className?: string
}

function TipButton({
  label,
  keys,
  children,
  ...props
}: React.ComponentProps<typeof Button> & { label: string; keys?: string[] }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="outline" size="icon-sm" aria-label={label} {...props}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent className="flex items-center gap-2">
        {label}
        {keys?.length ? (
          <KbdGroup>
            {keys.map((k) => (
              <Kbd key={k}>{k}</Kbd>
            ))}
          </KbdGroup>
        ) : null}
      </TooltipContent>
    </Tooltip>
  )
}

function FrameNumberInput({ frame, min, max, onCommit }: { frame: number; min: number; max: number; onCommit: (f: number) => void }) {
  const [draft, setDraft] = React.useState<string | null>(null)
  const commit = () => {
    if (draft !== null) {
      const n = Number.parseInt(draft, 10)
      if (Number.isFinite(n)) onCommit(Math.min(max, Math.max(min, n)))
    }
    setDraft(null)
  }
  return (
    <span className="flex items-center gap-1.5">
      <Input
        type="number"
        inputMode="numeric"
        min={min}
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
      <span className="text-xs text-muted-foreground">
        / <span className="font-mono tabular-nums">{max}</span>
      </span>
    </span>
  )
}

/** Shortcut legend shared by every player (words in sans, keys in Kbd). */
export function FrameKeyHints({ play = true, className }: { play?: boolean; className?: string }) {
  return (
    <p className={cn("flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground", className)}>
      {play ? (
        <span className="inline-flex items-center gap-1">
          <Kbd>Space</Kbd> play
        </span>
      ) : null}
      <span className="inline-flex items-center gap-1">
        <KbdGroup>
          <Kbd>←</Kbd>
          <Kbd>→</Kbd>
        </KbdGroup>
        step
      </span>
      <span className="inline-flex items-center gap-1">
        <KbdGroup>
          <Kbd>Shift</Kbd>
          <Kbd>←</Kbd>
          <Kbd>→</Kbd>
        </KbdGroup>
        ±10
      </span>
      <span className="inline-flex items-center gap-1">
        <KbdGroup>
          <Kbd>Home</Kbd>
          <Kbd>End</Kbd>
        </KbdGroup>
        ends
      </span>
    </p>
  )
}

/**
 * The one frame transport: play/pause, step, scrub (with a spoken value),
 * frame readout or input, and the shared keyboard shortcuts. Every video,
 * canvas and 3D player in the dashboard uses it.
 */
export function FramePlayer({
  frame,
  max,
  min = 0,
  playing = false,
  onTogglePlay,
  onSeek,
  fps,
  frameInput = false,
  readout,
  keyboard = true,
  keyHints = false,
  label = "Frame",
  children,
  className,
}: FramePlayerProps) {
  const hi = Math.max(min, max)
  const clamped = Math.min(hi, Math.max(min, frame))
  const clamp = (n: number) => Math.min(hi, Math.max(min, n))

  useFrameKeys({ enabled: keyboard, frame: clamped, min, max: hi, onTogglePlay, onSeek })

  const seconds = fps && fps > 0 ? ` (${(clamped / fps).toFixed(2)}s)` : ""
  const defaultReadout = (
    <span className="text-xs whitespace-nowrap text-muted-foreground">
      Frame <span className="font-mono tabular-nums">{clamped}</span> /{" "}
      <span className="font-mono tabular-nums">{hi}</span>
      {seconds ? <span className="font-mono tabular-nums">{seconds}</span> : null}
    </span>
  )

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <div className="flex items-center gap-1">
          <TipButton label="Previous frame" keys={["←"]} onClick={() => onSeek(clamp(clamped - 1))}>
            <ChevronLeftIcon />
          </TipButton>
          {onTogglePlay ? (
            <TipButton label={playing ? "Pause" : "Play"} keys={["Space"]} onClick={onTogglePlay}>
              {playing ? <PauseIcon /> : <PlayIcon />}
            </TipButton>
          ) : null}
          <TipButton label="Next frame" keys={["→"]} onClick={() => onSeek(clamp(clamped + 1))}>
            <ChevronRightIcon />
          </TipButton>
        </div>
        <Slider
          aria-label={label}
          valueText={`Frame ${clamped} of ${hi}`}
          className="order-last min-w-32 basis-full sm:order-none sm:basis-0 sm:flex-1"
          min={min}
          max={Math.max(hi, min + 1)}
          step={1}
          value={[clamped]}
          onValueChange={([v]) => onSeek(clamp(v ?? clamped))}
        />
        {frameInput ? <FrameNumberInput frame={clamped} min={min} max={hi} onCommit={onSeek} /> : (readout ?? defaultReadout)}
        {children}
      </div>
      {keyHints ? <FrameKeyHints play={!!onTogglePlay} /> : null}
    </div>
  )
}

import { MapPinIcon, MinusIcon, XIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { anchorSummary } from "./anchor-ops"
import type { AnchorFrame, AnchorMap } from "./types"

interface AnchorListProps {
  anchors: AnchorMap
  frame: number
  onSeek: (frame: number) => void
  onDeleteFrame: (frame: number) => void
  onDeletePoint: (name: string) => void
  onDeleteLine: (index: number) => void
}

interface FrameRowProps {
  frame: number
  anchor: AnchorFrame
  current: boolean
  onSeek: (frame: number) => void
  onDelete: (frame: number) => void
}

function FrameRow({ frame, anchor, current, onSeek, onDelete }: FrameRowProps) {
  return (
    <li
      className={cn(
        "flex items-center gap-1 rounded-md pr-1 transition-colors",
        current ? "bg-info/15 ring-1 ring-info/40" : "hover:bg-accent",
      )}
    >
      <button
        type="button"
        aria-current={current ? "true" : undefined}
        onClick={() => onSeek(frame)}
        className="flex min-w-0 flex-1 items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span className="font-mono text-xs font-medium tabular-nums">Frame {frame}</span>
        <Badge variant="secondary" className="font-normal tabular-nums">
          {anchorSummary(anchor)}
        </Badge>
      </button>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={`Delete anchor at frame ${frame}`}
            className="text-muted-foreground hover:text-destructive"
            onClick={() => onDelete(frame)}
          >
            <XIcon />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Delete this frame's anchor</TooltipContent>
      </Tooltip>
    </li>
  )
}

interface ItemRowProps {
  label: string
  onDelete: () => void
}

function ItemRow({ label, onDelete }: ItemRowProps) {
  return (
    <li className="flex items-center gap-1 rounded-md pr-1 pl-2 hover:bg-accent">
      <span className="min-w-0 flex-1 truncate text-xs">{label}</span>
      <Button
        variant="ghost"
        size="icon-xs"
        aria-label={`Remove ${label} from this frame`}
        className="text-muted-foreground hover:text-destructive"
        onClick={onDelete}
      >
        <MinusIcon />
      </Button>
    </li>
  )
}

function CurrentFrameItems({ frame, anchor, onDeletePoint, onDeleteLine }: Pick<AnchorListProps, "onDeletePoint" | "onDeleteLine"> & { frame: number; anchor: AnchorFrame }) {
  return (
    <div className="border-t p-2">
      <p className="px-2 pb-1 text-xs font-medium text-muted-foreground">
        On frame <span className="font-mono tabular-nums">{frame}</span>
      </p>
      {anchor.points.length + anchor.lines.length === 0 ? (
        <p className="px-2 py-1 text-xs text-muted-foreground">Nothing placed yet on this frame.</p>
      ) : (
        <ul className="flex max-h-40 flex-col overflow-y-auto">
          {anchor.points.map((p) => (
            <ItemRow key={`p:${p.name}`} label={p.name} onDelete={() => onDeletePoint(p.name)} />
          ))}
          {anchor.lines.map((l, i) => (
            <ItemRow key={`l:${i}:${l.name}`} label={`${l.name} (line)`} onDelete={() => onDeleteLine(i)} />
          ))}
        </ul>
      )}
    </div>
  )
}

export function AnchorList(props: AnchorListProps) {
  const { anchors, frame, onSeek, onDeleteFrame, onDeletePoint, onDeleteLine } = props
  const entries = [...anchors.entries()].sort((a, b) => a[0] - b[0])
  const here = anchors.get(frame)
  return (
    <div className="flex h-full min-h-0 flex-col">
      <ScrollArea className="min-h-0 flex-1">
        {entries.length === 0 ? (
          <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
            <MapPinIcon className="size-5 text-muted-foreground" aria-hidden />
            <p className="text-xs text-muted-foreground">
              No anchors yet. Pick a landmark in the palette, then click its position on the frame.
            </p>
          </div>
        ) : (
          <ul className="flex flex-col gap-0.5 p-2">
            {entries.map(([f, anchor]) => (
              <FrameRow
                key={f}
                frame={f}
                anchor={anchor}
                current={f === frame}
                onSeek={onSeek}
                onDelete={onDeleteFrame}
              />
            ))}
          </ul>
        )}
      </ScrollArea>
      {here ? (
        <CurrentFrameItems frame={frame} anchor={here} onDeletePoint={onDeletePoint} onDeleteLine={onDeleteLine} />
      ) : null}
    </div>
  )
}

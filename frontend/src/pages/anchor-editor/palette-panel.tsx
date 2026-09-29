import * as React from "react"
import { CheckIcon, SearchIcon } from "lucide-react"

import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { fmtCoord } from "./anchor-ops"
import type { Landmark, PaletteMode, PitchLine } from "./types"

interface PaletteRow {
  name: string
  detail: string
}

function landmarkRow(lm: Landmark): PaletteRow {
  return { name: lm.name, detail: fmtCoord(lm.world_xyz) }
}

function lineRow(ln: PitchLine): PaletteRow {
  if (ln.world_segment) {
    return { name: ln.name, detail: `${fmtCoord(ln.world_segment[0])} → ${fmtCoord(ln.world_segment[1])}` }
  }
  if (ln.world_direction) return { name: ln.name, detail: `direction (${fmtCoord(ln.world_direction)}) VP` }
  return { name: ln.name, detail: "unknown geometry" }
}

interface RowButtonProps {
  row: PaletteRow
  selected: boolean
  placed: boolean
  onSelect: (name: string) => void
}

function RowButton({ row, selected, placed, onSelect }: RowButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-pressed={selected}
          onClick={() => onSelect(row.name)}
          className={cn(
            "flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
            selected ? "bg-info/15 ring-1 ring-info/40" : "hover:bg-accent",
          )}
        >
          <span className="min-w-0 flex-1">
            <span className="block truncate text-xs font-medium">{row.name}</span>
            <span className="block truncate font-mono text-[11px] text-muted-foreground tabular-nums">
              {row.detail}
            </span>
          </span>
          {placed ? <CheckIcon className="mt-0.5 size-3.5 shrink-0 text-success" aria-label="Placed on this frame" /> : null}
        </button>
      </TooltipTrigger>
      <TooltipContent side="right" className="max-w-72">
        <div className="font-medium">{row.name}</div>
        <div className="font-mono tabular-nums">{row.detail}</div>
      </TooltipContent>
    </Tooltip>
  )
}

function GroupHeading({ children }: { children: React.ReactNode }) {
  return <p className="px-2 pt-3 pb-1 text-xs font-medium text-muted-foreground">{children}</p>
}

interface PalettePanelProps {
  mode: PaletteMode
  onModeChange: (mode: PaletteMode) => void
  landmarks: readonly Landmark[]
  pitchLines: readonly PitchLine[]
  selected: string | null
  /** Names already placed on the current frame. */
  placed: ReadonlySet<string>
  onSelect: (name: string) => void
}

export function PalettePanel(props: PalettePanelProps) {
  const { mode, onModeChange, landmarks, pitchLines, selected, placed, onSelect } = props
  const [query, setQuery] = React.useState("")
  const q = query.trim().toLowerCase()
  const match = (name: string) => !q || name.toLowerCase().includes(q)

  const pointRows = landmarks.filter((l) => match(l.name)).map(landmarkRow)
  const lineRows = pitchLines.filter((l) => l.category !== "mowing" && match(l.name)).map(lineRow)
  const mowRows = pitchLines.filter((l) => l.category === "mowing" && match(l.name)).map(lineRow)
  const total = mode === "points" ? pointRows.length : lineRows.length + mowRows.length

  const renderRows = (rows: PaletteRow[]) =>
    rows.map((r) => (
      <RowButton key={r.name} row={r} selected={selected === r.name} placed={placed.has(r.name)} onSelect={onSelect} />
    ))

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-col gap-2 border-b p-2">
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={mode}
          onValueChange={(v) => v && onModeChange(v as PaletteMode)}
          aria-label="Placement mode"
          className="w-full"
        >
          <ToggleGroupItem value="points" className="flex-1">Points</ToggleGroupItem>
          <ToggleGroupItem value="lines" className="flex-1">Lines</ToggleGroupItem>
        </ToggleGroup>
        <div className="relative">
          <SearchIcon className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={mode === "points" ? "Filter landmarks" : "Filter lines"}
            aria-label={mode === "points" ? "Filter landmarks" : "Filter lines"}
            className="h-8 pl-7 text-xs"
          />
        </div>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-0.5 p-2">
          {total === 0 ? (
            <p className="px-2 py-6 text-center text-xs text-muted-foreground">
              {q ? `Nothing matches "${query.trim()}".` : "No catalogue entries loaded."}
            </p>
          ) : mode === "points" ? (
            renderRows(pointRows)
          ) : (
            <>
              {renderRows(lineRows)}
              {mowRows.length > 0 ? (
                <>
                  <GroupHeading>Mowing stripes</GroupHeading>
                  {renderRows(mowRows)}
                </>
              ) : null}
            </>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

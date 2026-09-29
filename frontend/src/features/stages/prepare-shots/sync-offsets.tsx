import * as React from "react"

import { ToneBadge } from "@/components/status"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"

import type { AlignMethod } from "./sync-timeline"

interface OffsetRowsProps {
  shotIds: string[]
  referenceShot: string
  activeShot: string
  offsets: Record<string, number>
  methods: Record<string, AlignMethod>
  maxFrames: number
  onSetOffset: (shotId: string, offset: number) => void
  onPick: (shotId: string) => void
}

/** Integer input that tolerates in-progress text ("-", "") without fighting the value. */
export function OffsetInput({
  value,
  onCommit,
  id,
  label,
  className,
}: {
  value: number
  onCommit: (v: number) => void
  id?: string
  label: string
  className?: string
}) {
  const [draft, setDraft] = React.useState(String(value))
  React.useEffect(() => setDraft(String(value)), [value])
  return (
    <Input
      id={id}
      type="number"
      step={1}
      inputMode="numeric"
      aria-label={label}
      className={cn("h-8 w-24 tabular-nums", className)}
      value={draft}
      onChange={(e) => {
        setDraft(e.target.value)
        const v = Number(e.target.value)
        if (e.target.value.trim() !== "" && Number.isFinite(v)) onCommit(Math.round(v))
      }}
      onBlur={() => setDraft(String(value))}
    />
  )
}

export function MethodBadge({ method }: { method: AlignMethod | undefined }) {
  if (!method) return null
  if (method.method === "manual") return <ToneBadge tone="info">Manual</ToneBadge>
  return (
    <ToneBadge tone={method.confidence < 0.5 ? "destructive" : "success"} title={method.method}>
      Auto {method.confidence.toFixed(2)}
    </ToneBadge>
  )
}

/** One row per shot: pick it, see how it was aligned, set its offset by slider or number. */
export function SyncOffsetRows({
  shotIds,
  referenceShot,
  activeShot,
  offsets,
  methods,
  maxFrames,
  onSetOffset,
  onPick,
}: OffsetRowsProps) {
  const bound = Math.max(60, maxFrames + 120, ...shotIds.map((id) => Math.abs(offsets[id] ?? 0)))
  return (
    <ul className="flex flex-col divide-y rounded-lg border bg-stage/5">
      {shotIds.map((id) => {
        const isRef = id === referenceShot
        const isActive = id === activeShot
        const off = isRef ? 0 : (offsets[id] ?? 0)
        return (
          <li
            key={id}
            className={cn("flex flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2", isActive && !isRef && "bg-success/10")}
          >
            <button
              type="button"
              disabled={isRef}
              onClick={() => onPick(id)}
              aria-pressed={isActive}
              className="w-28 truncate text-left font-mono text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-default"
            >
              {id}
            </button>
            {isRef ? <ToneBadge tone="info">Reference</ToneBadge> : <MethodBadge method={methods[id]} />}
            <Slider
              className="min-w-40 flex-1"
              aria-label={`Offset for ${id}`}
              min={-bound}
              max={bound}
              step={1}
              disabled={isRef}
              value={[off]}
              onValueChange={([v]) => onSetOffset(id, v)}
            />
            <OffsetInput
              label={`Offset frames for ${id}`}
              value={off}
              onCommit={(v) => onSetOffset(id, v)}
              className={isRef ? "opacity-50" : undefined}
            />
          </li>
        )
      })}
    </ul>
  )
}

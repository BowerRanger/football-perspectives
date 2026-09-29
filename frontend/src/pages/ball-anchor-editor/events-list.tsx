import * as React from "react"
import {
  ArrowLeftToLineIcon,
  ArrowRightToLineIcon,
  PlusIcon,
  Trash2Icon,
  Undo2Icon,
  XIcon,
} from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { tagColour } from "./tags"
import { dismissKey, type AutoAnchor, type BallAnchor, type DismissedAuto } from "./types"
import type { AnchorDocApi } from "./use-anchor-doc"

interface EventRow {
  src: "manual" | "auto"
  anchor: BallAnchor | AutoAnchor
}

function detailOf(a: BallAnchor | AutoAnchor): string {
  const bits: string[] = []
  if (a.player_id) bits.push(`${a.player_id}/${a.bone ?? "?"}${a.touch_type ? ` ${a.touch_type}` : ""}`)
  else if (a.goal_element) bits.push(a.goal_element)
  if ("landmark" in a && a.landmark) bits.push(`landmark ${a.landmark}`)
  if ("spin" in a && a.spin && a.spin !== "none") bits.push(`spin ${a.spin}`)
  return bits.join(", ")
}

function Detail({ text }: { text: string }) {
  return text ? <span className="block truncate text-xs text-muted-foreground">{text}</span> : null
}

function IconAction(props: { label: string; tip?: string; onClick: () => void; tone?: "destructive"; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          size="icon-xs"
          variant="ghost"
          aria-label={props.label}
          className={props.tone === "destructive" ? "text-destructive hover:text-destructive" : undefined}
          onClick={props.onClick}
        >
          {props.children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{props.tip ?? props.label}</TooltipContent>
    </Tooltip>
  )
}

interface ListProps {
  anchors: BallAnchor[]
  autoAnchors: AutoAnchor[]
  dismissedAuto: DismissedAuto[]
  shotChains: number[][]
  currentFrame: number
  docApi: AnchorDocApi
  onSeek: (frame: number) => void
  onSetEnd: (anchorFrame: number) => void
}

export const EventsList = React.memo(function EventsList(p: ListProps) {
  const { anchors, autoAnchors, dismissedAuto, docApi } = p
  const dismissedKeys = React.useMemo(() => new Set(dismissedAuto.map(dismissKey)), [dismissedAuto])
  const rows = React.useMemo<EventRow[]>(
    () =>
      [
        ...anchors.map((a): EventRow => ({ src: "manual", anchor: a })),
        ...autoAnchors.map((a): EventRow => ({ src: "auto", anchor: a })),
      ].sort((x, y) => x.anchor.frame - y.anchor.frame || (x.src === "manual" ? -1 : 1)),
    [anchors, autoAnchors],
  )

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-medium">
        Events (manual + auto, {anchors.length}+{autoAnchors.length})
      </h3>
      {rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No events yet. Pick an anchor type, then click the ball on a frame to place the first anchor.
        </p>
      ) : (
        <ul className="flex flex-col gap-0.5">
          {rows.map((r) =>
            r.src === "manual" ? (
              <ManualRow key={`m${r.anchor.frame}`} a={r.anchor as BallAnchor} p={p} />
            ) : (
              <AutoRow
                key={`a${r.anchor.frame}-${r.anchor.state}-${r.anchor.player_id ?? ""}`}
                a={r.anchor as AutoAnchor}
                p={p}
                dismissed={dismissedKeys.has(dismissKey(r.anchor))}
              />
            ),
          )}
        </ul>
      )}
      <Chains chains={p.shotChains} docApi={docApi} onSeek={p.onSeek} />
    </div>
  )
})

function RowShell(props: {
  active: boolean
  muted?: boolean
  strike?: boolean
  onSeek: () => void
  dot: React.ReactNode
  title: React.ReactNode
  detail: string
  badge: React.ReactNode
  actions: React.ReactNode
}) {
  return (
    <li className={cn("flex items-center gap-1 rounded-md pr-1 hover:bg-muted/60", props.active && "bg-muted", props.muted && "text-muted-foreground")}>
      <button
        type="button"
        onClick={props.onSeek}
        className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
      >
        {props.dot}
        <span className={cn("min-w-0 flex-1", props.strike && "line-through")}>
          <span className="block truncate">{props.title}</span>
          <span className="flex items-center gap-1.5">
            {props.badge}
            <Detail text={props.detail} />
          </span>
        </span>
      </button>
      {props.actions}
    </li>
  )
}

function ManualRow({ a, p }: { a: BallAnchor; p: ListProps }) {
  const span = a.end_frame != null ? ` →${a.end_frame}` : ""
  return (
    <RowShell
      active={p.currentFrame === a.frame}
      onSeek={() => p.onSeek(a.frame)}
      dot={<span aria-hidden className="size-2 shrink-0 rounded-full" style={{ backgroundColor: tagColour(a.state) }} />}
      title={
        <>
          <span className="font-mono tabular-nums">Frame {a.frame}{span}</span> — {a.state}
        </>
      }
      detail={detailOf(a)}
      badge={<ToneBadge tone="info">manual</ToneBadge>}
      actions={
        <>
          {a.state === "player_touch" ? (
            <IconAction label="Set end frame" tip="Set end frame to the current video frame (span event, e.g. carry)" onClick={() => p.onSetEnd(a.frame)}>
              <ArrowRightToLineIcon />
            </IconAction>
          ) : null}
          {a.state === "player_touch" && a.end_frame != null ? (
            <IconAction label="Clear end frame" tip="Clear end frame (back to a point event)" onClick={() => p.docApi.setEndFrame(a.frame, null)}>
              <ArrowLeftToLineIcon />
            </IconAction>
          ) : null}
          <IconAction label={`Delete anchor at frame ${a.frame}`} tip="Delete this anchor (or right-click it on the frame)" tone="destructive" onClick={() => p.docApi.removeAnchor(a.frame)}>
            <Trash2Icon />
          </IconAction>
        </>
      }
    />
  )
}

function AutoRow({ a, p, dismissed }: { a: AutoAnchor; p: ListProps; dismissed: boolean }) {
  const suppressed = p.anchors.some((m) => Math.abs(m.frame - a.frame) <= 3)
  const conf = a.confidence != null ? `${Math.round(a.confidence * 100)}% confidence` : ""
  return (
    <RowShell
      active={p.currentFrame === a.frame}
      muted={dismissed || suppressed}
      strike={dismissed}
      onSeek={() => p.onSeek(a.frame)}
      dot={<span aria-hidden className="size-2 shrink-0 rounded-full border-[1.5px] border-dashed bg-transparent" style={{ borderColor: tagColour(a.state) }} />}
      title={
        <>
          <span className="font-mono tabular-nums">Frame {a.frame}</span> — {a.state}
        </>
      }
      detail={[detailOf(a), conf].filter(Boolean).join(" · ")}
      badge={
        dismissed ? <ToneBadge tone="destructive">dismissed</ToneBadge> : suppressed ? <ToneBadge tone="warning">suppressed</ToneBadge> : <ToneBadge tone="muted">auto</ToneBadge>
      }
      actions={
        dismissed ? (
          <IconAction label="Undo dismissal" onClick={() => p.docApi.undoDismiss(a)}>
            <Undo2Icon />
          </IconAction>
        ) : suppressed ? null : (
          <>
            <IconAction label="Promote suggestion" tip="Promote this suggestion to an editable anchor" onClick={() => p.docApi.promoteAuto(a)}>
              <PlusIcon />
            </IconAction>
            <IconAction label="Dismiss suggestion" tip="Dismiss this suggestion (persisted; won't return on re-runs)" tone="destructive" onClick={() => p.docApi.dismissAuto(a)}>
              <XIcon />
            </IconAction>
          </>
        )
      }
    />
  )
}

function Chains({ chains, docApi, onSeek }: { chains: number[][]; docApi: AnchorDocApi; onSeek: (f: number) => void }) {
  if (!chains.length) return null
  return (
    <div className="flex flex-col gap-1 border-t pt-3">
      <h3 className="text-sm font-medium">Shot chains ({chains.length})</h3>
      <ul className="flex flex-col gap-0.5">
        {chains.map((chain, idx) => (
          <li key={chain.join("-")} className="flex items-center gap-1 rounded-md pr-1 hover:bg-muted/60">
            <button
              type="button"
              className="min-w-0 flex-1 truncate rounded-md px-2 py-1.5 text-left font-mono text-sm tabular-nums outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
              onClick={() => onSeek(chain[0])}
            >
              {chain.join(" → ")}
            </button>
            <IconAction label="Delete chain" tip="Delete this chain (anchors stay)" tone="destructive" onClick={() => docApi.deleteChain(idx)}>
              <XIcon />
            </IconAction>
          </li>
        ))}
      </ul>
    </div>
  )
}

import { CrosshairIcon, PlusIcon, SaveIcon, Trash2Icon, XIcon } from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { cn } from "@/lib/utils"

import { IconButton } from "./icon-button"
import { fmtRate, momentsProblem, pairIssues, residualTone, type MomentFit, type MomentPair } from "./replay-speed"
import type { MomentsState } from "./use-moments"

interface MomentsTrayProps {
  referenceShot: string
  memberShot: string
  moments: MomentsState
  saving: boolean
  /** Why Save is unavailable for a reason outside the fit (e.g. ungrouped shots), or "". */
  blockedReason: string
  /** Current frame of each clip, read at click time. */
  readReference: () => number | null
  readMember: () => number | null
  onSave: () => void
  onClose: () => void
  onClear: () => void
  /** Return keyboard focus to the editor region so Enter / Esc work after a click. */
  onActed: () => void
}

function modeLine(m: MomentsState, reference: string, member: string): string {
  const n = m.pairs.length + 1
  if (m.pendingRef != null && m.pendingShot != null) {
    return `Pair ${n} ready: ${reference} frame ${m.pendingRef} = ${member} frame ${m.pendingShot}. Enter adds it.`
  }
  if (m.pendingRef != null) return `${reference} frame ${m.pendingRef} marked. Now mark the same instant in ${member}.`
  if (m.pendingShot != null) return `${member} frame ${m.pendingShot} marked. Now mark the same instant in ${reference}.`
  return "Scrub both clips to the same instant (a ball contact, a net impact), then mark each."
}

function ResultStrip({ fit, count, problem }: { fit: MomentFit | null; count: number; problem: string }) {
  if (count === 0) return <p className="text-xs text-muted-foreground">No pairs yet. Two pairs fix the rate and offset; a third checks them.</p>
  if (count === 1) {
    return <p className="text-xs text-muted-foreground">1 pair: offset only. Add a second moment to get the rate.</p>
  }
  if (!fit) return <p className="text-xs text-destructive">{problem}</p>
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm" aria-live="polite">
        <span>
          rate <span className="font-mono tabular-nums">{fit.rate.toFixed(3)}×</span>
        </span>
        <span>
          offset <span className="font-mono tabular-nums">{(-fit.offset).toFixed(1)}</span>
        </span>
        {count === 2 ? (
          <span className="text-xs text-muted-foreground">2 pairs fit exactly, add a third to check</span>
        ) : (
          <ToneBadge tone={residualTone(fit.residualFrames)} className="tabular-nums">
            residual {fit.residualFrames.toFixed(1)} f
          </ToneBadge>
        )}
      </div>
      {fit.ramp ? (
        <p className="text-xs text-warning">
          Speed ramp: {fit.intervalRates[0].toFixed(2)}× then {fit.intervalRates[fit.intervalRates.length - 1].toFixed(2)}×. A single rate
          saved from these pairs is an average.
        </p>
      ) : null}
      {problem && count >= 2 ? <p className="text-xs text-destructive">{problem}</p> : null}
    </div>
  )
}

function PairRow({
  index,
  pair,
  prev,
  fit,
  issue,
  onRemove,
}: {
  index: number
  pair: MomentPair
  prev: MomentPair | undefined
  fit: MomentFit | null
  issue: string | undefined
  onRemove: () => void
}) {
  const interval = prev ? (pair.reference_frame - prev.reference_frame) / (pair.shot_frame - prev.shot_frame) : null
  // Outlier: this pair sits more than 3 reference frames from the fitted line (three or more pairs only).
  const miss = fit && fit.n >= 3 ? Math.abs(pair.reference_frame - (fit.offset + fit.rate * pair.shot_frame)) : 0
  // A detected ramp already explains the misses; flag outliers only when it is not one.
  const off = miss > 3 && !fit?.ramp
  return (
    <li className="grid grid-cols-[1.5rem_1fr_1fr_6rem_1.5rem] items-center gap-2 px-3 py-1.5 text-sm">
      <span className="text-xs text-muted-foreground tabular-nums">{index + 1}</span>
      <span className="font-mono tabular-nums">{pair.reference_frame}</span>
      <span className="font-mono tabular-nums">{pair.shot_frame}</span>
      <span className={cn("flex items-center gap-1.5 text-xs tabular-nums", issue ? "text-destructive" : "text-muted-foreground")}>
        {issue
          ? issue === "order"
            ? "order inverted"
            : "repeated frame"
          : interval != null && Number.isFinite(interval)
            ? `${interval.toFixed(2)}×`
            : "—"}
        {off ? (
          <span
            role="img"
            tabIndex={0}
            className="size-2 shrink-0 rounded-full bg-warning"
            aria-label={`This pair disagrees with the others by ${miss.toFixed(0)} frames`}
            title={`This pair disagrees with the others by ${miss.toFixed(0)} frames`}
          />
        ) : null}
      </span>
      <IconButton label={`Remove pair ${index + 1}`} variant="ghost" size="icon-xs" onClick={onRemove}>
        <Trash2Icon />
      </IconButton>
    </li>
  )
}

/**
 * Camera-free alignment: mark the same instant in the live clip and the
 * replay (place, then commit), watch rate / offset / residual update, save a
 * manual alignment.
 */
export function MomentsTray({
  referenceShot,
  memberShot,
  moments,
  saving,
  blockedReason,
  readReference,
  readMember,
  onSave,
  onClose,
  onClear,
  onActed,
}: MomentsTrayProps) {
  const { pairs, fit, pendingRef, pendingShot } = moments
  const issues = pairIssues(pairs)
  const problem = momentsProblem(pairs, fit)
  const ready = pendingRef != null && pendingShot != null
  const saveBlock = blockedReason || problem
  return (
    <section aria-label={`Match moments for ${memberShot}`} className="flex flex-col gap-3 rounded-lg border p-3">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold">
          Match moments for <span className="font-mono">{memberShot}</span> against <span className="font-mono">{referenceShot}</span>
        </h3>
        <Button variant="ghost" size="xs" className="ml-auto" onClick={onClose}>
          <XIcon data-icon="inline-start" />
          Close <Kbd>M</Kbd>
        </Button>
      </div>

      <p className="text-sm" aria-live="polite" data-testid="moments-mode-line">
        {modeLine(moments, referenceShot, memberShot)}
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => {
            const f = readReference()
            if (f != null) moments.markRef(f)
            onActed()
          }}>
          <CrosshairIcon data-icon="inline-start" />
          Mark {referenceShot} <Kbd>1</Kbd>
        </Button>
        <Button variant="outline" size="sm" onClick={() => {
            const f = readMember()
            if (f != null) moments.markShot(f)
            onActed()
          }}>
          <CrosshairIcon data-icon="inline-start" />
          Mark {memberShot} <Kbd>2</Kbd>
        </Button>
        <Button
          size="sm"
          disabled={!ready}
          onClick={() => {
            moments.add()
            onActed()
          }}
        >
          <PlusIcon data-icon="inline-start" />
          Add pair <Kbd>Enter</Kbd>
        </Button>
        <Button variant="ghost" size="sm" disabled={pendingRef == null && pendingShot == null} onClick={() => {
            moments.discardPending()
            onActed()
          }}>
          Discard <Kbd>Esc</Kbd>
        </Button>
        {pendingRef != null ? (
          <ToneBadge tone="info" className="tabular-nums">
            <span className="font-mono">{referenceShot}</span> mark {pendingRef}
          </ToneBadge>
        ) : null}
        {pendingShot != null ? (
          <ToneBadge tone="info" className="tabular-nums">
            <span className="font-mono">{memberShot}</span> mark {pendingShot}
          </ToneBadge>
        ) : null}
      </div>

      <ResultStrip fit={fit} count={pairs.length} problem={problem} />

      {pairs.length > 0 ? (
        <div className="overflow-hidden rounded-md border">
          <div className="grid grid-cols-[1.5rem_1fr_1fr_6rem_1.5rem] gap-2 border-b bg-muted/40 px-3 py-1 text-xs text-muted-foreground">
            <span>#</span>
            <span className="font-mono">{referenceShot}</span>
            <span className="font-mono">{memberShot}</span>
            <span>interval</span>
            <span />
          </div>
          <ul className="divide-y">
            {pairs.map((p, i) => (
              <PairRow
                key={`${p.reference_frame}:${p.shot_frame}`}
                index={i}
                pair={p}
                prev={pairs[i - 1]}
                fit={fit}
                issue={issues.get(i)}
                onRemove={() => moments.remove(i)}
              />
            ))}
          </ul>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="ghost" size="sm" disabled={pairs.length === 0} onClick={onClear}>
          Clear all
        </Button>
        {saveBlock && pairs.length >= 1 ? <span className="text-xs text-muted-foreground">{saveBlock}</span> : null}
        <Button size="sm" className="ml-auto" disabled={!!saveBlock || saving} onClick={onSave}>
          <SaveIcon data-icon="inline-start" />
          Save alignment
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Saved as a manual alignment ({fit ? fmtRate(fit.rate) : "rate"} and offset); the automatic pass never overwrites it.
      </p>
    </section>
  )
}

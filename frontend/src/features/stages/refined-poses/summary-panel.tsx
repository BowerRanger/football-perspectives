import { Panel } from "@/components/panel"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { TriangleAlertIcon } from "lucide-react"
import { fmt, fmtInt } from "@/lib/format"

export type RefinedSummary = Record<string, unknown>

type Section = Record<string, unknown>

function section(s: RefinedSummary, key: string): Section | null {
  const v = s[key]
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Section) : null
}

function num(o: Section | RefinedSummary | null, key: string): number | null {
  const v = o?.[key]
  return typeof v === "number" && Number.isFinite(v) ? v : null
}

function strings(s: RefinedSummary, key: string): string[] {
  const v = s[key]
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []
}

function sumOf(o: Section | null, keys: string[]): number | null {
  if (!o) return null
  const vals = keys.map((k) => num(o, k)).filter((v): v is number => v !== null)
  return vals.length ? vals.reduce((a, b) => a + b, 0) : null
}

/** Rows only for keys the summary really has — never "undefined". */
function passRows(s: RefinedSummary): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = []
  const cleanup = section(s, "cleanup")
  const fixed = sumOf(cleanup, ["rejected_frames", "pop_rejected_frames", "clamped_frames", "accel_clamped_frames"])
  if (fixed !== null) rows.push({ label: "Cleanup: frames rejected or clamped", value: fmtInt(fixed) })
  const jitter = section(s, "jitter")
  if (jitter && num(jitter, "corrected_frames") !== null) {
    rows.push({ label: "Jitter: frames corrected", value: `${fmtInt(num(jitter, "corrected_frames"))} of ${fmtInt(num(jitter, "total_frames_evaluated"))}` })
  }
  const foot = section(s, "foot_lock")
  if (foot && num(foot, "spans_locked") !== null) {
    rows.push({
      label: "Foot lock: spans locked / skipped / unresolved",
      value: `${fmtInt(num(foot, "spans_locked"))} / ${fmtInt(num(foot, "spans_skipped"))} / ${fmtInt(num(foot, "spans_unresolved"))}`,
    })
    rows.push({
      label: "Foot lock: mean pin error before → after",
      value: `${fmt((num(foot, "mean_pin_err_m_before") ?? NaN) * 100, 1)} → ${fmt((num(foot, "mean_pin_err_m_after") ?? NaN) * 100, 2)} cm`,
    })
  }
  const phys = section(s, "physpt_takeover")
  if (phys && num(phys, "spans_flagged") !== null) {
    rows.push({
      label: "PhysPT takeover: spans flagged (translation / rotation accepted)",
      value: `${fmtInt(num(phys, "spans_flagged"))} (${fmtInt(num(phys, "translation_accepted"))} / ${fmtInt(num(phys, "rotation_accepted"))})`,
    })
  }
  const eff = section(s, "end_effectors")
  if (eff && num(eff, "toe_roll_frames") !== null) rows.push({ label: "End effectors: toe-roll frames", value: fmtInt(num(eff, "toe_roll_frames")) })
  return rows
}

function Warning({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null
  return (
    <Alert>
      <TriangleAlertIcon className="text-warning" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="font-mono text-xs">{items.join(", ")}</AlertDescription>
    </Alert>
  )
}

/** Pipeline-wide counters written by the refined_poses stage. */
export function SummaryPanel({ summary }: { summary: RefinedSummary }) {
  const refined = num(summary, "players_refined")
  const fused = num(summary, "total_fused_frames") ?? num(summary, "total_frames")
  const flagged = num(summary, "high_disagreement_frames")
  const singleView = num(summary, "single_view_frames")
  const items = [
    { label: "Players refined", value: fmtInt(refined) },
    { label: "Multi-shot / single-shot", value: `${fmtInt(num(summary, "multi_shot_players"))} / ${fmtInt(num(summary, "single_shot_players"))}` },
    { label: "Frames", value: fmtInt(fused) },
    ...(singleView !== null ? [{ label: "Single-view frames", value: fmtInt(singleView) }] : []),
    ...(flagged !== null ? [{ label: "Flagged (high disagreement)", value: fmtInt(flagged) }] : []),
    ...passRows(summary),
  ]
  return (
    <Panel title="Pipeline summary">
      <div className="flex flex-col gap-3">
        <dl className="grid gap-x-8 gap-y-2 text-sm sm:grid-cols-2 xl:grid-cols-3">
          {items.map((it) => (
            <div key={it.label} className="flex items-baseline justify-between gap-3 border-b border-border/50 pb-1.5">
              <dt className="text-muted-foreground">{it.label}</dt>
              <dd className="shrink-0 text-right font-medium tabular-nums">{it.value}</dd>
            </div>
          ))}
        </dl>
        <Warning title="Shots missing from sync_map" items={strings(summary, "shots_missing_sync")} />
        <Warning title="Beta disagreement" items={strings(summary, "beta_disagreement_warnings")} />
      </div>
    </Panel>
  )
}

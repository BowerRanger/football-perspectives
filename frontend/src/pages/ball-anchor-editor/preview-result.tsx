import { ToneBadge } from "@/components/status"
import type { PreviewResult } from "./types"

/** Summary of the last Solve & preview: state counts and shot-chain warnings. */
export function PreviewSummary({ result }: { result: PreviewResult }) {
  const counts = new Map<string, number>()
  for (const f of result.frames) counts.set(f.state, (counts.get(f.state) ?? 0) + 1)
  const warnings = (result.shot_chain_warnings ?? []).flatMap((c) =>
    c.warnings.map((w) => `[${c.frames.join("→")}] ${w.detail}`),
  )
  return (
    <div className="flex flex-col gap-2 rounded-lg border p-3 text-sm" role="status">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">Solve preview</span>
        {[...counts.entries()].map(([state, n]) => (
          <ToneBadge key={state} tone="muted" className="font-mono">
            {state}: {n}
          </ToneBadge>
        ))}
        <span className="text-xs text-muted-foreground">Faint white ring on the frame = solved ball position.</span>
      </div>
      {warnings.length ? (
        <ul className="flex list-disc flex-col gap-0.5 pl-5 text-warning">
          {warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

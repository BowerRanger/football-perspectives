import * as React from "react"
import { Link } from "react-router"
import { ExternalLinkIcon } from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Popover, PopoverContent, PopoverDescription, PopoverHeader, PopoverTitle, PopoverTrigger } from "@/components/ui/popover"
import { Spinner } from "@/components/ui/spinner"
import { errorMessage } from "@/lib/api"
import { postTriangulate } from "./api"
import { residualSeverity } from "./palette"
import type { TriangulateResult, TruthKey } from "./types"

const DELTAS = [-2, -1, 0, 1, 2] as const

interface Row {
  delta: number
  offset: number
  max: number | null
  gap: number | null
  ok: boolean
}

interface SyncProbeProps {
  groupId: string
  tkey: TruthKey
  /** Stored frame_offset per shot (the sync map's current values). */
  /** Stored frame_offset of the moving view (the sync map's current value). */
  storedOffset: Record<string, number>
  referenceShot: string
  children: React.ReactNode
}

/**
 * Read-only "what if view B's camera frame were off by a frame or two" probe.
 * The picked pixel is held and only the camera frame changes, so this mostly
 * measures camera drift - it is not a sync verdict. The Studio never writes
 * sync_map.json; the link goes to the Prepare Shots sync timeline.
 */
export function SyncProbe({ groupId, tkey, storedOffset, referenceShot, children }: SyncProbeProps) {
  const [open, setOpen] = React.useState(false)
  const [rows, setRows] = React.useState<Row[] | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const moving = tkey.observations.map((o) => o.shot_id).find((s) => s !== referenceShot)
  const stored = moving ? (storedOffset[moving] ?? 0) : 0
  // Latest key payload without making the effect depend on its identity.
  const keyRef = React.useRef(tkey)
  React.useEffect(() => {
    keyRef.current = tkey
  })

  React.useEffect(() => {
    if (!open || !moving) return
    let cancelled = false
    setRows(null)
    setError(null)
    const k = keyRef.current
    Promise.all(
      DELTAS.map(async (delta): Promise<Row> => {
        const r: TriangulateResult = await postTriangulate(groupId, {
          frame: k.frame,
          observations: k.observations,
          constraint: null,
          offsets: { [moving]: stored + delta },
        })
        return {
          delta,
          offset: stored + delta,
          max: r.max_residual_px ?? null,
          gap: r.skew_gap_cm ?? null,
          ok: r.ok,
        }
      }),
    )
      .then((out) => !cancelled && setRows(out))
      .catch((err: unknown) => !cancelled && setError(errorMessage(err)))
    return () => {
      cancelled = true
    }
  }, [open, moving, stored, groupId, tkey.id, tkey.frame])

  const best = rows?.reduce<Row | null>((b, r) => (r.max !== null && (b === null || (b.max ?? Infinity) > r.max) ? r : b), null)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button type="button" className="rounded-md outline-none focus-visible:ring-3 focus-visible:ring-ring/50" aria-label="Sync probe for this residual">
          {children}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 text-sm">
        <PopoverHeader>
          <PopoverTitle>Same click, other camera frame</PopoverTitle>
          <PopoverDescription>
            The pixel stays where you clicked in {moving ?? "view B"}; only its camera frame moves. This mostly shows camera drift, so it is a hint, not a sync verdict. To
            test a sync, re-click the ball in B at the candidate frame.
          </PopoverDescription>
        </PopoverHeader>
        {error ? <p className="text-xs text-destructive">{error}</p> : null}
        {!rows && !error ? (
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            <Spinner className="size-3.5" /> Probing five offsets…
          </p>
        ) : null}
        {rows ? (
          <table className="w-full text-xs tabular-nums">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="py-1 font-normal">Offset</th>
                <th className="font-normal">Residual</th>
                <th className="text-right font-normal">Skew gap</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.delta}>
                  <td className="py-1 font-mono">
                    {r.offset} <span className="text-muted-foreground">({r.delta > 0 ? `+${r.delta}` : r.delta === 0 ? "stored" : r.delta})</span>
                    {r === best ? <span className="ml-1 text-muted-foreground" title="Lowest residual of the five">·</span> : null}
                  </td>
                  <td>
                    {r.max === null ? (
                      "n/a"
                    ) : (
                      <ToneBadge tone={residualSeverity(r.max)} className="font-mono">
                        {r.max.toFixed(1)} px
                      </ToneBadge>
                    )}
                  </td>
                  <td className="text-right font-mono">{r.gap === null ? "n/a" : `${r.gap.toFixed(0)} cm`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
        {rows ? (
          <p className="text-xs text-muted-foreground">
            A held click mostly measures camera motion. To check sync, re-pick the ball on a neighbouring frame of this view.
          </p>
        ) : null}
        <Link to="/?stage=prepare_shots" className="inline-flex items-center gap-1 text-xs underline underline-offset-2">
          Open sync timeline <ExternalLinkIcon className="size-3" aria-hidden />
        </Link>
      </PopoverContent>
    </Popover>
  )
}

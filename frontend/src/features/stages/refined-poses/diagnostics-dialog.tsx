import * as React from "react"
import { ActivityIcon } from "lucide-react"

import { StatList } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Skeleton } from "@/components/ui/skeleton"
import { ApiError, errorMessage, getJson } from "@/lib/api"

interface Diagnostics {
  summary?: Record<string, unknown>
}

/** Flatten numeric leaves of the diagnostics summary into label/value rows. */
function flatten(obj: Record<string, unknown>, prefix = ""): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = []
  for (const [k, v] of Object.entries(obj)) {
    const name = prefix ? `${prefix} · ${k.replace(/_/g, " ")}` : k.replace(/_/g, " ")
    if (typeof v === "number" && Number.isFinite(v)) rows.push({ label: name, value: Number.isInteger(v) ? String(v) : v.toFixed(4) })
    else if (typeof v === "string") rows.push({ label: name, value: v })
    else if (Array.isArray(v) && v.every((x) => typeof x === "string")) rows.push({ label: name, value: v.join(", ") || "—" })
    else if (v && typeof v === "object" && !Array.isArray(v)) rows.push(...flatten(v as Record<string, unknown>, name))
  }
  return rows
}

function DiagnosticsBody({ playerId }: { playerId: string }) {
  const [rows, setRows] = React.useState<{ label: string; value: string }[] | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    let alive = true
    getJson<Diagnostics>(`/refined_poses/diagnostics?player_id=${encodeURIComponent(playerId)}`)
      .then((d) => alive && setRows(flatten(d.summary ?? {})))
      .catch((err: unknown) => {
        if (!alive) return
        setError(err instanceof ApiError && err.status === 404 ? "No diagnostics were written for this player. Re-run refined_poses to regenerate them." : errorMessage(err))
      })
    return () => {
      alive = false
    }
  }, [playerId])

  if (error) return <p className="text-sm text-destructive">{error}</p>
  if (!rows) return <Skeleton className="h-32 w-full" />
  if (rows.length === 0) return <p className="text-sm text-muted-foreground">Diagnostics file has no summary values.</p>
  return <StatList items={rows} className="max-h-80 overflow-y-auto" />
}

export function DiagnosticsDialog({ playerId, label }: { playerId: string; label: string }) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label={`Fusion diagnostics for ${label}`}>
          <ActivityIcon />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Fusion diagnostics — {label}</DialogTitle>
          <DialogDescription>Per-player refined_poses summary (foot lock, coverage, disagreement).</DialogDescription>
        </DialogHeader>
        <DiagnosticsBody playerId={playerId} />
      </DialogContent>
    </Dialog>
  )
}

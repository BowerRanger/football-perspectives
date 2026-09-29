
import { Panel, PanelError } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { ToneBadge, type Tone } from "@/components/status"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getJson } from "@/lib/api"
import { useResource } from "@/hooks/use-resource"
import { fmtInt } from "@/lib/format"

interface ShotStatus {
  shot_id: string
  has_anchors: boolean
  anchor_count: number
  has_camera: boolean
  camera_stale: boolean
  has_hmr: boolean
  hmr_player_count: number
  has_ball: boolean
  has_export: boolean
}

interface RefinedSummary {
  players_refined?: number
  single_shot_players?: number
  multi_shot_players?: number
  total_frames?: number
  total_fused_frames?: number
  high_disagreement_frames?: number
  cleanup?: { accel_clamped_frames?: number; clamped_frames?: number }
  foot_lock?: { spans_locked?: number; spans_skipped?: number }
}

interface QualityReport {
  refined_poses?: RefinedSummary
}

function present(ok: boolean, text = "Present"): { tone: Tone; text: string } {
  return ok ? { tone: "success", text } : { tone: "muted", text: "Missing" }
}

function Cell({ tone, text, title }: { tone: Tone; text: string; title?: string }) {
  return (
    <ToneBadge tone={tone} title={title}>
      {text}
    </ToneBadge>
  )
}

function StatusRow({ s }: { s: ShotStatus | null; id?: string }) {
  if (!s) return null
  const cam = s.camera_stale
    ? { tone: "warning" as Tone, text: "Stale" }
    : present(s.has_camera)
  const hmr = present(s.has_hmr, `${s.hmr_player_count} players`)
  return (
    <TableRow>
      <TableCell className="font-mono">{s.shot_id}</TableCell>
      <TableCell>
        <Cell {...(s.has_anchors ? { tone: "success" as Tone, text: `${s.anchor_count} anchors` } : present(false))} />
      </TableCell>
      <TableCell>
        <Cell
          {...cam}
          title={s.camera_stale ? "Anchors were edited after the last camera solve. Re-run camera." : undefined}
        />
      </TableCell>
      <TableCell>
        <Cell {...hmr} />
      </TableCell>
      <TableCell>
        <Cell {...present(s.has_ball)} />
      </TableCell>
      <TableCell>
        <Cell {...present(s.has_export)} />
      </TableCell>
    </TableRow>
  )
}

function refinedLine(rp: RefinedSummary | undefined): string {
  if (!rp) return "not run"
  const frames = rp.total_frames ?? rp.total_fused_frames
  const parts = [
    `${fmtInt(rp.players_refined)} players (${fmtInt(rp.multi_shot_players)} multi-shot, ${fmtInt(rp.single_shot_players)} single-shot)`,
    `${fmtInt(frames)} frames`,
  ]
  if (rp.foot_lock?.spans_locked !== undefined) parts.push(`${fmtInt(rp.foot_lock.spans_locked)} foot-lock spans`)
  const clamped = rp.cleanup?.accel_clamped_frames ?? rp.cleanup?.clamped_frames
  if (clamped !== undefined) parts.push(`${fmtInt(clamped)} clamped frames`)
  if (rp.high_disagreement_frames !== undefined) parts.push(`${fmtInt(rp.high_disagreement_frames)} flagged`)
  return parts.join(", ")
}

/** Per-shot artefact summary plus the refined-poses line from the quality report. */
export function MultiShotStatus({ shotIds }: { shotIds: string[] }) {
  const key = shotIds.join("|")
  // shot-status and quality-report answer 200 with empty payloads when
  // nothing has run, so any failure here is a real error worth a Retry.
  const { state, retry } = useResource(async (signal) => {
    const ids = key ? key.split("|") : []
    const [rows, report] = await Promise.all([
      Promise.all(ids.map((id) => getJson<ShotStatus>(`/api/output/shot-status/${encodeURIComponent(id)}`, { signal }))),
      getJson<QualityReport>("/api/output/quality-report", { signal }),
    ])
    return { rows, report }
  }, [key])

  return (
    <Panel title="Multi-shot status" description="Which shots have anchors, a current camera solve, poses, ball and export output.">
      {state.status === "error" ? (
        <PanelError
          title="Could not load shot status"
          message={state.error}
          action={
            <Button size="sm" variant="outline" className="mt-2" onClick={retry}>
              Retry
            </Button>
          }
        />
      ) : state.status === "loading" ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-6 w-full" />
          <Skeleton className="h-6 w-4/5" />
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <Table>
            <TableHeader className="sticky top-0 bg-card">
              <TableRow>
                <TableHead>Shot</TableHead>
                <TableHead>Anchors</TableHead>
                <TableHead>Camera</TableHead>
                <TableHead>HMR</TableHead>
                <TableHead>Ball</TableHead>
                <TableHead>Export</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {state.data.rows.map((s, i) =>
                s ? (
                  <StatusRow key={shotIds[i]} s={s} />
                ) : (
                  <TableRow key={shotIds[i]}>
                    <TableCell className="font-mono">{shotIds[i]}</TableCell>
                    <TableCell colSpan={5} className="text-muted-foreground">
                      Status unavailable
                    </TableCell>
                  </TableRow>
                ),
              )}
            </TableBody>
          </Table>
          <p className="text-sm">
            <strong className="font-medium">Refined poses:</strong>{" "}
            <span className="text-muted-foreground">{refinedLine(state.data.report.refined_poses)}</span>
          </p>
        </div>
      )}
    </Panel>
  )
}

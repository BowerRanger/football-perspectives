import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { useResource } from "@/hooks/use-resource"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getJson, getJsonOr404 } from "@/lib/api"
import { fmtInt, playerColour } from "@/lib/format"
import { ConfidenceCell, PlayerCell } from "../hmr-world/player-cells"
import { TrajectoryPanel } from "../hmr-world/trajectory-panel"
import type { Coloured, PosePreview } from "../hmr-world/types"
import { DiagnosticsDialog } from "./diagnostics-dialog"
import { SummaryPanel, type RefinedSummary } from "./summary-panel"

interface RefinedPlayer {
  player_id: string
  player_name?: string | null
  contributing_shots?: string[]
  n_frames?: number
  multi_view_frames?: number
  single_view_frames?: number
  mean_confidence?: number
}

interface Loaded {
  players: Coloured<RefinedPlayer>[]
  summary: RefinedSummary | null
  previews: (Coloured<RefinedPlayer> & { data: PosePreview })[]
}

// /players and /summary answer 200 ([] / {}) before the stage has run; a
// preview 404 means that one player's track is missing. Everything else throws.
async function load(signal: AbortSignal): Promise<Loaded> {
  const [list, summary] = await Promise.all([
    getJson<{ players?: RefinedPlayer[] }>("/refined_poses/players", { signal }),
    getJson<RefinedSummary>("/refined_poses/summary", { signal }),
  ])
  const players = (list.players ?? []).map((p, i) => ({ ...p, colour: playerColour(i) }))
  const loaded = await Promise.all(
    players.map(async (p) => ({
      ...p,
      data: await getJsonOr404<PosePreview>(`/refined_poses/preview?player_id=${encodeURIComponent(p.player_id)}`, { signal }),
    })),
  )
  const previews = loaded.flatMap((p) =>
    p.data && Array.isArray(p.data.root_t) && p.data.root_t.length > 0 ? [{ ...p, data: p.data }] : [],
  )
  return { players, summary: summary && Object.keys(summary).length ? summary : null, previews }
}

function PlayersTable({ players }: { players: readonly Coloured<RefinedPlayer>[] }) {
  return (
    <Panel title={`Refined poses — ${players.length} player${players.length === 1 ? "" : "s"}`} flush>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Player</TableHead>
            <TableHead className="text-right">Frames</TableHead>
            <TableHead className="text-right">Multi-view</TableHead>
            <TableHead className="text-right">Single-view</TableHead>
            <TableHead>Contributing shots</TableHead>
            <TableHead>Mean confidence</TableHead>
            <TableHead className="w-10" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {players.map((p) => (
            <TableRow key={p.player_id}>
              <TableCell>
                <PlayerCell colour={p.colour} playerId={p.player_id} playerName={p.player_name} />
              </TableCell>
              <TableCell className="text-right">{fmtInt(p.n_frames)}</TableCell>
              <TableCell className="text-right">{fmtInt(p.multi_view_frames)}</TableCell>
              <TableCell className="text-right">{fmtInt(p.single_view_frames)}</TableCell>
              <TableCell className="font-mono text-xs">{(p.contributing_shots ?? []).join(", ") || "—"}</TableCell>
              <TableCell>
                <ConfidenceCell value={p.mean_confidence} />
              </TableCell>
              <TableCell>
                <DiagnosticsDialog playerId={p.player_id} label={p.player_name || p.player_id} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Panel>
  )
}

export default function RefinedPosesStage() {
  const { state, retry } = useResource(load, [])

  if (state.status === "loading") return <PanelSkeleton rows={6} />
  if (state.status === "error") {
    return (
      <PanelError
        title="Could not load refined poses"
        message={state.error}
        action={
          <Button variant="outline" size="sm" className="mt-2" onClick={retry}>
            Retry
          </Button>
        }
      />
    )
  }
  const data = state.data
  if (data.players.length === 0) {
    return (
      <PanelEmpty
        title="No refined tracks yet"
        description="Refined poses fuse HMR World tracks across shots. Run hmr_world first, then run this stage from the header."
      />
    )
  }
  // The fused track lives on the reference timeline; play the first shot any player contributed to.
  const contextShot = data.previews.map((p) => p.contributing_shots?.[0] ?? p.data.contributing_shots?.[0]).find((s) => !!s) ?? null
  return (
    <div className="flex flex-col gap-4">
      {data.summary ? <SummaryPanel summary={data.summary} /> : null}
      <PlayersTable players={data.players} />
      {data.previews.length > 0 ? (
        <TrajectoryPanel players={data.previews} shotId={contextShot} title="Top-down player trajectories (refined)" />
      ) : null}
    </div>
  )
}

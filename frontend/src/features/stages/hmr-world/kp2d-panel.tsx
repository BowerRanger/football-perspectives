import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { useResource } from "@/hooks/use-resource"
import { ToneBadge } from "@/components/status"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getJson, getJsonOr404 } from "@/lib/api"
import { fmtInt, playerColour } from "@/lib/format"
import { Kp2dViewer, type Kp2dPlayer } from "./kp2d-viewer"
import { PlayerCell } from "./player-cells"
import type { Kp2dPreview, PlayerRef, PlayersResponse } from "./types"

async function loadKp2d(shotId: string, signal: AbortSignal): Promise<Kp2dPlayer[]> {
  const shotQ = `shot=${encodeURIComponent(shotId)}`
  const list = await getJson<PlayersResponse>(`/hmr_world/kp2d_players?${shotQ}`, { signal })
  const players = list.players ?? []
  return Promise.all(
    players.map(async (p: PlayerRef, i): Promise<Kp2dPlayer> => {
      const data = await getJsonOr404<Kp2dPreview>(
        `/hmr_world/kp2d_preview?player_id=${encodeURIComponent(p.player_id)}&${shotQ}`,
        { signal },
      )
      return { ...p, colour: playerColour(i), data: data ?? { player_id: p.player_id, frames: [] } }
    }),
  )
}

function FrameCountTable({ players }: { players: readonly Kp2dPlayer[] }) {
  return (
    <Panel title="Per-player keypoint frames" flush>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Player</TableHead>
            <TableHead>Shot</TableHead>
            <TableHead className="text-right">Frames</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {players.map((p) => {
            const n = p.data.frames.length
            return (
              <TableRow key={p.player_id}>
                <TableCell>
                  <PlayerCell colour={p.colour} playerId={p.player_id} playerName={p.player_name} />
                </TableCell>
                <TableCell className="font-mono text-xs">{p.data.shot_id ?? p.shot_id ?? "—"}</TableCell>
                <TableCell className="text-right">
                  <span className={n === 0 ? "text-muted-foreground" : "text-success"}>{fmtInt(n)}</span>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </Panel>
  )
}

/** kp2d summary, skeleton viewer and per-player frame counts for one shot. */
export function Kp2dPanel({ shotId }: { shotId: string }) {
  const { state, retry } = useResource((signal) => loadKp2d(shotId, signal), [shotId])

  if (state.status === "loading") return <PanelSkeleton rows={3} media />
  if (state.status === "error") {
    return (
      <PanelError
        title="Could not load 2D keypoints"
        message={state.error}
        action={
          <Button variant="outline" size="sm" className="mt-2" onClick={retry}>
            Retry
          </Button>
        }
      />
    )
  }
  const players = state.data
  if (players.length === 0) {
    return (
      <Panel title="2D keypoints">
        <PanelEmpty title="No 2D keypoints for this shot" description="Keypoints are written by hmr_world alongside each player's SMPL track. Run hmr_world for this shot to produce them." />
      </Panel>
    )
  }
  const withFrames = players.filter((p) => p.data.frames.length > 0)
  const total = players.reduce((s, p) => s + p.data.frames.length, 0)
  const empty = players.length - withFrames.length
  return (
    <>
      <Panel title={`2D keypoints (${players.length} players, from GVHMR ViTPose)`}>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          <span>
            Total keypoint frames <strong className="tabular-nums">{fmtInt(total)}</strong>
          </span>
          <span className="inline-flex items-center gap-2">
            Players with empty keypoints
            <ToneBadge tone={empty > 0 ? "warning" : "success"}>
              {empty} / {players.length}
            </ToneBadge>
          </span>
        </div>
      </Panel>
      {withFrames.length > 0 ? <Kp2dViewer shotId={shotId} players={withFrames} /> : null}
      <FrameCountTable players={players} />
    </>
  )
}

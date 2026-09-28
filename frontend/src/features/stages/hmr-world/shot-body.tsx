import * as React from "react"

import { Panel, PanelEmpty, PanelSkeleton } from "@/components/panel"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getJsonOrNull } from "@/lib/api"
import { fmtInt, playerColour } from "@/lib/format"
import { Kp2dPanel } from "./kp2d-panel"
import { ConfidenceCell, PlayerCell } from "./player-cells"
import { TrajectoryPanel } from "./trajectory-panel"
import type { Coloured, PlayerRef, PlayersResponse, PosePreview } from "./types"

type LoadedPlayer = Coloured<PlayerRef> & { data: PosePreview | null }

async function loadShot(shotId: string): Promise<LoadedPlayer[]> {
  const list = await getJsonOrNull<PlayersResponse>(`/hmr_world/players?shot=${encodeURIComponent(shotId)}`)
  return Promise.all(
    (list?.players ?? []).map(async (p, i) => ({
      ...p,
      colour: playerColour(i),
      data: await getJsonOrNull<PosePreview>(
        `/hmr_world/preview?shot=${encodeURIComponent(shotId)}&player_id=${encodeURIComponent(p.player_id)}`,
      ),
    })),
  )
}

function meanConfidence(conf: readonly number[] | undefined): number | null {
  return conf && conf.length ? conf.reduce((s, v) => s + v, 0) / conf.length : null
}

function PlayersTable({ shotId, players }: { shotId: string; players: readonly LoadedPlayer[] }) {
  return (
    <Panel title={`HMR World — ${shotId} (${players.length} player${players.length === 1 ? "" : "s"})`} flush>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Player</TableHead>
            <TableHead className="text-right">Frames</TableHead>
            <TableHead>Mean confidence</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {players.map((p) => (
            <TableRow key={p.player_id}>
              <TableCell>
                <PlayerCell colour={p.colour} playerId={p.player_id} playerName={p.player_name} />
              </TableCell>
              <TableCell className="text-right">{p.data ? fmtInt(p.data.frames?.length ?? 0) : "—"}</TableCell>
              <TableCell>
                <ConfidenceCell value={p.data ? meanConfidence(p.data.confidence) : null} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Panel>
  )
}

/** One shot's hmr_world output: player table, trajectories, keypoints. */
export function HmrShotBody({ shotId }: { shotId: string }) {
  const [players, setPlayers] = React.useState<LoadedPlayer[] | null>(null)

  React.useEffect(() => {
    let alive = true
    setPlayers(null)
    void loadShot(shotId).then((p) => {
      if (alive) setPlayers(p)
    })
    return () => {
      alive = false
    }
  }, [shotId])

  const withRoot = React.useMemo(
    () =>
      (players ?? []).flatMap((p) =>
        p.data && Array.isArray(p.data.root_t) && p.data.root_t.length > 0 ? [{ ...p, data: p.data }] : [],
      ),
    [players],
  )

  if (players === null) return <PanelSkeleton rows={5} />
  if (players.length === 0) {
    return (
      <Panel title="HMR World">
        <PanelEmpty
          title={`No hmr_world tracks for ${shotId} yet`}
          description="Pick a player above (or All players) and choose Run for selection."
        />
      </Panel>
    )
  }
  return (
    <>
      <PlayersTable shotId={shotId} players={players} />
      {withRoot.length > 0 ? <TrajectoryPanel players={withRoot} shotId={shotId} title="Top-down player trajectories (pitch-world)" /> : null}
      <Kp2dPanel shotId={shotId} />
    </>
  )
}

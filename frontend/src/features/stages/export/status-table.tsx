import { CheckIcon, DownloadIcon, MinusIcon } from "lucide-react"

import { Panel } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { fmtInt } from "@/lib/format"

export interface ShotExportRow {
  shotId: string
  exported: boolean
  /** null when there's no exported metadata to count from. */
  playersInScene: number | null
  hmrCount: number
}

/** Per-shot export status: baked scene, players in it, HMR players on disk. */
export function StatusTable({ rows }: { rows: readonly ShotExportRow[] }) {
  return (
    <Panel title="Export status" description="A scene can also be viewed live from hmr_world output before it has been exported." flush>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Shot</TableHead>
            <TableHead>Exported scene</TableHead>
            <TableHead className="text-right">Players in scene</TableHead>
            <TableHead className="text-right">HMR players on disk</TableHead>
            <TableHead className="w-10" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.shotId}>
              <TableCell className="font-mono text-xs">{r.shotId}</TableCell>
              <TableCell>
                {r.exported ? (
                  <span className="inline-flex items-center gap-1 text-success">
                    <CheckIcon className="size-4" /> glTF ready
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-muted-foreground">
                    <MinusIcon className="size-4" /> Not exported
                  </span>
                )}
              </TableCell>
              <TableCell className="text-right">{r.playersInScene === null ? "—" : fmtInt(r.playersInScene)}</TableCell>
              <TableCell className={`text-right ${r.hmrCount > 0 ? "text-success" : "text-muted-foreground"}`}>{fmtInt(r.hmrCount)}</TableCell>
              <TableCell>
                {r.exported ? (
                  <Button asChild variant="ghost" size="icon-sm">
                    <a href={`/api/export/scene.glb?shot=${encodeURIComponent(r.shotId)}`} download aria-label={`Download scene.glb for ${r.shotId}`}>
                      <DownloadIcon />
                    </a>
                  </Button>
                ) : null}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Panel>
  )
}

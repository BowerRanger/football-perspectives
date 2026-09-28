import { ChevronDownIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { fmt } from "@/lib/format"
import type { BallPreviewTrack } from "@/pages/ball-anchor-editor/api"

type Segments = NonNullable<BallPreviewTrack["flight_segments"]>

/** Flight segments, collapsed by default; columns that are empty for every row are hidden. */
export function SegmentsTable({ segments, fps }: { segments: Segments; fps: number | undefined }) {
  const hasResidual = segments.some((s) => s.fit_residual_px)
  const hasSpin = segments.some((s) => s.parabola?.spin_omega_rad_s != null)
  return (
    <Collapsible>
      <CollapsibleTrigger asChild>
        <Button variant="ghost" size="sm" className="group -ml-2">
          <ChevronDownIcon className="transition-transform group-data-[state=open]:rotate-180" />
          Flight segments ({segments.length})
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Segment</TableHead>
              <TableHead>Frames</TableHead>
              <TableHead>Duration</TableHead>
              {hasResidual ? <TableHead>Fit residual (px)</TableHead> : null}
              {hasSpin ? <TableHead>|ω| (rad/s)</TableHead> : null}
              {hasSpin ? <TableHead>Spin axis (world)</TableHead> : null}
              {hasSpin ? <TableHead>Spin confidence</TableHead> : null}
            </TableRow>
          </TableHeader>
          <TableBody>
            {segments.map((s) => {
              const [a, b] = s.frame_range ?? []
              const dur = fps && a != null && b != null ? `${fmt((b - a) / fps, 2)}s` : "—"
              const p = s.parabola
              return (
                <TableRow key={s.id}>
                  <TableCell className="font-mono">{s.id}</TableCell>
                  <TableCell className="font-mono">{a != null ? `${a}–${b}` : "—"}</TableCell>
                  <TableCell>{dur}</TableCell>
                  {hasResidual ? <TableCell>{fmt(s.fit_residual_px, 2)}</TableCell> : null}
                  {hasSpin ? <TableCell>{p?.spin_omega_rad_s != null ? fmt(p.spin_omega_rad_s, 2) : "—"}</TableCell> : null}
                  {hasSpin ? (
                    <TableCell className="font-mono">
                      {p?.spin_axis_world ? `[${p.spin_axis_world.map((v) => fmt(v, 2)).join(", ")}]` : "—"}
                    </TableCell>
                  ) : null}
                  {hasSpin ? <TableCell>{p?.spin_confidence != null ? fmt(p.spin_confidence, 2) : "—"}</TableCell> : null}
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </CollapsibleContent>
    </Collapsible>
  )
}

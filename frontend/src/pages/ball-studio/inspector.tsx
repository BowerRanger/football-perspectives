import * as React from "react"
import { Trash2Icon } from "lucide-react"

import { Panel } from "@/components/panel"
import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { SyncProbe } from "./sync-probe"
import { EVENT_KINDS, EVENT_STYLE, KEY_SOURCE_STYLE, SEGMENT_KINDS, SEGMENT_STYLE, residualSeverity } from "./palette"
import { removeSegment, setNotes, setOutcome, setSegmentKind, setStatus, updateEvent, updateKey } from "./truth-doc"
import type { EventKind, Outcome, SegmentKind, TruthStatus } from "./types"
import type { Studio } from "./use-studio"

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[88px_1fr] items-center gap-2 text-sm">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="min-w-0">{children}</div>
    </div>
  )
}

const fmt3 = (n: number) => n.toFixed(2)

function KeyInspector({ studio, id }: { studio: Studio; id: string }) {
  const { docApi, solvedKeyById, solver } = studio
  const shotsInScene = studio.scene.shots
  const storedOffsets = React.useMemo(() => Object.fromEntries(shotsInScene.map((s) => [s.shot_id, s.frame_offset])), [shotsInScene])
  const key = docApi.doc.keys.find((k) => k.id === id)
  if (!key) return <p className="text-sm text-muted-foreground">That key no longer exists.</p>
  const solvedKey = solvedKeyById.get(id)
  const xyz = solvedKey?.xyz ?? key.xyz
  const idx = docApi.doc.keys.findIndex((k) => k.id === id)
  const next = docApi.doc.keys[idx + 1]
  const seg = next ? solver.result?.segments.find((s) => s.from === id && s.to === next.id) : undefined
  const explicit = next ? docApi.doc.segments.find((s) => s.from === id && s.to === next.id) : undefined
  const obs = (solver.result?.observations ?? []).filter((o) => o.kind === "key" && o.key_id === id)
  const style = KEY_SOURCE_STYLE[key.source]

  return (
    <div className="flex flex-col gap-3">
      <Row label="Key">
        <span className="flex items-center gap-2">
          <span className="font-mono">{key.id}</span>
          <span className="text-muted-foreground">frame</span>
          <span className="font-mono tabular-nums">{key.frame}</span>
          <ToneBadge tone="muted">{style.label}</ToneBadge>
          {solvedKey && solvedKey.status !== "ok" ? (
            <ToneBadge tone={solvedKey.status === "error" ? "destructive" : "warning"}>{solvedKey.status}</ToneBadge>
          ) : null}
        </span>
      </Row>
      <Row label="Position">
        <span className="font-mono text-xs tabular-nums">
          x {fmt3(xyz[0])} · y {fmt3(xyz[1])} · z {fmt3(xyz[2])} m
        </span>
      </Row>
      {key.source === "ray_height" ? (
        <Row label="Height (m)">
          <Input
            type="number"
            step="0.05"
            className="h-7 w-24 font-mono text-xs"
            defaultValue={key.constraint.height_m ?? ""}
            key={`h-${key.id}-${key.constraint.height_m}`}
            onBlur={(e) => {
              const v = Number.parseFloat(e.target.value)
              if (Number.isFinite(v) && v !== key.constraint.height_m) {
                void studio.editDoc((d) => updateKey(d, id, { constraint: { ...key.constraint, height_m: v } }))
              }
            }}
          />
        </Row>
      ) : null}
      {key.source === "ray_depth" ? (
        <Row label="Depth (m)">
          <Input
            type="number"
            step="0.5"
            className="h-7 w-24 font-mono text-xs"
            defaultValue={key.constraint.depth_m ?? ""}
            key={`d-${key.id}-${key.constraint.depth_m}`}
            onBlur={(e) => {
              const v = Number.parseFloat(e.target.value)
              if (Number.isFinite(v) && v !== key.constraint.depth_m) {
                void studio.editDoc((d) => updateKey(d, id, { constraint: { ...key.constraint, depth_m: v } }))
              }
            }}
          />
        </Row>
      ) : null}
      {key.source === "player" ? (
        <Row label="Joint">
          <span className="font-mono text-xs">
            {key.constraint.player_id} · {key.constraint.bone}
          </span>
        </Row>
      ) : null}
      {next ? (
        <Row label="To next key">
          <Select
            value={explicit?.kind ?? "auto"}
            onValueChange={(v) => {
              if (v === "auto") void studio.editDoc((d) => removeSegment(d, id, next.id))
              else void studio.editDoc((d) => setSegmentKind(d, id, next.id, v as SegmentKind))
            }}
          >
            <SelectTrigger size="sm" className="h-7 w-44 text-xs" aria-label="Segment kind to the next key">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto{seg ? ` (${seg.kind})` : ""}</SelectItem>
              {SEGMENT_KINDS.map((k) => (
                <SelectItem key={k} value={k}>
                  {SEGMENT_STYLE[k].label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {seg ? (
            <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs whitespace-nowrap text-muted-foreground">
              <span className="font-mono tabular-nums">
                {seg.frame_range[0]}–{seg.frame_range[1]}
              </span>
              {seg.max_speed_m_s !== null ? <span>up to {seg.max_speed_m_s.toFixed(1)} m/s</span> : null}
              {seg.rms_obs_px !== null ? (
                <ToneBadge tone={residualSeverity(seg.rms_obs_px)} className="font-mono">
                  {seg.rms_obs_px.toFixed(1)} px rms · {seg.n_soft_obs} soft pick{seg.n_soft_obs === 1 ? "" : "s"}
                </ToneBadge>
              ) : null}
            </p>
          ) : null}
        </Row>
      ) : null}

      {obs.length ? (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="px-1">View · frame</TableHead>
              <TableHead className="px-1">Picked</TableHead>
              <TableHead className="px-1">Reprojected</TableHead>
              <TableHead className="px-1 text-right">Residual</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {obs.map((o) => (
              <TableRow key={o.shot_id}>
                <TableCell className="px-1 font-mono text-[11px]">
                  {o.shot_id} · {o.shot_frame}
                </TableCell>
                <TableCell className="px-1 font-mono text-[11px]">
                  {o.uv[0].toFixed(0)}, {o.uv[1].toFixed(0)}
                </TableCell>
                <TableCell className="px-1 font-mono text-[11px]">
                  {o.projected_uv ? `${o.projected_uv[0].toFixed(1)}, ${o.projected_uv[1].toFixed(1)}` : "n/a"}
                </TableCell>
                <TableCell className="px-1 text-right">
                  {o.residual_px === null ? (
                    "n/a"
                  ) : (
                    <SyncProbe
                      groupId={studio.group.group_id}
                      tkey={key}
                      storedOffset={storedOffsets}
                      referenceShot={studio.scene.reference_shot}
                    >
                      <ToneBadge tone={residualSeverity(o.residual_px)} className="font-mono">
                        {o.residual_px.toFixed(1)} px
                      </ToneBadge>
                    </SyncProbe>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : key.observations.length === 0 ? (
        <p className="text-xs text-muted-foreground">No picked pixels: this key is a 3-D position.</p>
      ) : null}
      {solvedKey?.messages.length ? (
        <ul className="list-disc pl-4 text-xs text-warning">
          {solvedKey.messages.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ul>
      ) : null}

      <Row label="Note">
        <Textarea
          rows={2}
          className="text-xs"
          defaultValue={key.note}
          key={`n-${key.id}`}
          onBlur={(e) => e.target.value !== key.note && void studio.editDoc((d) => updateKey(d, id, { note: e.target.value }))}
        />
      </Row>
      <Button variant="outline" size="sm" className="self-start" onClick={studio.deleteSelected}>
        <Trash2Icon /> Delete key
      </Button>
    </div>
  )
}

function EventInspector({ studio, index }: { studio: Studio; index: number }) {
  const ev = studio.docApi.doc.events[index]
  if (!ev) return <p className="text-sm text-muted-foreground">That event no longer exists.</p>
  const bones = studio.scene.bones
  return (
    <div className="flex flex-col gap-3">
      <Row label="Event">
        <Select value={ev.kind} onValueChange={(v) => void studio.editDoc((d) => updateEvent(d, index, { kind: v as EventKind }))}>
          <SelectTrigger size="sm" className="h-7 w-44 text-xs" aria-label="Event kind">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {EVENT_KINDS.map((k) => (
              <SelectItem key={k} value={k}>
                {EVENT_STYLE[k].label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Row>
      <Row label="Frame">
        <span className="font-mono text-sm tabular-nums">{ev.frame}</span>
      </Row>
      {ev.kind === "touch" || ev.kind === "keeper_save" ? (
        <>
          <Row label="Player">
            <Select
              value={ev.player_id ?? "none"}
              onValueChange={(v) => void studio.editDoc((d) => updateEvent(d, index, { player_id: v === "none" ? null : v }))}
            >
              <SelectTrigger size="sm" className="h-7 w-44 text-xs" aria-label="Player">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">Not set</SelectItem>
                {studio.scene.players.map((p) => (
                  <SelectItem key={p.player_id} value={p.player_id}>
                    {p.player_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Row>
          <Row label="Bone">
            <Select
              value={ev.bone ?? "none"}
              onValueChange={(v) => void studio.editDoc((d) => updateEvent(d, index, { bone: v === "none" ? null : v }))}
            >
              <SelectTrigger size="sm" className="h-7 w-44 text-xs" aria-label="Bone">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">Not set</SelectItem>
                {bones.map((b) => (
                  <SelectItem key={b} value={b}>
                    {b}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Row>
        </>
      ) : null}
      <Row label="Note">
        <Textarea
          rows={2}
          className="text-xs"
          defaultValue={ev.note}
          key={`en-${index}-${ev.frame}`}
          onBlur={(e) => e.target.value !== ev.note && void studio.editDoc((d) => updateEvent(d, index, { note: e.target.value }))}
        />
      </Row>
      <Button variant="outline" size="sm" className="self-start" onClick={studio.deleteSelected}>
        <Trash2Icon /> Delete event
      </Button>
    </div>
  )
}

function ObservationInspector({ studio, index }: { studio: Studio; index: number }) {
  const o = studio.docApi.doc.observations[index]
  if (!o) return <p className="text-sm text-muted-foreground">That observation no longer exists.</p>
  const sObs = studio.solver.result?.observations.find((s) => s.kind === "soft" && s.index === index)
  return (
    <div className="flex flex-col gap-3">
      <Row label="Soft pick">
        <span className="font-mono text-xs">
          {o.shot_id} · frame {o.shot_frame} · {o.uv[0].toFixed(0)}, {o.uv[1].toFixed(0)}
        </span>
      </Row>
      <Row label="Residual">
        {sObs?.residual_px != null ? (
          <ToneBadge tone={residualSeverity(sObs.residual_px)} className="font-mono">
            {sObs.residual_px.toFixed(1)} px
          </ToneBadge>
        ) : (
          <span className="text-xs text-muted-foreground">Not inside a solved span.</span>
        )}
      </Row>
      <p className="text-xs text-muted-foreground">Soft observations are fit targets for the segment between keys; they never become keys.</p>
      <Button variant="outline" size="sm" className="self-start" onClick={studio.deleteSelected}>
        <Trash2Icon /> Delete observation
      </Button>
    </div>
  )
}

function Lists({ studio, readOnly }: { studio: Studio; readOnly: boolean }) {
  const { docApi } = studio
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-muted-foreground">{readOnly ? "Tap a key or event to seek to it." : "Select a key, event or soft pick on the timeline or a view to edit it."}</p>
      <ul className="flex flex-col gap-1 text-sm" aria-label="Keys and events">
        {docApi.doc.keys.map((k) => (
          <li key={k.id}>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-full justify-start gap-2 font-normal"
              onClick={() => {
                studio.setFrame(k.frame)
                studio.setSelection({ type: "key", id: k.id })
              }}
            >
              <span className="font-mono">{k.id}</span>
              <span className="font-mono text-muted-foreground tabular-nums">{k.frame}</span>
              <span className="text-muted-foreground">{KEY_SOURCE_STYLE[k.source].label}</span>
            </Button>
          </li>
        ))}
        {docApi.doc.events.map((e, i) => (
          <li key={`e${i}`}>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-full justify-start gap-2 font-normal"
              onClick={() => {
                studio.setFrame(e.frame)
                studio.setSelection({ type: "event", index: i })
              }}
            >
              <span className="size-2 rounded-full" style={{ backgroundColor: EVENT_STYLE[e.kind].colour }} aria-hidden />
              <span>{EVENT_STYLE[e.kind].label}</span>
              <span className="font-mono text-muted-foreground tabular-nums">{e.frame}</span>
              {e.player_id ? <span className="font-mono text-muted-foreground">{e.player_id}</span> : null}
            </Button>
          </li>
        ))}
      </ul>
    </div>
  )
}

function FlagsTab({ studio }: { studio: Studio }) {
  const { solver } = studio
  const flags = solver.result?.flags ?? []
  if (solver.error) {
    return <p className="text-sm text-destructive">Solve failed: {solver.error}</p>
  }
  if (!flags.length) return <p className="text-sm text-muted-foreground">{solver.result ? "No physics flags. The track is self-consistent." : "Flags appear once there are keys to solve."}</p>
  return (
    <ul className="flex flex-col gap-1.5">
      {flags.map((f, i) => {
        const frame =
          f.frame ??
          (f.ref?.segment !== undefined ? solver.result?.segments[f.ref.segment]?.frame_range[0] : undefined) ??
          (f.ref?.key ? studio.docApi.doc.keys.find((k) => k.id === f.ref?.key)?.frame : undefined)
        return (
          <li key={i}>
            <Button
              variant="ghost"
              className="h-auto w-full justify-start gap-2 py-1.5 text-left font-normal whitespace-normal"
              disabled={frame === undefined}
              onClick={() => frame !== undefined && studio.setFrame(frame)}
            >
              <ToneBadge tone={f.level === "error" ? "destructive" : "warning"}>{f.level}</ToneBadge>
              <span className="min-w-0 flex-1 text-xs">
                <span className="font-mono">{f.code}</span> {f.message}
              </span>
              {frame !== undefined ? <span className="font-mono text-xs text-muted-foreground tabular-nums">{frame}</span> : null}
            </Button>
          </li>
        )
      })}
    </ul>
  )
}

function GroupTab({ studio, onHelp }: { studio: Studio; onHelp: () => void }) {
  const { doc } = studio.docApi
  const stats = studio.solver.result?.stats
  return (
    <div className="flex flex-col gap-3">
      <Row label="Outcome">
        <Select value={doc.outcome} onValueChange={(v) => void studio.editDoc((d) => setOutcome(d, v as Outcome))}>
          <SelectTrigger size="sm" className="h-7 w-44 text-xs" aria-label="Outcome">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="goal">Goal</SelectItem>
            <SelectItem value="no_goal">No goal</SelectItem>
            <SelectItem value="unknown">Unknown</SelectItem>
          </SelectContent>
        </Select>
      </Row>
      <Row label="Status">
        <Select value={doc.meta.status} onValueChange={(v) => void studio.editDoc((d) => setStatus(d, v as TruthStatus))}>
          <SelectTrigger size="sm" className="h-7 w-44 text-xs" aria-label="Truth status">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="draft">Draft</SelectItem>
            <SelectItem value="reviewed">Reviewed</SelectItem>
          </SelectContent>
        </Select>
      </Row>
      <Row label="Notes">
        <Textarea
          rows={3}
          className="text-xs"
          defaultValue={doc.meta.notes}
          key={`gn-${doc.group_id}`}
          onBlur={(e) => e.target.value !== doc.meta.notes && void studio.editDoc((d) => setNotes(d, e.target.value))}
        />
      </Row>
      {stats ? (
        <p className="text-xs text-muted-foreground">
          {stats.n_keys} keys · {stats.n_segments} segments · {stats.n_dense} dense frames
          {stats.max_speed_m_s != null ? ` · peak ${stats.max_speed_m_s.toFixed(1)} m/s` : ""}
        </p>
      ) : null}
      <Button variant="outline" size="sm" className="self-start" onClick={onHelp}>
        Keyboard shortcuts
      </Button>
    </div>
  )
}

export function Inspector({ studio, onHelp, readOnly = false }: { studio: Studio; onHelp: () => void; readOnly?: boolean }) {
  const sel = studio.selection
  const nFlags = studio.solver.result?.flags.length ?? 0
  const [tab, setTab] = React.useState("selection")
  React.useEffect(() => {
    if (sel) setTab("selection")
  }, [sel])
  return (
    <Panel title="Inspector" contentClassName="p-3">
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="w-full">
          <TabsTrigger value="selection">Selection</TabsTrigger>
          <TabsTrigger value="flags">Flags{nFlags ? ` (${nFlags})` : ""}</TabsTrigger>
          <TabsTrigger value="group">Group</TabsTrigger>
        </TabsList>
        <TabsContent value="selection" className="pt-1">
          {!readOnly && sel?.type === "key" ? <KeyInspector studio={studio} id={sel.id} /> : null}
          {!readOnly && sel?.type === "event" ? <EventInspector studio={studio} index={sel.index} /> : null}
          {!readOnly && sel?.type === "observation" ? <ObservationInspector studio={studio} index={sel.index} /> : null}
          {readOnly || !sel || sel.type === "segment" ? <Lists studio={studio} readOnly={readOnly} /> : null}
        </TabsContent>
        <TabsContent value="flags" className="pt-1">
          <FlagsTab studio={studio} />
        </TabsContent>
        <TabsContent value="group" className="pt-1">
          {readOnly ? (
            <p className="text-sm text-muted-foreground">
              Outcome <span className="font-medium text-foreground">{studio.docApi.doc.outcome.replace("_", " ")}</span>, status{" "}
              <span className="font-medium text-foreground">{studio.docApi.doc.meta.status}</span>.
            </p>
          ) : (
            <GroupTab studio={studio} onHelp={onHelp} />
          )}
        </TabsContent>
      </Tabs>
    </Panel>
  )
}

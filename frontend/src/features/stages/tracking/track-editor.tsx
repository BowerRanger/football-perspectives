import * as React from "react"
import { toast } from "sonner"

import { PanelEmpty, PanelError } from "@/components/panel"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { useConfirm } from "@/hooks/use-dialogs"
import { errorMessage } from "@/lib/api"
import * as api from "./api"
import { buildPlayerGroups, plural } from "./groups"
import { PlayerRow } from "./player-row"
import { TrackToolbar } from "./toolbar"
import { TrackVideo, type TrackVideoHandle } from "./track-video"
import { IGNORE_NAME, type FrameBox, type PlayerGroup } from "./types"
import { useTrackEditorData } from "./use-track-editor-data"

const FLASH_MS = 800
const EMPTY_BOXES: ReadonlyMap<number, FrameBox[]> = new Map()

function collisionNote(n: number | undefined): string {
  return n ? ` (${plural(n, "frame collision")} resolved by confidence)` : ""
}

function uniqueNames(names: Iterable<string | null | undefined>): string[] {
  return [...new Set([...names].filter((n): n is string => !!n && n !== IGNORE_NAME))]
}

/** Track annotation editor for one shot: video + bbox overlay + player list. */
export function TrackEditor({ shotId }: { shotId: string }) {
  const confirm = useConfirm()
  const { state, reload, applyName } = useTrackEditorData(shotId)
  const videoRef = React.useRef<TrackVideoHandle>(null)
  const rowEls = React.useRef(new Map<string, HTMLElement>())
  const [selected, setSelected] = React.useState<ReadonlySet<string>>(new Set())
  const [focusedKey, setFocusedKey] = React.useState<string | null>(null)
  const [flashKey, setFlashKey] = React.useState<string | null>(null)
  const [busy, setBusy] = React.useState<string | null>(null)
  const [status, setStatus] = React.useState("")

  const data = state.status === "ready" ? state.data : null
  const editable = React.useMemo(
    () => (data?.tracks ?? []).filter((t) => t.class_name === "player" || t.class_name === "goalkeeper"),
    [data],
  )
  const groups = React.useMemo(() => buildPlayerGroups(editable), [editable])
  const groupByKey = React.useMemo(() => new Map(groups.map((g) => [g.key, g])), [groups])
  const nameByTrack = React.useMemo(
    () => new Map(editable.map((t) => [t.track_id, t.player_name ?? ""])),
    [editable],
  )
  const highlightIds = React.useMemo(() => {
    const keys = new Set(selected)
    if (focusedKey) keys.add(focusedKey)
    return new Set([...keys].flatMap((k) => groupByKey.get(k)?.tracks.map((t) => t.track_id) ?? []))
  }, [selected, focusedKey, groupByKey])

  const allNames = React.useMemo(() => uniqueNames(editable.map((t) => t.player_name)), [editable])
  const roster = data?.roster ?? []
  const rosterFor = (team: string) => roster.filter((r) => r.team === team).map((r) => r.name)
  const listIds = { all: `player-names-${shotId}`, A: `player-names-${shotId}-A`, B: `player-names-${shotId}-B` }
  const datalistFor = (team: string) => (roster.length && (team === "A" || team === "B") ? listIds[team] : listIds.all)

  const fail = React.useCallback((title: string, err: unknown) => {
    const message = errorMessage(err)
    setStatus(`${title}: ${message}`)
    toast.error(title, { description: message })
  }, [])

  /** Run a mutation, then reload from the server and clear the selection. */
  async function mutate(label: string, work: () => Promise<string>, failTitle: string) {
    setBusy(label)
    try {
      const message = await work()
      setStatus(message)
      toast.success(message)
      setSelected(new Set())
      await reload()
    } catch (err) {
      fail(failTitle, err)
    } finally {
      setBusy(null)
    }
  }

  const selectedTrackIds = () =>
    [...selected].flatMap((k) => groupByKey.get(k)?.tracks.map((t) => t.track_id) ?? [])

  async function rename(g: PlayerGroup, name: string) {
    try {
      await Promise.all(g.tracks.map((t) => api.renameTrack(shotId, t.track_id, name)))
      applyName(new Set(g.tracks.map((t) => t.track_id)), name)
      setStatus(`Saved ${g.key}`)
    } catch (err) {
      fail("Rename failed", err)
    }
  }

  function toggle(key: string, checked: boolean) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (checked) next.add(key)
      else next.delete(key)
      return next
    })
  }

  function pickTrack(trackId: string) {
    const g = groups.find((x) => x.tracks.some((t) => t.track_id === trackId))
    const el = g ? rowEls.current.get(g.key) : undefined
    if (!g || !el) return
    el.scrollIntoView({ block: "nearest" })
    el.querySelector<HTMLInputElement>("input[type=text], input:not([type])")?.focus()
    setFlashKey(g.key)
    window.setTimeout(() => setFlashKey((k) => (k === g.key ? null : k)), FLASH_MS)
  }

  async function split(g: PlayerGroup) {
    const f = videoRef.current?.currentFrame() ?? 0
    const target = g.tracks.find((t) => f > t.frame_range[0] && f <= t.frame_range[1])
    if (!target) {
      const msg = "Current frame is outside every member track's range — scrub inside this player's frames first."
      setStatus(`Split failed: ${msg}`)
      toast.error("Split failed", { description: msg })
      return
    }
    await mutate("split", async () => {
      await api.splitTrack(shotId, target.track_id, f)
      return `Split ${target.track_id} at frame ${f}`
    }, "Split failed")
  }

  async function deleteRow(g: PlayerGroup) {
    const label = g.name && g.name !== IGNORE_NAME ? g.name : g.key
    const ok = await confirm({
      title: `Delete ${label}?`,
      description: `Removes ${plural(g.tracks.length, "track")} (${g.tracks.map((t) => t.track_id).join(", ")}) from ${shotId}. This cannot be undone.`,
      confirmLabel: "Delete",
      destructive: true,
    })
    if (!ok) return
    await mutate("delete", async () => {
      await Promise.all(g.tracks.map((t) => api.deleteTrack(shotId, t.track_id)))
      return `Deleted ${label}`
    }, "Delete failed")
  }

  async function deleteSelected() {
    const ids = selectedTrackIds()
    if (ids.length === 0) return
    const ok = await confirm({
      title: `Delete ${plural(selected.size, "player")}?`,
      description: `Removes ${plural(ids.length, "track")} from ${shotId}. This cannot be undone.`,
      confirmLabel: `Delete ${plural(ids.length, "track")}`,
      destructive: true,
    })
    if (!ok) return
    await mutate("delete", async () => {
      await api.deleteTracksBulk(shotId, ids)
      return `Deleted ${plural(ids.length, "track")}`
    }, "Delete failed")
  }

  async function merge() {
    const ids = selectedTrackIds()
    if (selected.size < 2) return
    await mutate("merge", async () => {
      const out = await api.mergeTracks(shotId, ids)
      return `Merged into ${out.merged_into}${collisionNote(out.frame_collisions)}`
    }, "Merge failed")
  }

  async function mergeByName() {
    const ok = await confirm({
      title: "Merge tracks by player name?",
      description: `Across EVERY shot, all tracks sharing a player name are rewritten to one player_id. Named tracks in this shot: ${allNames.length}. This cannot be undone.`,
      confirmLabel: "Merge by name",
    })
    if (!ok) return
    await mutate("merge-by-name", async () => {
      const out = await api.mergeByName()
      return `Merged ${plural(out.tracks_removed, "track")} across ${plural(out.merged_groups, "name")}${collisionNote(out.frame_collisions)}`
    }, "Merge by name failed")
  }

  async function ignoreUnknown() {
    const unnamed = groups.filter((g) => !g.name).length
    const ok = await confirm({
      title: "Mark unnamed players as ignore?",
      description: `Every unnamed player/goalkeeper track in ${shotId} (${plural(unnamed, "row")} now) is renamed to 'ignore'. You can rename them again afterwards.`,
      confirmLabel: "Ignore unknown",
    })
    if (!ok) return
    await mutate("ignore", async () => {
      const out = await api.ignoreUnknown(shotId)
      return `Marked ${out.count} as 'ignore'`
    }, "Ignore failed")
  }

  async function deleteIgnored() {
    const ignored = editable.filter((t) => t.player_name === IGNORE_NAME).length
    const ok = await confirm({
      title: "Delete every ignored track?",
      description: `Removes all tracks named 'ignore' across every shot (${plural(ignored, "track")} in ${shotId}). This cannot be undone.`,
      confirmLabel: "Delete ignored",
      destructive: true,
    })
    if (!ok) return
    await mutate("delete-ignored", async () => {
      const out = await api.deleteIgnored()
      return `Deleted ${plural(out.deleted, "ignored track")}`
    }, "Delete ignored failed")
  }

  async function interpolate() {
    const ids = selectedTrackIds()
    if (ids.length === 0) return
    setBusy("interpolate")
    try {
      const out = await api.interpolateGaps(shotId, ids)
      if (out.total_frames_added === 0) {
        setStatus("No gaps to fill within max_gap.")
        toast.info("No gaps to fill within max_gap.")
        return
      }
      const msg = `Filled ${plural(out.total_frames_added, "frame")} across ${plural(out.results.length, "track")}`
      setStatus(msg)
      toast.success(msg)
      setSelected(new Set())
      await reload()
    } catch (err) {
      fail("Interpolate failed", err)
    } finally {
      setBusy(null)
    }
  }

  if (state.status === "loading") return <EditorSkeleton />
  if (state.status === "error") {
    return <PanelError title="Failed to load tracks for this shot" message={state.message} />
  }
  if (groups.length === 0) {
    return (
      <PanelEmpty
        title="No player tracks in this shot"
        description="The shot has no player or goalkeeper tracks. Re-run tracking, or pick another shot."
      />
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <TrackToolbar
        selectedCount={selected.size}
        busy={busy}
        onMerge={merge}
        onMergeByName={mergeByName}
        onIgnoreUnknown={ignoreUnknown}
        onDeleteSelected={deleteSelected}
        onInterpolate={interpolate}
        onDeleteIgnored={deleteIgnored}
      />
      <p role="status" aria-live="polite" className="min-h-4 text-xs text-muted-foreground">
        {status}
      </p>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <TrackVideo
          ref={videoRef}
          shotId={shotId}
          fps={data?.fps ?? 30}
          boxesByFrame={data?.boxesByFrame ?? EMPTY_BOXES}
          nameByTrack={nameByTrack}
          highlightIds={highlightIds}
          onPickTrack={pickTrack}
        />
        <div className="flex min-h-0 flex-col border-t pt-3 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-4">
          <div className="pb-2 text-sm font-medium">
            Players <span className="font-normal text-muted-foreground">· {shotId} · {groups.length}</span>
          </div>
          <ScrollArea className="h-[min(60vh,34rem)]">
            <ul>
              {groups.map((g) => (
                <PlayerRow
                    key={g.key}
                    rowRef={(el) => {
                      if (el) rowEls.current.set(g.key, el)
                      else rowEls.current.delete(g.key)
                    }}
                    group={g}
                    selected={selected.has(g.key)}
                    flash={flashKey === g.key}
                    busy={busy !== null}
                    datalistId={datalistFor(g.team)}
                    onToggle={toggle}
                    onFocusChange={setFocusedKey}
                    onRename={rename}
                    onJump={(grp) => videoRef.current?.seekToFrame(Math.round((grp.frameRange[0] + grp.frameRange[1]) / 2))}
                    onSplit={split}
                    onDelete={deleteRow}
                  />
              ))}
            </ul>
          </ScrollArea>
        </div>
      </div>
      <datalist id={listIds.all}>
        {allNames.map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
      {(["A", "B"] as const).map((team) => (
        <datalist key={team} id={listIds[team]}>
          {uniqueNames([...rosterFor(team), ...allNames]).map((n) => (
            <option key={n} value={n} />
          ))}
        </datalist>
      ))}
    </div>
  )
}

function EditorSkeleton() {
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <Skeleton className="aspect-video w-full" />
      <div className="flex flex-col gap-2">
        {Array.from({ length: 6 }, (_, i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    </div>
  )
}

import * as React from "react"
import { toast } from "sonner"

import { useConfirm } from "@/hooks/use-dialogs"
import { errorMessage, postJson } from "@/lib/api"

import { fmtRate } from "./replay-speed"

interface RetimeResponse {
  retime: { frames_in: number; frames_out: number; tracks_remapped?: boolean; camera_remapped?: boolean } | null
  alignment: unknown
  speed_factor: number
}

interface RetimeTarget {
  shotId: string
  /** Playback rate relative to the clip as it is now (= the native clip: not retimed yet). */
  rate: number
  /** Frames in the clip today. */
  frames: number
}

/**
 * Retime / restore with their confirms, toasts and undo. `onChanged` must
 * reload the manifest AND the sync map (the editor reseeds from it).
 */
export function useRetimeActions(onChanged: () => void | Promise<void>, gate: () => string = () => "") {
  const confirm = useConfirm()
  const [busyId, setBusyId] = React.useState<string | null>(null)

  const restore = React.useCallback(
    async (shotId: string, opts: { confirmFirst: boolean; wasRate?: number } = { confirmFirst: true }): Promise<boolean> => {
      // Re-checked at click time: toast actions outlive the state they were created in.
      const blocked = gate()
      if (blocked) {
        toast.error(`Can't restore ${shotId} now`, { description: blocked })
        return false
      }
      if (opts.confirmFirst) {
        const ok = await confirm({
          title: `Restore the native clip for ${shotId}?`,
          description: (
            <div className="flex flex-col gap-3">
              <p>
                This puts back the original slow-motion clip, its tracks and its camera, and returns the alignment to
                rate {opts.wasRate ? fmtRate(opts.wasRate) : "the native rate"}. Anything you edited on the retimed tracks since the retime is lost.
              </p>
              <ul className="rounded-md bg-muted px-3 py-2 font-mono text-xs leading-relaxed text-foreground">
                <li>{`shots/${shotId}.mp4  replaced by shots/native/${shotId}.mp4`}</li>
                <li>{`tracks/${shotId}_tracks.json  native restored`}</li>
                <li>{`camera/${shotId}_camera_track.json  native restored`}</li>
                <li>{`shots/sync_map.json  rate 1.0 → ${opts.wasRate ? fmtRate(opts.wasRate) : "native"}, offset unchanged`}</li>
              </ul>
            </div>
          ),
          confirmLabel: "Restore clip",
        })
        if (!ok) return false
      }
      setBusyId(shotId)
      try {
        await postJson(`/api/shots/${encodeURIComponent(shotId)}/restore-native`)
        toast.success(`Restored the native clip for ${shotId}`)
        await onChanged()
        return true
      } catch (err) {
        toast.error(`Could not restore ${shotId}`, { description: errorMessage(err) })
        return false
      } finally {
        setBusyId(null)
      }
    },
    [confirm, onChanged, gate],
  )

  const retime = React.useCallback(
    async ({ shotId, rate, frames }: RetimeTarget): Promise<boolean> => {
      const blocked = gate()
      if (blocked) {
        toast.error(`Can't retime ${shotId} now`, { description: blocked })
        return false
      }
      const out = Math.max(1, Math.round(frames * rate))
      const ok = await confirm({
        title: `Retime ${shotId} to real time?`,
        description: (
          <div className="flex flex-col gap-3">
            <p>
              {shotId} plays at {fmtRate(rate)}. This re-encodes the clip so it plays at the live shot&apos;s speed
              (repeated slow-motion frames are dropped), then remaps its tracks and camera frame for frame without
              re-solving. The original clip, tracks and camera are kept and you can restore them.
            </p>
            <ul className="rounded-md bg-muted px-3 py-2 font-mono text-xs leading-relaxed text-foreground">
              <li>{`shots/${shotId}.mp4  replaced (${frames} → ${out} frames)`}</li>
              <li>{`shots/native/${shotId}.mp4  kept`}</li>
              <li>{`tracks/${shotId}_tracks.json  remapped`}</li>
              <li>{`camera/${shotId}_camera_track.json  remapped`}</li>
              <li>shots/sync_map.json  rate 1.0</li>
            </ul>
            <p>
              hmr_world and later outputs for {shotId} were made from the slow clip. Re-run from hmr_world after
              retiming.
            </p>
          </div>
        ),
        confirmLabel: "Retime clip",
      })
      if (!ok) return false
      setBusyId(shotId)
      const loading = toast.loading(`Retiming ${shotId}…`)
      try {
        const res = await postJson<RetimeResponse>(`/api/shots/${encodeURIComponent(shotId)}/retime`, { rate })
        toast.success(`Retimed ${shotId} to real time`, {
          id: loading,
          description: res.retime ? `${res.retime.frames_in} → ${res.retime.frames_out} frames; native clip kept.` : undefined,
          action: { label: "Undo", onClick: () => void restore(shotId, { confirmFirst: false, wasRate: rate }) },
        })
        await onChanged()
        return true
      } catch (err) {
        toast.error(`Could not retime ${shotId}`, { id: loading, description: errorMessage(err) })
        return false
      } finally {
        setBusyId(null)
      }
    },
    [confirm, onChanged, restore, gate],
  )

  return { retime, restore, busyId }
}

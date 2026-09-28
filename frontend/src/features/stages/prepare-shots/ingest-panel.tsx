import * as React from "react"
import { FilmIcon, PlusIcon, RefreshCwIcon, UploadIcon } from "lucide-react"

import { Panel } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { errorMessage, postForm, postJson } from "@/lib/api"
import { useConfirm } from "@/hooks/use-dialogs"
import { usePipeline } from "@/hooks/use-pipeline"
import { cn } from "@/lib/utils"

interface UploadReelResponse {
  saved: string
  job_id: string
}
interface UploadClipsResponse {
  saved?: string[]
  skipped?: { name: string; reason: string }[]
  job_id?: string
}
interface ResplitResponse {
  job_id: string
}

type Tone = "muted" | "success" | "warning" | "destructive"
const STATUS_TONE: Record<Tone, string> = {
  muted: "text-muted-foreground",
  success: "text-success",
  warning: "text-warning",
  destructive: "text-destructive",
}

/** Reel drop zone + Add shots + Re-run split. All three start a prepare_shots job. */
export function IngestPanel({ onChanged }: { onChanged: () => Promise<void> }) {
  const { attachToJob, isRunning } = usePipeline()
  const confirm = useConfirm()
  const reelInput = React.useRef<HTMLInputElement>(null)
  const clipsInput = React.useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = React.useState(false)
  const [busy, setBusy] = React.useState(false)
  const [status, setStatus] = React.useState<{ tone: Tone; text: string } | null>(null)

  const say = (tone: Tone, text: string) => setStatus({ tone, text })
  const attach = (jobId: string) => {
    attachToJob(jobId, "prepare_shots", () => {
      setBusy(false)
      void onChanged()
    })
  }

  const uploadReel = async (file: File) => {
    const fd = new FormData()
    fd.append("file", file)
    setBusy(true)
    say("muted", `Uploading ${file.name} (${(file.size / 1e6).toFixed(0)} MB)…`)
    try {
      const body = await postForm<UploadReelResponse>("/api/shots/upload-reel", fd)
      say("success", `Saved ${body.saved}. Splitting into shots — follow the run log.`)
      attach(body.job_id)
    } catch (err) {
      say("destructive", `Upload failed: ${errorMessage(err)}`)
      setBusy(false)
    }
  }

  const uploadClips = async (files: FileList) => {
    const fd = new FormData()
    for (const f of Array.from(files)) fd.append("files", f)
    setBusy(true)
    say("muted", `Uploading ${files.length} clip(s)…`)
    try {
      const body = await postForm<UploadClipsResponse>("/api/shots/upload", fd)
      const skipped = body.skipped ?? []
      const note = skipped.length
        ? ` (skipped ${skipped.length}: ${skipped.map((s) => `${s.name} — ${s.reason}`).join("; ")})`
        : ""
      if (!body.saved || body.saved.length === 0) {
        say("warning", `Nothing uploaded${note}`)
        setBusy(false)
        return
      }
      say("success", `Uploaded ${body.saved.length} shot(s)${note}. Registering…`)
      if (body.job_id) attach(body.job_id)
      else {
        setBusy(false)
        await onChanged()
      }
    } catch (err) {
      say("destructive", `Upload failed: ${errorMessage(err)}`)
      setBusy(false)
    }
  }

  const resplit = async () => {
    const ok = await confirm({
      title: "Re-run the split from scratch?",
      description: (
        <span>
          This re-ingests the original reel and <strong>replaces all shots, groups, discards and sync offsets</strong>
          , including manual offsets. Match details survive.
        </span>
      ),
      confirmLabel: "Re-split reel",
      destructive: true,
    })
    if (!ok) return
    setBusy(true)
    say("muted", "Re-splitting the source reel…")
    try {
      const body = await postJson<ResplitResponse>("/api/shots/resplit")
      attach(body.job_id)
    } catch (err) {
      say("destructive", `Re-split failed: ${errorMessage(err)}`)
      setBusy(false)
    }
  }

  const disabled = busy || isRunning

  return (
    <Panel title="Ingest" description="Bring footage in: a full highlights reel, or pre-trimmed clips.">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-stretch gap-3">
          <div
            role="group"
            aria-label="Highlights reel drop zone"
            className={cn(
              "flex min-h-24 flex-1 basis-72 flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed p-4 text-center transition-colors",
              dragOver ? "border-info bg-info/10" : "border-border",
              disabled && "opacity-60",
            )}
            onDragOver={(e) => {
              e.preventDefault()
              if (!disabled) setDragOver(true)
            }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragOver(false)
            }}
            onDrop={(e) => {
              e.preventDefault()
              setDragOver(false)
              const f = e.dataTransfer.files?.[0]
              if (f && !disabled) void uploadReel(f)
            }}
          >
            <FilmIcon className="size-5 text-muted-foreground" aria-hidden />
            <p className="text-sm font-medium">Drop a full highlights reel here</p>
            <p className="text-xs text-muted-foreground">
              Auto-splits into shots, drops reactions, groups highlights and aligns replays.
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-1"
              disabled={disabled}
              onClick={() => reelInput.current?.click()}
            >
              <UploadIcon data-icon="inline-start" />
              Choose .mp4
            </Button>
            <input
              ref={reelInput}
              type="file"
              accept="video/mp4,.mp4"
              className="sr-only"
              aria-label="Highlights reel file"
              tabIndex={-1}
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void uploadReel(f)
                e.target.value = ""
              }}
            />
          </div>

          <div className="flex w-full flex-col justify-center gap-2 sm:w-auto">
            <Button variant="secondary" className="w-full sm:w-auto" disabled={disabled} onClick={() => clipsInput.current?.click()}>
              <PlusIcon data-icon="inline-start" />
              Add shots
            </Button>
            <p className="text-xs sm:max-w-44 text-muted-foreground">
              Pre-trimmed .mp4 clips, added as ungrouped shots (no splitting).
            </p>
            <input
              ref={clipsInput}
              type="file"
              accept="video/mp4,.mp4"
              multiple
              className="sr-only"
              aria-label="Pre-trimmed clip files"
              tabIndex={-1}
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) void uploadClips(e.target.files)
                e.target.value = ""
              }}
            />
          </div>

          <div className="flex w-full flex-col justify-center gap-2 sm:w-auto">
            <Button variant="outline" className="w-full sm:w-auto" disabled={disabled} onClick={() => void resplit()}>
              <RefreshCwIcon data-icon="inline-start" />
              Re-run split
            </Button>
            <p className="text-xs sm:max-w-44 text-muted-foreground">
              Re-ingest the same source video from scratch with the current settings.
            </p>
          </div>
        </div>

        <p
          role="status"
          aria-live="polite"
          className={cn("flex items-center gap-2 text-sm empty:hidden", status ? STATUS_TONE[status.tone] : "")}
        >
          {busy && status?.tone === "muted" ? <Spinner /> : null}
          {status?.text}
        </p>
      </div>
    </Panel>
  )
}

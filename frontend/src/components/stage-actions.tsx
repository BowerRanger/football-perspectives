import * as React from "react"
import { CircleAlertIcon, ListRestartIcon, PlayIcon, RotateCcwIcon, StepForwardIcon } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { ButtonGroup } from "@/components/ui/button-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { usePipeline } from "@/hooks/use-pipeline"
import { useConfirm, usePrompt } from "@/hooks/use-dialogs"
import { errorMessage, getJson } from "@/lib/api"
import { humanizeStageName, type StageName } from "@/lib/stages"

// Stages whose outputs hold operator edits (shot groups + manual sync
// offsets; track names, merges and splits): re-running them needs the
// stage name typed, not just a click.
const TYPED_CONFIRM_STAGES: ReadonlySet<StageName> = new Set(["prepare_shots", "tracking"])

interface ArtifactsPayload {
  paths: { path: string; is_dir: boolean }[]
}

interface GatedButtonProps extends React.ComponentProps<typeof Button> {
  reason: string | null
  hint?: string
}

/** Disabled buttons swallow pointer events, so the tooltip wraps a span. */
function GatedButton({ reason, hint, children, ...props }: GatedButtonProps) {
  const tip = reason ?? hint
  const button = (
    <Button {...props} disabled={!!reason || props.disabled}>
      {children}
    </Button>
  )
  if (!tip) return button
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={reason ? 0 : -1} className="inline-flex">
          {button}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">{tip}</TooltipContent>
    </Tooltip>
  )
}

function ClearList({ paths }: { paths: ArtifactsPayload["paths"] }) {
  if (!paths.length) return <p>Nothing on disk to clear yet.</p>
  return (
    <div className="grid gap-2">
      <p>These generated outputs are deleted once the run is accepted. Anchors and other operator input are kept.</p>
      <ul className="max-h-40 overflow-auto rounded-md border bg-muted/40 p-2 font-mono text-xs text-foreground">
        {paths.map((p) => (
          <li key={p.path}>
            {p.path}
            {p.is_dir ? "/" : ""}
          </li>
        ))}
      </ul>
    </div>
  )
}

function useRerunFlow(stage: StageName) {
  const { rerunStage } = usePipeline()
  const confirm = useConfirm()
  const prompt = usePrompt()
  const label = humanizeStageName(stage)

  return async () => {
    let paths: ArtifactsPayload["paths"] = []
    try {
      paths = (await getJson<ArtifactsPayload>(`/api/output/${stage}/artifacts`)).paths
    } catch (err) {
      toast.error(`Could not list ${label} output`, { description: errorMessage(err) })
      return
    }
    if (paths.length && TYPED_CONFIRM_STAGES.has(stage)) {
      const typed = await prompt({
        title: `Re-run ${label} from scratch?`,
        description: <ClearList paths={paths} />,
        label: `Type ${stage} to confirm — this output includes your manual edits`,
        placeholder: stage,
        confirmLabel: "Clear and re-run",
        validate: (v) => (v === stage ? null : `Type “${stage}” exactly`),
      })
      if (typed !== stage) return
    } else if (paths.length) {
      const ok = await confirm({
        title: `Re-run ${label} from scratch?`,
        description: <ClearList paths={paths} />,
        confirmLabel: "Clear and re-run",
        destructive: true,
      })
      if (!ok) return
    }
    await rerunStage(stage)
  }
}

export function useBlockedReason(stage: StageName): string | null {
  const { isRunning, runningLabel, missingDeps } = usePipeline()
  const missing = missingDeps(stage)
  if (isRunning) return `${runningLabel} is running — one job at a time.`
  if (missing.length) return `Needs ${missing.map(humanizeStageName).join(", ")} first.`
  return null
}

/** Visible (not tooltip-only) reason the run controls are disabled. */
export function BlockedNote({ stage }: { stage: StageName }) {
  const reason = useBlockedReason(stage)
  const { isRunning } = usePipeline()
  if (!reason || isRunning) return null
  return (
    <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
      <CircleAlertIcon className="size-3.5" /> {reason}
    </span>
  )
}

export function StageActions({ stage, hasOutput }: { stage: StageName; hasOutput: boolean }) {
  const { isRunning, startRun } = usePipeline()
  const confirm = useConfirm()
  const onRerun = useRerunFlow(stage)
  const blockedReason = useBlockedReason(stage)

  const onRunAll = async () => {
    const ok = await confirm({
      title: "Run the whole pipeline?",
      description:
        "Runs every stage in order. Completed stages are skipped from cache; HMR World alone can take most of an hour.",
      confirmLabel: "Run all stages",
    })
    if (ok) await startRun("all")
  }

  return (
    <>
      <GatedButton
        variant="ghost"
        size="sm"
        reason={isRunning ? blockedReason : null}
        hint="Run every stage in order"
        onClick={() => void onRunAll()}
      >
        <ListRestartIcon /> Run all
      </GatedButton>
      <ButtonGroup>
        <GatedButton
          variant="outline"
          size="sm"
          reason={blockedReason}
          hint={hasOutput ? "Clear this stage's generated output, then run it again" : undefined}
          onClick={() => void (hasOutput ? onRerun() : startRun(stage))}
        >
          {hasOutput ? <RotateCcwIcon /> : <PlayIcon />}
          {hasOutput ? "Re-run clean" : "Run"}
        </GatedButton>
        {hasOutput ? (
          <GatedButton
            size="sm"
            reason={blockedReason}
            hint="Run without clearing output — resumes from cached per-player results."
            onClick={() => void startRun(stage, stage)}
          >
            <StepForwardIcon /> Continue
          </GatedButton>
        ) : null}
      </ButtonGroup>
    </>
  )
}

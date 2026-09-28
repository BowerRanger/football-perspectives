import { ListRestartIcon, PlayIcon, RotateCcwIcon, StepForwardIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ButtonGroup } from "@/components/ui/button-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { usePipeline } from "@/hooks/use-pipeline"
import { useConfirm } from "@/hooks/use-dialogs"
import { humanizeStageName, type StageName } from "@/lib/stages"

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

export function StageActions({ stage, complete }: { stage: StageName; complete: boolean }) {
  const { isRunning, runningLabel, missingDeps, startRun, rerunStage } = usePipeline()
  const confirm = useConfirm()
  const missing = missingDeps(stage)
  const label = humanizeStageName(stage)

  const blockedReason = isRunning
    ? `${runningLabel} is running — one job at a time.`
    : missing.length
      ? `Needs ${missing.map(humanizeStageName).join(", ")} first.`
      : null

  const onRerun = async () => {
    if (complete) {
      const ok = await confirm({
        title: `Re-run ${label} from scratch?`,
        description: `This deletes everything in the ${stage}/ output directory before running. Use Continue to keep cached per-player results.`,
        confirmLabel: "Delete output and re-run",
        destructive: true,
      })
      if (!ok) return
    }
    await rerunStage(stage)
  }

  const onRunAll = async () => {
    const ok = await confirm({
      title: "Run the whole pipeline?",
      description: "Runs every stage in order. Completed stages are skipped from cache; heavy stages (HMR World) can take most of an hour.",
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
          hint="Run without wiping the output directory — resumes from cached per-player results."
          onClick={() => void startRun(stage, stage)}
        >
          <StepForwardIcon /> Continue
        </GatedButton>
        <GatedButton size="sm" reason={blockedReason} onClick={() => void onRerun()}>
          {complete ? <RotateCcwIcon /> : <PlayIcon />}
          {complete ? "Re-run stage" : "Run stage"}
        </GatedButton>
      </ButtonGroup>
    </>
  )
}

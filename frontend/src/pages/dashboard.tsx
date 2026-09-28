import * as React from "react"
import { useSearchParams } from "react-router"

import { PageHeader } from "@/components/page-header"
import { PanelError, PanelSkeleton } from "@/components/panel"
import { BlockedNote, StageActions } from "@/components/stage-actions"
import { StatusBadge, resolveStageStatus } from "@/components/status"
import { Button } from "@/components/ui/button"
import { STAGE_PANELS } from "@/features/stages/registry"
import { usePipeline } from "@/hooks/use-pipeline"
import { STAGE_DESCRIPTIONS, humanizeStageName, isStageName, type StageName } from "@/lib/stages"

const STAGE_KEY = "fp.currentStage"

function readStoredStage(): string | null {
  try {
    return localStorage.getItem(STAGE_KEY)
  } catch {
    return null
  }
}

/**
 * Resolve the stage to show: ?stage= wins, then the stage the operator was
 * on before leaving for an editor, then the first incomplete stage.
 */
export function useActiveStage(): StageName | null {
  const [params] = useSearchParams()
  const { stages } = usePipeline()
  const fromUrl = params.get("stage")
  if (isStageName(fromUrl)) return fromUrl
  const stored = readStoredStage()
  if (isStageName(stored) && stages.some((s) => s.name === stored)) return stored
  const firstIncomplete = stages.find((s) => !s.complete) ?? stages[0]
  return firstIncomplete?.name ?? null
}

class PanelBoundary extends React.Component<{ children: React.ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    if (this.state.error) {
      return (
        <PanelError
          title="This panel crashed"
          message={this.state.error.message}
          action={
            <Button size="sm" variant="outline" className="mt-2" onClick={() => this.setState({ error: null })}>
              Try again
            </Button>
          }
        />
      )
    }
    return this.props.children
  }
}

export default function DashboardPage() {
  const stage = useActiveStage()
  const { stages, liveState, outputVersion } = usePipeline()

  React.useEffect(() => {
    if (!stage) return
    try {
      localStorage.setItem(STAGE_KEY, stage)
    } catch {
      /* storage unavailable */
    }
    document.title = `${humanizeStageName(stage)} · Football Perspectives`
  }, [stage])

  if (!stage) {
    return (
      <div className="flex flex-1 flex-col gap-4 p-4">
        <PanelSkeleton rows={6} />
      </div>
    )
  }

  const info = stages.find((s) => s.name === stage)
  const status = resolveStageStatus(info?.complete, liveState[stage], info?.partial)
  const Panel = STAGE_PANELS[stage]

  return (
    <>
      <PageHeader
        title={humanizeStageName(stage)}
        status={
          <>
            <StatusBadge status={status} />
            <BlockedNote stage={stage} />
          </>
        }
        description={STAGE_DESCRIPTIONS[stage]}
        actions={<StageActions stage={stage} hasOutput={!!(info?.complete || info?.partial)} />}
      />
      <div className="flex flex-1 flex-col gap-4 p-4 md:p-6">
        <PanelBoundary key={`${stage}:${outputVersion}`}>
          <React.Suspense fallback={<PanelSkeleton rows={5} media />}>
            <Panel />
          </React.Suspense>
        </PanelBoundary>
      </div>
    </>
  )
}

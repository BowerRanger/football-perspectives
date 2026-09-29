import * as React from "react"
import { InfoIcon } from "lucide-react"

import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { Skeleton } from "@/components/ui/skeleton"
import { ToneBadge, type Tone } from "@/components/status"
import { fmt } from "@/lib/format"
import { getJson, qs } from "@/lib/api"
import { useResource } from "@/hooks/use-resource"
import { Button } from "@/components/ui/button"
import type { CameraMetrics } from "./types"

// Explanations + thresholds for each honest camera metric (ported verbatim
// from the legacy CAM_METRIC_HELP), delivered via keyboard-reachable HoverCards.
interface MetricHelp {
  text: string
  legend?: string
}

const HELP: Record<string, MetricHelp> = {
  coverage: {
    text: "How much of the clip actually got a camera solution, versus frames that are interpolated or have no painted lines. A low percentage means large guessed spans that aren't really tracked.",
    legend: "Good ≥ 90%, marginal ≥ 60%",
  },
  line: {
    text: "Average pixel distance between detected painted lines and where the model projects them. SELF-REFERENTIAL — the solver minimises this exact quantity, so a low value mainly means the detected lines are crisp, NOT that the camera is globally correct. Treat it as one signal, never as the success metric.",
    legend: "Good ≤ 2 px, marginal ≤ 4 px",
  },
  jitter: {
    text: "Frame-to-frame camera rotation (p95 = typical worst-case jump). A high value means a glitchy or shaky track between frames, even when each frame fits its lines well.",
    legend: "Good ≤ 0.5°, marginal ≤ 1.5°",
  },
  circle: {
    text: "HELD-OUT check: the painted centre circle is detected directly in the image and compared to where the model projects it. Because it isn't used by the line solve, it exposes lens / wide-field error that line crispness can't see. 'not detected' means the ring wasn't clearly visible in the sampled frames.",
    legend: "Good ≤ 5 px, marginal ≤ 15 px",
  },
  manual: {
    text: "Geometric difference from the hand-anchored manual track: median camera rotation and centre-position Δ. This is true camera accuracy (not line-fit), so a large value flags a genuine disagreement worth eyeballing — though the manual itself isn't always perfect ground truth.",
    legend: "Good ≤ 1° / 1 m, marginal ≤ 3° / 3 m",
  },
}

/** good <= g, marginal <= a, else bad. */
function toneFor(v: number | null | undefined, good: number, ok: number): Tone {
  if (v === null || v === undefined) return "muted"
  return v <= good ? "success" : v <= ok ? "warning" : "destructive"
}

function MetricLabel({ label, help }: { label: string; help: MetricHelp }) {
  return (
    <HoverCard openDelay={100}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded-sm text-muted-foreground underline decoration-dotted underline-offset-4 outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {label}
          <InfoIcon className="size-3" aria-hidden />
        </button>
      </HoverCardTrigger>
      <HoverCardContent className="w-80 text-xs leading-relaxed">
        <p>{help.text}</p>
        {help.legend ? <p className="mt-2 font-medium text-foreground">{help.legend}</p> : null}
      </HoverCardContent>
    </HoverCard>
  )
}

function Row({ label, help, children }: { label: string; help: MetricHelp; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-sm">
      <MetricLabel label={label} help={help} />
      <span className="flex items-center gap-1.5 text-right font-medium tabular-nums">{children}</span>
    </div>
  )
}

function MetricsRows({ m }: { m: CameraMetrics }) {
  const pct = m.clip_frames ? Math.round((100 * m.covered) / m.clip_frames) : 0
  const pctTone: Tone = pct >= 90 ? "success" : pct >= 60 ? "warning" : "destructive"
  return (
    <div className="flex flex-col gap-1.5">
      <Row label="Coverage" help={HELP.coverage}>
        <span className="text-muted-foreground">{m.covered}/{m.clip_frames}</span>
        <ToneBadge tone={pctTone}>{pct}%</ToneBadge>
      </Row>
      <Row label="Line crispness" help={HELP.line}>
        <ToneBadge tone={toneFor(m.line_rms_mean, 2, 4)}>{fmt(m.line_rms_mean, 2)} px</ToneBadge>
      </Row>
      {m.jitter_p95 != null ? (
        <Row label="Jitter (p95)" help={HELP.jitter}>
          <ToneBadge tone={toneFor(m.jitter_p95, 0.5, 1.5)}>{fmt(m.jitter_p95, 2)}°</ToneBadge>
        </Row>
      ) : null}
      <Row label="Circle (held-out)" help={HELP.circle}>
        {!m.circle ? (
          <ToneBadge tone="muted">n/a</ToneBadge>
        ) : m.circle.misfit == null ? (
          <ToneBadge tone="muted">not detected</ToneBadge>
        ) : (
          <>
            <ToneBadge tone={toneFor(m.circle.misfit, 5, 15)}>{fmt(m.circle.misfit, 1)} px</ToneBadge>
            <span className="text-xs text-muted-foreground">({m.circle.frames}f)</span>
          </>
        )}
      </Row>
      {m.vs_manual ? (
        <Row label="vs manual" help={HELP.manual}>
          <ToneBadge tone={toneFor(m.vs_manual.rotation, 1, 3)}>{fmt(m.vs_manual.rotation, 2)}°</ToneBadge>
          <ToneBadge tone={toneFor(m.vs_manual.centre, 1, 3)}>{fmt(m.vs_manual.centre, 2)} m</ToneBadge>
        </Row>
      ) : null}
    </div>
  )
}

/** Lazily loaded: the held-out circle check reads the clip video, so it takes a few seconds. */
export function CameraMetricsBlock({ shot }: { shot: string }) {
  // 200 {available:false} means "no metrics for this shot"; a 500 is an error.
  const { state, retry } = useResource(
    (signal) => getJson<CameraMetrics>(`/api/camera/metrics${qs({ shot })}`, { signal }),
    [shot],
  )

  if (state.status === "loading") {
    return (
      <div className="flex flex-col gap-1.5" aria-busy="true" aria-label="Loading quality metrics">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-5 w-full" />
        ))}
      </div>
    )
  }
  if (state.status === "error") {
    return (
      <div className="flex flex-col items-start gap-2 text-sm">
        <p className="text-destructive">Could not compute quality metrics: {state.error}</p>
        <Button size="sm" variant="outline" onClick={retry}>
          Retry
        </Button>
      </div>
    )
  }
  const m = state.data
  if (!m || m.available === false) {
    return <p className="text-sm text-muted-foreground">No quality metrics available for this shot.</p>
  }
  return <MetricsRows m={m} />
}

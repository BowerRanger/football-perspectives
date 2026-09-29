import * as React from "react"

import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { errorMessage } from "@/lib/api"
import { loadShotOptions, type ShotOption } from "./api"

interface ShotOptionsState {
  options: ShotOption[]
  loading: boolean
  error: string | null
}

/** Shots from /api/output/shots with saved ball-anchor counts. */
export function useShotOptions(): ShotOptionsState {
  const [state, setState] = React.useState<ShotOptionsState>({ options: [], loading: true, error: null })
  React.useEffect(() => {
    let cancelled = false
    loadShotOptions()
      .then((options) => !cancelled && setState({ options, loading: false, error: null }))
      .catch((err: unknown) => !cancelled && setState({ options: [], loading: false, error: errorMessage(err) }))
    return () => {
      cancelled = true
    }
  }, [])
  return state
}

interface ShotSelectProps {
  value: string
  options: ShotOption[]
  onChange: (shot: string) => void
  id?: string
  className?: string
}

export function ShotSelect({ value, options, onChange, id = "ball-shot-select", className }: ShotSelectProps) {
  return (
    <div className="flex items-center gap-2">
      <Label htmlFor={id} className="text-sm text-muted-foreground">
        Shot
      </Label>
      <Select value={value || undefined} onValueChange={onChange} disabled={options.length === 0}>
        <SelectTrigger id={id} size="sm" className={className ?? "min-w-40"}>
          <SelectValue placeholder={options.length ? "Choose a shot" : "No shots"} />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.id} value={o.id}>
              {o.id}
              {o.anchorCount != null ? (
                <span className="text-muted-foreground"> · {o.anchorCount} anchors</span>
              ) : null}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

import { Link } from "react-router"
import { CheckIcon, ChevronDownIcon, EyeIcon, ExternalLinkIcon, RefreshCwIcon, SaveIcon } from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Kbd } from "@/components/ui/kbd"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Spinner } from "@/components/ui/spinner"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { Stadium, ViewOptions } from "./types"

export const VIEW_ITEMS: { key: keyof ViewOptions; label: string; hint: string; keys: string }[] = [
  { key: "snap", label: "Snap to lines", hint: "Refine each click to the nearest painted feature", keys: "S" },
  { key: "pitch", label: "Projected pitch", hint: "Catalogue pitch lines projected through the solved camera", keys: "P" },
  { key: "detected", label: "Detected lines", hint: "Painted lines the camera stage found (cyan)", keys: "D" },
  { key: "labels", label: "Landmark labels", hint: "Catalogue landmark names projected onto the frame", keys: "L" },
  { key: "anchors", label: "Anchors", hint: "Your placed points and lines for this frame", keys: "A" },
]

const NO_STADIUM = "__none__"

export type StatusTone = "muted" | "warning" | "destructive"

export interface ToolbarProps {
  shots: readonly string[]
  shot: string
  showShotSelect: boolean
  onShotChange: (shot: string) => void
  stadiums: readonly Stadium[]
  stadium: string
  onStadiumChange: (stadium: string) => void
  view: ViewOptions
  onToggleView: (key: keyof ViewOptions) => void
  status: string
  statusTone: StatusTone
  /** True when `status` is a transient message (save/rerun result) rather than the shot baseline. */
  statusIsFlash: boolean
  anchorCount: number
  dirty: boolean
  saving: boolean
  onSave: () => void
  rerunPhase: "idle" | "saving" | "starting"
  /** Why a rerun can't start right now (a job is running), or null. */
  rerunBlockedReason: string | null
  onRerun: () => void
  viewerHref: string | null
}

function ViewMenu({ view, onToggle }: { view: ViewOptions; onToggle: (key: keyof ViewOptions) => void }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <EyeIcon />
          View
          <ChevronDownIcon />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64">
        <DropdownMenuLabel>Overlays and click behaviour</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {VIEW_ITEMS.map((item) => (
          <DropdownMenuCheckboxItem
            key={item.key}
            checked={view[item.key]}
            onCheckedChange={() => onToggle(item.key)}
            onSelect={(e) => e.preventDefault()}
          >
            <span className="flex min-w-0 flex-1 flex-col">
              <span>{item.label}</span>
              <span className="text-xs text-muted-foreground">{item.hint}</span>
            </span>
            <Kbd>{item.keys}</Kbd>
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function RerunButton(props: Pick<ToolbarProps, "rerunPhase" | "rerunBlockedReason" | "onRerun" | "shot">) {
  const { rerunPhase, rerunBlockedReason, onRerun, shot } = props
  const busy = rerunPhase !== "idle"
  const disabled = busy || rerunBlockedReason !== null || !shot
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span>
          <Button variant="outline" size="sm" disabled={disabled} onClick={onRerun}>
            {busy ? <Spinner /> : <RefreshCwIcon />}
            {rerunPhase === "saving" ? "Saving anchors" : rerunPhase === "starting" ? "Starting" : "Rerun camera tracking"}
          </Button>
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">
        {rerunBlockedReason ?? "Saves your anchors, then re-solves the camera for this shot. Overwrites its camera output."}
      </TooltipContent>
    </Tooltip>
  )
}

/** Dirty / saved / transient-message badge shown beside the title (page) or at the toolbar's right (embedded). */
export function AnchorStatusBadge(props: Pick<ToolbarProps, "dirty" | "anchorCount" | "status" | "statusTone" | "statusIsFlash">) {
  if (props.dirty) return <ToneBadge tone="warning">Unsaved changes · {props.anchorCount} anchor frames</ToneBadge>
  if (props.statusIsFlash) {
    const tone = props.statusTone === "muted" ? "success" : props.statusTone
    return (
      <ToneBadge tone={tone} role="status" aria-live="polite" className="max-w-full truncate">
        {props.status}
      </ToneBadge>
    )
  }
  return (
    <ToneBadge tone="muted">
      <CheckIcon /> {props.anchorCount} anchor frames saved
    </ToneBadge>
  )
}

/** Shot / stadium / view / rerun / save. Shared by the page header and the embedded toolbar. */
export function EditorControls(props: ToolbarProps) {
  const { shots, shot, showShotSelect, onShotChange, stadiums, stadium, onStadiumChange } = props
  return (
    <>
      {showShotSelect ? (
        <Select value={shot || undefined} onValueChange={onShotChange} disabled={shots.length === 0}>
          <SelectTrigger size="sm" aria-label="Shot" className="w-40 font-mono text-xs">
            <SelectValue placeholder={shots.length === 0 ? "No shots found" : "Select shot"} />
          </SelectTrigger>
          <SelectContent>
            {shots.map((id) => (
              <SelectItem key={id} value={id} className="font-mono text-xs">
                {id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : null}
      <Tooltip>
        <TooltipTrigger asChild>
          <span>
            <Select
              value={stadium || NO_STADIUM}
              onValueChange={(v) => onStadiumChange(v === NO_STADIUM ? "" : v)}
            >
              <SelectTrigger size="sm" aria-label="Stadium" className="w-44 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={NO_STADIUM}>No stadium</SelectItem>
                {stadiums.map((s) => (
                  <SelectItem key={s.id} value={s.id}>
                    {s.display_name || s.id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </span>
        </TooltipTrigger>
        <TooltipContent>Pick the stadium to enable mowing-stripe entries in the Lines palette</TooltipContent>
      </Tooltip>
      <ViewMenu view={props.view} onToggle={props.onToggleView} />
      {props.viewerHref ? (
        <Button asChild variant="ghost" size="sm">
          <Link to={props.viewerHref}>
            <ExternalLinkIcon />
            Open viewer
          </Link>
        </Button>
      ) : null}
      <RerunButton
        shot={shot}
        rerunPhase={props.rerunPhase}
        rerunBlockedReason={props.rerunBlockedReason}
        onRerun={props.onRerun}
      />
      <Tooltip>
        <TooltipTrigger asChild>
          <Button size="sm" disabled={!shot || props.saving} onClick={props.onSave}>
            {props.saving ? <Spinner /> : <SaveIcon />}
            Save anchors
          </Button>
        </TooltipTrigger>
        <TooltipContent className="flex items-center gap-2">
          Save to anchors.json <Kbd>⌘S</Kbd>
        </TooltipContent>
      </Tooltip>
    </>
  )
}

/** Compact single toolbar row for the embedded (Camera stage) editor. */
export function Toolbar(props: ToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
      <EditorControls {...props} />
      <div className="ml-auto min-w-0">
        <AnchorStatusBadge {...props} />
      </div>
    </div>
  )
}

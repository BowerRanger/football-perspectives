import { PlusIcon, XIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

import { IconButton } from "./icon-button"
import { newRosterRow, type RosterRow } from "./match-types"

interface RosterColumnProps {
  teamLabel: string
  teamCode: "A" | "B"
  rows: RosterRow[]
  onChange: (rows: RosterRow[]) => void
}

/** One team's roster editor (name, position, shirt number per row). */
export function RosterColumn({ teamLabel, teamCode, rows, onChange }: RosterColumnProps) {
  const patch = (key: string, change: Partial<RosterRow>) =>
    onChange(rows.map((r) => (r.key === key ? { ...r, ...change } : r)))

  return (
    <div className="flex min-w-0 flex-1 basis-72 flex-col gap-2">
      <h4 className="text-sm font-medium">
        {teamLabel} <span className="font-normal text-muted-foreground">(team {teamCode})</span>
      </h4>
      {rows.length === 0 ? <p className="text-xs text-muted-foreground">No players yet.</p> : null}
      <ul className="flex flex-col gap-1.5">
        {rows.map((r, i) => (
          <li key={r.key} className="flex items-center gap-1.5">
            <Input
              aria-label={`${teamLabel} player ${i + 1} name`}
              placeholder="Name"
              className="min-w-0 flex-[2]"
              value={r.name}
              onChange={(e) => patch(r.key, { name: e.target.value })}
            />
            <Input
              aria-label={`${teamLabel} player ${i + 1} position`}
              placeholder="Pos"
              className="w-16"
              value={r.position}
              onChange={(e) => patch(r.key, { position: e.target.value })}
            />
            <Input
              aria-label={`${teamLabel} player ${i + 1} shirt number`}
              placeholder="#"
              type="number"
              className="w-16"
              value={r.shirt}
              onChange={(e) => patch(r.key, { shirt: e.target.value })}
            />
            <IconButton
              label={`Remove ${r.name || `player ${i + 1}`}`}
              variant="ghost"
              onClick={() => onChange(rows.filter((x) => x.key !== r.key))}
            >
              <XIcon />
            </IconButton>
          </li>
        ))}
      </ul>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="self-start"
        onClick={() => onChange([...rows, newRosterRow()])}
      >
        <PlusIcon data-icon="inline-start" />
        Add {teamLabel.toLowerCase()} player
      </Button>
    </div>
  )
}

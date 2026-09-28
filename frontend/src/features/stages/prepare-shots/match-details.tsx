import * as React from "react"
import { ChevronDownIcon, SaveIcon } from "lucide-react"
import { toast } from "sonner"

import { Panel } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { errorMessage, getJson, putJson } from "@/lib/api"
import { cn } from "@/lib/utils"

import { MatchLookup } from "./match-lookup"
import {
  applyCandidate,
  formFromMatch,
  payloadFromForm,
  type KitsPayload,
  type MatchForm,
  type MatchPayload,
} from "./match-types"
import { RosterColumn } from "./roster-columns"

function Field({ label, htmlFor, className, children }: { label: string; htmlFor: string; className?: string; children: React.ReactNode }) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1.5", className)}>
      <Label htmlFor={htmlFor}>{label}</Label>
      {children}
    </div>
  )
}

const KIT_FIELDS: { key: keyof KitsPayload; label: string }[] = [
  { key: "home_primary", label: "Home primary" },
  { key: "away_primary", label: "Away primary" },
  { key: "home_goalkeeper", label: "Home GK" },
  { key: "away_goalkeeper", label: "Away GK" },
  { key: "referee", label: "Referee" },
]

/**
 * Match / roster editor, collapsed by default. The tracking stage's player-name
 * picker reads /api/match itself, so saving here is all that is needed.
 */
export default function MatchDetails() {
  const [open, setOpen] = React.useState(false)
  const [form, setForm] = React.useState<MatchForm | null>(null)
  const [loadError, setLoadError] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [saving, setSaving] = React.useState(false)
  const loaded = React.useRef(false)

  React.useEffect(() => {
    if (!open || loaded.current) return
    loaded.current = true
    getJson<MatchPayload | null>("/api/match")
      .then((m) => setForm(formFromMatch(m)))
      .catch((err) => {
        loaded.current = false
        setLoadError(errorMessage(err))
      })
  }, [open])

  const set = <K extends keyof MatchForm>(key: K, value: MatchForm[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f))

  const save = async () => {
    if (!form) return
    const payload = payloadFromForm(form)
    if (!payload.home_team || !payload.away_team || !payload.venue) {
      setError("Home team, away team and venue are required.")
      return
    }
    setError(null)
    setSaving(true)
    try {
      await putJson("/api/match", payload)
      toast.success("Match details saved")
    } catch (err) {
      setError(`Save failed: ${errorMessage(err)}`)
      toast.error("Could not save match details", { description: errorMessage(err) })
    } finally {
      setSaving(false)
    }
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Panel
        title="Match details"
        description="Teams, score, kit colours and roster. Feeds the tracking stage's player-name picker."
        actions={
          <CollapsibleTrigger asChild>
            <Button variant="outline" size="sm" aria-label={open ? "Collapse match details" : "Expand match details"}>
              {open ? "Hide" : "Edit"}
              <ChevronDownIcon data-icon="inline-end" className={cn("transition-transform", open && "rotate-180")} />
            </Button>
          </CollapsibleTrigger>
        }
        contentClassName={open ? undefined : "hidden"}
      >
        <CollapsibleContent>
          {loadError ? (
            <p className="text-sm text-destructive">Could not load match details: {loadError}</p>
          ) : !form ? (
            <div className="flex flex-col gap-2">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          ) : (
            <MatchFormBody
              form={form}
              set={set}
              error={error}
              saving={saving}
              onApply={(cand) => setForm((f) => (f ? applyCandidate(f, cand) : f))}
              onSave={() => void save()}
            />
          )}
        </CollapsibleContent>
      </Panel>
    </Collapsible>
  )
}

interface BodyProps {
  form: MatchForm
  set: <K extends keyof MatchForm>(key: K, value: MatchForm[K]) => void
  error: string | null
  saving: boolean
  onApply: Parameters<typeof MatchLookup>[0]["onApply"]
  onSave: () => void
}

function MatchFormBody({ form, set, error, saving, onApply, onSave }: BodyProps) {
  const text = (id: string, key: "home_team" | "away_team" | "venue" | "competition" | "date", type = "text") => (
    <Input id={id} type={type} value={form[key]} onChange={(e) => set(key, e.target.value)} />
  )
  return (
    <div className="flex flex-col gap-5">
      <MatchLookup
        initialHome={form.home_team}
        initialAway={form.away_team}
        initialDate={form.date}
        onApply={onApply}
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Field label="Home team" htmlFor="m-home">{text("m-home", "home_team")}</Field>
        <Field label="Away team" htmlFor="m-away">{text("m-away", "away_team")}</Field>
        <Field label="Home score" htmlFor="m-hs">
          <Input id="m-hs" type="number" value={form.home_score} onChange={(e) => set("home_score", e.target.value)} />
        </Field>
        <Field label="Away score" htmlFor="m-as">
          <Input id="m-as" type="number" value={form.away_score} onChange={(e) => set("away_score", e.target.value)} />
        </Field>
        <Field label="Venue (required)" htmlFor="m-venue">{text("m-venue", "venue")}</Field>
        <Field label="Date" htmlFor="m-date">{text("m-date", "date", "date")}</Field>
        <Field label="Competition" htmlFor="m-comp" className="col-span-2">{text("m-comp", "competition")}</Field>
      </div>

      <fieldset className="flex flex-col gap-3">
        <legend className="mb-1 text-sm font-medium">Moment</legend>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Field label="Minute" htmlFor="m-min">
            <Input id="m-min" type="number" min={0} max={130} placeholder="1-120" value={form.minute} onChange={(e) => set("minute", e.target.value)} />
          </Field>
          <Field label="Added time" htmlFor="m-added">
            <Input id="m-added" type="number" min={0} max={20} value={form.added_time} onChange={(e) => set("added_time", e.target.value)} />
          </Field>
          <Field label="Event" htmlFor="m-event">
            <NativeSelect id="m-event" disabled value="goal" title="Only goals are supported for now.">
              <NativeSelectOption value="goal">goal</NativeSelectOption>
            </NativeSelect>
          </Field>
        </div>
        <Field label="Description" htmlFor="m-desc">
          <Textarea id="m-desc" placeholder="e.g. Salah header from corner" value={form.description} onChange={(e) => set("description", e.target.value)} />
        </Field>
      </fieldset>

      <fieldset className="flex flex-col gap-3">
        <legend className="mb-1 text-sm font-medium">Kits</legend>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
          {KIT_FIELDS.map(({ key, label }) => (
            <Field key={key} label={label} htmlFor={`m-kit-${key}`}>
              <Input
                id={`m-kit-${key}`}
                type="color"
                className="h-8 cursor-pointer p-1"
                value={form.kits[key]}
                onChange={(e) => set("kits", { ...form.kits, [key]: e.target.value })}
              />
            </Field>
          ))}
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-3">
        <legend className="mb-1 text-sm font-medium">Roster (players on the pitch)</legend>
        <div className="flex flex-wrap gap-6">
          <RosterColumn teamLabel="Home" teamCode="A" rows={form.home} onChange={(r) => set("home", r)} />
          <RosterColumn teamLabel="Away" teamCode="B" rows={form.away} onChange={(r) => set("away", r)} />
        </div>
      </fieldset>

      <div className="flex flex-wrap items-center gap-3">
        <Button disabled={saving} onClick={onSave}>
          <SaveIcon data-icon="inline-start" />
          Save match details
        </Button>
        {error ? (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        ) : null}
      </div>
    </div>
  )
}

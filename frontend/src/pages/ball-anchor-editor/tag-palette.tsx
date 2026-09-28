import { Kbd } from "@/components/ui/kbd"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { TAGS, tagById } from "./tags"

interface TagPaletteProps {
  selected: string
  onSelect: (id: string) => void
}

/** Single-select anchor type list; the selected tag's help text shows inline (not in a title tooltip). */
export function TagPalette({ selected, onSelect }: TagPaletteProps) {
  const tag = tagById(selected)
  return (
    <div className="flex flex-col gap-3">
      <ToggleGroup
        type="single"
        orientation="vertical"
        spacing={1}
        value={selected}
        onValueChange={(v) => v && onSelect(v)}
        aria-label="Anchor type"
        className="w-full flex-col items-stretch"
      >
        {TAGS.map((t) => (
          <ToggleGroupItem
            key={t.id}
            value={t.id}
            aria-label={`${t.label}, ${t.range}`}
            className="h-auto w-full justify-start gap-2 px-2 py-1 text-left"
          >
            <span aria-hidden className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: t.color }} />
            <span className="flex min-w-0 flex-1 flex-col leading-tight">
              <span className="text-sm">{t.label}</span>
              <span className="truncate font-mono text-xs font-normal text-muted-foreground">{t.range}</span>
            </span>
            <Kbd>{t.key}</Kbd>
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      {tag ? (
        <p className="rounded-md bg-muted/50 p-2 text-xs leading-relaxed text-muted-foreground">{tag.description}</p>
      ) : null}
    </div>
  )
}

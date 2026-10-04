import { FlagIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Command, CommandGroup, CommandItem, CommandList } from "@/components/ui/command"
import { Kbd } from "@/components/ui/kbd"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { EVENT_KINDS, EVENT_STYLE } from "./palette"
import type { EventKind } from "./types"

interface EventMenuProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  frame: number
  onPick: (kind: EventKind) => void
}

/** Event menu at the playhead (touch, bounce, post, ...). Opens with E; arrows + Enter choose. */
export function EventMenu({ open, onOpenChange, frame, onPick }: EventMenuProps) {
  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" aria-label={`Add an event at frame ${frame}`}>
          <FlagIcon /> Event <Kbd>E</Kbd>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-52 p-0">
        <Command>
          <CommandList>
            <CommandGroup heading={`At frame ${frame}`}>
              {EVENT_KINDS.map((k) => (
                <CommandItem
                  key={k}
                  value={k}
                  onSelect={() => {
                    onPick(k)
                    onOpenChange(false)
                  }}
                >
                  <span className="size-2.5 rounded-full" style={{ backgroundColor: EVENT_STYLE[k].colour }} aria-hidden />
                  {EVENT_STYLE[k].label}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

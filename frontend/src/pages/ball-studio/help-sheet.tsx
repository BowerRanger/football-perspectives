import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"

const GROUPS: { title: string; rows: [string[], string][] }[] = [
  {
    title: "Transport",
    rows: [
      [["Space"], "Play or pause all views"],
      [["←", "→"], "Step one frame (also , and .)"],
      [["Shift", "←"], "Step ten frames (also < and >)"],
      [["Home", "End"], "Jump to the ends"],
    ],
  },
  {
    title: "Picking",
    rows: [
      [["Click"], "Pending pick in that view (nothing is saved yet)"],
      [["Enter"], "Commit: triangulated key, or constrained key"],
      [["K"], "Commit the pick as a key"],
      [["O"], "Commit the pick as a soft observation"],
      [["Esc"], "Drop pick, then clear selection, then leave focus"],
      [["Alt", "←"], "Nudge the pending pick 1 px (Shift: 5 px)"],
      [["Alt", "Click"], "Free click, no epipolar snap"],
    ],
  },
  {
    title: "Single-view constraints",
    rows: [
      [["G"], "Ground"],
      [["H"], "Height"],
      [["L"], "Goal-line plane"],
      [["D"], "Depth along the ray"],
      [["P"], "Player joint"],
      [["T"], "Back to triangulate"],
    ],
  },
  {
    title: "Navigation",
    rows: [
      [["["], "Previous key"],
      [["]"], "Next key"],
      [["N"], "Next flag or unsupported span (Shift: previous)"],
      [["Tab"], "Next active view"],
      [["F"], "Focus view on or off"],
      [["Z"], "Hold for the 4x loupe"],
      [["Wheel"], "Zoom (Alt+drag or middle drag pans), 0 resets"],
    ],
  },
  {
    title: "Edit",
    rows: [
      [["E"], "Event menu at the playhead"],
      [["Del"], "Delete the selected key, event or observation"],
      [["Ctrl", "Z"], "Undo (Shift: redo)"],
      [["Ctrl", "S"], "Save (also S)"],
    ],
  },
]

export function HelpSheet({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Keyboard shortcuts</SheetTitle>
          <SheetDescription>Ignored while you type in a field or have a menu open.</SheetDescription>
        </SheetHeader>
        <div className="flex flex-col gap-5 px-4 pb-6">
          {GROUPS.map((g) => (
            <section key={g.title} aria-label={g.title}>
              <h3 className="mb-2 text-sm font-semibold">{g.title}</h3>
              <dl className="flex flex-col gap-1.5 text-sm">
                {g.rows.map(([keys, desc]) => (
                  <div key={desc} className="grid grid-cols-[110px_1fr] items-center gap-2">
                    <dt>
                      <KbdGroup>
                        {keys.map((k) => (
                          <Kbd key={k}>{k}</Kbd>
                        ))}
                      </KbdGroup>
                    </dt>
                    <dd className="text-muted-foreground">{desc}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  )
}

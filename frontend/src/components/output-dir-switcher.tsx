import * as React from "react"
import { ChevronsUpDownIcon, FolderIcon, FolderPlusIcon, CheckIcon } from "lucide-react"
import { toast } from "sonner"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar"
import { errorMessage, getJson, postJson, putJson } from "@/lib/api"
import { usePipeline } from "@/hooks/use-pipeline"
import { usePrompt } from "@/hooks/use-dialogs"

interface OutputDirsPayload {
  dirs: string[]
  current: string
}

const NAME_RE = /^[A-Za-z0-9_-]+$/

/**
 * The active output directory is server-wide: switching re-points every
 * endpoint at once, so a successful switch reloads the page to refresh all
 * panels. Locked while a run is in flight (the job keeps writing to the dir
 * it started on).
 */
export function OutputDirSwitcher() {
  const { isRunning, runningLabel } = usePipeline()
  const prompt = usePrompt()
  const [data, setData] = React.useState<OutputDirsPayload | null>(null)

  React.useEffect(() => {
    getJson<OutputDirsPayload>("/api/output-dirs")
      .then(setData)
      .catch(() => setData(null))
  }, [])

  const switchTo = async (name: string) => {
    if (!data || name === data.current) return
    try {
      await putJson("/api/output-dirs/active", { name })
      window.location.reload()
    } catch (err) {
      toast.error("Could not switch output directory", { description: errorMessage(err) })
    }
  }

  const createNew = async () => {
    const name = await prompt({
      title: "New output directory",
      description: "Creates an empty sibling directory and makes it active. “exp1” becomes output-exp1.",
      label: "Name",
      placeholder: "exp1",
      confirmLabel: "Create and switch",
      validate: (v) => (NAME_RE.test(v) ? null : "Use letters, numbers, dashes and underscores only"),
    })
    if (!name) return
    try {
      await postJson("/api/output-dirs", { name })
      window.location.reload()
    } catch (err) {
      toast.error("Could not create output directory", { description: errorMessage(err) })
    }
  }

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild disabled={isRunning || !data}>
            <SidebarMenuButton
              size="lg"
              className="data-[state=open]:bg-sidebar-accent"
              tooltip={isRunning ? `Locked while ${runningLabel} runs` : "Active output directory"}
            >
              <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                <FolderIcon className="size-4" />
              </div>
              <div className="grid flex-1 text-left leading-tight">
                <span className="truncate text-xs text-muted-foreground">Output directory</span>
                <span className="truncate font-mono text-sm font-medium">{data?.current ?? "…"}</span>
              </div>
              <ChevronsUpDownIcon className="ml-auto" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="min-w-56" align="start" side="bottom">
            <DropdownMenuLabel className="text-xs text-muted-foreground">Switch reconstruction</DropdownMenuLabel>
            {data?.dirs.map((d) => (
              <DropdownMenuItem key={d} onSelect={() => void switchTo(d)} className="font-mono">
                {d}
                {d === data.current ? <CheckIcon className="ml-auto" /> : null}
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => void createNew()}>
              <FolderPlusIcon /> New output…
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}

import { Link, useLocation, useSearchParams } from "react-router"
import { BoxIcon, CircleDotIcon, CrosshairIcon, LoaderCircleIcon, OrbitIcon, TerminalIcon } from "lucide-react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  SidebarRail,
  SidebarSeparator,
} from "@/components/ui/sidebar"
import { OutputDirSwitcher } from "@/components/output-dir-switcher"
import { StatusDot, resolveStageStatus } from "@/components/status"
import { ThemeToggle } from "@/components/theme-toggle"
import { usePipeline } from "@/hooks/use-pipeline"
import { humanizeStageName } from "@/lib/stages"

const EDITORS = [
  { to: "/anchor_editor", label: "Pitch anchors", icon: CrosshairIcon },
  { to: "/ball-anchor-editor", label: "Ball anchors", icon: CircleDotIcon },
  { to: "/viewer", label: "3D viewer", icon: BoxIcon },
  { to: "/ball-studio", label: "Ball studio", icon: OrbitIcon },
] as const

export function AppSidebar({ activeStage }: { activeStage: string | null }) {
  const { stages, stagesLoaded, liveState, runningLabel, log, setLogOpen } = usePipeline()
  const location = useLocation()
  const [searchParams] = useSearchParams()
  const onDashboard = location.pathname === "/"
  // Carry ?shot= across editor links so the operator stays on the same shot.
  const shot = searchParams.get("shot")

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center gap-2 px-2 pt-1 pb-2 group-data-[collapsible=icon]:hidden">
          <span className="text-sm font-semibold tracking-tight">Football Perspectives</span>
        </div>
        <OutputDirSwitcher />
      </SidebarHeader>
      <SidebarContent>
        <nav aria-label="Dashboard" className="contents">
        <SidebarGroup>
          <SidebarGroupLabel id="nav-pipeline">Pipeline</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu aria-labelledby="nav-pipeline">
              {!stagesLoaded
                ? Array.from({ length: 8 }, (_, i) => (
                    <SidebarMenuItem key={i}>
                      <SidebarMenuSkeleton showIcon />
                    </SidebarMenuItem>
                  ))
                : stages.map((s) => {
                    const status = resolveStageStatus(s.complete, liveState[s.name], s.partial)
                    const label = humanizeStageName(s.name)
                    return (
                      <SidebarMenuItem key={s.name}>
                        <SidebarMenuButton
                          asChild
                          isActive={onDashboard && activeStage === s.name}
                          tooltip={`${s.index}. ${label}`}
                        >
                          <Link to={`/?stage=${s.name}`}>
                            <span className="flex size-4 items-center justify-center">
                              <StatusDot status={status} />
                            </span>
                            <span>{label}</span>
                          </Link>
                        </SidebarMenuButton>
                        <SidebarMenuBadge className="font-mono text-muted-foreground">{s.index}</SidebarMenuBadge>
                      </SidebarMenuItem>
                    )
                  })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarSeparator />
        <SidebarGroup>
          <SidebarGroupLabel id="nav-editors">Editors</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu aria-labelledby="nav-editors">
              {EDITORS.map((e) => (
                <SidebarMenuItem key={e.to}>
                  <SidebarMenuButton asChild isActive={location.pathname === e.to} tooltip={e.label}>
                    <Link to={shot && e.to !== "/ball-studio" ? `${e.to}?shot=${encodeURIComponent(shot)}` : e.to}>
                      <e.icon />
                      <span>{e.label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        </nav>
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          {log.status !== "idle" ? (
            <SidebarMenuItem>
              <SidebarMenuButton onClick={() => setLogOpen(true)} tooltip="Show run log">
                {runningLabel ? <LoaderCircleIcon className="animate-spin text-warning" /> : <TerminalIcon />}
                <span className="truncate">{runningLabel ? `${runningLabel} running…` : "Last run log"}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ) : null}
          <SidebarMenuItem className="flex items-center justify-between gap-2 px-1 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-0">
            <span className="text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">Theme</span>
            <ThemeToggle />
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}

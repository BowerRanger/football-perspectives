import * as React from "react"
import { Outlet, Route, Routes } from "react-router"

import { AppSidebar } from "@/components/app-sidebar"
import { LogDock } from "@/components/log-dock"
import { PanelSkeleton } from "@/components/panel"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import DashboardPage, { useActiveStage } from "@/pages/dashboard"

const AnchorEditorPage = React.lazy(() => import("@/pages/anchor-editor"))
const BallAnchorEditorPage = React.lazy(() => import("@/pages/ball-anchor-editor"))
const ViewerPage = React.lazy(() => import("@/pages/viewer"))

function readSidebarCookie(): boolean {
  const m = document.cookie.match(/(?:^|; )sidebar_state=(true|false)/)
  return m ? m[1] === "true" : true
}

function Shell() {
  const activeStage = useActiveStage()
  return (
    <SidebarProvider defaultOpen={readSidebarCookie()}>
      <AppSidebar activeStage={activeStage} />
      <SidebarInset className="min-w-0">
        <React.Suspense
          fallback={
            <div className="p-6">
              <PanelSkeleton rows={6} media />
            </div>
          }
        >
          <Outlet />
        </React.Suspense>
        <LogDock />
      </SidebarInset>
    </SidebarProvider>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<DashboardPage />} />
        <Route path="anchor_editor" element={<AnchorEditorPage />} />
        <Route path="ball-anchor-editor" element={<BallAnchorEditorPage />} />
        <Route path="viewer" element={<ViewerPage />} />
        <Route path="*" element={<DashboardPage />} />
      </Route>
    </Routes>
  )
}

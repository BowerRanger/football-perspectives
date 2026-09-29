import * as React from "react"
import { Outlet, createBrowserRouter } from "react-router"

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

// A data router (not <BrowserRouter>) so editors can use useBlocker to stop
// in-app navigation from discarding unsaved operator edits.
export const router = createBrowserRouter([
  {
    element: <Shell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "anchor_editor", element: <AnchorEditorPage /> },
      { path: "ball-anchor-editor", element: <BallAnchorEditorPage /> },
      { path: "viewer", element: <ViewerPage /> },
      { path: "*", element: <DashboardPage /> },
    ],
  },
])

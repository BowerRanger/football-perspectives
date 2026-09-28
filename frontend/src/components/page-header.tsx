import * as React from "react"

import { Separator } from "@/components/ui/separator"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { cn } from "@/lib/utils"

interface PageHeaderProps {
  title: React.ReactNode
  /** Inline status next to the title (e.g. a StatusBadge). */
  status?: React.ReactNode
  description?: React.ReactNode
  /** Right-aligned actions; wraps below the title on narrow screens. */
  actions?: React.ReactNode
  className?: string
}

/** Sticky page header shared by the dashboard and every editor page. */
export function PageHeader({ title, status, description, actions, className }: PageHeaderProps) {
  return (
    <header
      className={cn(
        "sticky top-0 z-30 flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b bg-background/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/80",
        className,
      )}
    >
      {/* min-w keeps the title from collapsing to nothing; actions wrap below it instead. */}
      <div className="flex min-w-60 flex-1 items-center gap-2">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-1 data-[orientation=vertical]:h-4" />
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="truncate text-base font-semibold tracking-tight">{title}</h1>
            {status}
          </div>
          {description ? <p className="hidden truncate text-xs text-muted-foreground md:block">{description}</p> : null}
        </div>
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  )
}

import * as React from "react"
import { AlertCircleIcon, InboxIcon } from "lucide-react"

import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { cn } from "@/lib/utils"

interface PanelProps extends Omit<React.ComponentProps<typeof Card>, "title"> {
  title: React.ReactNode
  description?: React.ReactNode
  actions?: React.ReactNode
  /** Remove body padding for full-bleed media (video, canvas, tables). */
  flush?: boolean
  contentClassName?: string
}

/**
 * The dashboard's one container: a shadcn Card with a sentence-case title,
 * optional description and header actions. Never nest Panels.
 */
export function Panel({
  title,
  description,
  actions,
  flush,
  className,
  contentClassName,
  children,
  ...props
}: PanelProps) {
  return (
    <Card className={cn("gap-0 py-0", className)} {...props}>
      <CardHeader className="border-b px-4 py-3 [.border-b]:pb-3">
        <CardTitle className="text-sm font-semibold">{title}</CardTitle>
        {description ? <CardDescription className="text-xs">{description}</CardDescription> : null}
        {actions ? <CardAction className="flex items-center gap-2">{actions}</CardAction> : null}
      </CardHeader>
      <CardContent className={cn(flush ? "p-0" : "p-4", contentClassName)}>{children}</CardContent>
    </Card>
  )
}

interface PanelEmptyProps {
  title: string
  description?: React.ReactNode
  icon?: React.ReactNode
  children?: React.ReactNode
  className?: string
}

/** Empty state that teaches the next step instead of saying "nothing here". */
export function PanelEmpty({ title, description, icon, children, className }: PanelEmptyProps) {
  return (
    <Empty className={cn("py-10", className)}>
      <EmptyHeader>
        <EmptyMedia variant="icon">{icon ?? <InboxIcon />}</EmptyMedia>
        <EmptyTitle>{title}</EmptyTitle>
        {description ? <EmptyDescription>{description}</EmptyDescription> : null}
      </EmptyHeader>
      {children ? <EmptyContent>{children}</EmptyContent> : null}
    </Empty>
  )
}

interface PanelErrorProps {
  title?: string
  message: React.ReactNode
  action?: React.ReactNode
}

export function PanelError({ title = "Something went wrong", message, action }: PanelErrorProps) {
  return (
    <Alert variant="destructive">
      <AlertCircleIcon />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        <p>{message}</p>
        {action}
      </AlertDescription>
    </Alert>
  )
}

/** Skeleton shaped like a panel while its data loads. */
export function PanelSkeleton({ rows = 4, media = false }: { rows?: number; media?: boolean }) {
  return (
    <Card className="gap-0 py-0">
      <div className="border-b px-4 py-3">
        <Skeleton className="h-4 w-40" />
      </div>
      <div className="flex flex-col gap-3 p-4">
        {media ? <Skeleton className="aspect-video w-full" /> : null}
        {Array.from({ length: rows }, (_, i) => (
          <Skeleton key={i} className="h-4" style={{ width: `${90 - i * 12}%` }} />
        ))}
      </div>
    </Card>
  )
}

/** Compact label/value grid for stats inside a panel. */
export function StatList({ items, className }: { items: { label: React.ReactNode; value: React.ReactNode; hint?: string }[]; className?: string }) {
  return (
    <dl className={cn("grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm", className)}>
      {items.map((it, i) => (
        <React.Fragment key={i}>
          <dt className="text-muted-foreground" title={it.hint}>
            {it.label}
          </dt>
          <dd className="text-right font-medium tabular-nums">{it.value}</dd>
        </React.Fragment>
      ))}
    </dl>
  )
}

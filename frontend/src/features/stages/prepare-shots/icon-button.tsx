import * as React from "react"

import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

interface IconButtonProps extends Omit<React.ComponentProps<typeof Button>, "aria-label" | "size"> {
  label: string
  size?: "icon-xs" | "icon-sm" | "icon"
}

/** Icon-only button with an aria-label and a visible Tooltip. */
export function IconButton({ label, size = "icon-sm", variant = "outline", children, ...props }: IconButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button type="button" variant={variant} size={size} aria-label={label} {...props}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

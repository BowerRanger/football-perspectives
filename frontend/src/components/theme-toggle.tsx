import { MoonIcon, SunIcon } from "lucide-react"
import { useTheme } from "next-themes"

import { Button } from "@/components/ui/button"

// One click flips light <-> dark (bright-sunlight use). Flipping from "system"
// pins the opposite of what the OS currently resolves to.
export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const isDark = (resolvedTheme ?? "dark") === "dark"
  const label = isDark ? "Switch to light mode" : "Switch to dark mode"
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      title={label}
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      <SunIcon className="scale-100 rotate-0 transition-transform dark:scale-0 dark:-rotate-90" />
      <MoonIcon className="absolute scale-0 rotate-90 transition-transform dark:scale-100 dark:rotate-0" />
    </Button>
  )
}

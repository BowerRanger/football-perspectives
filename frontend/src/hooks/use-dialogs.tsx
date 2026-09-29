import * as React from "react"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"

// Promise-based replacements for window.confirm / window.prompt, rendered
// with shadcn AlertDialog / Dialog. Usage:
//   const confirm = useConfirm()
//   if (!(await confirm({ title: "Delete track?", destructive: true }))) return
//   const prompt = usePrompt()
//   const name = await prompt({ title: "New output", label: "Name" })  // null on cancel

export interface ConfirmOptions {
  title: string
  description?: React.ReactNode
  confirmLabel?: string
  cancelLabel?: string
  destructive?: boolean
}

export interface PromptOptions {
  title: string
  description?: React.ReactNode
  label: string
  defaultValue?: string
  placeholder?: string
  confirmLabel?: string
  /** Return an error message to block submit, or null when valid. */
  validate?: (value: string) => string | null
}

type ConfirmFn = (opts: ConfirmOptions) => Promise<boolean>
type PromptFn = (opts: PromptOptions) => Promise<string | null>

const DialogsContext = React.createContext<{ confirm: ConfirmFn; prompt: PromptFn } | null>(null)

export function useConfirm(): ConfirmFn {
  const ctx = React.useContext(DialogsContext)
  if (!ctx) throw new Error("useConfirm must be used within <DialogsProvider>")
  return ctx.confirm
}

export function usePrompt(): PromptFn {
  const ctx = React.useContext(DialogsContext)
  if (!ctx) throw new Error("usePrompt must be used within <DialogsProvider>")
  return ctx.prompt
}

interface ConfirmState extends ConfirmOptions {
  resolve: (v: boolean) => void
}
interface PromptState extends PromptOptions {
  resolve: (v: string | null) => void
}

export function DialogsProvider({ children }: { children: React.ReactNode }) {
  const [confirmState, setConfirmState] = React.useState<ConfirmState | null>(null)
  const [promptState, setPromptState] = React.useState<PromptState | null>(null)
  const [promptValue, setPromptValue] = React.useState("")
  const [promptError, setPromptError] = React.useState<string | null>(null)

  const confirm = React.useCallback<ConfirmFn>(
    (opts) => new Promise<boolean>((resolve) => setConfirmState({ ...opts, resolve })),
    [],
  )
  const prompt = React.useCallback<PromptFn>((opts) => {
    setPromptValue(opts.defaultValue ?? "")
    setPromptError(null)
    return new Promise<string | null>((resolve) => setPromptState({ ...opts, resolve }))
  }, [])

  const closeConfirm = (result: boolean) => {
    confirmState?.resolve(result)
    setConfirmState(null)
  }
  const closePrompt = (result: string | null) => {
    promptState?.resolve(result)
    setPromptState(null)
  }
  const submitPrompt = (e: React.FormEvent) => {
    e.preventDefault()
    const value = promptValue.trim()
    const err = promptState?.validate?.(value) ?? (value ? null : `${promptState?.label ?? "Value"} is required`)
    if (err) {
      setPromptError(err)
      return
    }
    closePrompt(value)
  }

  const value = React.useMemo(() => ({ confirm, prompt }), [confirm, prompt])

  return (
    <DialogsContext.Provider value={value}>
      {children}
      <AlertDialog open={!!confirmState} onOpenChange={(open) => !open && closeConfirm(false)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{confirmState?.title}</AlertDialogTitle>
            {confirmState?.description ? (
              <AlertDialogDescription asChild={typeof confirmState.description !== "string"}>
                {typeof confirmState.description === "string" ? (
                  confirmState.description
                ) : (
                  <div>{confirmState.description}</div>
                )}
              </AlertDialogDescription>
            ) : null}
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => closeConfirm(false)}>
              {confirmState?.cancelLabel ?? "Cancel"}
            </AlertDialogCancel>
            <AlertDialogAction
              className={cn(confirmState?.destructive && buttonVariants({ variant: "destructive" }))}
              onClick={() => closeConfirm(true)}
            >
              {confirmState?.confirmLabel ?? "Confirm"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <Dialog open={!!promptState} onOpenChange={(open) => !open && closePrompt(null)}>
        <DialogContent className="sm:max-w-md">
          <form onSubmit={submitPrompt} className="grid gap-4">
            <DialogHeader>
              <DialogTitle>{promptState?.title}</DialogTitle>
              {promptState?.description ? (
                <DialogDescription asChild={typeof promptState.description !== "string"}>
                  {typeof promptState.description === "string" ? (
                    promptState.description
                  ) : (
                    <div>{promptState.description}</div>
                  )}
                </DialogDescription>
              ) : null}
            </DialogHeader>
            <div className="grid gap-2">
              <Label htmlFor="fp-prompt-input">{promptState?.label}</Label>
              <Input
                id="fp-prompt-input"
                autoFocus
                value={promptValue}
                placeholder={promptState?.placeholder}
                aria-invalid={!!promptError}
                onChange={(e) => {
                  setPromptValue(e.target.value)
                  setPromptError(null)
                }}
              />
              {promptError ? <p className="text-sm text-destructive">{promptError}</p> : null}
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => closePrompt(null)}>
                Cancel
              </Button>
              <Button type="submit">{promptState?.confirmLabel ?? "Save"}</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </DialogsContext.Provider>
  )
}

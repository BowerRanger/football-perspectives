/** Inputs of the "can we retime / restore right now" gate. */
export interface RetimeGateInputs {
  isRunning: boolean
  runningLabel: string | null
  /** Offsets edited in the sync editor but not saved. */
  dirty: boolean
  /** Marked moment pairs not saved. */
  hasUnsaved: boolean
}

export const CLEAN_GATE: RetimeGateInputs = { isRunning: false, runningLabel: null, dirty: false, hasUnsaved: false }

/** Why retime / restore must not run now, or "" when it may. */
export function retimeGateReason(g: RetimeGateInputs): string {
  if (g.isRunning) return `${g.runningLabel ?? "A job"} is running.`
  if (g.hasUnsaved) return "Save or clear the marked pairs first."
  if (g.dirty) return "Save the group first."
  return ""
}

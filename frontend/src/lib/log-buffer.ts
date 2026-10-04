// Bounded run-log buffer: classifies each line, caps memory, and tracks the
// first error line so the dock can jump to it.

export type LogLevel = "info" | "warn" | "error"

export interface LogLine {
  text: string
  level: LogLevel
}

export interface LogBuffer {
  lines: LogLine[]
  /** Lines discarded from the head once the cap was hit. */
  dropped: number
  /** Index into `lines` of the first error line, or null. */
  firstError: number | null
}

export const EMPTY_LOG_BUFFER: LogBuffer = { lines: [], dropped: 0, firstError: null }

export const MAX_LOG_LINES = 20_000

const ERROR_RE = /(Traceback \(most recent call last\)|\bERROR\b|Error:|Exception:|\[FAIL)/
const WARN_RE = /(\bWARN(ING)?\b|\[SKIP\])/

export function classifyLine(text: string): LogLevel {
  return ERROR_RE.test(text) ? "error" : WARN_RE.test(text) ? "warn" : "info"
}

/** Append raw lines, returning a new buffer (never mutates `buf`). */
export function appendLogLines(buf: LogBuffer, raw: string[], max = MAX_LOG_LINES): LogBuffer {
  if (!raw.length) return buf
  const added: LogLine[] = raw.map((text) => ({ text, level: classifyLine(text) }))
  let lines = buf.lines.concat(added)
  let firstError = buf.firstError
  if (firstError === null) {
    const i = added.findIndex((l) => l.level === "error")
    if (i >= 0) firstError = buf.lines.length + i
  }
  let dropped = buf.dropped
  const overflow = lines.length - max
  if (overflow > 0) {
    lines = lines.slice(overflow)
    dropped += overflow
    if (firstError !== null) {
      firstError -= overflow
      if (firstError < 0) {
        const i = lines.findIndex((l) => l.level === "error")
        firstError = i >= 0 ? i : null
      }
    }
  }
  return { lines, dropped, firstError }
}

export function logText(buf: LogBuffer): string {
  return buf.lines.map((l) => l.text).join("\n")
}

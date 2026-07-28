/** Formatting helpers. Every number in the UI goes through one of these. */

export function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`
}

export function fixed(value: number, digits = 3): string {
  return value.toFixed(digits)
}

export function seconds(ms: number): string {
  return `${Math.floor(ms / 1000)}s`
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function classNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(' ')
}

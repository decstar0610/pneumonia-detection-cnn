import type { ReactNode } from 'react'

/** Chart chrome, matched to the app tokens (and to src/viz_style.py). */
export const CHART = {
  grid: '#1F2A35',
  axis: '#647686',
  text: '#93A4B3',
  internal: '#4EA3D6',
  external: '#D1495B',
  reference: '#6C757D',
  caution: '#E9C46A',
  stable: '#2A9D8F',
} as const

export const AXIS_TICK = {
  fill: CHART.axis,
  fontSize: 10,
  fontFamily: 'IBM Plex Mono, monospace',
} as const

export const CHART_MARGIN = { top: 8, right: 12, bottom: 4, left: 0 } as const

/**
 * Draw two curves that have different x-samples on one shared numeric axis
 * without interpolating either of them: interleave the raw points and let
 * `connectNulls` bridge the gaps each series leaves behind.
 */
export function mergeCurves<T>(
  a: readonly T[],
  b: readonly T[],
  x: (point: T) => number,
  y: (point: T) => number,
  names: [string, string],
): Array<Record<string, number>> {
  const rows: Array<Record<string, number>> = []
  for (const point of a) rows.push({ x: x(point), [names[0]]: y(point) })
  for (const point of b) rows.push({ x: x(point), [names[1]]: y(point) })
  return rows.sort((p, q) => (p['x'] ?? 0) - (q['x'] ?? 0))
}

interface TooltipEntry {
  name?: string | number
  value?: number | string | Array<number | string>
  color?: string
  dataKey?: string | number
}

interface ChartTooltipProps {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string | number
  labelName?: string
  digits?: number
}

export function ChartTooltip({ active, payload, label, labelName, digits = 3 }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div className="rounded-sm border border-line-strong bg-ink-900/95 px-2.5 py-2 shadow-lg">
      {labelName !== undefined && typeof label === 'number' && (
        <div className="text-label text-low uppercase">
          {labelName} <span className="tnum text-mid">{label.toFixed(digits)}</span>
        </div>
      )}
      <ul className="mt-1 space-y-0.5">
        {payload.map((entry, index) => (
          <li key={`${String(entry.dataKey)}-${index}`} className="flex items-center gap-2 text-micro">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: entry.color ?? CHART.axis }}
              aria-hidden="true"
            />
            <span className="text-mid">{String(entry.name ?? entry.dataKey)}</span>
            <span className="tnum ml-auto text-hi">
              {typeof entry.value === 'number' ? entry.value.toFixed(digits) : String(entry.value)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** Question-framed caption above every panel — the §7.5 design rule. */
export function Takeaway({ children }: { children: ReactNode }) {
  return <p className="mb-3 text-micro text-mid">{children}</p>
}

export function ChartFrame({ height, children }: { height: number; children: ReactNode }) {
  return (
    <div style={{ height }} className="w-full">
      {children}
    </div>
  )
}

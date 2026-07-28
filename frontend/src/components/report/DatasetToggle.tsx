import { classNames } from '../../lib/format'
import { CHART } from './chart'
import type { DatasetKey } from '../../types/report'

interface DatasetToggleProps {
  value: DatasetKey
  onChange: (value: DatasetKey) => void
  labels: Record<DatasetKey, string>
}

const DOT: Record<DatasetKey, string> = {
  internal: CHART.internal,
  external: CHART.external,
}

export function DatasetToggle({ value, onChange, labels }: DatasetToggleProps) {
  return (
    <div
      className="flex rounded-sm border border-line-strong p-0.5"
      role="group"
      aria-label="Evaluation dataset"
    >
      {(['internal', 'external'] as const).map((key) => (
        <button
          key={key}
          type="button"
          onClick={() => onChange(key)}
          aria-pressed={value === key}
          className={classNames(
            'flex items-center gap-2 rounded-[3px] px-2.5 py-1 text-label uppercase transition-colors duration-150',
            value === key ? 'bg-ink-600 text-hi' : 'text-low hover:text-mid',
          )}
        >
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: DOT[key] }}
            aria-hidden="true"
          />
          {labels[key]}
        </button>
      ))}
    </div>
  )
}

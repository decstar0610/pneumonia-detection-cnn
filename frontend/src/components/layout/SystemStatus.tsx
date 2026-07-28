import { classNames, seconds } from '../../lib/format'
import type { HealthStatus } from '../../hooks/useHealth'

const COPY: Record<HealthStatus['state'], { label: string; dot: string; text: string }> = {
  checking: { label: 'Connecting', dot: 'bg-low', text: 'text-mid' },
  online: { label: 'Service online', dot: 'bg-stable', text: 'text-mid' },
  warming: { label: 'Waking up', dot: 'bg-caution animate-blink', text: 'text-caution' },
  offline: { label: 'Service unreachable', dot: 'bg-alert', text: 'text-alert' },
}

/** Compact liveness readout in the header rail; the Analyze tab carries the long form. */
export function SystemStatus({ health }: { health: HealthStatus }) {
  const copy = COPY[health.state]
  return (
    <div className="flex items-center gap-4">
      {health.modelVersion !== null && (
        <span className="hidden text-label text-low uppercase sm:inline">
          Model <span className="tnum text-mid">v{health.modelVersion}</span>
        </span>
      )}
      <div
        className="flex items-center gap-2"
        role="status"
        aria-live="polite"
        aria-label={`Backend status: ${copy.label}`}
      >
        <span className={classNames('h-1.5 w-1.5 rounded-full', copy.dot)} aria-hidden="true" />
        <span className={classNames('text-micro', copy.text)}>
          {copy.label}
          {health.state === 'warming' && (
            <span className="tnum ml-1.5 text-low">{seconds(health.elapsedMs)}</span>
          )}
        </span>
        {health.state === 'offline' && (
          <button
            type="button"
            onClick={health.retry}
            className="rounded-sm border border-line-strong px-1.5 py-0.5 text-label text-mid uppercase transition-colors duration-150 hover:border-clinical-lit hover:text-hi"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  )
}

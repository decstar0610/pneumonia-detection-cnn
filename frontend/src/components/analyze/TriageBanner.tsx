import { classNames } from '../../lib/format'
import { TONE_CLASSES, zoneMeta } from '../../lib/triage'
import type { PredictionResponse } from '../../types/api'

/**
 * The triage disposition. Abstention gets the widest, most deliberate treatment
 * in the whole UI — a model that refuses to decide on a borderline case is the
 * product working, so it is never styled as an error.
 */
export function TriageBanner({ result }: { result: PredictionResponse }) {
  const meta = zoneMeta(result.triage_zone)
  const c = TONE_CLASSES[meta.tone]
  const abstained = result.is_uncertain

  return (
    <div className={classNames('rounded-panel border px-4 py-3', c.border, c.bg)}>
      <div className="flex items-center gap-2">
        <span className={classNames('h-1.5 w-1.5 rounded-full', c.dot)} aria-hidden="true" />
        <p className={classNames('text-label font-medium uppercase', c.text)}>
          {abstained ? 'Escalated — model abstained' : meta.label}
        </p>
      </div>
      <p className="mt-2 text-body text-hi">{result.recommendation}</p>
      <p className="mt-1.5 text-micro text-mid">{meta.blurb}</p>
    </div>
  )
}

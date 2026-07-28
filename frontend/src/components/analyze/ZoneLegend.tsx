import { TRIAGE_ZONES } from '../../types/api'
import { TONE_CLASSES, zoneMeta } from '../../lib/triage'
import { classNames } from '../../lib/format'

/** Shown before any study is loaded: what the three dispositions actually mean. */
export function ZoneLegend() {
  return (
    <dl className="space-y-3">
      {TRIAGE_ZONES.map((zone) => {
        const meta = zoneMeta(zone)
        const c = TONE_CLASSES[meta.tone]
        return (
          <div key={zone} className="flex gap-3">
            <span
              className={classNames('mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full', c.dot)}
              aria-hidden="true"
            />
            <div>
              <dt className={classNames('text-micro font-medium', c.text)}>{meta.label}</dt>
              <dd className="mt-0.5 text-micro text-low">{meta.blurb}</dd>
            </div>
          </div>
        )
      })}
    </dl>
  )
}

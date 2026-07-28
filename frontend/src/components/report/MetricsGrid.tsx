import { AnimatedNumber } from '../ui/AnimatedNumber'
import { classNames } from '../../lib/format'
import type { DatasetKey, DatasetResult } from '../../types/report'

interface MetricsGridProps {
  active: DatasetKey
  results: Record<DatasetKey, DatasetResult>
  labels: Record<DatasetKey, string>
}

const METRICS = [
  { key: 'sensitivity', label: 'Sensitivity', note: 'Primary metric — missed pneumonia is the costly error' },
  { key: 'specificity', label: 'Specificity', note: 'Share of non-pneumonia studies left alone' },
  { key: 'precision', label: 'Precision', note: 'Of the studies flagged, how many were pneumonia' },
  { key: 'roc_auc', label: 'ROC-AUC', note: 'Ranking quality, threshold-independent' },
  { key: 'pr_auc', label: 'PR-AUC', note: 'Ranking quality under class imbalance' },
] as const

export function MetricsGrid({ active, results, labels }: MetricsGridProps) {
  const other: DatasetKey = active === 'internal' ? 'external' : 'internal'

  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-panel border border-line bg-line md:grid-cols-5">
      {METRICS.map((metric) => {
        const value = results[active][metric.key]
        const reference = results[other][metric.key]
        const delta = value - reference
        const worse = delta < -0.005
        return (
          <div key={metric.key} className="bg-ink-800 p-3.5">
            <dt className="text-label text-low uppercase">{metric.label}</dt>
            <dd>
              <div className="mt-1.5 text-metric font-medium text-hi">
                <AnimatedNumber value={value} digits={3} />
              </div>
              <div className="mt-1.5 flex items-baseline gap-1.5 text-micro">
                <span
                  className={classNames(
                    'tnum',
                    worse ? 'text-alert' : delta > 0.005 ? 'text-stable' : 'text-low',
                  )}
                >
                  {delta >= 0 ? '+' : '−'}
                  {Math.abs(delta).toFixed(3)}
                </span>
                <span className="text-low">vs {labels[other].toLowerCase()}</span>
              </div>
              <p className="mt-2 text-micro leading-snug text-low">{metric.note}</p>
            </dd>
          </div>
        )
      })}
    </dl>
  )
}

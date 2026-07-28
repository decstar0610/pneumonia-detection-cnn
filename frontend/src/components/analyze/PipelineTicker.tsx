import { classNames } from '../../lib/format'
import { COLD_START_HINT_MS } from '../../lib/api'

/** The server-side pipeline, in order, with the elapsed time each stage is expected by. */
const STAGES = [
  { label: 'Transferring study', at: 0 },
  { label: 'Preprocess · 224² · ImageNet norm', at: 350 },
  { label: 'DenseNet121 backbone · onnxruntime', at: 800 },
  { label: 'Calibrated head + Grad-CAM', at: 1500 },
] as const

interface PipelineTickerProps {
  elapsedMs: number
  /** Backend is known to be asleep, so stage timings are meaningless. */
  waking: boolean
}

export function PipelineTicker({ elapsedMs, waking }: PipelineTickerProps) {
  const cold = waking || elapsedMs > COLD_START_HINT_MS
  const activeIndex = cold
    ? 0
    : STAGES.reduce((found, stage, index) => (elapsedMs >= stage.at ? index : found), 0)

  return (
    <ol className="space-y-1.5" aria-label="Analysis pipeline">
      {STAGES.map((stage, index) => {
        const done = !cold && index < activeIndex
        const active = !cold && index === activeIndex
        return (
          <li key={stage.label} className="flex items-center gap-2.5 text-micro">
            <span
              className={classNames(
                'h-1.5 w-1.5 shrink-0 rounded-full',
                done && 'bg-clinical',
                active && 'bg-clinical-lit animate-blink',
                !done && !active && 'bg-ink-600',
              )}
              aria-hidden="true"
            />
            <span
              className={classNames(
                done && 'text-mid',
                active && 'text-hi',
                !done && !active && 'text-low',
              )}
            >
              {stage.label}
            </span>
          </li>
        )
      })}
      {cold && (
        <li className="flex items-center gap-2.5 pt-1 text-micro text-caution">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-caution animate-blink" aria-hidden="true" />
          Waiting on a sleeping container — the first request cold-starts it.
        </li>
      )}
    </ol>
  )
}

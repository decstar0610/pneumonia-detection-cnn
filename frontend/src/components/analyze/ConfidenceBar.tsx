import { motion } from 'framer-motion'

import { AnimatedNumber } from '../ui/AnimatedNumber'
import { classNames } from '../../lib/format'
import { transitions } from '../../lib/motion'
import { TONE_CLASSES, type Tone } from '../../lib/triage'

interface ConfidenceBarProps {
  label: string
  value: number
  tone: Tone
  hint?: string
  /** Thinner variant used for the per-class split. */
  compact?: boolean
}

/** A bar that fills to its value once, then holds. No pulsing, no shimmer. */
export function ConfidenceBar({ label, value, tone, hint, compact = false }: ConfidenceBarProps) {
  const c = TONE_CLASSES[tone]
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className={classNames('text-micro', compact ? 'text-mid' : 'text-hi')}>{label}</span>
        <AnimatedNumber
          value={value}
          scale={100}
          digits={compact ? 1 : 2}
          suffix="%"
          className={classNames(compact ? 'text-micro text-mid' : 'text-body font-medium text-hi')}
        />
      </div>
      <div
        className={classNames('mt-1.5 w-full overflow-hidden rounded-full bg-ink-700', compact ? 'h-1' : 'h-1.5')}
        role="meter"
        aria-valuenow={Math.round(value * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <motion.div
          className={classNames('h-full rounded-full', c.fill)}
          initial={{ scaleX: 0 }}
          animate={{ scaleX: value }}
          transition={{ ...transitions.entrance, duration: 0.6 }}
          style={{ transformOrigin: 'left' }}
        />
      </div>
      {hint !== undefined && <p className="mt-1 text-micro text-low">{hint}</p>}
    </div>
  )
}

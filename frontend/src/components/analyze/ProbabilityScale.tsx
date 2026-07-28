import { motion } from 'framer-motion'

import { fixed } from '../../lib/format'
import { transitions } from '../../lib/motion'

interface ProbabilityScaleProps {
  /** Calibrated P(pneumonia) from the API. */
  probability: number
  /** Calibrated decision threshold t_cal the server actually compared against. */
  threshold: number
}

/**
 * Where this study landed on the 0→1 calibrated probability axis, relative to the
 * tuned decision threshold. This is the §4 story made literal: the threshold is
 * not 0.5, and you can see exactly how far the case sits from it.
 */
export function ProbabilityScale({ probability, threshold }: ProbabilityScaleProps) {
  const positive = probability >= threshold
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-label text-low uppercase">Calibrated P(pneumonia)</span>
        <span className="tnum text-micro text-mid">{fixed(probability, 4)}</span>
      </div>

      <div className="relative mt-2 h-6">
        <div className="absolute inset-x-0 top-2 h-1.5 overflow-hidden rounded-full bg-ink-700">
          <motion.div
            className={positive ? 'h-full bg-alert' : 'h-full bg-stable'}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: probability }}
            transition={{ ...transitions.entrance, duration: 0.6 }}
            style={{ transformOrigin: 'left' }}
          />
        </div>

        {/* Decision threshold marker */}
        <div
          className="absolute top-0 h-5 w-px bg-hi/70"
          style={{ left: `${threshold * 100}%` }}
          aria-hidden="true"
        />
        <div
          className="absolute top-5 -translate-x-1/2 text-label whitespace-nowrap text-low"
          style={{ left: `${threshold * 100}%` }}
        >
          <span className="tnum">{fixed(threshold, 3)}</span> threshold
        </div>
      </div>

      <div className="mt-5 flex justify-between text-label text-low tnum">
        <span>0.0</span>
        <span>0.5</span>
        <span>1.0</span>
      </div>
    </div>
  )
}

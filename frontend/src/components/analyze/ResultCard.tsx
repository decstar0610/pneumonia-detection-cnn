import { motion } from 'framer-motion'

import { AnimatedNumber } from '../ui/AnimatedNumber'
import { Readout } from '../ui/Readout'
import { StatusPill } from '../ui/StatusPill'
import { ConfidenceBar } from './ConfidenceBar'
import { ProbabilityScale } from './ProbabilityScale'
import { TriageBanner } from './TriageBanner'
import { fixed } from '../../lib/format'
import { transitions } from '../../lib/motion'
import { zoneMeta } from '../../lib/triage'
import type { PredictionResponse } from '../../types/api'

interface ResultCardProps {
  result: PredictionResponse
  /** Client-observed round trip, reported as such — not as server inference time. */
  roundTripMs: number
}

export function ResultCard({ result, roundTripMs }: ResultCardProps) {
  const meta = zoneMeta(result.triage_zone)

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={transitions.entrance}
      className="space-y-5"
    >
      <p className="sr-only" role="status">
        {`${result.prediction}, ${meta.label}, calibrated confidence ${(result.calibrated_confidence * 100).toFixed(1)} percent. ${result.recommendation}`}
      </p>

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-label text-low uppercase">Model call</div>
          <div className="mt-1 text-display font-semibold tracking-tight text-hi">
            {result.prediction}
          </div>
        </div>
        <StatusPill label={meta.label} tone={meta.tone} pulse={result.is_uncertain} />
      </div>

      <ConfidenceBar
        label={`Calibrated confidence in "${result.prediction}"`}
        value={result.calibrated_confidence}
        tone={meta.tone}
        hint="Temperature-scaled (T = 0.72) so the number means what it says."
      />

      <ProbabilityScale probability={result.probability.pneumonia} threshold={result.threshold_used} />

      <div className="grid grid-cols-3 gap-4 border-t border-line pt-4">
        <Readout label="P(normal)">
          <AnimatedNumber value={result.probability.normal} digits={4} />
        </Readout>
        <Readout label="P(pneumonia)">
          <AnimatedNumber value={result.probability.pneumonia} digits={4} />
        </Readout>
        <Readout label="Round trip" hint={`threshold ${fixed(result.threshold_used, 3)}`}>
          <AnimatedNumber value={roundTripMs / 1000} digits={2} suffix="s" />
        </Readout>
      </div>

      <TriageBanner result={result} />

      <p className="text-micro text-low">{result.disclaimer}</p>
    </motion.div>
  )
}

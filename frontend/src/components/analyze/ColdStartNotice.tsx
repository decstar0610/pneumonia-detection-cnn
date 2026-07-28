import { AnimatePresence, motion } from 'framer-motion'

import { clamp, seconds } from '../../lib/format'
import { transitions } from '../../lib/motion'
import type { HealthStatus } from '../../hooks/useHealth'

/** Typical Render free-tier cold start, used only to pace the progress hint. */
const EXPECTED_WAKE_MS = 55_000

/**
 * The backend sleeps when idle. Rather than hiding that behind a spinner, we
 * name it, show elapsed time, and keep the UI usable while it wakes.
 */
export function ColdStartNotice({ health }: { health: HealthStatus }) {
  const visible = health.state === 'warming' || health.state === 'offline'
  const progress = clamp(health.elapsedMs / EXPECTED_WAKE_MS, 0.04, 0.97)
  const offline = health.state === 'offline'

  return (
    <AnimatePresence initial={false}>
      {visible && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={transitions.state}
          className="overflow-hidden"
        >
          <div
            className={`hairline rounded-panel px-4 py-3 ${offline ? 'border-alert/40 bg-alert/5' : 'border-caution/35 bg-caution/5'}`}
            role="status"
            aria-live="polite"
          >
            <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
              <p className={`text-micro font-medium ${offline ? 'text-alert' : 'text-caution'}`}>
                {offline ? 'Analysis service unreachable' : 'Analysis service is waking up'}
              </p>
              <p className="tnum text-label text-low uppercase">{seconds(health.elapsedMs)} elapsed</p>
            </div>
            <p className="mt-1 text-micro text-mid">
              {offline ? (
                <>
                  The container did not answer within the wake-up window. Uploads will fail until it
                  responds — retry from the header, or check the service status.
                </>
              ) : (
                <>
                  It runs on a free tier that sleeps when idle, so the first request cold-starts the
                  container (~50s). You can pick an image now; analysis will run as soon as it
                  answers.
                </>
              )}
            </p>
            {!offline && (
              <div className="mt-2.5 h-0.5 w-full overflow-hidden rounded-full bg-ink-700">
                <motion.div
                  className="h-full bg-caution"
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: progress }}
                  transition={transitions.state}
                  style={{ transformOrigin: 'left' }}
                />
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

/**
 * Backend liveness with cold-start awareness.
 *
 * The API sleeps on Render's free tier, so the first request of the day can take
 * ~50s. A failed probe therefore means "probably waking up", not "broken" — we
 * keep retrying with backoff for a budget, then declare it offline.
 */
import { useCallback, useEffect, useState } from 'react'

import { getHealth } from '../lib/api'

export type HealthState = 'checking' | 'online' | 'warming' | 'offline'

/** How long we're willing to wait for a sleeping container before giving up. */
const WARMUP_BUDGET_MS = 150_000
const BACKOFF_MS = [1_500, 3_000, 5_000, 8_000, 10_000] as const

export interface HealthStatus {
  state: HealthState
  modelVersion: string | null
  /** Milliseconds since the current probing cycle started (drives the counter). */
  elapsedMs: number
  retry: () => void
}

export function useHealth(): HealthStatus {
  const [state, setState] = useState<HealthState>('checking')
  const [modelVersion, setModelVersion] = useState<string | null>(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [cycle, setCycle] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    const startedAt = Date.now()
    let cancelled = false

    setState('checking')
    setElapsedMs(0)

    const ticker = window.setInterval(() => {
      if (!cancelled) setElapsedMs(Date.now() - startedAt)
    }, 500)

    const sleep = (ms: number) =>
      new Promise<void>((resolve) => {
        const timer = window.setTimeout(resolve, ms)
        controller.signal.addEventListener(
          'abort',
          () => {
            window.clearTimeout(timer)
            resolve()
          },
          { once: true },
        )
      })

    void (async () => {
      for (let attempt = 0; !cancelled; attempt += 1) {
        try {
          const health = await getHealth(controller.signal)
          if (cancelled) return
          setModelVersion(health.model_version)
          setState(health.status === 'ok' ? 'online' : 'offline')
          return
        } catch {
          if (cancelled || controller.signal.aborted) return
          if (Date.now() - startedAt > WARMUP_BUDGET_MS) {
            setState('offline')
            return
          }
          setState('warming')
          await sleep(BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)] ?? 8_000)
        }
      }
    })()

    return () => {
      cancelled = true
      controller.abort()
      window.clearInterval(ticker)
    }
  }, [cycle])

  const retry = useCallback(() => setCycle((value) => value + 1), [])

  return { state, modelVersion, elapsedMs, retry }
}

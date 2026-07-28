/**
 * The upload → predict flow as an explicit state machine.
 *
 * Selecting a study starts analysis immediately (one gesture, like a real
 * viewer), but every stage is separately observable so the UI can show a
 * distinct rejection, analyzing, error and result state instead of a spinner.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, describeError, predictImage, validateUpload } from '../lib/api'
import type { PredictionResponse } from '../types/api'

export type Phase = 'idle' | 'analyzing' | 'done' | 'error'

export interface PredictionFlow {
  phase: Phase
  file: File | null
  /** Object URL for the selected study; revoked when replaced. */
  previewUrl: string | null
  result: PredictionResponse | null
  /** API/transport failure for the last run. */
  error: string | null
  /** Client-side gate message (wrong type, too large) — not an API failure. */
  rejection: string | null
  /** Milliseconds the current (or last) analysis has been running. */
  elapsedMs: number
  select: (file: File) => void
  reject: (reason: string) => void
  rerun: () => void
  clear: () => void
}

export function usePrediction(): PredictionFlow {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [rejection, setRejection] = useState<string | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [elapsedMs, setElapsedMs] = useState(0)
  const [runToken, setRunToken] = useState(0)
  const objectUrl = useRef<string | null>(null)

  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    }
  }, [])

  const select = useCallback((next: File) => {
    const gate = validateUpload(next)
    if (!gate.ok) {
      setRejection(gate.reason)
      return
    }
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = URL.createObjectURL(next)

    setRejection(null)
    setError(null)
    setResult(null)
    setPreviewUrl(objectUrl.current)
    setFile(next)
    setRunToken((token) => token + 1)
  }, [])

  const reject = useCallback((reason: string) => setRejection(reason), [])

  const rerun = useCallback(() => {
    if (!file) return
    setError(null)
    setResult(null)
    setRunToken((token) => token + 1)
  }, [file])

  const clear = useCallback(() => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = null
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setRejection(null)
    setElapsedMs(0)
    setPhase('idle')
  }, [])

  // Runs whenever a new study is selected or the user asks for a re-run.
  useEffect(() => {
    if (!file || runToken === 0) return

    const controller = new AbortController()
    const startedAt = Date.now()
    let cancelled = false

    setPhase('analyzing')
    setElapsedMs(0)
    const ticker = window.setInterval(() => {
      if (!cancelled) setElapsedMs(Date.now() - startedAt)
    }, 100)

    void (async () => {
      try {
        const response = await predictImage(file, controller.signal)
        if (cancelled) return
        setResult(response)
        setPhase('done')
      } catch (caught) {
        // An abort is our own cleanup (StrictMode remount, replaced study) — not an error.
        if (cancelled || (caught instanceof ApiError && caught.kind === 'aborted')) return
        setError(describeError(caught))
        setPhase('error')
      } finally {
        if (!cancelled) setElapsedMs(Date.now() - startedAt)
      }
    })()

    return () => {
      cancelled = true
      controller.abort()
      window.clearInterval(ticker)
    }
  }, [file, runToken])

  return { phase, file, previewUrl, result, error, rejection, elapsedMs, select, reject, rerun, clear }
}

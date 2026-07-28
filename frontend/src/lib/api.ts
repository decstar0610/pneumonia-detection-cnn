/**
 * Typed client for the PneumoScan FastAPI backend.
 *
 * Only two endpoints exist and both are real: `GET /health` and `POST /predict`
 * (multipart, field name "file"). Errors are normalised into `ApiError` so the UI
 * can distinguish "backend asleep" from "you uploaded a PDF".
 */
import {
  parseHealthResponse,
  parsePredictionResponse,
  isContractError,
  type HealthResponse,
  type PredictionResponse,
} from '../types/api'

const RAW_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
export const API_BASE = RAW_BASE.replace(/\/+$/, '')

/** Free-tier Render sleeps; a cold container needs ~50s to answer the first call. */
export const COLD_START_HINT_MS = 6_000
const HEALTH_TIMEOUT_MS = 8_000
const PREDICT_TIMEOUT_MS = 120_000

/** Content types api/main.py accepts (`ALLOWED`); anything else is rejected with 422. */
export const ACCEPTED_MIME = ['image/jpeg', 'image/jpg', 'image/png'] as const
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024

export type ApiErrorKind = 'network' | 'timeout' | 'http' | 'contract' | 'aborted'

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number | undefined

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
  }
}

interface RequestOptions {
  signal?: AbortSignal
  timeoutMs?: number
}

/** fetch + a timeout that is distinguishable from a caller-initiated abort. */
async function request(path: string, init: RequestInit, options: RequestOptions): Promise<Response> {
  const timeoutMs = options.timeoutMs ?? HEALTH_TIMEOUT_MS
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  const onOuterAbort = () => controller.abort()
  options.signal?.addEventListener('abort', onOuterAbort, { once: true })

  try {
    return await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal })
  } catch (error) {
    if (options.signal?.aborted) throw new ApiError('aborted', 'Request cancelled.')
    if (controller.signal.aborted) {
      throw new ApiError('timeout', `The backend did not respond within ${Math.round(timeoutMs / 1000)}s.`)
    }
    throw new ApiError(
      'network',
      error instanceof Error ? error.message : 'Could not reach the backend.',
    )
  } finally {
    window.clearTimeout(timer)
    options.signal?.removeEventListener('abort', onOuterAbort)
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'string') return detail
    }
  } catch {
    /* non-JSON error body — fall through to the generic message */
  }
  return `Request failed (HTTP ${response.status}).`
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await request(
    '/health',
    { method: 'GET' },
    signal ? { signal, timeoutMs: HEALTH_TIMEOUT_MS } : { timeoutMs: HEALTH_TIMEOUT_MS },
  )
  if (!response.ok) {
    throw new ApiError('http', await readErrorDetail(response), response.status)
  }
  try {
    return parseHealthResponse(await response.json())
  } catch (error) {
    throw new ApiError('contract', isContractError(error) ? error.message : 'Malformed health response.')
  }
}

export async function predictImage(file: File, signal?: AbortSignal): Promise<PredictionResponse> {
  const form = new FormData()
  form.append('file', file) // field name must stay "file" — api/main.py signature

  const response = await request(
    '/predict',
    { method: 'POST', body: form },
    signal ? { signal, timeoutMs: PREDICT_TIMEOUT_MS } : { timeoutMs: PREDICT_TIMEOUT_MS },
  )
  if (!response.ok) {
    throw new ApiError('http', await readErrorDetail(response), response.status)
  }
  try {
    return parsePredictionResponse(await response.json())
  } catch (error) {
    throw new ApiError(
      'contract',
      isContractError(error) ? error.message : 'Malformed prediction response.',
    )
  }
}

export type FileRejection = { ok: false; reason: string }
export type FileAcceptance = { ok: true }

/** Client-side gate so obvious mistakes never cost a 50s round trip. */
export function validateUpload(file: File): FileAcceptance | FileRejection {
  const named = file.name || 'This file'
  if (!(ACCEPTED_MIME as readonly string[]).includes(file.type)) {
    const kind = file.type || 'unknown type'
    return { ok: false, reason: `${named} is ${kind}. Upload a JPEG or PNG chest X-ray.` }
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      reason: `${named} is ${(file.size / 1024 / 1024).toFixed(1)} MB — the limit is ${MAX_UPLOAD_BYTES / 1024 / 1024} MB.`,
    }
  }
  if (file.size === 0) {
    return { ok: false, reason: `${named} is empty.` }
  }
  return { ok: true }
}

/** Human-readable message for any error the flow can produce. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.kind) {
      case 'timeout':
        return `${error.message} It may still be waking up — try again in a moment.`
      case 'network':
        return `Could not reach the analysis service at ${API_BASE}.`
      case 'contract':
        return `The service returned an unexpected response. ${error.message}`
      default:
        return error.message
    }
  }
  return error instanceof Error ? error.message : 'Something went wrong.'
}

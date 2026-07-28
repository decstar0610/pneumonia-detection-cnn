/**
 * The PneumoScan API contract (PRD §7.2), mirrored exactly from api/inference.py.
 *
 * Nothing here is invented: every field is produced by `predict_image()` on the
 * server. The runtime parsers below mean a contract drift surfaces as a clear
 * error instead of an `undefined` rendered into the UI.
 */

export const TRIAGE_ZONES = ['routine', 'urgent_review', 'needs_human_review'] as const
export type TriageZone = (typeof TRIAGE_ZONES)[number]

export const PREDICTIONS = ['Normal', 'Pneumonia'] as const
export type PredictionLabel = (typeof PREDICTIONS)[number]

export interface ClassProbabilities {
  readonly normal: number
  readonly pneumonia: number
}

export interface PredictionResponse {
  readonly prediction: PredictionLabel
  readonly triage_zone: TriageZone
  /** Confidence in `prediction` — NOT P(pneumonia). Already temperature-scaled. */
  readonly calibrated_confidence: number
  readonly is_uncertain: boolean
  /** Calibrated class probabilities; `pneumonia` is the model's positive class. */
  readonly probability: ClassProbabilities
  /** Calibrated decision threshold t_cal (0.2653), comparable to probability.pneumonia. */
  readonly threshold_used: number
  /** Ready-to-render data URI: "data:image/png;base64,…" (224×224, pre-blended). */
  readonly gradcam_overlay: string
  readonly recommendation: string
  readonly disclaimer: string
}

export interface HealthResponse {
  readonly status: string
  readonly model_version: string
}

/** FastAPI's error shape: `{"detail": "…"}` (HTTPException) */
export interface ApiErrorBody {
  readonly detail?: string
}

// --- runtime validation -----------------------------------------------------

class ContractError extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function requireNumber(source: Record<string, unknown>, key: string): number {
  const value = source[key]
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ContractError(`Expected numeric "${key}" in API response.`)
  }
  return value
}

function requireString(source: Record<string, unknown>, key: string): string {
  const value = source[key]
  if (typeof value !== 'string') {
    throw new ContractError(`Expected string "${key}" in API response.`)
  }
  return value
}

function requireOneOf<T extends string>(
  source: Record<string, unknown>,
  key: string,
  allowed: readonly T[],
): T {
  const value = requireString(source, key)
  if (!(allowed as readonly string[]).includes(value)) {
    throw new ContractError(`Unexpected "${key}" value: ${value}`)
  }
  return value as T
}

export function parsePredictionResponse(raw: unknown): PredictionResponse {
  if (!isRecord(raw)) throw new ContractError('Malformed prediction response.')
  const probability = raw['probability']
  if (!isRecord(probability)) throw new ContractError('Missing "probability" in response.')

  return {
    prediction: requireOneOf(raw, 'prediction', PREDICTIONS),
    triage_zone: requireOneOf(raw, 'triage_zone', TRIAGE_ZONES),
    calibrated_confidence: requireNumber(raw, 'calibrated_confidence'),
    is_uncertain: raw['is_uncertain'] === true,
    probability: {
      normal: requireNumber(probability, 'normal'),
      pneumonia: requireNumber(probability, 'pneumonia'),
    },
    threshold_used: requireNumber(raw, 'threshold_used'),
    gradcam_overlay: requireString(raw, 'gradcam_overlay'),
    recommendation: requireString(raw, 'recommendation'),
    disclaimer: requireString(raw, 'disclaimer'),
  }
}

export function parseHealthResponse(raw: unknown): HealthResponse {
  if (!isRecord(raw)) throw new ContractError('Malformed health response.')
  return {
    status: requireString(raw, 'status'),
    model_version: requireString(raw, 'model_version'),
  }
}

export function isContractError(error: unknown): error is ContractError {
  return error instanceof ContractError
}

/**
 * Triage-zone presentation — the single source of truth for how each of the
 * three §7.2 zones looks and reads. `needs_human_review` is deliberately the
 * most prominent state: the model abstaining is the product, not a failure.
 */
import type { TriageZone } from '../types/api'

export type Tone = 'stable' | 'alert' | 'caution' | 'clinical' | 'neutral'

export interface ZoneMeta {
  readonly label: string
  readonly tone: Tone
  /** One line explaining what the zone means operationally. */
  readonly blurb: string
}

export const ZONE_META: Record<TriageZone, ZoneMeta> = {
  routine: {
    label: 'Routine',
    tone: 'stable',
    blurb: 'Confidently below the decision threshold — no urgent finding to escalate.',
  },
  urgent_review: {
    label: 'Urgent review',
    tone: 'alert',
    blurb: 'Confidently above the decision threshold — queue ahead of routine studies.',
  },
  needs_human_review: {
    label: 'Needs human review',
    tone: 'caution',
    blurb:
      'Inside the abstention band, where the model is not reliable enough to decide. It hands the case to a radiologist by design.',
  },
}

export function zoneMeta(zone: TriageZone): ZoneMeta {
  return ZONE_META[zone]
}

/** Static class strings per tone — written out so Tailwind's scanner sees them. */
export const TONE_CLASSES: Record<
  Tone,
  { text: string; bg: string; border: string; dot: string; fill: string }
> = {
  stable: {
    text: 'text-stable',
    bg: 'bg-stable/10',
    border: 'border-stable/35',
    dot: 'bg-stable',
    fill: 'bg-stable',
  },
  alert: {
    text: 'text-alert',
    bg: 'bg-alert/10',
    border: 'border-alert/35',
    dot: 'bg-alert',
    fill: 'bg-alert',
  },
  caution: {
    text: 'text-caution',
    bg: 'bg-caution/10',
    border: 'border-caution/35',
    dot: 'bg-caution',
    fill: 'bg-caution',
  },
  clinical: {
    text: 'text-clinical-lit',
    bg: 'bg-clinical-lit/10',
    border: 'border-clinical-lit/35',
    dot: 'bg-clinical-lit',
    fill: 'bg-clinical-lit',
  },
  neutral: {
    text: 'text-mid',
    bg: 'bg-ink-700',
    border: 'border-line-strong',
    dot: 'bg-neutral',
    fill: 'bg-neutral',
  },
}

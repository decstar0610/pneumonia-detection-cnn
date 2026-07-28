import { classNames } from '../../lib/format'
import { TONE_CLASSES, type Tone } from '../../lib/triage'

interface StatusPillProps {
  label: string
  tone: Tone
  /** Slow ring pulse — used for the escalation state, never for errors. */
  pulse?: boolean
  size?: 'sm' | 'md'
  className?: string
}

export function StatusPill({ label, tone, pulse = false, size = 'md', className }: StatusPillProps) {
  const c = TONE_CLASSES[tone]
  return (
    <span
      className={classNames(
        'inline-flex items-center gap-2 rounded-full border font-medium whitespace-nowrap',
        size === 'sm' ? 'px-2 py-0.5 text-label uppercase' : 'px-3 py-1 text-micro',
        c.bg,
        c.border,
        c.text,
        pulse && 'animate-ring',
        className,
      )}
    >
      <span className={classNames('h-1.5 w-1.5 shrink-0 rounded-full', c.dot)} aria-hidden="true" />
      {label}
    </span>
  )
}

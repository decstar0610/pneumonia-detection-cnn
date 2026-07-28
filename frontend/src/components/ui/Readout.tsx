import type { ReactNode } from 'react'

import { classNames } from '../../lib/format'

interface ReadoutProps {
  label: string
  children: ReactNode
  hint?: string
  className?: string
}

/** Label-over-value instrument readout. Values are always mono + tabular. */
export function Readout({ label, children, hint, className }: ReadoutProps) {
  return (
    <div className={classNames('min-w-0', className)}>
      <div className="text-label text-low uppercase">{label}</div>
      <div className="mt-1 text-body text-hi tnum">{children}</div>
      {hint !== undefined && <div className="mt-0.5 text-micro text-low">{hint}</div>}
    </div>
  )
}

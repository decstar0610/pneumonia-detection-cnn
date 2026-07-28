import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

import { classNames } from '../../lib/format'
import { panelVariants } from '../../lib/motion'

interface PanelProps {
  /** Uppercase instrument label shown in the panel's header rail. */
  label?: string
  /** Right-aligned header content (counts, toggles, legends). */
  aside?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
  /** Panels animate in by default; disable inside already-animating regions. */
  animate?: boolean
}

export function Panel({
  label,
  aside,
  children,
  className,
  bodyClassName,
  animate = true,
}: PanelProps) {
  return (
    <motion.section
      variants={animate ? panelVariants : undefined}
      className={classNames(
        'hairline rounded-panel bg-ink-800/70 backdrop-blur-[1px]',
        className,
      )}
    >
      {(label !== undefined || aside != null) && (
        <header className="flex min-h-9 items-center justify-between gap-3 border-b border-line px-4">
          {label !== undefined && (
            <h2 className="text-label font-medium text-low uppercase">{label}</h2>
          )}
          {aside}
        </header>
      )}
      <div className={classNames('p-4', bodyClassName)}>{children}</div>
    </motion.section>
  )
}

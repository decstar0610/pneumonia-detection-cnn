import { animate } from 'framer-motion'
import { useEffect, useRef } from 'react'

import { classNames } from '../../lib/format'
import { EASE_ENTRANCE, useMotionSafe } from '../../lib/motion'

interface AnimatedNumberProps {
  value: number
  /** Multiplier applied before formatting (100 turns 0.94 into 94.2). */
  scale?: number
  digits?: number
  suffix?: string
  duration?: number
  className?: string
}

/**
 * Counts up to `value` in tabular mono, writing straight to the DOM node so a
 * running number never triggers a React re-render or shifts layout.
 */
export function AnimatedNumber({
  value,
  scale = 1,
  digits = 1,
  suffix = '',
  duration = 0.6,
  className,
}: AnimatedNumberProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const previous = useRef(0)
  const motionSafe = useMotionSafe()

  useEffect(() => {
    const node = ref.current
    if (!node) return
    const render = (raw: number) => {
      node.textContent = `${(raw * scale).toFixed(digits)}${suffix}`
    }

    if (!motionSafe) {
      render(value)
      previous.current = value
      return
    }

    const controls = animate(previous.current, value, {
      duration,
      ease: EASE_ENTRANCE,
      onUpdate: render,
    })
    previous.current = value
    return () => controls.stop()
  }, [value, scale, digits, suffix, duration, motionSafe])

  return (
    <span ref={ref} className={classNames('tnum', className)}>
      {`${(value * scale).toFixed(digits)}${suffix}`}
    </span>
  )
}

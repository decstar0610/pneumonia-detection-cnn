/**
 * Motion tokens. Everything eased, nothing bouncy, nothing over 400ms except
 * the two continuous loops (scan-line, escalation ring) defined in index.css.
 */
import { useReducedMotion, type Transition, type Variants } from 'framer-motion'

export const DURATION = {
  fast: 0.15,
  base: 0.22,
  slow: 0.38,
} as const

/** Entrances decelerate; state changes use the standard curve. */
export const EASE_ENTRANCE = [0.22, 1, 0.36, 1] as const
export const EASE_STATE = [0.4, 0, 0.2, 1] as const

export const transitions = {
  entrance: { duration: DURATION.slow, ease: EASE_ENTRANCE } satisfies Transition,
  state: { duration: DURATION.base, ease: EASE_STATE } satisfies Transition,
  fast: { duration: DURATION.fast, ease: EASE_STATE } satisfies Transition,
}

export const panelVariants: Variants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: transitions.entrance },
}

export const staggerParent: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06, delayChildren: 0.02 } },
}

/**
 * `true` when the viewer has asked for reduced motion. Components use it to
 * render final states directly instead of animating into them.
 */
export function useMotionSafe(): boolean {
  return !useReducedMotion()
}

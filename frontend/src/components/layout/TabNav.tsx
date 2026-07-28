import { motion } from 'framer-motion'
import { useRef } from 'react'

import { classNames } from '../../lib/format'
import { transitions } from '../../lib/motion'

export const TABS = [
  { id: 'analyze', label: 'Analyze' },
  { id: 'report', label: 'Model report' },
] as const

export type TabId = (typeof TABS)[number]['id']

interface TabNavProps {
  active: TabId
  onChange: (tab: TabId) => void
}

/** ARIA tablist with roving focus: ← → move between tabs, Home/End jump. */
export function TabNav({ active, onChange }: TabNavProps) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({})

  function onKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const index = TABS.findIndex((tab) => tab.id === active)
    let next = index
    if (event.key === 'ArrowRight') next = (index + 1) % TABS.length
    else if (event.key === 'ArrowLeft') next = (index - 1 + TABS.length) % TABS.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = TABS.length - 1
    else return

    event.preventDefault()
    const target = TABS[next]
    if (!target) return
    onChange(target.id)
    refs.current[target.id]?.focus()
  }

  return (
    <div role="tablist" aria-label="Sections" onKeyDown={onKeyDown} className="flex gap-1">
      {TABS.map((tab) => {
        const selected = tab.id === active
        return (
          <button
            key={tab.id}
            ref={(node) => {
              refs.current[tab.id] = node
            }}
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={selected}
            aria-controls={`panel-${tab.id}`}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(tab.id)}
            className={classNames(
              'relative px-3 py-2.5 text-micro font-medium transition-colors duration-150',
              selected ? 'text-hi' : 'text-low hover:text-mid',
            )}
          >
            {tab.label}
            {selected && (
              <motion.span
                layoutId="tab-underline"
                transition={transitions.state}
                className="absolute inset-x-2 -bottom-px h-px bg-clinical-lit"
              />
            )}
          </button>
        )
      })}
    </div>
  )
}

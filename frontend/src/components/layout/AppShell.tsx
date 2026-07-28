import type { ReactNode } from 'react'

import { Wordmark } from './Wordmark'
import { SystemStatus } from './SystemStatus'
import { TabNav, type TabId } from './TabNav'
import { DisclaimerBar } from './DisclaimerBar'
import { Footer } from './Footer'
import type { HealthStatus } from '../../hooks/useHealth'

interface AppShellProps {
  health: HealthStatus
  tab: TabId
  onTabChange: (tab: TabId) => void
  children: ReactNode
}

export function AppShell({ health, tab, onTabChange, children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-ink-900">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-sm focus:bg-ink-700 focus:px-3 focus:py-2 focus:text-micro"
      >
        Skip to content
      </a>

      <header className="sticky top-0 z-30 border-b border-line bg-ink-900/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1180px] items-center justify-between gap-4 px-5 py-3.5">
          <Wordmark />
          <SystemStatus health={health} />
        </div>
        <div className="mx-auto max-w-[1180px] px-5">
          <TabNav active={tab} onChange={onTabChange} />
        </div>
      </header>

      <DisclaimerBar />

      <main id="main" className="mx-auto max-w-[1180px] px-5 py-6">
        {children}
      </main>

      <Footer />
    </div>
  )
}

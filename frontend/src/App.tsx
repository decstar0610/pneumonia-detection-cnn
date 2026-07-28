import { Suspense, lazy, useState } from 'react'
import { motion } from 'framer-motion'

import { AppShell } from './components/layout/AppShell'
import type { TabId } from './components/layout/TabNav'
import { AnalyzeTab } from './components/analyze/AnalyzeTab'
import { useHealth } from './hooks/useHealth'
import { staggerParent } from './lib/motion'

// The report tab carries the chart library and the exported evaluation data;
// keeping it in its own chunk means the upload flow paints without them.
const ReportTab = lazy(() => import('./components/report/ReportTab'))

function ReportFallback() {
  return (
    <div className="hairline rounded-panel bg-ink-800/70 p-6 text-micro text-low">
      Loading evaluation report…
    </div>
  )
}

export function App() {
  const health = useHealth()
  const [tab, setTab] = useState<TabId>('analyze')

  return (
    <AppShell health={health} tab={tab} onTabChange={setTab}>
      <motion.div
        key={tab}
        variants={staggerParent}
        initial="hidden"
        animate="visible"
        role="tabpanel"
        id={`panel-${tab}`}
        aria-labelledby={`tab-${tab}`}
      >
        {tab === 'analyze' ? (
          <AnalyzeTab health={health} />
        ) : (
          <Suspense fallback={<ReportFallback />}>
            <ReportTab />
          </Suspense>
        )}
      </motion.div>
    </AppShell>
  )
}

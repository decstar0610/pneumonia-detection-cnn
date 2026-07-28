import { useState } from 'react'

import { Panel } from '../ui/Panel'
import { CalibrationChart } from './CalibrationChart'
import { CurvesPanel } from './CurvesPanel'
import { DatasetToggle } from './DatasetToggle'
import { FairnessPanel } from './FairnessPanel'
import { GeneralizationNote } from './GeneralizationNote'
import { MetricsGrid } from './MetricsGrid'
import { ThresholdExplorer } from './ThresholdExplorer'
import { TriagePanel } from './TriagePanel'
import { REPORT } from '../../data/report'
import type { DatasetKey } from '../../types/report'

/** Marks a panel whose data comes from one fixed dataset, whatever the toggle says. */
function ScopeChip({ children }: { children: string }) {
  return (
    <span className="rounded-full border border-line-strong px-2 py-0.5 text-label text-low uppercase">
      {children}
    </span>
  )
}

export default function ReportTab() {
  const [dataset, setDataset] = useState<DatasetKey>('internal')
  const labels = {
    internal: REPORT.datasets.internal.label,
    external: REPORT.datasets.external.label,
  }
  const results = { internal: REPORT.internal, external: REPORT.external }
  const active = results[dataset]
  const meta = REPORT.datasets[dataset]

  return (
    <div className="space-y-4">
      <Panel
        label="Evaluation summary"
        aside={<DatasetToggle value={dataset} onChange={setDataset} labels={labels} />}
      >
        <p className="mb-4 text-micro text-mid">
          <span className="text-hi">{meta.label}</span> — {meta.source}.{' '}
          <span className="text-low">{meta.note}</span>{' '}
          <span className="tnum text-low">
            n = {active.n.toLocaleString()}, {active.positives.toLocaleString()} positive (
            {(active.prevalence * 100).toFixed(0)}% prevalence)
          </span>
        </p>
        <MetricsGrid active={dataset} results={results} labels={labels} />
        <p className="mt-3 text-micro text-low">
          All figures at the frozen calibrated threshold{' '}
          <span className="tnum">{REPORT.threshold.calibrated.toFixed(3)}</span> (raw{' '}
          <span className="tnum">{REPORT.threshold.raw.toFixed(3)}</span>, chosen on{' '}
          {REPORT.threshold.chosen_on} for sensitivity ≥{' '}
          {REPORT.threshold.target_sensitivity.toFixed(2)}), probabilities temperature-scaled at T ={' '}
          {REPORT.temperature.value}.
        </p>
      </Panel>

      <Panel label="Does it generalize?">
        <GeneralizationNote report={REPORT} />
      </Panel>

      <Panel
        label="Operating point explorer"
        aside={<ScopeChip>{labels[dataset]}</ScopeChip>}
      >
        <ThresholdExplorer
          active={dataset}
          result={active}
          datasetLabel={labels[dataset]}
          deployedThreshold={REPORT.threshold.calibrated}
          targetSensitivity={REPORT.threshold.target_sensitivity}
        />
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel label="Precision–recall" aside={<ScopeChip>Both datasets</ScopeChip>}>
          <CurvesPanel internal={REPORT.internal} external={REPORT.external} labels={labels} />
        </Panel>
        <Panel label="Calibration" aside={<ScopeChip>{labels.internal}</ScopeChip>}>
          <CalibrationChart calibration={REPORT.calibration} />
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel label="Subgroup audit" aside={<ScopeChip>{labels.external}</ScopeChip>}>
          <FairnessPanel fairness={REPORT.fairness} />
        </Panel>
        <Panel label="Uncertainty triage" aside={<ScopeChip>{labels.internal}</ScopeChip>}>
          <TriagePanel triage={REPORT.triage} />
        </Panel>
      </div>

      <p className="px-1 text-micro text-low">
        Generated {REPORT.generated_at} by <span className="text-mid">python -m src.export_report</span>{' '}
        for model v{REPORT.model_version}. The internal split is re-scored through the same ONNX
        path the API serves with; external figures come from the per-image predictions persisted
        during evaluation. Nothing on this tab is hand-entered.
      </p>
    </div>
  )
}

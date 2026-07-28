import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { classNames } from '../../lib/format'
import { useMotionSafe } from '../../lib/motion'
import { AXIS_TICK, CHART, CHART_MARGIN, ChartFrame, ChartTooltip, Takeaway } from './chart'
import type { DatasetKey, DatasetResult, SweepPoint } from '../../types/report'

interface ThresholdExplorerProps {
  active: DatasetKey
  result: DatasetResult
  datasetLabel: string
  deployedThreshold: number
  targetSensitivity: number
}

function nearestIndex(sweep: readonly SweepPoint[], threshold: number): number {
  let best = 0
  let bestGap = Number.POSITIVE_INFINITY
  sweep.forEach((point, index) => {
    const gap = Math.abs(point.t - threshold)
    if (gap < bestGap) {
      bestGap = gap
      best = index
    }
  })
  return best
}

function OpposingBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-label text-low uppercase">{label}</span>
        <span className="tnum text-body font-medium text-hi">{value.toFixed(3)}</span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-ink-700">
        <div
          className="h-full rounded-full transition-[width] duration-[120ms] ease-[cubic-bezier(0.4,0,0.2,1)]"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function ConfusionCell({
  label,
  value,
  tone,
}: {
  label: string
  value: number
  tone?: 'alert' | 'caution'
}) {
  return (
    <div className="bg-ink-800 p-3">
      <div
        className={classNames(
          'text-label uppercase',
          tone === 'alert' ? 'text-alert' : tone === 'caution' ? 'text-caution' : 'text-low',
        )}
      >
        {label}
      </div>
      <div className="tnum mt-1 text-h2 text-hi">{value}</div>
    </div>
  )
}

/**
 * The centerpiece: every point on this slider is a real operating point measured
 * on the selected dataset (201-step sweep exported by src/export_report.py), so
 * dragging it shows the actual sensitivity/specificity trade-off rather than a
 * smoothed model of it.
 */
export function ThresholdExplorer({
  active,
  result,
  datasetLabel,
  deployedThreshold,
  targetSensitivity,
}: ThresholdExplorerProps) {
  const motionSafe = useMotionSafe()
  const deployedIndex = useMemo(
    () => nearestIndex(result.sweep, deployedThreshold),
    [result.sweep, deployedThreshold],
  )
  const [index, setIndex] = useState(deployedIndex)

  const point = result.sweep[Math.min(index, result.sweep.length - 1)]
  const deployed = result.sweep[deployedIndex]
  if (!point || !deployed) return null

  const atDeployed = index === deployedIndex
  const deltaFn = point.fn - deployed.fn
  const deltaFp = point.fp - deployed.fp
  const rocData = result.roc.map((p) => ({ x: p.fpr, y: p.tpr }))
  const seriesColor = active === 'internal' ? CHART.internal : CHART.external

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <Takeaway>
          What does moving the decision threshold actually cost? Every point below is measured on{' '}
          {datasetLabel.toLowerCase()} — {result.n} studies, {result.positives} with pneumonia.
        </Takeaway>
        <button
          type="button"
          onClick={() => setIndex(deployedIndex)}
          disabled={atDeployed}
          className="shrink-0 rounded-sm border border-line-strong px-2.5 py-1 text-label text-mid uppercase transition-colors duration-150 hover:border-clinical-lit hover:text-hi disabled:cursor-not-allowed disabled:opacity-40"
        >
          Reset to deployed
        </button>
      </div>

      <div className="rounded-panel border border-line bg-ink-850 p-4">
        <div className="flex items-baseline justify-between gap-4">
          <div>
            <span className="text-label text-low uppercase">Decision threshold</span>
            <div className="tnum mt-1 text-metric font-medium text-hi">{point.t.toFixed(3)}</div>
          </div>
          <span
            className={classNames(
              'rounded-full border px-2.5 py-1 text-label uppercase',
              atDeployed
                ? 'border-stable/35 bg-stable/10 text-stable'
                : 'border-line-strong text-low',
            )}
          >
            {atDeployed ? 'Deployed operating point' : `Deployed: ${deployed.t.toFixed(3)}`}
          </span>
        </div>

        <input
          type="range"
          min={0}
          max={result.sweep.length - 1}
          step={1}
          value={index}
          onChange={(event) => setIndex(Number(event.target.value))}
          aria-label="Decision threshold"
          aria-valuetext={`Threshold ${point.t.toFixed(3)}, sensitivity ${point.sensitivity.toFixed(3)}, specificity ${point.specificity.toFixed(3)}`}
          className="mt-4 h-1.5 w-full cursor-ew-resize accent-clinical-lit"
        />
        <div className="mt-1.5 flex justify-between text-label text-low tnum">
          <span>0.000</span>
          <span>0.500</span>
          <span>1.000</span>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div className="space-y-4">
          <OpposingBar label="Sensitivity" value={point.sensitivity} color={CHART.stable} />
          <OpposingBar label="Specificity" value={point.specificity} color={CHART.caution} />

          <div className="grid grid-cols-2 gap-4 border-t border-line pt-3">
            <div>
              <div className="text-label text-low uppercase">Precision</div>
              <div className="tnum mt-1 text-body text-hi">{point.precision.toFixed(3)}</div>
            </div>
            <div>
              <div className="text-label text-low uppercase">Accuracy</div>
              <div className="tnum mt-1 text-body text-hi">{point.accuracy.toFixed(3)}</div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-panel border border-line bg-line">
            <ConfusionCell label="True positive" value={point.tp} />
            <ConfusionCell label="Missed pneumonia" value={point.fn} tone="alert" />
            <ConfusionCell label="False alarm" value={point.fp} tone="caution" />
            <ConfusionCell label="True negative" value={point.tn} />
          </div>

          <p className="text-micro text-mid" role="status" aria-live="polite">
            {atDeployed ? (
              <>
                This is the deployed threshold, tuned on validation to reach sensitivity ≥{' '}
                {targetSensitivity.toFixed(2)} — not left at 0.5. It is frozen: the external set was
                scored at this exact value, with no re-tuning.
              </>
            ) : (
              <>
                Versus the deployed {deployed.t.toFixed(3)}, this threshold{' '}
                {deltaFn === 0 ? (
                  'misses the same number of pneumonia cases'
                ) : (
                  <span className={deltaFn > 0 ? 'text-alert' : 'text-stable'}>
                    misses {Math.abs(deltaFn)} {deltaFn > 0 ? 'more' : 'fewer'} pneumonia{' '}
                    {Math.abs(deltaFn) === 1 ? 'case' : 'cases'}
                  </span>
                )}{' '}
                and raises{' '}
                {deltaFp === 0 ? (
                  'the same number of false alarms'
                ) : (
                  <span className={deltaFp > 0 ? 'text-caution' : 'text-stable'}>
                    {Math.abs(deltaFp)} {deltaFp > 0 ? 'more' : 'fewer'} false{' '}
                    {Math.abs(deltaFp) === 1 ? 'alarm' : 'alarms'}
                  </span>
                )}
                .
              </>
            )}
          </p>
        </div>

        <ChartFrame height={300}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rocData} margin={CHART_MARGIN}>
              <CartesianGrid stroke={CHART.grid} strokeDasharray="2 4" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={{ stroke: CHART.grid }}
                label={{
                  value: '1 − specificity',
                  position: 'insideBottom',
                  offset: -2,
                  fill: CHART.axis,
                  fontSize: 10,
                }}
                height={34}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={{ stroke: CHART.grid }}
                width={38}
                label={{
                  value: 'sensitivity',
                  angle: -90,
                  position: 'insideLeft',
                  fill: CHART.axis,
                  fontSize: 10,
                }}
              />
              <Tooltip
                content={<ChartTooltip labelName="1 − spec" />}
                cursor={{ stroke: CHART.grid }}
              />
              <ReferenceLine
                segment={[
                  { x: 0, y: 0 },
                  { x: 1, y: 1 },
                ]}
                stroke={CHART.reference}
                strokeDasharray="3 3"
              />
              <Line
                type="monotone"
                dataKey="y"
                name="ROC"
                stroke={seriesColor}
                strokeWidth={2}
                dot={false}
                isAnimationActive={motionSafe}
                animationDuration={700}
                animationEasing="ease-out"
              />
              <ReferenceDot
                x={1 - deployed.specificity}
                y={deployed.sensitivity}
                r={5}
                fill="none"
                stroke={CHART.text}
                strokeWidth={1.5}
                isFront
              />
              <ReferenceDot
                x={1 - point.specificity}
                y={point.sensitivity}
                r={5}
                fill={seriesColor}
                stroke="#0A0E12"
                strokeWidth={2}
                isFront
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="mt-1 text-micro text-low">
            Filled dot = the threshold you picked · hollow dot = the deployed operating point.
          </p>
        </ChartFrame>
      </div>
    </div>
  )
}

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

import { useMotionSafe } from '../../lib/motion'
import { AXIS_TICK, CHART, CHART_MARGIN, ChartFrame, ChartTooltip, Takeaway } from './chart'
import type { ModelReport } from '../../types/report'

export function TriagePanel({ triage }: { triage: ModelReport['triage'] }) {
  const motionSafe = useMotionSafe()
  const data = triage.curve.map((point) => ({ x: point.coverage, y: point.accuracy }))
  const escalated = 1 - triage.coverage
  const bandLow = triage.band[0] ?? 0
  const bandHigh = triage.band[1] ?? 1

  return (
    <div>
      <Takeaway>
        How good is it on the cases it chooses to decide? Abstaining on the least-confident{' '}
        {(escalated * 100).toFixed(0)}% lifts accuracy on the rest from{' '}
        {triage.full_accuracy.toFixed(3)} to {triage.decided_accuracy.toFixed(3)}.
      </Takeaway>

      <ChartFrame height={230}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={CHART_MARGIN}>
            <CartesianGrid stroke={CHART.grid} strokeDasharray="2 4" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: CHART.grid }}
              height={34}
              label={{
                value: 'coverage (share of studies decided)',
                position: 'insideBottom',
                offset: -2,
                fill: CHART.axis,
                fontSize: 10,
              }}
            />
            <YAxis
              type="number"
              domain={[0.9, 1]}
              ticks={[0.9, 0.925, 0.95, 0.975, 1]}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: CHART.grid }}
              width={44}
              label={{
                value: 'accuracy on decided',
                angle: -90,
                position: 'insideLeft',
                fill: CHART.axis,
                fontSize: 10,
              }}
            />
            <Tooltip content={<ChartTooltip labelName="coverage" />} cursor={{ stroke: CHART.grid }} />
            <ReferenceLine
              y={triage.target_accuracy}
              stroke={CHART.caution}
              strokeDasharray="3 3"
              label={{
                value: `target ${triage.target_accuracy}`,
                fill: CHART.caution,
                fontSize: 10,
                position: 'insideBottomRight',
              }}
            />
            <Line
              type="monotone"
              dataKey="y"
              name="accuracy"
              stroke={CHART.internal}
              strokeWidth={2}
              dot={false}
              isAnimationActive={motionSafe}
              animationDuration={700}
            />
            <ReferenceDot
              x={triage.coverage}
              y={triage.decided_accuracy}
              r={5}
              fill={CHART.caution}
              stroke="#0A0E12"
              strokeWidth={2}
              isFront
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartFrame>

      <dl className="mt-3 grid grid-cols-3 gap-px overflow-hidden rounded-panel border border-line bg-line">
        {[
          { label: 'Routine', value: triage.zones.routine, tone: 'text-stable' },
          { label: 'Escalated', value: triage.zones.uncertain, tone: 'text-caution' },
          { label: 'Urgent', value: triage.zones.urgent, tone: 'text-alert' },
        ].map((zone) => (
          <div key={zone.label} className="bg-ink-800 p-3">
            <dt className={`text-label uppercase ${zone.tone}`}>{zone.label}</dt>
            <dd className="tnum mt-1 text-h2 text-hi">{zone.value}</dd>
          </div>
        ))}
      </dl>

      <p className="mt-2 text-micro text-low">
        Abstention band [{bandLow.toFixed(3)}, {bandHigh.toFixed(3)}] on the calibrated probability —
        chosen as the widest band still meeting the {triage.target_accuracy} decided-accuracy target.
        Of the {triage.zones.uncertain} escalated studies, {triage.zones.uncertain_true_pneumonia}{' '}
        were truly pneumonia: cases a silent model would have called wrong.
      </p>
    </div>
  )
}

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useMotionSafe } from '../../lib/motion'
import { AXIS_TICK, CHART, CHART_MARGIN, ChartFrame, ChartTooltip, Takeaway, mergeCurves } from './chart'
import type { DatasetResult } from '../../types/report'

interface CurvesPanelProps {
  internal: DatasetResult
  external: DatasetResult
  labels: { internal: string; external: string }
}

/**
 * Precision–recall for both datasets on one axis. The curves are the raw exported
 * points, interleaved rather than resampled, so neither is smoothed to fit the other.
 */
export function CurvesPanel({ internal, external, labels }: CurvesPanelProps) {
  const motionSafe = useMotionSafe()
  const data = mergeCurves(
    internal.pr,
    external.pr,
    (point) => point.recall,
    (point) => point.precision,
    ['internal', 'external'],
  )

  return (
    <div>
      <Takeaway>
        How much does ranking quality shift off-source? PR-AUC {internal.pr_auc.toFixed(3)} →{' '}
        {external.pr_auc.toFixed(3)}, on a set where pneumonia prevalence drops from{' '}
        {(internal.prevalence * 100).toFixed(0)}% to {(external.prevalence * 100).toFixed(0)}%.
      </Takeaway>
      <ChartFrame height={260}>
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
                value: 'recall (sensitivity)',
                position: 'insideBottom',
                offset: -2,
                fill: CHART.axis,
                fontSize: 10,
              }}
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: CHART.grid }}
              width={38}
              label={{
                value: 'precision',
                angle: -90,
                position: 'insideLeft',
                fill: CHART.axis,
                fontSize: 10,
              }}
            />
            <Tooltip content={<ChartTooltip labelName="recall" />} cursor={{ stroke: CHART.grid }} />
            <Line
              type="monotone"
              dataKey="internal"
              name={labels.internal}
              stroke={CHART.internal}
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={motionSafe}
              animationDuration={700}
              animationEasing="ease-out"
            />
            <Line
              type="monotone"
              dataKey="external"
              name={labels.external}
              stroke={CHART.external}
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={motionSafe}
              animationDuration={700}
              animationEasing="ease-out"
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartFrame>
    </div>
  )
}

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useMotionSafe } from '../../lib/motion'
import { AXIS_TICK, CHART, CHART_MARGIN, ChartFrame, ChartTooltip, Takeaway, mergeCurves } from './chart'
import type { ModelReport } from '../../types/report'

export function CalibrationChart({ calibration }: { calibration: ModelReport['calibration'] }) {
  const motionSafe = useMotionSafe()
  const improved = calibration.ece_scaled < calibration.ece_raw
  const data = mergeCurves(
    calibration.bins_raw,
    calibration.bins_scaled,
    (bin) => bin.confidence,
    (bin) => bin.empirical,
    ['raw', 'scaled'],
  )

  return (
    <div>
      <Takeaway>
        Can the confidence numbers be trusted? Points above the diagonal mean the model is
        under-confident. Temperature scaling (T = {calibration.temperature}) moved expected
        calibration error {calibration.ece_raw.toFixed(3)} → {calibration.ece_scaled.toFixed(3)}
        {improved ? '' : ' (no improvement — reported as measured)'}.
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
                value: 'mean predicted probability',
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
                value: 'empirical rate',
                angle: -90,
                position: 'insideLeft',
                fill: CHART.axis,
                fontSize: 10,
              }}
            />
            <Tooltip content={<ChartTooltip labelName="predicted" />} cursor={{ stroke: CHART.grid }} />
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke={CHART.reference}
              strokeDasharray="3 3"
            />
            <Line
              type="linear"
              dataKey="raw"
              name={`raw (ECE ${calibration.ece_raw.toFixed(3)})`}
              stroke={CHART.external}
              strokeWidth={2}
              dot={{ r: 2.5, fill: CHART.external, strokeWidth: 0 }}
              connectNulls
              isAnimationActive={motionSafe}
              animationDuration={700}
            />
            <Line
              type="linear"
              dataKey="scaled"
              name={`temp-scaled (ECE ${calibration.ece_scaled.toFixed(3)})`}
              stroke={CHART.internal}
              strokeWidth={2}
              dot={{ r: 2.5, fill: CHART.internal, strokeWidth: 0 }}
              connectNulls
              isAnimationActive={motionSafe}
              animationDuration={700}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartFrame>
      <p className="mt-2 text-micro text-low">
        Fit on validation, reported on the held-out internal test set. Scaling is monotonic, so the
        decision threshold and every ranking metric are unchanged.
      </p>
    </div>
  )
}

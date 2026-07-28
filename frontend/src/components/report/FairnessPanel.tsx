import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useMotionSafe } from '../../lib/motion'
import { AXIS_TICK, CHART, ChartFrame, ChartTooltip, Takeaway } from './chart'
import type { ModelReport } from '../../types/report'

export function FairnessPanel({ fairness }: { fairness: ModelReport['fairness'] }) {
  const motionSafe = useMotionSafe()
  const flagged = fairness.by_subgroup.filter((row) => row.flag)
  const data = fairness.by_subgroup.map((row) => ({
    name: `${row.subgroup} (n=${row.n_pos})`,
    sensitivity: row.sensitivity,
    flag: row.flag,
  }))

  return (
    <div>
      <Takeaway>
        Who does it fail? Per-subgroup sensitivity on the external set, against the
        {' '}{fairness.flag_floor.toFixed(2)} floor this project set itself.
      </Takeaway>

      <ChartFrame height={252}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, bottom: 16, left: 4 }}>
            <XAxis
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: CHART.grid }}
              height={28}
              label={{
                value: 'sensitivity',
                position: 'insideBottom',
                offset: -6,
                fill: CHART.axis,
                fontSize: 10,
              }}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: CHART.grid }}
              width={104}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: '#ffffff08' }} />
            <ReferenceLine
              x={fairness.flag_floor}
              stroke={CHART.caution}
              strokeDasharray="3 3"
              label={{
                value: `floor ${fairness.flag_floor}`,
                fill: CHART.caution,
                fontSize: 10,
                position: 'top',
              }}
            />
            <Bar
              dataKey="sensitivity"
              name="sensitivity"
              barSize={12}
              radius={2}
              isAnimationActive={motionSafe}
              animationDuration={700}
            >
              {data.map((row) => (
                <Cell key={row.name} fill={row.flag ? CHART.external : CHART.internal} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartFrame>

      {flagged.length > 0 && (
        <p className="mt-2 text-micro text-mid">
          {flagged.map((row) => (
            <span key={row.subgroup}>
              <span className="text-alert">
                {row.dimension} = {row.subgroup}
              </span>{' '}
              falls to {row.sensitivity.toFixed(3)} on {row.n_pos} positive studies — a genuine
              disparity, not a small-sample artifact, and disclosed in the model card.
            </span>
          ))}
        </p>
      )}

      <div className="mt-4 border-t border-line pt-3">
        <p className="mb-2 text-label text-low uppercase">
          Specificity by negative class — the mechanism
        </p>
        <table className="w-full text-micro">
          <thead>
            <tr className="text-low">
              <th scope="col" className="pb-1 text-left font-normal">
                Negative class
              </th>
              <th scope="col" className="pb-1 text-right font-normal">
                n
              </th>
              <th scope="col" className="pb-1 text-right font-normal">
                Specificity
              </th>
            </tr>
          </thead>
          <tbody>
            {fairness.by_negative_class.map((row) => (
              <tr key={row.neg_class} className="border-t border-line/60">
                <td className="py-1.5 text-mid">{row.neg_class}</td>
                <td className="tnum py-1.5 text-right text-low">{row.n_neg}</td>
                <td className="tnum py-1.5 text-right text-hi">{row.specificity.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-micro text-low">
          The specificity collapse is class-driven: the model holds up on genuinely normal chests
          but false-alarms on abnormal-but-not-pneumonia studies, a category absent from its
          pediatric training data.
        </p>
      </div>
    </div>
  )
}

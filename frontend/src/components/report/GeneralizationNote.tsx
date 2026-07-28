import type { ModelReport } from '../../types/report'

/**
 * The headline finding, stated plainly. Every number is read from the exported
 * report — if the model is re-evaluated, this paragraph updates itself.
 */
export function GeneralizationNote({ report }: { report: ModelReport }) {
  const { internal, external, fairness } = report
  const worstView = fairness.by_subgroup.find((row) => row.flag)
  const negClasses = [...fairness.by_negative_class].sort((a, b) => a.specificity - b.specificity)
  const worstNeg = negClasses[0]
  const bestNeg = negClasses[negClasses.length - 1]

  return (
    <div className="space-y-3 text-body text-mid">
      <p className="text-hi">
        Sensitivity survives the move to another hospital&rsquo;s data. Specificity does not.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <p>
          Scored once at the frozen threshold, sensitivity goes{' '}
          <span className="tnum text-hi">{internal.sensitivity.toFixed(3)}</span> →{' '}
          <span className="tnum text-hi">{external.sensitivity.toFixed(3)}</span> — it still finds
          pneumonia. Specificity goes{' '}
          <span className="tnum text-hi">{internal.specificity.toFixed(3)}</span> →{' '}
          <span className="tnum text-alert">{external.specificity.toFixed(3)}</span>, so on the
          external set it raises {external.confusion.fp.toLocaleString()} false alarms against{' '}
          {external.confusion.tp.toLocaleString()} true ones.
        </p>
        <p>
          The cause is visible in the breakdown, not guessed at:{' '}
          {bestNeg !== undefined && worstNeg !== undefined && (
            <>
              specificity holds at{' '}
              <span className="tnum text-hi">{bestNeg.specificity.toFixed(3)}</span> on studies
              labelled &ldquo;{bestNeg.neg_class}&rdquo; but falls to{' '}
              <span className="tnum text-alert">{worstNeg.specificity.toFixed(3)}</span> on
              &ldquo;{worstNeg.neg_class}&rdquo;.{' '}
            </>
          )}
          The model was trained on pediatric chests that were either normal or pneumonic; adult
          studies that are abnormal for some other reason are a category it has never seen, so it
          reads their opacity as pneumonia.
          {worstView !== undefined && (
            <>
              {' '}
              The subgroup audit adds a second, independent finding: sensitivity on{' '}
              {worstView.dimension} = {worstView.subgroup} is{' '}
              <span className="tnum text-alert">{worstView.sensitivity.toFixed(3)}</span>.
            </>
          )}
        </p>
      </div>
      <p className="text-micro text-low">
        Reported rather than hidden: the external set was scored once, at the threshold tuned on
        internal validation, with no retraining and no re-tuning. A drop of this size is the
        expected behaviour of a single-source model — the point of measuring it is to know how big
        it is before anyone relies on the tool.
      </p>
    </div>
  )
}

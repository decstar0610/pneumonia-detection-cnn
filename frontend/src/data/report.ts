import raw from './model_report.json'
import type { ModelReport } from '../types/report'

/**
 * The one place the generated evaluation data enters the app. `model_report.json`
 * is a build-time artifact of `python -m src.export_report`, so its shape is fixed
 * by that script — asserted once here rather than re-validated at every use site.
 */
export const REPORT: ModelReport = raw as unknown as ModelReport

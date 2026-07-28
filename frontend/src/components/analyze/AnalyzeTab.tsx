import { Panel } from '../ui/Panel'
import { ErrorState } from '../ui/ErrorState'
import { FileButton } from '../ui/FileButton'
import { ColdStartNotice } from './ColdStartNotice'
import { GradCamViewer } from './GradCamViewer'
import { PipelineTicker } from './PipelineTicker'
import { SampleGallery } from './SampleGallery'
import { ResultCard } from './ResultCard'
import { ScanViewport } from './ScanViewport'
import { UploadDropzone } from './UploadDropzone'
import { ZoneLegend } from './ZoneLegend'
import { usePrediction } from '../../hooks/usePrediction'
import type { HealthStatus } from '../../hooks/useHealth'
import { seconds } from '../../lib/format'

function fileSize(bytes: number): string {
  return bytes < 1024 * 1024
    ? `${Math.round(bytes / 1024)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function AnalyzeTab({ health }: { health: HealthStatus }) {
  const flow = usePrediction()
  const analyzing = flow.phase === 'analyzing'

  return (
    <div className="space-y-4">
      <ColdStartNotice health={health} />

      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.02fr)_minmax(0,1fr)]">
        <Panel
          label="Study"
          aside={
            flow.file && (
              <div className="flex items-center gap-3">
                <span className="max-w-[16ch] truncate text-micro text-low sm:max-w-[28ch]">
                  {flow.file.name}
                </span>
                <span className="tnum text-label text-low">{fileSize(flow.file.size)}</span>
              </div>
            )
          }
        >
          {flow.previewUrl === null ? (
            <div className="space-y-5">
              <UploadDropzone
                onSelect={flow.select}
                onReject={flow.reject}
                rejection={flow.rejection}
              />
              <div className="border-t border-line pt-4">
                <SampleGallery onSelect={flow.select} />
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              {flow.phase === 'done' && flow.result !== null ? (
                <GradCamViewer original={flow.previewUrl} overlay={flow.result.gradcam_overlay} />
              ) : (
                <ScanViewport
                  src={flow.previewUrl}
                  alt="Uploaded chest X-ray"
                  analyzing={analyzing}
                  footer={
                    analyzing
                      ? 'Acquiring — the model sees a 224×224 normalised version of this study.'
                      : 'Uploaded study, shown as provided.'
                  }
                />
              )}
              <div className="flex flex-wrap items-center gap-2">
                <FileButton onSelect={flow.select}>Replace study</FileButton>
                <button
                  type="button"
                  onClick={flow.rerun}
                  disabled={analyzing}
                  className="rounded-sm border border-line-strong px-2.5 py-1 text-label text-mid uppercase transition-colors duration-150 hover:border-clinical-lit hover:text-hi disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Re-run
                </button>
                <button
                  type="button"
                  onClick={flow.clear}
                  className="rounded-sm px-2.5 py-1 text-label text-low uppercase transition-colors duration-150 hover:text-mid"
                >
                  Clear
                </button>
              </div>
              {flow.rejection !== null && (
                <p className="text-micro text-alert" role="alert">
                  {flow.rejection}
                </p>
              )}
              <div className="border-t border-line pt-4">
                <SampleGallery onSelect={flow.select} disabled={analyzing} />
              </div>
            </div>
          )}
        </Panel>

        <Panel label="Assessment">
          {flow.phase === 'idle' && (
            <div className="space-y-4">
              <p className="text-micro text-mid">
                Load a study to get a calibrated probability, a triage disposition and a Grad-CAM
                explanation. Every prediction is produced live by the deployed model — nothing here
                is pre-computed.
              </p>
              <div className="border-t border-line pt-4">
                <p className="mb-3 text-label text-low uppercase">Dispositions</p>
                <ZoneLegend />
              </div>
            </div>
          )}

          {analyzing && (
            <div className="space-y-4">
              <div className="flex items-baseline justify-between">
                <p className="text-body text-hi">Analyzing study</p>
                <p className="tnum text-micro text-low">{seconds(flow.elapsedMs)}</p>
              </div>
              <PipelineTicker elapsedMs={flow.elapsedMs} waking={health.state === 'warming'} />
            </div>
          )}

          {flow.phase === 'error' && flow.error !== null && (
            <ErrorState
              title="Analysis failed"
              message={flow.error}
              actionLabel="Try again"
              onAction={flow.rerun}
            />
          )}

          {flow.phase === 'done' && flow.result !== null && (
            <ResultCard result={flow.result} roundTripMs={flow.elapsedMs} />
          )}
        </Panel>
      </div>
    </div>
  )
}

import { useState } from 'react'

import { classNames } from '../../lib/format'

interface GradCamViewerProps {
  /** Object URL of the uploaded study. */
  original: string
  /** Base64 PNG overlay returned by the API (224×224, pre-blended). */
  overlay: string
}

type Mode = 'fade' | 'split'

/**
 * The heatmap is computed on the model's 224×224 input, so both panes stretch the
 * study to that same square geometry — otherwise the overlay would not line up
 * with the anatomy underneath it, and a crossfade would be meaningless.
 */
const PANE = 'relative aspect-square w-full overflow-hidden rounded-panel border border-line bg-black'

/** Stops sampled from the server's `_jet` colormap in api/inference.py. */
const JET_STOPS =
  'linear-gradient(to right, #000080 0%, #0080ff 25%, #00ffff 37.5%, #80ff80 50%, #ffff00 62.5%, #ff8000 75%, #800000 100%)'

export function GradCamViewer({ original, overlay }: GradCamViewerProps) {
  const [mode, setMode] = useState<Mode>('fade')
  const [opacity, setOpacity] = useState(0.75)
  const [dragging, setDragging] = useState(false)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="text-label text-low uppercase">Grad-CAM · where the model looked</div>
        <div
          className="flex rounded-sm border border-line-strong p-0.5"
          role="group"
          aria-label="Comparison mode"
        >
          {(['fade', 'split'] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setMode(value)}
              aria-pressed={mode === value}
              className={classNames(
                'rounded-[3px] px-2 py-0.5 text-label uppercase transition-colors duration-150',
                mode === value ? 'bg-ink-600 text-hi' : 'text-low hover:text-mid',
              )}
            >
              {value === 'fade' ? 'Fade' : 'Side by side'}
            </button>
          ))}
        </div>
      </div>

      {mode === 'fade' ? (
        <figure className={PANE}>
          <img src={original} alt="Uploaded chest X-ray" className="absolute inset-0 h-full w-full object-fill" />
          <img
            src={overlay}
            alt="Grad-CAM attention heatmap over the study"
            className={classNames(
              'absolute inset-0 h-full w-full object-fill',
              !dragging && 'transition-opacity duration-150 ease-[cubic-bezier(0.4,0,0.2,1)]',
            )}
            style={{ opacity }}
          />
        </figure>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <figure>
            <div className={PANE}>
              <img src={original} alt="Uploaded chest X-ray" className="absolute inset-0 h-full w-full object-fill" />
            </div>
            <figcaption className="mt-1.5 text-label text-low uppercase">Study · 224²</figcaption>
          </figure>
          <figure>
            <div className={PANE}>
              <img
                src={overlay}
                alt="Grad-CAM attention heatmap over the study"
                className="absolute inset-0 h-full w-full object-fill"
              />
            </div>
            <figcaption className="mt-1.5 text-label text-low uppercase">Grad-CAM</figcaption>
          </figure>
        </div>
      )}

      {mode === 'fade' && (
        <div className="flex items-center gap-3">
          <span className="text-label text-low uppercase">X-ray</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={Math.round(opacity * 100)}
            onChange={(event) => setOpacity(Number(event.target.value) / 100)}
            onPointerDown={() => setDragging(true)}
            onPointerUp={() => setDragging(false)}
            onBlur={() => setDragging(false)}
            aria-label="Heatmap opacity"
            aria-valuetext={`Heatmap at ${Math.round(opacity * 100)} percent`}
            className="h-1 w-full cursor-ew-resize accent-clinical-lit"
          />
          <span className="text-label text-low uppercase">Heatmap</span>
          <span className="tnum w-10 shrink-0 text-right text-micro text-mid">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}

      <div className="flex items-center gap-3">
        <span className="text-label text-low uppercase">Attention</span>
        <span className="h-1.5 flex-1 rounded-full" style={{ background: JET_STOPS }} aria-hidden="true" />
        <span className="text-label text-low uppercase">Low → high</span>
      </div>

      <p className="text-micro text-low">
        Gradient of the calibrated output with respect to the final 7×7 convolutional map, upsampled
        to the input. It is a debugging aid, not evidence: on some false positives the model attends
        to borders, markers or the diaphragm rather than lung fields.
      </p>
    </div>
  )
}

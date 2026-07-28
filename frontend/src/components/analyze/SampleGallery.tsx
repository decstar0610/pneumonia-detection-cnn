import { useState } from 'react'

import { classNames } from '../../lib/format'
import { SAMPLE_STUDIES, type SampleStudy } from '../../data/samples'

interface SampleGalleryProps {
  onSelect: (file: File) => void
  disabled?: boolean
}

/**
 * Four held-out studies a reviewer can run without having an X-ray to hand.
 * Each tile fetches the real file and posts it to /predict like any upload —
 * the tiles carry dataset ground truth, never a stored prediction.
 */
export function SampleGallery({ onSelect, disabled = false }: SampleGalleryProps) {
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function load(sample: SampleStudy) {
    setLoadingId(sample.id)
    setError(null)
    try {
      const response = await fetch(sample.src)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      onSelect(new File([blob], sample.origin, { type: 'image/jpeg' }))
    } catch {
      setError('Could not load that sample study.')
    } finally {
      setLoadingId(null)
    }
  }

  return (
    <section aria-labelledby="samples-heading">
      <div className="flex items-baseline justify-between gap-3">
        <h3 id="samples-heading" className="text-label text-low uppercase">
          Sample studies
        </h3>
        <span className="text-micro text-low">Held-out test split</span>
      </div>

      <ul className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {SAMPLE_STUDIES.map((sample) => (
          <li key={sample.id}>
            <button
              type="button"
              onClick={() => void load(sample)}
              disabled={disabled || loadingId !== null}
              className={classNames(
                'group w-full overflow-hidden rounded-panel border border-line bg-ink-850 text-left transition-colors duration-150',
                'hover:border-clinical/70 disabled:cursor-not-allowed disabled:opacity-50',
              )}
              aria-label={`Analyze sample study: ${sample.truth.toLowerCase()}, ${sample.note}`}
            >
              <span className="relative block aspect-square overflow-hidden bg-black">
                <img
                  src={sample.thumb}
                  alt=""
                  loading="lazy"
                  className="h-full w-full object-cover opacity-85 transition-opacity duration-150 group-hover:opacity-100"
                />
                {loadingId === sample.id && (
                  <span className="absolute inset-0 flex items-center justify-center bg-ink-900/70 text-label text-clinical-lit uppercase">
                    Loading
                  </span>
                )}
              </span>
              <span className="block px-2 py-1.5">
                <span className="block text-label text-mid uppercase">{sample.truth}</span>
                <span className="block text-micro text-low">{sample.note}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>

      {error !== null && (
        <p className="mt-2 text-micro text-alert" role="alert">
          {error}
        </p>
      )}
      <p className="mt-2 text-micro text-low">
        Labels are the dataset&rsquo;s ground truth, not the model&rsquo;s answer — every prediction
        is computed live. Images: Kermany et al., CC BY 4.0.
      </p>
    </section>
  )
}

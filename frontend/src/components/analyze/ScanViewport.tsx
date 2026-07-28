import type { ReactNode } from 'react'

import { classNames } from '../../lib/format'

interface ScanViewportProps {
  src: string
  alt: string
  analyzing: boolean
  /** Overlay layer (Grad-CAM) rendered above the study. */
  overlay?: ReactNode
  footer?: ReactNode
}

/** Corner reticles — the frame reads as an acquisition viewport, not a photo card. */
function Corners() {
  const shared = 'absolute h-3 w-3 border-clinical-lit/40'
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0">
      <span className={classNames(shared, 'top-2 left-2 border-t border-l')} />
      <span className={classNames(shared, 'top-2 right-2 border-t border-r')} />
      <span className={classNames(shared, 'bottom-2 left-2 border-b border-l')} />
      <span className={classNames(shared, 'bottom-2 right-2 border-b border-r')} />
    </div>
  )
}

export function ScanViewport({ src, alt, analyzing, overlay, footer }: ScanViewportProps) {
  return (
    <figure className="space-y-2">
      <div className="relative aspect-square w-full overflow-hidden rounded-panel border border-line bg-black">
        <img
          src={src}
          alt={alt}
          className={classNames(
            'h-full w-full object-contain transition-opacity duration-[380ms]',
            analyzing ? 'opacity-70' : 'opacity-100',
          )}
        />
        {overlay}
        <Corners />
        {analyzing && (
          <>
            <div className="raster pointer-events-none absolute inset-0 opacity-50" aria-hidden="true" />
            <div
              className="scan-band animate-scanline pointer-events-none absolute inset-0"
              aria-hidden="true"
            />
          </>
        )}
      </div>
      {footer !== undefined && <figcaption className="text-micro text-low">{footer}</figcaption>}
    </figure>
  )
}

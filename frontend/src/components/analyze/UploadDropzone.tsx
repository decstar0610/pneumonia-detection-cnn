import { useEffect, useRef, useState } from 'react'

import { ACCEPTED_MIME, MAX_UPLOAD_BYTES } from '../../lib/api'
import { classNames } from '../../lib/format'

interface UploadDropzoneProps {
  onSelect: (file: File) => void
  onReject: (reason: string) => void
  /** Rejection message from the client-side gate, rendered inline. */
  rejection: string | null
  disabled?: boolean
}

/**
 * Drag-and-drop / click / paste study intake. It is a real <button>, so it is
 * reachable and operable from the keyboard without any extra key handling.
 */
export function UploadDropzone({ onSelect, onReject, rejection, disabled = false }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)
  const [dragging, setDragging] = useState(false)

  // Paste an X-ray straight from the clipboard — no file manager round trip.
  useEffect(() => {
    function onPaste(event: ClipboardEvent) {
      const item = Array.from(event.clipboardData?.files ?? [])[0]
      if (item) onSelect(item)
    }
    document.addEventListener('paste', onPaste)
    return () => document.removeEventListener('paste', onPaste)
  }, [onSelect])

  function handleDrop(event: React.DragEvent) {
    event.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    const dropped = Array.from(event.dataTransfer.files)
    const first = dropped[0]
    if (!first) {
      onReject('That drop contained no file. Drag a JPEG or PNG image.')
      return
    }
    onSelect(first)
  }

  return (
    <div className="space-y-3">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => {
          event.preventDefault()
          dragDepth.current += 1
          setDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => {
          dragDepth.current -= 1
          if (dragDepth.current <= 0) setDragging(false)
        }}
        onDrop={handleDrop}
        disabled={disabled}
        aria-describedby="upload-help"
        className={classNames(
          'flex w-full flex-col items-center justify-center gap-3 rounded-panel border border-dashed px-6 py-14 text-center transition-[border-color,background-color,transform] duration-150 ease-[cubic-bezier(0.4,0,0.2,1)]',
          dragging
            ? 'scale-[0.99] border-clinical-lit bg-clinical-lit/8'
            : 'border-line-strong bg-ink-850 hover:border-clinical/70 hover:bg-ink-800',
          disabled && 'cursor-not-allowed opacity-50',
        )}
      >
        <svg width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">
          <rect
            x="4.5"
            y="2.5"
            width="25"
            height="29"
            rx="2.5"
            className={classNames('stroke-line-strong', dragging && 'stroke-clinical-lit')}
          />
          <path
            d="M17 10v13M11.5 15.5 17 10l5.5 5.5"
            className={classNames(
              'transition-colors duration-150',
              dragging ? 'stroke-clinical-lit' : 'stroke-low',
            )}
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="text-body font-medium text-hi">
          {dragging ? 'Release to load study' : 'Drop a chest X-ray'}
        </span>
        <span id="upload-help" className="text-micro text-low">
          or click to browse · paste from clipboard · JPEG or PNG up to{' '}
          {MAX_UPLOAD_BYTES / 1024 / 1024} MB
        </span>
      </button>

      {rejection !== null && (
        <p className="text-micro text-alert" role="alert">
          {rejection}
        </p>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_MIME.join(',')}
        className="sr-only"
        tabIndex={-1}
        onChange={(event) => {
          const chosen = event.target.files?.[0]
          if (chosen) onSelect(chosen)
          event.target.value = '' // allow re-selecting the same file
        }}
      />
    </div>
  )
}

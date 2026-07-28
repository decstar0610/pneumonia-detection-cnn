import { useRef, type ReactNode } from 'react'

import { ACCEPTED_MIME } from '../../lib/api'
import { classNames } from '../../lib/format'

interface FileButtonProps {
  onSelect: (file: File) => void
  children: ReactNode
  className?: string
}

/** Secondary "replace study" affordance — same accept list as the dropzone. */
export function FileButton({ onSelect, children, className }: FileButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  return (
    <>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className={classNames(
          'rounded-sm border border-line-strong px-2.5 py-1 text-label text-mid uppercase transition-colors duration-150 hover:border-clinical-lit hover:text-hi',
          className,
        )}
      >
        {children}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_MIME.join(',')}
        className="sr-only"
        tabIndex={-1}
        onChange={(event) => {
          const chosen = event.target.files?.[0]
          if (chosen) onSelect(chosen)
          event.target.value = ''
        }}
      />
    </>
  )
}

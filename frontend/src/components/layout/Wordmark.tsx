/** The mark: a reticle over a rib-cage arc — drawn, not an emoji or stock icon. */
export function Wordmark() {
  return (
    <div className="flex items-center gap-3">
      <svg
        width="30"
        height="30"
        viewBox="0 0 30 30"
        fill="none"
        aria-hidden="true"
        className="shrink-0"
      >
        <rect x="0.5" y="0.5" width="29" height="29" rx="5.5" className="fill-ink-700 stroke-line-strong" />
        <path
          d="M15 7v16M9.5 10.5c0 5 1.5 8 5.5 9.5M20.5 10.5c0 5-1.5 8-5.5 9.5"
          className="stroke-clinical-lit"
          strokeWidth="1.25"
          strokeLinecap="round"
        />
        <circle cx="15" cy="15" r="7" className="stroke-clinical-lit/35" strokeWidth="1" />
        <path d="M15 4v3M15 23v3M4 15h3M23 15h3" className="stroke-clinical-lit/60" strokeWidth="1" strokeLinecap="round" />
      </svg>
      <div className="leading-none">
        <div className="text-h2 font-semibold tracking-tight">PneumoScan</div>
        <div className="mt-1 text-label text-low uppercase">Chest X-ray triage · research prototype</div>
      </div>
    </div>
  )
}

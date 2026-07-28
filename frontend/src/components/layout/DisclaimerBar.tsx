/**
 * Persistent, non-alarming scope statement (PRD §7.3). It is always visible and
 * intentionally styled as instrument chrome rather than a warning banner.
 */
export function DisclaimerBar() {
  return (
    <div className="border-b border-line bg-ink-850">
      <p className="mx-auto flex max-w-[1180px] items-baseline gap-2 px-5 py-2 text-micro text-mid">
        <span className="text-label font-medium text-caution uppercase">Research prototype</span>
        <span className="text-low">·</span>
        <span>
          Decision-support demonstration only. Not a diagnostic device, not FDA/CE cleared, and
          never a substitute for a radiologist.
        </span>
      </p>
    </div>
  )
}

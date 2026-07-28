interface ErrorStateProps {
  title: string
  message: string
  actionLabel?: string
  onAction?: () => void
}

export function ErrorState({ title, message, actionLabel, onAction }: ErrorStateProps) {
  return (
    <div
      className="rounded-panel border border-alert/40 bg-alert/5 p-4"
      role="alert"
      aria-live="assertive"
    >
      <p className="text-micro font-medium text-alert">{title}</p>
      <p className="mt-1 text-micro text-mid">{message}</p>
      {actionLabel !== undefined && onAction !== undefined && (
        <button
          type="button"
          onClick={onAction}
          className="mt-3 rounded-sm border border-line-strong px-2.5 py-1 text-label text-mid uppercase transition-colors duration-150 hover:border-clinical-lit hover:text-hi"
        >
          {actionLabel}
        </button>
      )}
    </div>
  )
}

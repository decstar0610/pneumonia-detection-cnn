const LINKS = [
  { href: 'https://github.com/decstar0610/pneumonia-detection-cnn', label: 'Source' },
  {
    href: 'https://github.com/decstar0610/pneumonia-detection-cnn/blob/main/MODEL_CARD.md',
    label: 'Model card',
  },
  { href: 'https://huggingface.co/decstzz06/pneumoscan-model', label: 'Weights' },
] as const

export function Footer() {
  return (
    <footer className="mt-8 border-t border-line bg-ink-850">
      <div className="mx-auto flex max-w-[1180px] flex-col gap-3 px-5 py-6 text-micro text-low md:flex-row md:items-start md:justify-between">
        <div className="max-w-2xl space-y-1.5">
          <p>
            DenseNet121 trained on pediatric chest X-rays (Kaggle, Paul Mooney), externally
            validated without retraining on the adult RSNA set. Decision threshold tuned on
            validation for sensitivity ≥ 0.92, then frozen; probabilities temperature-scaled.
          </p>
          <p>
            Known limits: external specificity falls to 0.47, PA-view sensitivity to 0.615, and
            Grad-CAM sometimes attends to image borders. See the model report and model card before
            drawing any conclusion.
          </p>
        </div>
        <nav className="flex shrink-0 gap-4" aria-label="Project links">
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              target="_blank"
              rel="noreferrer"
              className="text-mid transition-colors duration-150 hover:text-clinical-lit"
            >
              {link.label}
            </a>
          ))}
        </nav>
      </div>
    </footer>
  )
}

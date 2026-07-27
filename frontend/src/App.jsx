import { useState, useRef } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Triage zone -> presentation. "needs_human_review" is shown prominently: a feature, not a bug.
const ZONES = {
  urgent_review:      { label: 'Urgent review',        cls: 'bg-rose-50 text-rose-700 ring-rose-600/20',    dot: 'bg-rose-500' },
  needs_human_review: { label: 'Needs human review',   cls: 'bg-amber-50 text-amber-700 ring-amber-600/20',  dot: 'bg-amber-500' },
  routine:            { label: 'Routine',              cls: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20', dot: 'bg-emerald-500' },
}

function Bar({ label, value, color }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-sm text-slate-600">
        <span>{label}</span><span className="tabular-nums">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-slate-100">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  function choose(f) {
    if (!f) return
    setFile(f); setResult(null); setError(null)
    setPreview(URL.createObjectURL(f))
  }

  async function analyze() {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    try {
      const body = new FormData()
      body.append('file', file)
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)
      setResult(data)
    } catch (e) {
      setError(e.message || 'Could not reach the API. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const zone = result && (ZONES[result.triage_zone] || ZONES.routine)

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6 py-5">
          <h1 className="text-2xl font-bold tracking-tight">PneumoScan</h1>
          <p className="text-sm text-slate-500">Chest X-ray pneumonia triage — sensitivity-first, uncertainty-aware.</p>
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-6 py-4">
        <div className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-800 ring-1 ring-amber-600/20">
          <strong>Research prototype — not a diagnostic device.</strong> For demonstration only; do not use for clinical decisions.
        </div>
      </div>

      <main className="mx-auto grid max-w-5xl gap-6 px-6 pb-16 md:grid-cols-2">
        {/* Upload */}
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="mb-4 font-semibold">1 · Upload a chest X-ray</h2>
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); choose(e.dataTransfer.files?.[0]) }}
            className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 p-8 text-center hover:border-slate-400"
          >
            {preview
              ? <img src={preview} alt="uploaded X-ray" className="max-h-64 rounded-md" />
              : <p className="text-sm text-slate-500">Click or drag a JPEG/PNG chest X-ray here</p>}
            <input ref={inputRef} type="file" accept="image/png,image/jpeg" className="hidden"
                   onChange={(e) => choose(e.target.files?.[0])} />
          </div>
          <button onClick={analyze} disabled={!file || loading}
                  className="mt-4 w-full rounded-lg bg-slate-800 py-2.5 font-medium text-white hover:bg-slate-700 disabled:opacity-40">
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
          {error && <p className="mt-3 text-sm text-rose-600">{error}</p>}
        </section>

        {/* Result */}
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="mb-4 font-semibold">2 · Result</h2>
          {!result && <p className="text-sm text-slate-400">Upload an image and click Analyze.</p>}
          {result && (
            <div>
              <div className="mb-4 flex items-center justify-between">
                <span className="text-xl font-bold">{result.prediction}</span>
                <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset ${zone.cls}`}>
                  <span className={`h-2 w-2 rounded-full ${zone.dot}`} />{zone.label}
                </span>
              </div>

              <p className="mb-4 text-sm text-slate-600">
                Calibrated confidence <strong className="tabular-nums">{(result.calibrated_confidence * 100).toFixed(1)}%</strong>
                {result.is_uncertain && <span className="ml-1 text-amber-700">· flagged uncertain → escalated to a human</span>}
              </p>

              <Bar label="Pneumonia" value={result.probability.pneumonia} color="bg-rose-500" />
              <Bar label="Normal" value={result.probability.normal} color="bg-emerald-500" />

              <div className="mt-5">
                <p className="mb-2 text-sm font-medium text-slate-600">Grad-CAM — where the model looked</p>
                <img src={result.gradcam_overlay} alt="Grad-CAM overlay" className="w-full max-w-xs rounded-md" />
              </div>

              <p className="mt-4 rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-600">{result.recommendation}</p>
            </div>
          )}
        </section>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6 py-4 text-xs text-slate-400">
          PneumoScan · trained on pediatric chest X-rays, externally validated on RSNA (adult). Decision threshold tuned for sensitivity ≥ 0.92. Not a diagnostic device.
        </div>
      </footer>
    </div>
  )
}

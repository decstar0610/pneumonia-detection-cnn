/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the PneumoScan FastAPI backend (no trailing slash). */
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

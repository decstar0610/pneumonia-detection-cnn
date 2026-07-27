# PneumoScan — Deployment (Phase 8)

Public demo = **HF Docker Space** (API, model baked in) + **Vercel** (React frontend).
You create the accounts and authenticate; the scripts/configs here do the rest.

---

## Part A — API → Hugging Face Docker Space

**Prereqs (you do once):**
1. Create a free account at <https://huggingface.co/join>.
2. Create a **WRITE** token at <https://huggingface.co/settings/tokens>.
3. Log in from this repo:
   ```
   ! ./.venv/Scripts/huggingface-cli.exe login
   ```
   (paste the WRITE token; the `!` prefix runs it in this session so I see the result.)

**Deploy:**
```
./.venv/Scripts/python.exe deploy/deploy_hf_space.py --repo-id <your-username>/pneumoscan-api
```
This assembles `deploy/_space_build/` (Dockerfile + README + code + model, LFS-tracked)
and uploads it as a Docker Space. The Space then builds the image automatically —
watch the **Logs** tab; first build takes a few minutes (installs tensorflow-cpu).

- Dry run without uploading: add `--stage-only`.
- Private Space: add `--private`.

**API base URL** once the build is green:
```
https://<your-username>-pneumoscan-api.hf.space
```
Verify:
```
curl https://<your-username>-pneumoscan-api.hf.space/health
# -> {"status":"ok","model_version":"2.0"}
```
Interactive docs: append `/docs` to that URL.

---

## Part B — Frontend → Vercel

**Prereqs:** account at <https://vercel.com/signup> (sign in with GitHub is easiest).

**Option 1 — Vercel dashboard (no CLI):**
1. Push this repo to GitHub, then **New Project** → import it.
2. Set **Root Directory** = `frontend` (Vercel reads `frontend/vercel.json`; framework = Vite auto-detected).
3. Add an **Environment Variable**:
   - `VITE_API_URL` = `https://<your-username>-pneumoscan-api.hf.space`  (no trailing slash)
4. Deploy. Vercel gives you `https://<project>.vercel.app`.

**Option 2 — Vercel CLI:**
```
cd frontend
npx vercel --prod -e VITE_API_URL=https://<your-username>-pneumoscan-api.hf.space
```

> `VITE_API_URL` is read at **build** time (baked into the bundle). If you change it,
> trigger a redeploy.

---

## Part C — Lock down CORS (after both URLs exist)

The API defaults to `CORS_ALLOW_ORIGINS=*`. Tighten it to your Vercel origin:

1. On the HF Space → **Settings → Variables and secrets** → add:
   - `CORS_ALLOW_ORIGINS` = `https://<project>.vercel.app`
2. The Space restarts. (Comma-separate if you have a preview + prod origin.)

---

## Part D — Verify end-to-end

1. `GET /health` on the Space returns `{"status":"ok",...}`.
2. Open the Vercel URL, upload a chest X-ray (e.g. one from
   `data/raw/kaggle_chest_xray/chest_xray/test/`), confirm a prediction + triage
   badge + Grad-CAM overlay render, cross-origin, with no CORS error in the console.
3. Note both URLs for the README hero/badges (Phase 9).

---

## Notes / gotchas
- HF free Spaces sleep after inactivity; first request after a nap is a cold start.
- The model (37 MB `.keras`) is baked into the image via Git LFS — no separate HF
  model repo is required. (If you later want a standalone model card on the Hub,
  that's an optional extra, not needed for the demo.)
- Rerun `deploy_hf_space.py` any time to push updates; it commits over the same Space.

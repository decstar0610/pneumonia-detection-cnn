# PneumoScan — Deployment (Phase 8)

Public demo = **Render** (API, Docker) + **Vercel** (React frontend).
The trained model lives in a **free HF Hub model repo**
(<https://huggingface.co/decstzz06/pneumoscan-model>) and is pulled into the API
image at build time, so nothing large is committed to git.

> Note: HF *Docker Spaces* now require paid PRO, so the API moved to Render's free
> tier. The old `deploy/deploy_hf_space.py` + `deploy/Dockerfile` are kept only for
> reference if you ever get PRO.

---

## Step 0 — Push the repo to GitHub (both hosts need it)

The local repo is already initialized and committed on `main`. Create an empty
GitHub repo (no README), then:
```
git remote add origin https://github.com/<you>/pneumonia-detection-cnn.git
git push -u origin main
```

---

## Part A — API → Render (free Docker web service)

1. Sign up at <https://render.com> (sign in with GitHub).
2. **New → Blueprint**, connect the GitHub repo. Render reads `render.yaml` and
   provisions the `pneumoscan-api` web service (Docker, free plan). Apply.
   - The build runs `api/Dockerfile`, which `pip install`s the CPU stack and
     downloads the model from HF Hub. First build takes several minutes.
3. When live, the API base URL is `https://pneumoscan-api.onrender.com`
   (Render shows the exact URL). Verify:
   ```
   curl https://pneumoscan-api.onrender.com/health
   # -> {"status":"ok","model_version":"2.0"}
   ```
   Docs at `/docs`.

> Free tier sleeps after ~15 min idle; the first request after a nap cold-starts
> (~50s). Fine for a demo — the frontend shows a spinner.

**Manual alternative (no Blueprint):** New → Web Service → connect repo →
Runtime **Docker**, Dockerfile path `./api/Dockerfile`, context `.`, plan Free,
health check `/health`, add env var `CORS_ALLOW_ORIGINS=*`.

---

## Part B — Frontend → Vercel

1. Sign up at <https://vercel.com/signup> (GitHub sign-in).
2. **New Project** → import the repo. Set:
   - **Root Directory** = `frontend`  (Vercel reads `frontend/vercel.json`; Vite auto-detected)
   - **Environment Variable**: `VITE_API_URL` = your Render URL (no trailing slash),
     e.g. `https://pneumoscan-api.onrender.com`
3. Deploy → you get `https://<project>.vercel.app`.

> `VITE_API_URL` is baked in at **build** time. If you change it, redeploy.

---

## Part C — Lock down CORS (after both URLs exist)

The API defaults to `CORS_ALLOW_ORIGINS=*`. Tighten it:
- Render → your service → **Environment** → set `CORS_ALLOW_ORIGINS` =
  `https://<project>.vercel.app` → save (service redeploys).

---

## Part D — Verify end-to-end

1. `GET /health` on Render returns `{"status":"ok",...}`.
2. Open the Vercel URL, upload a chest X-ray from
   `data/raw/kaggle_chest_xray/chest_xray/test/`, confirm the prediction + triage
   badge + Grad-CAM overlay render with no CORS error in the browser console.
3. Save both URLs for the README badges (Phase 9).

---

## Updating later
- **Code**: `git push` → Render and Vercel auto-redeploy from `main`.
- **Model**: re-upload to the HF Hub model repo, then trigger a Render rebuild
  (Manual Deploy → Clear build cache & deploy) so it re-downloads.

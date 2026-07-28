"""Phase 6 — FastAPI service exposing the PneumoScan model (PRD §7.2 / §7.3).

Endpoints:
  GET  /health   -> {"status":"ok","model_version":"2.0"}
  POST /predict  -> multipart image upload, returns the §7.2 JSON contract (incl. base64 Grad-CAM)

The model loads once at startup; bad uploads return 422; CORS is open for the frontend.
Run locally:
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m uvicorn api.main:app --port 8000
Interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import MODEL_VERSION, predict_image, warmup

ALLOWED = {"image/jpeg", "image/png", "image/jpg", "application/octet-stream"}

# Comma-separated origins allowed to call the API. Default "*" for local dev;
# in production set CORS_ALLOW_ORIGINS to the deployed frontend origin,
# e.g. CORS_ALLOW_ORIGINS="https://pneumoscan.vercel.app"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]

# `allow_origins` is exact-match, which would block Vercel *preview* deployments —
# every branch and commit preview gets its own generated subdomain. Set
# CORS_ALLOW_ORIGIN_REGEX to admit them without widening the policy back to "*".
CORS_ORIGIN_REGEX = os.getenv("CORS_ALLOW_ORIGIN_REGEX") or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup()  # load model + artifacts once, before serving
    yield


app = FastAPI(title="PneumoScan API", version=MODEL_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # set CORS_ALLOW_ORIGINS env var to the frontend origin in production
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type and file.content_type not in ALLOWED:
        raise HTTPException(status_code=422, detail=f"Unsupported content type: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty upload.")
    try:
        return predict_image(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

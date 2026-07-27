---
title: PneumoScan API
emoji: 🫁
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Chest X-ray pneumonia triage API (DenseNet121 + Grad-CAM)
---

# PneumoScan API

FastAPI inference service for **PneumoScan** — a chest X-ray pneumonia *triage*
model (DenseNet121, calibrated, with abstention-based triage and Grad-CAM).

## Endpoints

- `GET /health` → `{"status":"ok","model_version":"2.0"}`
- `POST /predict` → multipart image upload; returns the prediction JSON contract
  (label, calibrated confidence, triage zone, recommendation, base64 Grad-CAM overlay).
- Interactive docs at `/docs`.

## Notes

Research/education demo only — **not a medical device**. The model was trained on
pediatric data (Kaggle) and false-alarms on adult negatives (see the project model
card for external-validation, fairness, and Grad-CAM limitations).

Build context is this repo root; the model is baked into the image at build time.

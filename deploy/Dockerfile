# PneumoScan API — HF Docker Space image (CPU inference).
# The push script (deploy/deploy_hf_space.py) assembles a staging folder whose
# root mirrors the repo (src/, api/, models/) so these COPY paths resolve.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install -r /app/api/requirements.txt

# App code + the trained artifacts the API loads at startup.
COPY src /app/src
COPY api /app/api
COPY models/best_model.keras /app/models/best_model.keras
COPY models/threshold.json /app/models/threshold.json
COPY models/temperature.json /app/models/temperature.json
COPY models/triage.json /app/models/triage.json

# HF Spaces routes to $PORT (7860 by default; matches app_port in README frontmatter).
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

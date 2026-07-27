# PneumoScan API — CPU inference image for Render (or any Docker host).
# Build context is the repo root:  docker build -f api/Dockerfile -t pneumoscan-api .
#
# The trained model is NOT in the git repo (it's gitignored). Instead it is
# pulled at build time from the public HF Hub model repo, so the build works on
# Render straight from GitHub with no large files committed.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install -r /app/api/requirements.txt

# Fetch the trained artifacts from the public HF Hub model repo (no token needed).
# Override MODEL_REPO at build time (--build-arg) to point at a fork.
ARG MODEL_REPO=decstzz06/pneumoscan-model
RUN pip install "huggingface_hub>=0.23,<2" && \
    python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', repo_type='model', local_dir='/app/models', \
allow_patterns=['*.keras','*.json'])"

# App code.
COPY src /app/src
COPY api /app/api

# Render provides $PORT at runtime; default 7860 for local runs.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

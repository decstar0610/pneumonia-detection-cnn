# PneumoScan API — CPU inference image for Render (or any Docker host).
# TensorFlow-free (onnxruntime + NumPy) so it fits Render's 512MB free tier.
# Build context is the repo root:  docker build -f api/Dockerfile -t pneumoscan-api .
#
# The model artifacts are NOT in the git repo. They're pulled at build time from the
# public HF Hub model repo, so the build works on Render straight from GitHub.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install -r /app/api/requirements.txt

# Fetch the ONNX backbone + NumPy head weights + calibration jsons from the public
# HF Hub model repo (no token needed). Override MODEL_REPO with --build-arg for a fork.
ARG MODEL_REPO=decstzz06/pneumoscan-model
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', repo_type='model', local_dir='/app/models', \
allow_patterns=['backbone.onnx','gradcam_head.npz','*.json'])"

# App code. The serve path uses only src/config (pure) + api/onnx_head (NumPy);
# the TF-importing src modules are present but never imported at serve time.
COPY src /app/src
COPY api /app/api

# Render provides $PORT at runtime; default 7860 for local runs.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

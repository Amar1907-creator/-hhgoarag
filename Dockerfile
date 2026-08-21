# HHGOARAG — self-contained image for the live deployment.
#
# The embedding model and the FAISS index are baked in, so the container starts
# without reaching the network and serves the same corpus that was benchmarked.
# Answer generation falls back to quoting retrieved evidence: a local Ollama
# model is not shipped here, and the fallback is still fully grounded and cited.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HHGOARAG_PREFIX=hi-train-5k \
    HF_HUB_OFFLINE=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the encoder at build time. Without this the first request would
# wait on a 470 MB download, and HF_HUB_OFFLINE would make it fail outright.
RUN HF_HUB_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small')"

COPY src/ src/
COPY scripts/ scripts/
COPY static/ static/
COPY tests/fixtures/ tests/fixtures/
COPY data/manifests/ data/manifests/
# The built corpus and index. Build the image from a checkout where these exist.
COPY data/processed/hi-train-5k-corpus.jsonl data/processed/
COPY data/processed/hi-train-5k-index/ data/processed/hi-train-5k-index/
COPY data/processed/hi-validation-5k-evaluation.jsonl data/processed/
COPY data/demo/ data/demo/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD curl -fsS http://127.0.0.1:${PORT:-8000}/health || exit 1

CMD ["python", "scripts/run_app.py", "--host", "0.0.0.0", "--no-browser"]

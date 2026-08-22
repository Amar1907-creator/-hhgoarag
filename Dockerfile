# HHGOARAG — self-contained image for the live deployment.
#
# The embedding model and the FAISS index are baked in, so the container starts
# without reaching the network and serves the same corpus that was benchmarked.
# Answer generation falls back to quoting retrieved evidence: a local Ollama
# model is not shipped here, and the fallback is still fully grounded and cited.
#
# Embedding runs on ONNX Runtime, not PyTorch: torch alone costs ~190 MB of
# resident memory just to import, before any model weights, which does not fit
# a 512 MB deployment target. The ONNX file below is the official export
# published inside the intfloat/multilingual-e5-small repository itself --
# same vocabulary and weights, just quantized and run without torch. See
# src/retrieval/embedding.py:OnnxE5Embedder and requirements-prod.txt.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HHGOARAG_PREFIX=hi-train-5k \
    HHGOARAG_EMBEDDING_BACKEND=onnx

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Fetch the ONNX model and tokenizer at build time with curl (already
# installed above) rather than any Python HF client -- there is no
# huggingface_hub in this image, and none is needed for two files.
RUN mkdir -p data/models/multilingual-e5-small-onnx \
 && curl -fsSL -o data/models/multilingual-e5-small-onnx/model.onnx \
      https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model_qint8_avx512_vnni.onnx \
 && curl -fsSL -o data/models/multilingual-e5-small-onnx/tokenizer.json \
      https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json

COPY src/ src/
COPY scripts/ scripts/
COPY static/ static/
COPY tests/fixtures/ tests/fixtures/
COPY data/manifests/ data/manifests/
# The built corpus and index. Build the image from a checkout where these exist.
COPY data/processed/hi-train-5k-corpus.jsonl data/processed/
COPY data/processed/hi-train-5k-index/ data/processed/hi-train-5k-index/
COPY data/demo/ data/demo/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD curl -fsS http://127.0.0.1:${PORT:-8000}/health || exit 1

CMD ["python", "scripts/run_app.py", "--host", "0.0.0.0", "--generator", "extractive", "--no-browser"]

# Deploying the live link

The image is self-contained: the encoder and the FAISS index are baked in, so the
container starts without reaching the network.

## Build and run locally first

```bash
cd ~/hhgoarag
docker build -t hhgoarag .
docker run --rm -p 8000:8000 -e SARVAM_API_KEY=your-key hhgoarag
open http://127.0.0.1:8000
```

The build copies `data/processed/hi-train-5k-*`, so run it from a checkout where
the corpus and index exist. Image is roughly 2 GB, mostly PyTorch and the encoder.

## Render (blueprint included)

1. Push to GitHub.
2. Render → **New → Blueprint** → pick the repo. `render.yaml` configures a Docker
   web service with `/health` as the health check.
3. Add **`SARVAM_API_KEY`** in the dashboard. It is marked `sync: false` so it is
   never committed.
4. Deploy. First boot takes a couple of minutes while the model loads.

Use an instance with at least **2 GB RAM** — the encoder plus a 90 MB index will
not fit in 512 MB.

## Railway / Fly.io

Both read the `Dockerfile` directly.

```bash
fly launch --dockerfile Dockerfile --no-deploy
fly secrets set SARVAM_API_KEY=your-key
fly deploy
```

Set `PORT` if the platform expects a specific one — `scripts/run_app.py` honours it.

## What the live link can and cannot do

- Retrieval, grounding, citations, abstention, PDF upload and speech-to-text all
  work.
- **Answer generation is extractive**: no Ollama model ships in the image, so
  answers quote the retrieved passage verbatim. That is still fully grounded and
  cited. To demonstrate generated answers, run locally with Ollama.

## Verify a deployment

```bash
curl -s https://your-app/health | python3 -m json.tool
curl -s https://your-app/api/speech | python3 -m json.tool
curl -s https://your-app/api/query -H 'content-type: application/json' \
  -d '{"question":"मैग्नीशियम क्या है?"}' | python3 -m json.tool
```

`/health` must report `"status": "ok"` with `corpus_passages` equal to
`index_vectors`.

#!/usr/bin/env bash
# HHGOARAG — one command from a clean checkout to a running application.
#
#   ./run.sh              build what is missing, then start the app
#   ./run.sh --limit 15000  build a larger corpus
#   ./run.sh --pull       also pull a local language model for generated answers
#   ./run.sh --app-only   skip all building, just start the app
#
# No API key is required at any point.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
LIMIT=5000
PULL=0
APP_ONLY=0
PORT=8000
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2;;
    --pull) PULL=1; shift;;
    --app-only) APP_ONLY=1; shift;;
    --port) PORT="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "Python and dependencies"
$PY --version
$PY -m pip install --quiet --disable-pip-version-check -r requirements.txt
$PY - <<'CHECK'
import importlib, sys
missing = [m for m in ("pyarrow", "faiss", "sentence_transformers", "fastapi", "uvicorn")
           if not importlib.util.find_spec(m)]
if missing:
    print("missing packages:", ", ".join(missing)); sys.exit(1)
print("dependencies ok")
CHECK

PREFIX="hi-train-$((LIMIT / 1000))k"
CORPUS="data/processed/${PREFIX}-corpus.jsonl"
INDEX="data/processed/${PREFIX}-index/index.faiss"
EVAL="data/processed/hi-validation-$((LIMIT / 1000))k-evaluation.jsonl"

if [[ $APP_ONLY -eq 0 ]]; then
  if [[ -f "$CORPUS" && -f "$INDEX" ]]; then
    say "Corpus and index already built ($PREFIX) — skipping the build"
  else
    say "Building corpus, evaluation and index ($LIMIT records, unattended, ~20 min)"
    $PY scripts/run_pipeline.py --limit "$LIMIT"
  fi

  if [[ -f "$EVAL" && ! -f data/demo/questions.json ]]; then
    say "Selecting demonstration questions from real data"
    $PY scripts/pick_demo_questions.py --corpus "$CORPUS" \
        --index-dir "data/processed/${PREFIX}-index" --evaluation "$EVAL" || true
  fi
fi

say "Local language model"
if command -v ollama >/dev/null 2>&1; then
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "starting ollama serve in the background…"
    (ollama serve >/dev/null 2>&1 &) ; sleep 3
  fi
  MODELS="$(curl -sf http://127.0.0.1:11434/api/tags 2>/dev/null || echo '')"
  if [[ $PULL -eq 1 || "$MODELS" != *"qwen2.5"* ]]; then
    if [[ $PULL -eq 1 ]]; then
      echo "pulling qwen2.5:3b-instruct (~2 GB)…"
      ollama pull qwen2.5:3b-instruct
    else
      echo "no qwen2.5 model installed. Answers will quote evidence verbatim (still fully grounded)."
      echo "For generated Hindi answers, run:  ./run.sh --pull"
    fi
  else
    echo "local model available"
  fi
else
  echo "ollama not installed — answers will quote evidence verbatim (still fully grounded)."
  echo "For generated Hindi answers: https://ollama.com/download then ./run.sh --pull"
fi

say "Starting HHGOARAG"
exec $PY scripts/run_app.py --port "$PORT" "${EXTRA[@]+"${EXTRA[@]}"}"

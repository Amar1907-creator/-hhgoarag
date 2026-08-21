#!/usr/bin/env python3
"""Start the HHGOARAG application.

Loads the embedding model, FAISS index and passage store once, then serves the
API and the web interface. No hosted API and no API key are involved.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.api import create_app  # noqa: E402
from src.app.service import Service, discover_prefix, resolve_device  # noqa: E402
from src.rag.generator import RECOMMENDED_SMALL, installed_models  # noqa: E402


def free_port(preferred: int, host: str) -> int:
    for candidate in range(preferred, preferred + 20):
        with socket.socket() as probe:
            if probe.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", candidate)) != 0:
                return candidate
    return preferred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefix", default=None, help="artifact prefix, e.g. hi-train-5k")
    parser.add_argument("--generator", choices=("auto", "ollama", "extractive"), default="auto")
    parser.add_argument("--device", default=None, help="cpu, mps or cuda; auto-detected by default")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    prefix = discover_prefix(args.prefix or "hi-train-5k")
    if prefix is None:
        print("No built corpus found under data/processed/.\n"
              "Build one first (about 20 minutes, unattended):\n"
              "  python3 scripts/run_pipeline.py --limit 5000", file=sys.stderr)
        raise SystemExit(2)

    device = resolve_device(args.device)
    models = installed_models()
    print(f"HHGOARAG starting")
    print(f"  corpus prefix : {prefix}")
    print(f"  device        : {device}")
    if args.generator == "extractive":
        print(f"  generator     : extractive (evidence-only, no model)")
    elif models:
        print(f"  ollama models : {', '.join(models[:4])}{' …' if len(models) > 4 else ''}")
    else:
        print(f"  ollama        : not reachable -- answers will quote evidence verbatim.")
        print(f"                  For generated answers: ollama serve && ollama pull {RECOMMENDED_SMALL}")

    service = Service(prefix=prefix, generator_kind=args.generator, device=device,
                      top_k=args.top_k, min_score=args.min_score)
    started = time.perf_counter()
    status = service.load()
    if not status.ready:
        print(f"\nfailed to start: {status.error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  passages      : {status.corpus_passages:,} (index {status.index_vectors:,} vectors)")
    print(f"  loaded in     : {time.perf_counter() - started:.1f}s")

    port = free_port(args.port, args.host)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{port}"
    print(f"\n  ready at {url}\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(create_app(service, load=False), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

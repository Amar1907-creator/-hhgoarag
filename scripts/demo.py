#!/usr/bin/env python3
"""End-to-end demo: Hindi question -> FAISS retrieval -> grounded answer.

Retrieval decides what is true; the model only phrases it. If retrieval is weak
or the answer cannot be tied to a retrieved passage, the demo abstains and says
why rather than producing something plausible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.generator import build_generator, installed_models
from src.rag.pipeline import RagPipeline

SAMPLE_QUESTIONS = [
    "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",
    "एक सख्त उबला हुआ अंडा कितने समय तक पकाते हैं?",
    "मैग्नीशियम क्या है?",
]


def render(result, show_evidence: bool) -> None:
    print("=" * 78)
    print(f"Q: {result.question}")
    print("-" * 78)
    if result.abstained:
        print(f"ABSTAINED ({result.reason})")
        if result.retrieval:
            best = result.retrieval[0]
            print(f"  best retrieval score {best['score']:.4f} for {best['passage_id'][:22]}...")
        if result.invented_citations:
            print(f"  citations not in evidence, dropped: {result.invented_citations}")
    else:
        print(f"A: {result.answer}")
        print(f"\nEvidence ({len(result.citations)} cited, generator {result.generator}):")
        for citation in result.citations:
            snippet = citation["text"][:180].replace("\n", " ")
            print(f"  [{citation['rank']}] score {citation['score']:.4f}  {citation['passage_id'][:22]}...")
            print(f"      {snippet}{'...' if len(citation['text']) > 180 else ''}")
        if result.invented_citations:
            print(f"  dropped uncheckable citations: {result.invented_citations}")
    timings = result.timings_ms
    print("\ntimings ms: " + "  ".join(f"{name}={value:.1f}" for name, value in timings.items()))
    if result.usage:
        print(f"tokens: {result.usage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/hi-train-5k-corpus.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/hi-train-5k-index"))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--generator", choices=("auto", "ollama", "extractive"), default="auto")
    parser.add_argument("--llm-model", default=None, help="Ollama model tag; best installed by default")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-evidence", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=None,
                        help="override the evidence score floor (default 0.80)")
    parser.add_argument("--question", action="append", help="ask one question; repeatable")
    parser.add_argument("--samples", action="store_true", help="run the built-in sample questions")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    parser.add_argument("--output", type=Path, help="also write the JSON results here")
    args = parser.parse_args()

    for path in (args.corpus, args.index_dir):
        if not path.exists():
            raise SystemExit(f"missing {path}. Run scripts/run_pipeline.py first.")

    generator = build_generator(args.generator, model=args.llm_model)
    if args.generator != "extractive" and not installed_models():
        print("[demo] no local Ollama model reachable; answers will quote retrieved "
              "evidence verbatim (still fully grounded).", file=sys.stderr)

    pipeline = RagPipeline.load(corpus=args.corpus, index_dir=args.index_dir, model=args.model,
                                device=args.device, generator=generator, top_k=args.top_k,
                                min_score=args.min_score, max_evidence=args.max_evidence)
    questions = list(args.question or [])
    if args.samples or not questions:
        questions = questions or list(SAMPLE_QUESTIONS)

    results = []
    try:
        for question in questions:
            result = pipeline.answer(question)
            results.append(result.to_dict())
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                render(result, show_evidence=True)

        if args.interactive:
            print("\nInteractive mode. Blank line or Ctrl-D to exit.")
            while True:
                try:
                    question = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    break
                result = pipeline.answer(question)
                results.append(result.to_dict())
                render(result, show_evidence=True)
    finally:
        pipeline.close()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(f"\nwrote {args.output}")

    grounded = sum(1 for r in results if r["grounded"])
    print(f"\n{grounded}/{len(results)} questions answered with cited evidence; "
          f"{len(results) - grounded} abstained.")


if __name__ == "__main__":
    main()

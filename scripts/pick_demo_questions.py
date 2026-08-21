#!/usr/bin/env python3
"""Choose demonstration questions that are known to behave, from real data.

Curating demo questions by hand is guesswork: whether a question has strong
evidence depends on what actually landed in the corpus. This runs candidate
questions through the real pipeline and keeps the ones that demonstrate each
behaviour, so the demo cannot embarrass anyone on the day.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.generator import ExtractiveGenerator  # noqa: E402
from src.rag.pipeline import RagPipeline  # noqa: E402

# Questions that cannot be supported by an MSMARCO passage corpus, used to show
# that the system declines rather than improvises.
ABSTENTION_CANDIDATES = [
    "क्या बृहस्पति ग्रह पर मानव बस्तियाँ स्थापित हो चुकी हैं?",
    "मेरे बैंक खाते में इस समय कितना पैसा है?",
    "अगले सप्ताह मुंबई में सोने का भाव क्या होगा?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="evaluation JSONL to draw real, answerable questions from")
    parser.add_argument("--output", type=Path, default=Path("data/demo/questions.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidates", type=int, default=40)
    args = parser.parse_args()

    pipeline = RagPipeline.load(corpus=args.corpus, index_dir=args.index_dir,
                                device=args.device, generator=ExtractiveGenerator(), top_k=10)
    with args.evaluation.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()][: args.candidates]

    scored = []
    for row in rows:
        result = pipeline.answer(row["query"])
        if not result.grounded:
            continue
        best = result.retrieval[0]["score"]
        strong_support = sum(1 for hit in result.retrieval[:5] if hit["score"] >= 0.86)
        hit_at_1 = result.retrieval and result.retrieval[0]["passage_id"] in row["positive_passage_ids"]
        scored.append({"question": row["query"], "best": best, "support": strong_support,
                       "correct": bool(hit_at_1), "citations": len(result.citations)})

    scored.sort(key=lambda item: (item["correct"], item["best"]), reverse=True)
    chosen: list[dict] = []
    used: set[str] = set()

    def take(predicate, label, limit=1):
        taken = 0
        for item in scored:
            if taken >= limit or item["question"] in used:
                continue
            if predicate(item):
                chosen.append({"question": item["question"], "expect": "evidence", "label": label,
                               "best_score": round(item["best"], 4)})
                used.add(item["question"])
                taken += 1

    take(lambda i: i["correct"] and i["best"] >= 0.90, "strong single-passage evidence")
    take(lambda i: i["support"] >= 3, "answer drawn from several passages")
    take(lambda i: i["citations"] >= 2, "multiple citations")
    take(lambda i: True, "realistic Hindi query", limit=2)

    for question in ABSTENTION_CANDIDATES:
        result = pipeline.answer(question)
        if not result.grounded:
            chosen.append({"question": question, "expect": "abstention",
                           "label": "safe abstention — no supporting evidence",
                           "best_score": round(result.retrieval[0]["score"], 4) if result.retrieval else 0.0})
            break

    pipeline.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chosen, ensure_ascii=False, indent=2) + "\n")
    print(f"selected {len(chosen)} demonstration questions -> {args.output}")
    for item in chosen:
        print(f"  [{item['expect']:<10}] {item['best_score']:.3f}  {item['question'][:60]}  ({item['label']})")
    if len(chosen) < 5:
        print("\nfewer than 5 demonstration questions were found; the interface will "
              "top up from its built-in list.", file=sys.stderr)


if __name__ == "__main__":
    main()

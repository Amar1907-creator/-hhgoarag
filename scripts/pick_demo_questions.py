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

# Questions no web-passage corpus can support: private facts, the future, and
# entities that do not exist. Several are tried and the one that retrieves
# WORST is kept, so the abstention demo is chosen by measurement rather than by
# assuming which question will fail.
ABSTENTION_BY_LANGUAGE = {
    "hi": [
    "मेरे बैंक खाते में इस समय कितना पैसा है?",
    "मेरा आधार नंबर क्या है?",
    "कल दोपहर मेरी मीटिंग किसके साथ है?",
    "अगले सप्ताह मुंबई में सोने का भाव क्या होगा?",
    "इस कमरे में इस समय कितने लोग बैठे हैं?",
    "क्या बृहस्पति ग्रह पर मानव बस्तियाँ स्थापित हो चुकी हैं?",
    "ज़ोर्ब्लैक्स ग्रह की राजधानी का नाम क्या है?",
    "क्विंबल्टन विश्वविद्यालय की स्थापना किस वर्ष हुई थी?",
    "मेरे पड़ोसी की बिल्ली का नाम क्या है?",
        "अगले महीने का लॉटरी नंबर क्या होगा?",
    ],
}
# Deliberately not machine-translated into the other twelve languages. A
# demonstration question that is subtly ungrammatical is worse than none, so a
# language without curated candidates is reported rather than guessed at.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="evaluation JSONL to draw real, answerable questions from")
    parser.add_argument("--output", type=Path, default=Path("data/demo/questions.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--language", default="hi", help="language code, for the abstention candidates")
    args = parser.parse_args()

    from src.languages import find as find_language
    language = find_language(args.language)
    candidates = ABSTENTION_BY_LANGUAGE.get(args.language, [])

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

    refusals, weakest = [], None
    if not candidates:
        print(f"\nNOTE: no curated abstention questions exist for "
              f"{language.english if language else args.language}. The evidence questions above are "
              f"verified; the abstention example must be written and checked by a speaker before it "
              f"is demonstrated.", file=sys.stderr)
    for question in candidates:
        result = pipeline.answer(question)
        score = round(result.retrieval[0]["score"], 4) if result.retrieval else 0.0
        if not result.grounded:
            refusals.append((score, question))
        if weakest is None or score < weakest[0]:
            weakest = (score, question)

    if refusals:
        refusals.sort()
        score, question = refusals[0]
        chosen.append({"question": question, "expect": "abstention",
                       "label": "safe abstention — no supporting evidence", "best_score": score})
    elif weakest:
        # Nothing was refused at the current floor. Say so rather than shipping a
        # demo question that promises an abstention and then answers.
        score, question = weakest
        print(f"\nWARNING: none of the {len(candidates)} unanswerable candidates was "
              f"refused; the weakest still scored {score:.4f}, above the evidence floor.\n"
              f"         The abstention demo is only trustworthy if the floor is above that.\n"
              f"         Re-run with:  HHGOARAG_MIN_SCORE={min(0.95, score + 0.01):.2f} "
              f"python3 scripts/pick_demo_questions.py ...", file=sys.stderr)
        chosen.append({"question": question, "expect": "abstention_unverified",
                       "label": f"weakest retrieval found ({score:.3f}) — verify before demonstrating",
                       "best_score": score})

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

"""Query -> embedding -> FAISS -> evidence -> grounded answer.

Retrieval decides what is true; the model only phrases it. Every stage is timed
and every answer carries the reason it was produced or withheld.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.rag.evidence import EvidenceSet, select_evidence, validate_citations
from src.rag.generator import ExtractiveGenerator, Generator
from src.rag.store import PassageTextStore

REASON_UNGROUNDED = "answer_without_valid_citations"
REASON_MODEL_ABSTAINED = "model_reported_insufficient_evidence"


@dataclass
class Answer:
    question: str
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    grounded: bool = False
    abstained: bool = True
    reason: str = ""
    generator: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieval: list[dict] = field(default_factory=list)
    invented_citations: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RagPipeline:
    def __init__(self, *, embedder, index, texts: PassageTextStore,
                 generator: Generator | None = None, top_k: int = 10,
                 min_score: float | None = None, max_evidence: int = 5) -> None:
        self.embedder = embedder
        self.index = index
        self.texts = texts
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k
        self.max_evidence = max_evidence
        self.min_score = min_score

    @classmethod
    def load(cls, *, corpus: Path, index_dir: Path, model: str = "intfloat/multilingual-e5-small",
             device: str | None = None, generator: Generator | None = None, top_k: int = 10,
             min_score: float | None = None, max_evidence: int = 5) -> "RagPipeline":
        from src.retrieval.embedding import SentenceTransformerE5
        from src.retrieval.index import FaissHNSWIndex
        embedder = SentenceTransformerE5(model, device=device)
        loaded = FaissHNSWIndex.load(index_dir)
        if loaded.dimension != embedder.dimension:
            raise SystemExit(
                f"index dimension {loaded.dimension} does not match {model} ({embedder.dimension}); "
                f"the index was built with a different embedding model")
        return cls(embedder=embedder, index=loaded, texts=PassageTextStore.build(corpus),
                   generator=generator, top_k=top_k, min_score=min_score, max_evidence=max_evidence)

    def retrieve(self, question: str) -> tuple[list[tuple[str, float]], dict[str, float]]:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        vector = self.embedder.embed_queries([question])
        timings["embed"] = (time.perf_counter() - start) * 1e3
        start = time.perf_counter()
        hits = self.index.search(vector, self.top_k)[0]
        timings["search"] = (time.perf_counter() - start) * 1e3
        return hits, timings

    def answer(self, question: str) -> Answer:
        overall = time.perf_counter()
        result = Answer(question=question, generator=self.generator.name)
        if not question or not question.strip():
            result.reason = "empty_question"
            return result

        hits, timings = self.retrieve(question)
        start = time.perf_counter()
        texts = self.texts.texts([passage_id for passage_id, _ in hits])
        timings["lookup"] = (time.perf_counter() - start) * 1e3
        result.retrieval = [{"passage_id": passage_id, "score": round(float(score), 4)}
                            for passage_id, score in hits]

        kwargs = {"max_items": self.max_evidence}
        if self.min_score is not None:
            kwargs["min_score"] = self.min_score
        evidence: EvidenceSet = select_evidence(hits, texts, **kwargs)
        if not evidence.eligible:
            result.reason = evidence.reason
            timings["total"] = (time.perf_counter() - overall) * 1e3
            result.timings_ms = {name: round(value, 2) for name, value in timings.items()}
            return result

        start = time.perf_counter()
        generated = self.generator.generate(question, evidence)
        timings["generate"] = (time.perf_counter() - start) * 1e3
        result.usage = generated.usage

        if generated.insufficient:
            result.reason = REASON_MODEL_ABSTAINED
        else:
            kept, invented = validate_citations(generated.citations, evidence)
            result.invented_citations = invented
            if not kept:
                # An answer nobody can check is worse than no answer.
                result.reason = REASON_UNGROUNDED
            else:
                by_id = {item.passage_id: item for item in evidence.items}
                result.answer = generated.answer
                result.citations = [{"passage_id": passage_id, "score": round(by_id[passage_id].score, 4),
                                     "rank": by_id[passage_id].rank,
                                     "text": by_id[passage_id].text} for passage_id in kept]
                result.grounded = True
                result.abstained = False
                result.reason = evidence.reason

        timings["total"] = (time.perf_counter() - overall) * 1e3
        result.timings_ms = {name: round(value, 2) for name, value in timings.items()}
        return result

    def close(self) -> None:
        self.texts.close()

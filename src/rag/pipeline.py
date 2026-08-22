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
from src.rag.guardrails import screen_input, screen_output, screen_relevance
from src.rag.generator import ExtractiveGenerator, Generator
from src.rag.sources import CorpusSource, KnowledgeSource
from src.rag.store import PassageTextStore

REASON_UNGROUNDED = "answer_without_valid_citations"
REASON_MODEL_ABSTAINED = "model_reported_insufficient_evidence"
REASON_ENTITY_MISMATCH = "evidence_entity_mismatch"


@dataclass
class Answer:
    question: str
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    source: str = "corpus"
    sources_used: list[dict] = field(default_factory=list)
    grounded: bool = False
    abstained: bool = True
    reason: str = ""
    generator: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieval: list[dict] = field(default_factory=list)
    invented_citations: list[str] = field(default_factory=list)
    guardrail: dict = field(default_factory=dict)
    blocked: bool = False
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RagPipeline:
    def __init__(self, *, embedder, index=None, texts: PassageTextStore | None = None,
                 generator: Generator | None = None, top_k: int = 10,
                 min_score: float | None = None, max_evidence: int = 5,
                 sources: dict[str, KnowledgeSource] | None = None,
                 default_source: str = "corpus") -> None:
        self.embedder = embedder
        self.index = index
        self.texts = texts
        self.sources: dict[str, KnowledgeSource] = dict(sources or {})
        if index is not None and texts is not None and "corpus" not in self.sources:
            self.sources["corpus"] = CorpusSource(index, texts)
        self.default_source = default_source
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

    # -- sources ---------------------------------------------------------
    def add_source(self, source: KnowledgeSource) -> None:
        self.sources[source.key] = source

    def remove_source(self, key: str) -> None:
        self.sources.pop(key, None)

    def resolve_source(self, key: str | None) -> KnowledgeSource:
        wanted = key or self.default_source
        if wanted in self.sources:
            return self.sources[wanted]
        raise KeyError(f"unknown knowledge source {wanted!r}; "
                       f"available: {', '.join(sorted(self.sources)) or 'none'}")

    def describe_sources(self) -> list[dict]:
        return [{"key": source.key, "label": source.label, "kind": source.kind,
                 "size": source.size()} for source in self.sources.values()]

    def retrieve(self, question: str, source: KnowledgeSource | None = None):
        timings: dict[str, float] = {}
        target = source or self.resolve_source(None)
        start = time.perf_counter()
        vector = self.embedder.embed_queries([question])
        timings["embed"] = (time.perf_counter() - start) * 1e3
        start = time.perf_counter()
        hits = target.search(vector, self.top_k)
        timings["search"] = (time.perf_counter() - start) * 1e3
        return hits, timings

    def answer(self, question: str, source: str | None = None) -> Answer:
        overall = time.perf_counter()
        target = self.resolve_source(source)
        result = Answer(question=question, generator=self.generator.name, source=target.key)

        # Guardrail 1: refuse before spending an embedding, let alone a model call.
        verdict = screen_input(question)
        result.guardrail = verdict.to_dict()
        if not verdict.allowed:
            result.blocked = True
            result.reason = verdict.code
            result.timings_ms = {"total": round((time.perf_counter() - overall) * 1e3, 2)}
            return result

        hits, timings = self.retrieve(question, target)
        start = time.perf_counter()
        texts = target.hydrate([passage_id for passage_id, _ in hits])
        timings["lookup"] = (time.perf_counter() - start) * 1e3
        result.retrieval = [{"passage_id": passage_id, "score": round(float(score), 4)}
                            for passage_id, score in hits]

        kwargs = {"max_items": self.max_evidence}
        floor = self.min_score if self.min_score is not None else getattr(target, "min_score", None)
        if floor is not None:
            kwargs["min_score"] = floor
        evidence: EvidenceSet = select_evidence(hits, texts, **kwargs)
        if not evidence.eligible:
            result.reason = evidence.reason
            timings["total"] = (time.perf_counter() - overall) * 1e3
            result.timings_ms = {name: round(value, 2) for name, value in timings.items()}
            return result

        # Guardrail 2: refuse before the (expensive) model call when the
        # evidence is plainly about a different country than the question.
        relevance = screen_relevance(question, [item.text for item in evidence.items])
        result.guardrail = {**result.guardrail, "relevance": relevance.to_dict()}
        if not relevance.allowed:
            result.reason = REASON_ENTITY_MISMATCH
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
                # Guardrail 3: the answer must be carried by what it cites.
                supported = screen_output(generated.answer, [by_id[pid].text for pid in kept])
                result.guardrail = {**result.guardrail, "output": supported.to_dict()}
                if not supported.allowed:
                    result.reason = REASON_UNGROUNDED
                    timings["total"] = (time.perf_counter() - overall) * 1e3
                    result.timings_ms = {n: round(v, 2) for n, v in timings.items()}
                    return result
                result.answer = generated.answer
                result.citations = [{"passage_id": passage_id, "score": round(by_id[passage_id].score, 4),
                                     "rank": by_id[passage_id].rank,
                                     "text": by_id[passage_id].text,
                                     **by_id[passage_id].meta} for passage_id in kept]
                result.sources_used = summarise_sources(result.citations)
                result.grounded = True
                result.abstained = False
                result.reason = evidence.reason

        timings["total"] = (time.perf_counter() - overall) * 1e3
        result.timings_ms = {name: round(value, 2) for name, value in timings.items()}
        return result

    def close(self) -> None:
        if self.texts is not None:
            self.texts.close()


def summarise_sources(citations: list[dict]) -> list[dict]:
    """The "Sources" list: which document, which pages, deduplicated in order."""
    summary: list[dict] = []
    for citation in citations:
        name = citation.get("document")
        if not name:
            continue
        entry = next((item for item in summary if item["document"] == name), None)
        if entry is None:
            entry = {"document": name, "document_id": citation.get("document_id", ""), "pages": []}
            summary.append(entry)
        if citation.get("page") and citation["page"] not in entry["pages"]:
            entry["pages"].append(citation["page"])
    for entry in summary:
        entry["pages"].sort()
    return summary

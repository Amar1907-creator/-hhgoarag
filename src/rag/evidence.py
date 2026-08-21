"""Deterministic evidence selection and grounding checks.

These run before and after the model call. They are plain thresholds on
retrieval scores, not model judgements, so an abstention is reproducible and
explainable without a second API round trip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Inner-product score on L2-normalised E5 vectors, i.e. cosine similarity.
# E5 puts unrelated multilingual pairs around 0.70-0.78, so the floor sits above
# that band rather than at an arbitrary round number.
MIN_SCORE = float(os.environ.get("HHGOARAG_MIN_SCORE", "0.80"))
# Document chunks are longer and more heterogeneous than curated corpus passages,
# so E5 similarity for a correct hit sits a little lower. Calibrated conservatively:
# too high and every PDF question abstains, too low and weak matches get answered.
DOCUMENT_MIN_SCORE = float(os.environ.get("HHGOARAG_DOC_MIN_SCORE", "0.78"))
# The best hit must lead the field by this much, otherwise the top-k is a wash
# and picking a winner would be arbitrary.
MIN_MARGIN = 0.0
MAX_EVIDENCE = 5
MAX_EVIDENCE_CHARS = 6000

REASON_OK = "sufficient_evidence"
REASON_NO_HITS = "no_retrieval_hits"
REASON_LOW_SCORE = "best_score_below_threshold"
REASON_NO_MARGIN = "insufficient_score_margin"


@dataclass
class Evidence:
    passage_id: str
    text: str
    score: float
    rank: int
    # Source-specific fields travelling with the evidence: for an uploaded PDF
    # this is what turns a citation into "GOA Task-2.pdf — Page 7".
    meta: dict = field(default_factory=dict)


@dataclass
class EvidenceSet:
    items: list[Evidence] = field(default_factory=list)
    eligible: bool = False
    reason: str = REASON_NO_HITS
    best_score: float | None = None
    margin: float | None = None

    @property
    def passage_ids(self) -> list[str]:
        return [item.passage_id for item in self.items]


def select_evidence(hits: list[tuple[str, float]], texts: dict[str, str], *,
                    min_score: float = MIN_SCORE, min_margin: float = MIN_MARGIN,
                    max_items: int = MAX_EVIDENCE, max_chars: int = MAX_EVIDENCE_CHARS) -> EvidenceSet:
    """Turn ranked hits into a bounded evidence set, or refuse to.

    `texts` maps an ID either to its text, or to a record dict containing "text"
    plus whatever metadata the source attaches (document name, page number).
    """
    def body(value) -> str:
        return value["text"] if isinstance(value, dict) else value

    def extra(value) -> dict:
        return {k: v for k, v in value.items() if k != "text"} if isinstance(value, dict) else {}

    scored = [(passage_id, score) for passage_id, score in hits if passage_id in texts]
    if not scored:
        return EvidenceSet(reason=REASON_NO_HITS)

    best_score = scored[0][1]
    margin = best_score - scored[1][1] if len(scored) > 1 else best_score
    if best_score < min_score:
        return EvidenceSet(reason=REASON_LOW_SCORE, best_score=best_score, margin=margin)
    if min_margin > 0 and margin < min_margin:
        return EvidenceSet(reason=REASON_NO_MARGIN, best_score=best_score, margin=margin)

    items: list[Evidence] = []
    budget = max_chars
    for rank, (passage_id, score) in enumerate(scored[:max_items], start=1):
        if score < min_score:
            break
        record = texts[passage_id]
        text = body(record)
        if len(text) > budget:
            if not items:
                text = text[:budget]
            else:
                break
        budget -= len(text)
        items.append(Evidence(passage_id=passage_id, text=text, score=float(score),
                              rank=rank, meta=extra(record)))
    if not items:
        return EvidenceSet(reason=REASON_LOW_SCORE, best_score=best_score, margin=margin)
    return EvidenceSet(items=items, eligible=True, reason=REASON_OK, best_score=best_score, margin=margin)


def validate_citations(citations: list[str], evidence: EvidenceSet) -> tuple[list[str], list[str]]:
    """Split cited IDs into those present in the evidence set and those invented."""
    allowed = set(evidence.passage_ids)
    kept = [citation for citation in citations if citation in allowed]
    invented = [citation for citation in citations if citation not in allowed]
    return kept, invented

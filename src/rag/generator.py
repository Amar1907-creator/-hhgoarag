"""Answer generation constrained to retrieved evidence.

Evidence is presented to the model as numbered items and citations come back as
those numbers, not as 66-character passage hashes: a model asked to copy hashes
will eventually corrupt one, and a corrupted citation is indistinguishable from
an invented one. Numbers are mapped back to passage IDs here, and anything out
of range is dropped by the caller's citation check.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.rag.evidence import EvidenceSet

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 800
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = """You answer questions using ONLY the numbered evidence passages supplied to you.

Rules, in priority order:
1. Never state a fact that is not present in the evidence. You have no other knowledge for this task.
2. If the evidence does not answer the question, reply with insufficient=true and leave answer empty. Answering weakly is worse than abstaining.
3. Answer in the SAME language as the question. The evidence is Hindi; if the question is Hindi, answer in Hindi.
4. Cite the evidence numbers you actually used. Every claim must be traceable to a cited passage.
5. Be brief: two or three sentences at most.

Reply with a single JSON object and nothing else:
{"answer": "...", "citations": [1, 2], "insufficient": false}"""


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[str]
    insufficient: bool = False
    model: str = ""
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class Generator(Protocol):
    name: str

    def generate(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer: ...


def format_evidence(evidence: EvidenceSet) -> str:
    return "\n\n".join(
        f"[{index}] (retrieval score {item.score:.3f})\n{item.text}"
        for index, item in enumerate(evidence.items, start=1)
    )


def build_prompt(question: str, evidence: EvidenceSet) -> str:
    return f"Question:\n{question}\n\nEvidence:\n{format_evidence(evidence)}"


def parse_response(text: str, evidence: EvidenceSet) -> tuple[str, list[str], bool]:
    """Parse the model's JSON and map evidence numbers back to passage IDs."""
    payload: dict[str, Any] | None = None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        # No parseable JSON: treat the whole reply as prose with no citations,
        # which the caller's grounding check will then reject.
        stripped = text.strip()
        return stripped, [], stripped.upper().startswith(INSUFFICIENT) or not stripped

    answer = str(payload.get("answer") or "").strip()
    insufficient = bool(payload.get("insufficient", False)) or not answer
    citations: list[str] = []
    for value in payload.get("citations") or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= len(evidence.items):
            passage_id = evidence.items[number - 1].passage_id
            if passage_id not in citations:
                citations.append(passage_id)
    return answer, citations, insufficient


class ExtractiveGenerator:
    """Zero-dependency fallback: quote the best passage verbatim.

    Not an answer, but it is never ungrounded, needs no API key, and keeps the
    demo runnable offline. ARCHITECTURE.md names this as the fallback when
    generation is unavailable or too slow.
    """

    name = "extractive"

    def generate(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        if not evidence.items:
            return GeneratedAnswer(answer="", citations=[], insufficient=True, model=self.name)
        best = evidence.items[0]
        return GeneratedAnswer(answer=best.text.strip(), citations=[best.passage_id],
                               insufficient=False, model=self.name)


class ClaudeGenerator:
    """Anthropic-backed generator. The API key is read from ANTHROPIC_API_KEY."""

    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = DEFAULT_MAX_TOKENS,
                 api_key: str | None = None, client: Any = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"claude:{model}"
        if client is not None:
            self.client = client
            return
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "install the anthropic package to generate answers "
                "(python3 -m pip install anthropic), or run with --generator extractive"
            ) from exc
        import os
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or run with --generator extractive "
                "to use the offline evidence-only fallback."
            )
        self.client = anthropic.Anthropic(api_key=key)

    def _resolve_model(self) -> None:
        """If the configured model id is unknown, pick the newest available one."""
        try:
            available = [entry.id for entry in self.client.models.list(limit=50).data]
        except Exception:
            return
        if self.model in available:
            return
        preferred = [name for name in available if "sonnet" in name] or available
        if preferred:
            self.model = preferred[0]
            self.name = f"claude:{self.model}"

    def generate(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        prompt = build_prompt(question, evidence)
        for attempt in (1, 2):
            try:
                response = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=0,
                    system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}])
                break
            except Exception as exc:
                if attempt == 1 and "not_found" in str(exc).lower():
                    self._resolve_model()
                    continue
                raise
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        answer, citations, insufficient = parse_response(text, evidence)
        usage = {}
        if getattr(response, "usage", None):
            usage = {"input_tokens": response.usage.input_tokens,
                     "output_tokens": response.usage.output_tokens}
        return GeneratedAnswer(answer=answer, citations=citations, insufficient=insufficient,
                               model=self.model, raw=text, usage=usage)

"""Answer generation constrained to retrieved evidence.

The runtime generator is a LOCAL open-source model served by Ollama. No hosted
API and no API key is involved anywhere in this module: the product must keep
working on a machine with no accounts attached to it. If Ollama is not running
or has no usable model, generation degrades to a verbatim extractive answer,
which is never ungrounded and needs nothing installed at all.

Evidence is presented to the model as numbered items and citations come back as
those numbers, not 66-character passage hashes: a model asked to copy hashes
will eventually corrupt one, and a corrupted citation is indistinguishable from
an invented one. Numbers are mapped back to passage IDs here, and anything out
of range is dropped by the caller's citation check.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.rag.evidence import EvidenceSet

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_TIMEOUT = float(os.environ.get("HHGOARAG_LLM_TIMEOUT", "120"))
DEFAULT_NUM_PREDICT = 400

# Ranked by Hindi instruction-following quality per GB of RAM. The first entry
# that is actually installed wins; nothing is downloaded implicitly.
MODEL_PREFERENCE = (
    "qwen2.5:7b-instruct",
    "qwen2.5:7b",
    "aya-expanse:8b",
    "gemma2:9b-instruct",
    "gemma2:9b",
    "llama3.1:8b-instruct",
    "llama3.1:8b",
    "qwen2.5:3b-instruct",
    "qwen2.5:3b",
    "gemma2:2b",
    "qwen2.5:1.5b-instruct",
)
# What to suggest pulling when nothing suitable is installed.
RECOMMENDED_LARGE = "qwen2.5:7b-instruct"   # ~4.7 GB, best Hindi of the small models
RECOMMENDED_SMALL = "qwen2.5:3b-instruct"   # ~2.0 GB, for 8 GB machines

SYSTEM_PROMPT = """You answer questions using ONLY the numbered evidence passages supplied to you.

Rules, in priority order:
1. Never state a fact that is not present in the evidence. You have no other knowledge for this task.
2. If the evidence does not answer the question, reply with insufficient=true and an empty answer. Answering weakly is worse than abstaining.
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
        stripped = text.strip()
        return stripped, [], not stripped

    answer = str(payload.get("answer") or "").strip()
    insufficient = bool(payload.get("insufficient", False)) or not answer
    citations: list[str] = []
    raw_citations = payload.get("citations") or []
    if isinstance(raw_citations, (int, str)):
        raw_citations = [raw_citations]
    for value in raw_citations:
        try:
            number = int(str(value).strip().lstrip("[").rstrip("]"))
        except (TypeError, ValueError):
            continue
        if 1 <= number <= len(evidence.items):
            passage_id = evidence.items[number - 1].passage_id
            if passage_id not in citations:
                citations.append(passage_id)
    return answer, citations, insufficient


class ExtractiveGenerator:
    """Quote the best passage verbatim.

    Not a synthesised answer, but it is never ungrounded, needs no model, no
    network and no account, and keeps the product demonstrable on any machine.
    """

    name = "extractive"
    available = True

    def generate(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        if not evidence.items:
            return GeneratedAnswer(answer="", citations=[], insufficient=True, model=self.name)
        best = evidence.items[0]
        return GeneratedAnswer(answer=best.text.strip(), citations=[best.passage_id],
                               insufficient=False, model=self.name)


class OllamaError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def installed_models(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 5.0) -> list[str]:
    """Model tags currently pulled into Ollama. Empty if it is not reachable."""
    try:
        payload = _get_json(f"{host.rstrip('/')}/api/tags", timeout)
    except Exception:
        return []
    return [entry.get("name", "") for entry in payload.get("models", []) if entry.get("name")]


def choose_model(available: list[str], preference: tuple[str, ...] = MODEL_PREFERENCE) -> str | None:
    """Best installed model by preference order, tolerant of quantisation tags."""
    for wanted in preference:
        for name in available:
            if name == wanted or name.startswith(wanted + "-") or name.split(":")[0] == wanted.split(":")[0] \
                    and name.startswith(wanted):
                return name
    # Nothing preferred: fall back to any instruct-looking model rather than none.
    for name in available:
        if "instruct" in name.lower():
            return name
    return available[0] if available else None


class OllamaGenerator:
    """Local open-source generation. No API key, no hosted service."""

    def __init__(self, model: str | None = None, *, host: str = DEFAULT_OLLAMA_HOST,
                 timeout: float = DEFAULT_TIMEOUT, num_predict: int = DEFAULT_NUM_PREDICT,
                 transport=None) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_predict = num_predict
        self._transport = transport or (lambda payload: _post_json(f"{self.host}/api/chat", payload, self.timeout))
        requested = model or os.environ.get("HHGOARAG_LLM_MODEL")
        self.available_models = installed_models(self.host) if transport is None else [requested or RECOMMENDED_SMALL]
        self.model = requested or choose_model(self.available_models) or ""
        self.available = bool(self.model)
        self.name = f"ollama:{self.model}" if self.model else "ollama:unavailable"

    def status(self) -> dict:
        return {"runtime": "ollama", "host": self.host, "model": self.model,
                "available": self.available, "installed_models": self.available_models,
                "requires_api_key": False}

    def generate(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        if not self.available:
            raise OllamaError(
                f"no local model available from Ollama at {self.host}. Start it with `ollama serve` "
                f"and pull a model, e.g. `ollama pull {RECOMMENDED_SMALL}`.")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": build_prompt(question, evidence)}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": self.num_predict},
        }
        try:
            response = self._transport(payload)
        except urllib.error.URLError as exc:
            raise OllamaError(f"Ollama at {self.host} is not reachable: {exc.reason}") from exc
        except Exception as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        text = (response.get("message") or {}).get("content", "") or response.get("response", "")
        answer, citations, insufficient = parse_response(text, evidence)
        usage = {}
        if response.get("eval_count") is not None:
            usage = {"prompt_tokens": response.get("prompt_eval_count"),
                     "output_tokens": response.get("eval_count"),
                     "generation_ms": round((response.get("eval_duration") or 0) / 1e6, 1)}
        return GeneratedAnswer(answer=answer, citations=citations, insufficient=insufficient,
                               model=self.model, raw=text, usage=usage)


class FallbackGenerator:
    """Try the local model; fall back to extraction rather than failing a query."""

    def __init__(self, primary: Generator, fallback: Generator | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or ExtractiveGenerator()
        self.name = getattr(primary, "name", "primary")
        self.degraded = False

    def generate(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        try:
            answer = self.primary.generate(question, evidence)
            self.degraded = False
            return answer
        except Exception:
            self.degraded = True
            answer = self.fallback.generate(question, evidence)
            answer.model = f"{self.fallback.name} (local model unavailable)"
            return answer


def build_generator(kind: str = "auto", *, model: str | None = None,
                    host: str = DEFAULT_OLLAMA_HOST) -> Generator:
    """Resolve the runtime generator. 'auto' prefers local generation, then extraction."""
    if kind == "extractive":
        return ExtractiveGenerator()
    generator = OllamaGenerator(model=model, host=host)
    if kind == "ollama":
        return generator
    return FallbackGenerator(generator) if generator.available else ExtractiveGenerator()

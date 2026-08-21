"""Speech-to-text providers.

The task specification requires Sarvam or ElevenLabs for voice-to-text, so the
transcription stage is provider-backed rather than browser-backed. Sarvam is the
default: its saarika model is built for Indian languages and handles Hindi
noticeably better than a general-purpose recogniser.

Requests are made with stdlib urllib and a hand-rolled multipart body so the
speech stage adds no dependency. Every call is bounded by a timeout, retried
with backoff on transient faults, and returns a structured result carrying its
own latency -- the harness needs timings per stage, not one number at the end.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Protocol

SARVAM_ENDPOINT = os.environ.get("SARVAM_STT_ENDPOINT", "https://api.sarvam.ai/speech-to-text")
SARVAM_MODEL = os.environ.get("SARVAM_STT_MODEL", "saarika:v2")
ELEVENLABS_ENDPOINT = os.environ.get("ELEVENLABS_STT_ENDPOINT",
                                     "https://api.elevenlabs.io/v1/speech-to-text")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")

DEFAULT_TIMEOUT = float(os.environ.get("HHGOARAG_STT_TIMEOUT", "30"))
MAX_ATTEMPTS = int(os.environ.get("HHGOARAG_STT_ATTEMPTS", "3"))
MAX_AUDIO_BYTES = 25 * 1024 * 1024

TRANSIENT_MARKERS = ("timeout", "timed out", "connection", "temporarily",
                     "429", "500", "502", "503", "504", "too many requests")


class SpeechError(RuntimeError):
    """Transcription failed in a way the caller must surface, not swallow."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class Transcript:
    text: str
    provider: str
    model: str = ""
    language: str = ""
    latency_ms: float = 0.0
    attempts: int = 1
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "provider": self.provider, "model": self.model,
                "language": self.language, "latency_ms": round(self.latency_ms, 2),
                "attempts": self.attempts}


class SpeechToText(Protocol):
    name: str
    available: bool

    def transcribe(self, audio: bytes, *, filename: str, language: str | None) -> Transcript: ...
    def status(self) -> dict: ...


def encode_multipart(fields: dict[str, str], filename: str, audio: bytes) -> tuple[bytes, str]:
    """Build a multipart/form-data body without pulling in an HTTP library."""
    boundary = f"----hhgoarag{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            .encode("utf-8"))
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    parts.append(audio)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _is_transient(message: str, status: int | None) -> bool:
    if status is not None:
        return status == 429 or 500 <= status < 600
    lowered = message.lower()
    return any(marker in lowered for marker in TRANSIENT_MARKERS)


def _post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise SpeechError(f"HTTP {exc.code} from the speech provider: {detail}",
                          retryable=_is_transient(detail, exc.code), status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise SpeechError(f"could not reach the speech provider: {exc.reason}",
                          retryable=_is_transient(str(exc.reason), None)) from exc
    except TimeoutError as exc:
        raise SpeechError("the speech provider timed out", retryable=True) from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SpeechError(f"unreadable response from the speech provider: {payload[:200]}") from exc


class _HttpProvider:
    """Shared retry, timing and validation for the HTTP-backed providers."""

    name = "http"
    key_env = ""

    def __init__(self, api_key: str | None = None, *, timeout: float = DEFAULT_TIMEOUT,
                 attempts: int = MAX_ATTEMPTS, transport=None, sleep=time.sleep) -> None:
        self.api_key = api_key or os.environ.get(self.key_env, "")
        self.timeout = timeout
        self.attempts = max(1, attempts)
        self.available = bool(self.api_key) or transport is not None
        self._transport = transport
        self._sleep = sleep

    def _send(self, audio: bytes, filename: str, language: str | None) -> dict:
        raise NotImplementedError

    def _parse(self, payload: dict) -> tuple[str, str]:
        raise NotImplementedError

    def transcribe(self, audio: bytes, *, filename: str = "audio.webm",
                   language: str | None = None) -> Transcript:
        if not audio:
            raise SpeechError("no audio was supplied")
        if len(audio) > MAX_AUDIO_BYTES:
            raise SpeechError(f"audio is {len(audio) / 1e6:.1f} MB; the limit is "
                              f"{MAX_AUDIO_BYTES / 1e6:.0f} MB")
        if not self.available:
            raise SpeechError(
                f"{self.name} is not configured. Set {self.key_env} to enable speech input.")

        started = time.perf_counter()
        last: SpeechError | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                payload = self._send(audio, filename, language)
                text, detected = self._parse(payload)
                if not text.strip():
                    raise SpeechError("the provider returned an empty transcript")
                return Transcript(text=text.strip(), provider=self.name, model=self.model,
                                  language=detected or (language or ""),
                                  latency_ms=(time.perf_counter() - started) * 1e3,
                                  attempts=attempt, raw=payload)
            except SpeechError as exc:
                last = exc
                if not exc.retryable or attempt == self.attempts:
                    raise
                self._sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))
        raise last or SpeechError("transcription failed")

    def status(self) -> dict:
        return {"provider": self.name, "model": self.model, "available": self.available,
                "configured_via": self.key_env}


class SarvamSTT(_HttpProvider):
    """Sarvam AI speech-to-text (saarika). Built for Indian languages."""

    name = "sarvam"
    key_env = "SARVAM_API_KEY"

    def __init__(self, *args, model: str = SARVAM_MODEL, endpoint: str = SARVAM_ENDPOINT, **kwargs):
        self.model = model
        self.endpoint = endpoint
        super().__init__(*args, **kwargs)

    def _send(self, audio: bytes, filename: str, language: str | None) -> dict:
        # Sarvam expects a BCP-47-style code such as hi-IN; "unknown" asks it to detect.
        fields = {"model": self.model, "language_code": language or "unknown"}
        if self._transport is not None:
            return self._transport({"fields": fields, "audio": audio, "filename": filename})
        body, content_type = encode_multipart(fields, filename, audio)
        return _post(self.endpoint, body,
                     {"api-subscription-key": self.api_key, "Content-Type": content_type},
                     self.timeout)

    def _parse(self, payload: dict) -> tuple[str, str]:
        return str(payload.get("transcript") or ""), str(payload.get("language_code") or "")


class ElevenLabsSTT(_HttpProvider):
    """ElevenLabs Scribe speech-to-text."""

    name = "elevenlabs"
    key_env = "ELEVENLABS_API_KEY"

    def __init__(self, *args, model: str = ELEVENLABS_MODEL, endpoint: str = ELEVENLABS_ENDPOINT,
                 **kwargs):
        self.model = model
        self.endpoint = endpoint
        super().__init__(*args, **kwargs)

    def _send(self, audio: bytes, filename: str, language: str | None) -> dict:
        fields = {"model_id": self.model}
        if language:
            fields["language_code"] = language.split("-")[0]
        if self._transport is not None:
            return self._transport({"fields": fields, "audio": audio, "filename": filename})
        body, content_type = encode_multipart(fields, filename, audio)
        return _post(self.endpoint, body,
                     {"xi-api-key": self.api_key, "Content-Type": content_type}, self.timeout)

    def _parse(self, payload: dict) -> tuple[str, str]:
        return str(payload.get("text") or ""), str(payload.get("language_code") or "")


PROVIDERS = {"sarvam": SarvamSTT, "elevenlabs": ElevenLabsSTT}


def build_speech(provider: str | None = None) -> SpeechToText | None:
    """Resolve the configured provider, preferring whichever has a key present."""
    wanted = (provider or os.environ.get("HHGOARAG_STT_PROVIDER", "")).strip().lower()
    if wanted in PROVIDERS:
        return PROVIDERS[wanted]()
    for name in ("sarvam", "elevenlabs"):
        candidate = PROVIDERS[name]()
        if candidate.available:
            return candidate
    return SarvamSTT()          # unconfigured, but reports itself honestly

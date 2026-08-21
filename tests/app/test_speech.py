"""Speech-to-text: provider-backed, retried, and honest when unconfigured."""

import unittest

from fastapi.testclient import TestClient

from src.app.api import create_app
from src.speech.providers import (
    ElevenLabsSTT, SarvamSTT, SpeechError, Transcript, build_speech, encode_multipart,
)
from tests.app.test_api import build_service


class MultipartTests(unittest.TestCase):
    def test_body_carries_fields_and_the_file(self):
        body, content_type = encode_multipart({"model": "saarika:v2", "language_code": "hi-IN"},
                                              "question.webm", b"AUDIOBYTES")
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        boundary = content_type.split("boundary=")[1].encode()
        self.assertIn(b'name="model"', body)
        self.assertIn(b"saarika:v2", body)
        self.assertIn(b'name="language_code"', body)
        self.assertIn(b'filename="question.webm"', body)
        self.assertIn(b"AUDIOBYTES", body)
        self.assertTrue(body.rstrip().endswith(b"--" + boundary + b"--"))


class SarvamTests(unittest.TestCase):
    def provider(self, reply=None, fail_times=0, **kwargs):
        state = {"calls": 0}
        def transport(request):
            state["calls"] += 1
            if state["calls"] <= fail_times:
                raise SpeechError("503 service unavailable", retryable=True, status=503)
            return reply if reply is not None else {"transcript": "गोवा कहाँ है?",
                                                    "language_code": "hi-IN"}
        provider = SarvamSTT(transport=transport, sleep=lambda _: None, **kwargs)
        return provider, state

    def test_transcribes_and_reports_provenance(self):
        provider, _ = self.provider()
        result = provider.transcribe(b"AUDIO", filename="q.webm", language="hi-IN")
        self.assertEqual(result.text, "गोवा कहाँ है?")
        self.assertEqual(result.provider, "sarvam")
        self.assertEqual(result.model, "saarika:v2")
        self.assertEqual(result.language, "hi-IN")
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertIn("latency_ms", result.to_dict())

    def test_language_is_passed_through_and_defaults_to_detection(self):
        seen = {}
        def transport(request):
            seen.update(request["fields"])
            return {"transcript": "ok"}
        SarvamSTT(transport=transport).transcribe(b"A", filename="q.webm", language="ta-IN")
        self.assertEqual(seen["language_code"], "ta-IN")
        SarvamSTT(transport=transport).transcribe(b"A", filename="q.webm", language=None)
        self.assertEqual(seen["language_code"], "unknown")

    def test_transient_failures_are_retried(self):
        provider, state = self.provider(fail_times=2)
        result = provider.transcribe(b"AUDIO", filename="q.webm", language="hi-IN")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(state["calls"], 3)

    def test_retries_are_bounded(self):
        provider, state = self.provider(fail_times=99, attempts=3)
        with self.assertRaises(SpeechError):
            provider.transcribe(b"AUDIO", filename="q.webm", language="hi-IN")
        self.assertEqual(state["calls"], 3)

    def test_permanent_failures_are_not_retried(self):
        state = {"calls": 0}
        def transport(request):
            state["calls"] += 1
            raise SpeechError("401 unauthorized", retryable=False, status=401)
        with self.assertRaises(SpeechError):
            SarvamSTT(transport=transport, sleep=lambda _: None).transcribe(
                b"A", filename="q.webm", language="hi-IN")
        self.assertEqual(state["calls"], 1, "a bad key must not be retried")

    def test_empty_transcript_is_an_error_not_an_empty_answer(self):
        provider, _ = self.provider(reply={"transcript": "   "})
        with self.assertRaises(SpeechError):
            provider.transcribe(b"AUDIO", filename="q.webm", language="hi-IN")

    def test_rejects_empty_and_oversized_audio(self):
        provider, _ = self.provider()
        with self.assertRaises(SpeechError):
            provider.transcribe(b"", filename="q.webm", language="hi-IN")
        with self.assertRaises(SpeechError):
            provider.transcribe(b"x" * (26 * 1024 * 1024), filename="q.webm", language="hi-IN")

    def test_unconfigured_provider_says_so(self):
        provider = SarvamSTT(api_key="")
        self.assertFalse(provider.available)
        self.assertEqual(provider.status()["configured_via"], "SARVAM_API_KEY")
        with self.assertRaises(SpeechError) as caught:
            provider.transcribe(b"AUDIO", filename="q.webm", language="hi-IN")
        self.assertIn("SARVAM_API_KEY", str(caught.exception))


class ElevenLabsTests(unittest.TestCase):
    def test_parses_its_own_response_shape(self):
        provider = ElevenLabsSTT(transport=lambda r: {"text": "where is Goa", "language_code": "en"})
        result = provider.transcribe(b"AUDIO", filename="q.webm", language="en-IN")
        self.assertEqual(result.text, "where is Goa")
        self.assertEqual(result.provider, "elevenlabs")


class SelectionTests(unittest.TestCase):
    def test_explicit_provider_wins(self):
        self.assertIsInstance(build_speech("elevenlabs"), ElevenLabsSTT)
        self.assertIsInstance(build_speech("sarvam"), SarvamSTT)

    def test_defaults_to_sarvam_when_nothing_is_configured(self):
        provider = build_speech("")
        self.assertIsInstance(provider, SarvamSTT)
        self.assertFalse(provider.available)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        service, directory = build_service()
        self.addCleanup(directory.cleanup)
        self.service = service
        self.client = TestClient(create_app(service, load=False))

    def test_status_endpoint_reports_the_provider(self):
        body = self.client.get("/api/speech").json()
        self.assertEqual(body["endpoint"], "/api/transcribe")
        self.assertIn("provider", body)

    def test_transcribe_returns_text(self):
        self.service.speech = SarvamSTT(
            transport=lambda r: {"transcript": "गोवा कहाँ है?", "language_code": "hi-IN"})
        response = self.client.post("/api/transcribe?language=hi",
                                    files={"file": ("q.webm", b"AUDIO", "audio/webm")})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["text"], "गोवा कहाँ है?")
        self.assertEqual(body["provider"], "sarvam")

    def test_empty_audio_is_rejected(self):
        response = self.client.post("/api/transcribe",
                                    files={"file": ("q.webm", b"", "audio/webm")})
        self.assertEqual(response.status_code, 422)

    def test_unconfigured_provider_returns_503_not_a_crash(self):
        self.service.speech = SarvamSTT(api_key="")
        response = self.client.post("/api/transcribe",
                                    files={"file": ("q.webm", b"AUDIO", "audio/webm")})
        self.assertEqual(response.status_code, 503)
        self.assertIn("SARVAM_API_KEY", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

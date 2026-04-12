import unittest

from llm_agent_platform.services.quota_transport import (
    Quota429Type,
    classify_429_error_payload,
    classify_429_exception,
    classify_429_response,
)


class Quota429ClassificationTests(unittest.TestCase):
    def test_classify_rate_limit_response(self):
        kind = classify_429_response(429, "too many requests: retry after 60 seconds")
        self.assertEqual(kind, Quota429Type.RATE_LIMIT)

    def test_classify_quota_exhausted_response(self):
        kind = classify_429_response(429, "insufficient_quota: daily quota exceeded")
        self.assertEqual(kind, Quota429Type.QUOTA_EXHAUSTED)

    def test_classify_non_429_response(self):
        kind = classify_429_response(503, "service unavailable")
        self.assertEqual(kind, Quota429Type.NOT_429)

    def test_classify_error_payload(self):
        payload = {"code": 429, "message": "rate limit exceeded"}
        kind = classify_429_error_payload(payload)
        self.assertEqual(kind, Quota429Type.RATE_LIMIT)

    def test_classify_stream_exception(self):
        err = RuntimeError("429:insufficient_quota")
        kind = classify_429_exception(err)
        self.assertEqual(kind, Quota429Type.QUOTA_EXHAUSTED)


if __name__ == "__main__":
    unittest.main()

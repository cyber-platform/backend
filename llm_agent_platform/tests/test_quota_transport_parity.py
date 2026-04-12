import json
from pathlib import Path
import unittest

from llm_agent_platform.api.openai.transform import transform_openai_to_gemini
from llm_agent_platform.services.quota_transport import build_quota_payload, parse_cloud_code_sse_line


class QuotaTransportParityTests(unittest.TestCase):
    def test_build_quota_payload_matches_golden_shape(self):
        payload = build_quota_payload(
            model="gemini-3-flash-preview",
            project="demo-project",
            user_prompt_id="<user_prompt_id>",
            session_id="<session_id>",
            request_payload={
                "contents": [{"role": "user", "parts": [{"text": "hello parity"}]}],
                "generationConfig": {"temperature": 0.7, "topP": 1.0, "topK": 40},
            },
        )

        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "parity"
            / "golden_cli_generate_request.json"
        )
        golden = json.loads(fixture_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, golden)

    def test_parse_cloud_code_sse_line_unwraps_response_field(self):
        line = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}'
        parsed = parse_cloud_code_sse_line(line)
        self.assertIsNotNone(parsed)
        self.assertIn("candidates", parsed)
        self.assertEqual(parsed["candidates"][0]["content"]["parts"][0]["text"], "ok")

    def test_transform_does_not_inject_legacy_thought_signature(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_docs",
                            "arguments": '{"query":"quota"}',
                        }
                    }
                ],
            }
        ]

        contents, _ = transform_openai_to_gemini(messages)
        self.assertTrue(contents)

        part = contents[0]["parts"][0]
        self.assertIn("functionCall", part)
        self.assertNotIn("thoughtSignature", part)


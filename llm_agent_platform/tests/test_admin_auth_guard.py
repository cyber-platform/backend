"""Admin auth guard tests.

Suite: TS-ADMIN-JWT-GUARD
"""

import unittest
from unittest.mock import patch

from llm_agent_platform.__main__ import app
from llm_agent_platform.tests.admin_auth_test_utils import (
    build_admin_headers,
    patch_admin_auth_config,
)


class AdminAuthGuardTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        patch_admin_auth_config(self)

    def test_developer_role_is_mapped_to_admin_for_admin_routes(self):
        with patch(
            "llm_agent_platform.api.admin.routes._monitoring_service"
        ) as mock_service:
            mock_service.return_value.list_providers.return_value = [
                {"provider_id": "openai-chatgpt"}
            ]

            response = self.client.get(
                "/admin/monitoring/providers",
                headers=build_admin_headers(role="developer", roles=["developer"]),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [{"provider_id": "openai-chatgpt"}])

    def test_admin_routes_reject_missing_or_malformed_bearer_token(self):
        with patch(
            "llm_agent_platform.api.admin.routes._monitoring_service"
        ) as mock_service:
            mock_service.return_value.list_providers.return_value = []

            missing_response = self.client.get("/admin/monitoring/providers")
            malformed_response = self.client.get(
                "/admin/monitoring/providers",
                headers={"Authorization": "Token not-bearer"},
            )

        self.assertEqual(missing_response.status_code, 401)
        self.assertEqual(malformed_response.status_code, 401)
        self.assertEqual(
            missing_response.get_json()["error"]["code"], "missing_bearer_token"
        )
        self.assertEqual(
            malformed_response.get_json()["error"]["code"], "missing_bearer_token"
        )

    def test_admin_routes_reject_non_admin_roles(self):
        with patch(
            "llm_agent_platform.api.admin.routes._monitoring_service"
        ) as mock_service:
            mock_service.return_value.list_providers.return_value = []

            response = self.client.get(
                "/admin/monitoring/providers",
                headers=build_admin_headers(role="observer", roles=["observer"]),
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["error"]["code"], "insufficient_role")


if __name__ == "__main__":
    unittest.main()

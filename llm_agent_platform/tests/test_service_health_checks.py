"""Service health-check verification tests.

Suite: TS-SERVICE-HEALTH-CHECKS
"""

import unittest
from pathlib import Path

from llm_agent_platform.__main__ import app

REPO_ROOT = Path(__file__).resolve().parents[4]


class ServiceHealthChecksTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_backend_health_endpoint_returns_stable_payload(self):
        """Checks the explicit backend health endpoint contract.

        Test case: TC-SERVICE-HEALTH-CHECKS-001
        Requirement: backend exposes a lightweight `/health` route for compose and runtime probing.
        """

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok", "service": "backend"})

    def test_frontend_runtime_and_compose_keep_lightweight_health_probe(self):
        """Verifies the frontend health contour in checked-in runtime files.

        Test case: TC-SERVICE-HEALTH-CHECKS-002
        Requirement: frontend exposes a stable health endpoint and compose-level probe.
        """

        nginx_text = (REPO_ROOT / "services/frontend/nginx.conf").read_text(
            encoding="utf-8"
        )
        compose_text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        compose_dev_text = (REPO_ROOT / "docker-compose-dev.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("location = /health", nginx_text)
        self.assertIn('"service":"frontend"', nginx_text)
        self.assertIn("wget -qO- http://127.0.0.1/health", compose_text)
        self.assertIn("wget -qO- http://127.0.0.1/health", compose_dev_text)

    def test_user_service_keeps_existing_endpoint_and_compose_probe(self):
        """Verifies the user_service baseline health probe stays normalized.

        Test case: TC-SERVICE-HEALTH-CHECKS-003
        Requirement: user_service keeps `/health` and exposes a compose-level healthcheck.
        """

        app_text = (REPO_ROOT / "services/user_service/app/main.py").read_text(
            encoding="utf-8"
        )
        compose_dev_text = (
            REPO_ROOT / "services/user_service/docker-compose-dev.yml"
        ).read_text(encoding="utf-8")
        compose_prod_text = (
            REPO_ROOT / "services/user_service/docker-compose-prod.yml"
        ).read_text(encoding="utf-8")

        self.assertIn('@app.get("/health")', app_text)
        self.assertIn('"service": "user-service"', app_text)
        self.assertIn("http://127.0.0.1:8000/health", compose_dev_text)
        self.assertIn("http://127.0.0.1:8000/health", compose_prod_text)


if __name__ == "__main__":
    unittest.main()

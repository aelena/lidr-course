import os

# Settings are validated when app.main is imported, so set them first. Real
# environment variables take priority over .env, which keeps tests independent
# of the developer's local configuration.
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = "gpt-4o-mini"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

"""Drives the chat interface with Streamlit's AppTest; the LLM is replaced by a double."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.llm_service import LLMServiceError

APP = Path(__file__).parent.parent / "streamlit_app.py"
BASIC = ESTIMATION_EXAMPLES[0]["estimation"]
RESULT = {
    "estimation": BASIC,
    "model": "gpt-4o-mini",
    "provider": "openai",
    "finish_reason": "stop",
    "total_hours": 175,
    "input_tokens": 800,
    "output_tokens": 90,
    "latency_ms": 4200,
    "first_token_ms": 600,
    "extracted_requirements": None,
}


class FakeStream:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.result = None

    def __iter__(self):
        if self.error:
            raise self.error
        yield from (BASIC[:60], BASIC[60:])
        self.result = RESULT


@pytest.fixture
def conversations(monkeypatch):
    """The conversations the app sends to the service, one per call."""
    sent = []

    def fake_stream_estimation(messages, opts=None):
        sent.append(list(messages))
        return FakeStream()

    monkeypatch.setattr(llm_service, "stream_estimation", fake_stream_estimation)
    return sent


@pytest.fixture
def app():
    return AppTest.from_file(str(APP), default_timeout=10).run()


def _ask(app: AppTest, text: str) -> AppTest:
    return app.chat_input[0].set_value(text).run()


def _chat(app: AppTest) -> list[tuple[str, str]]:
    return [(message.name, message.markdown[0].value) for message in app.chat_message]


def test_it_starts_with_an_empty_chat(app):
    assert not app.exception
    assert app.title[0].value == "Estimador CAG"
    assert len(app.chat_input) == 1
    assert len(app.chat_message) == 0


def test_a_transcription_gets_an_estimation(app, conversations):
    _ask(app, "El cliente quiere un inventario")

    assert not app.exception
    assert _chat(app) == [("user", "El cliente quiere un inventario"), ("assistant", BASIC)]
    assert conversations == [[{"role": "user", "content": "El cliente quiere un inventario"}]]


def test_the_conversation_stays_on_screen_and_travels_with_each_question(app, conversations):
    _ask(app, "El cliente quiere un inventario")
    _ask(app, "¿Y sin el dashboard?")

    assert [name for name, _ in _chat(app)] == ["user", "assistant", "user", "assistant"]
    assert conversations[1] == [
        {"role": "user", "content": "El cliente quiere un inventario"},
        {"role": "assistant", "content": BASIC},
        {"role": "user", "content": "¿Y sin el dashboard?"},
    ]
    assert app.session_state["messages"][-1] == {"role": "assistant", "content": BASIC}


def test_a_provider_failure_is_shown_and_not_stored(app, monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "stream_estimation",
        lambda messages, opts=None: FakeStream(LLMServiceError("OpenAI request failed")),
    )

    _ask(app, "El cliente quiere un inventario")

    assert not app.exception
    assert "OpenAI request failed" in app.error[0].value
    assert app.session_state["messages"] == []


# --- API key ---------------------------------------------------------------


@pytest.fixture
def no_api_key(monkeypatch, tmp_path):
    """No key in the environment and no .env in the working directory."""
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    st.cache_resource.clear()
    yield
    get_settings.cache_clear()
    st.cache_resource.clear()


def test_the_api_key_can_come_from_streamlit_secrets(no_api_key):
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.secrets["OPENAI_API_KEY"] = "key-from-secrets"

    app.run()

    assert not app.exception
    assert not app.error
    assert get_settings().llm_api_key.get_secret_value() == "key-from-secrets"


def test_without_an_api_key_the_app_explains_what_is_missing(no_api_key):
    app = AppTest.from_file(str(APP), default_timeout=10).run()

    assert not app.exception
    assert "OPENAI_API_KEY is required" in app.error[0].value
    assert len(app.chat_input) == 0

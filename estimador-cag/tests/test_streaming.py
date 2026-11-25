"""Streams estimations from canned provider answers; no LLM is called."""

import pytest

from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.llm_service import Completion, GenerationOptions, stream_estimation

BASIC = ESTIMATION_EXAMPLES[0]["estimation"]
QUESTION = [{"role": "user", "content": "Una reunión"}]


def _chunks(text: str, size: int = 40) -> list[str]:
    return [text[start : start + size] for start in range(0, len(text), size)]


def _stub_streams(monkeypatch, *texts, finish_reason="stop"):
    """Answer each provider call with the next text, in chunks, and record the prompts."""
    calls = []
    answers = iter(texts)

    def fake_stream(settings, model, system, messages, *, max_tokens, thinking_budget=None):
        calls.append(
            {
                "model": model,
                "system": system,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            }
        )
        text = next(answers)
        yield from _chunks(text)
        return Completion(text, finish_reason, 800, 90)

    monkeypatch.setattr(llm_service, "_stream", fake_stream)
    return calls


def test_the_deltas_arrive_one_by_one_and_add_up_to_the_estimation(monkeypatch):
    _stub_streams(monkeypatch, BASIC)

    deltas = list(stream_estimation(QUESTION))

    assert len(deltas) > 1
    assert "".join(deltas) == BASIC


def test_the_result_is_available_once_the_stream_ends(monkeypatch):
    _stub_streams(monkeypatch, BASIC)
    stream = stream_estimation(QUESTION)

    assert stream.result is None
    list(stream)

    result = stream.result
    assert result["estimation"] == BASIC
    assert result["model"] == "gpt-4o-mini"
    assert result["provider"] == "openai"
    assert result["finish_reason"] == "stop"
    assert result["total_hours"] == 175
    assert (result["input_tokens"], result["output_tokens"]) == (800, 90)
    assert 0 <= result["first_token_ms"] <= result["latency_ms"]
    assert result["extracted_requirements"] is None


def test_nothing_is_sent_until_the_stream_is_iterated(monkeypatch):
    calls = _stub_streams(monkeypatch, BASIC)

    stream_estimation(QUESTION)

    assert calls == []


def test_it_sends_the_same_system_prompt_as_the_endpoint(monkeypatch):
    calls = _stub_streams(monkeypatch, BASIC, BASIC)

    list(stream_estimation(QUESTION))
    list(stream_estimation(QUESTION, GenerationOptions(preprocessing="inline_cleaning")))

    assert calls[0]["system"] == llm_service.build_system_prompt()
    assert calls[1]["system"] == llm_service.build_system_prompt(inline_cleaning=True)


def test_the_whole_conversation_reaches_the_provider(monkeypatch):
    calls = _stub_streams(monkeypatch, "Sin la app serían 275 horas.")
    conversation = [
        *QUESTION,
        {"role": "assistant", "content": BASIC},
        {"role": "user", "content": "¿Y sin el dashboard?"},
    ]

    list(stream_estimation(conversation, GenerationOptions(model="otro", max_tokens=500)))

    assert calls[0]["messages"] == conversation
    assert calls[0]["model"] == "otro"
    assert calls[0]["max_tokens"] == 500


def test_two_phase_extracts_the_last_message_without_streaming_it(monkeypatch):
    calls = _stub_streams(monkeypatch, "- Control de stock\n- Roles\n", BASIC)
    stream = stream_estimation(QUESTION, GenerationOptions(preprocessing="two_phase"))

    assert "".join(stream) == BASIC
    assert calls[0]["system"] == llm_service.EXTRACTION_SYSTEM_PROMPT
    assert calls[0]["messages"] == QUESTION
    assert calls[1]["messages"] == [
        {"role": "user", "content": "- Control de stock\n- Roles"}
    ]
    assert stream.result["extracted_requirements"] == "- Control de stock\n- Roles"
    assert (stream.result["input_tokens"], stream.result["output_tokens"]) == (1600, 180)


def test_a_truncated_stream_keeps_its_finish_reason(monkeypatch):
    _stub_streams(monkeypatch, "## Estimación: Port", finish_reason="length")
    stream = stream_estimation(QUESTION)

    list(stream)

    assert stream.result["finish_reason"] == "length"
    assert stream.result["total_hours"] is None


def test_an_empty_stream_is_an_error(monkeypatch):
    _stub_streams(monkeypatch, "  ")

    with pytest.raises(llm_service.LLMServiceError):
        list(stream_estimation(QUESTION))

import pytest

from app.context.examples import ESTIMATION_EXAMPLES
from app.routers import estimations
from app.services.llm_service import GenerationOptions, LLMServiceError

URL = "/api/v1/estimate"
EXAMPLE = ESTIMATION_EXAMPLES[0]
RESULT = {
    "estimation": EXAMPLE["estimation"],
    "model": "gpt-4o-mini",
    "provider": "openai",
    "finish_reason": "stop",
    "total_hours": 175,
    "input_tokens": 900,
    "output_tokens": 120,
}


@pytest.fixture
def generated(monkeypatch):
    """Replace the LLM service and record what the endpoint passed to it."""
    calls = {}

    async def fake_generate(transcription, options):
        calls["transcription"] = transcription
        calls["options"] = options
        return dict(RESULT)

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate)
    return calls


def test_returns_the_generated_estimation(client, generated):
    response = client.post(URL, json={"transcription": EXAMPLE["meeting_summary"]})

    assert response.status_code == 200
    assert response.json() == {**RESULT, "validation": None}
    assert generated["transcription"] == EXAMPLE["meeting_summary"]
    assert generated["options"] == GenerationOptions()


def test_passes_the_generation_options_to_the_service(client, generated):
    client.post(
        URL,
        json={
            "transcription": "Una reunión",
            "preprocessing": False,
            "example_format": "json",
            "num_examples": 1,
            "use_examples": False,
            "model": "gpt-4o",
            "max_tokens": 2000,
            "thinking_budget": 1024,
        },
    )

    assert generated["options"] == GenerationOptions(
        preprocessing=False,
        example_format="json",
        num_examples=1,
        use_examples=False,
        model="gpt-4o",
        max_tokens=2000,
        thinking_budget=1024,
    )


def test_includes_the_validation_when_asked(client, generated):
    response = client.post(URL, json={"transcription": "Una reunión", "evaluate": True})

    validation = response.json()["validation"]
    assert validation["is_valid"] is True
    assert validation["task_count"] == 5
    assert validation["issues"] == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"transcription": ""},
        {"transcription": "   "},
        {"transcription": 42},
        {"transcription": "ok", "example_format": "yaml"},
        {"transcription": "ok", "num_examples": 0},
        {"transcription": "ok", "max_tokens": 0},
        {"transcription": "ok", "thinking_budget": 100},
    ],
)
def test_rejects_invalid_requests(client, generated, body):
    assert client.post(URL, json=body).status_code == 422
    assert generated == {}


def test_maps_service_failure_to_500(client, monkeypatch):
    async def failing_generate(transcription, options):
        raise LLMServiceError("provider down")

    monkeypatch.setattr(estimations, "generate_estimation", failing_generate)

    response = client.post(URL, json={"transcription": "Una reunión"})

    assert response.status_code == 500
    assert response.json() == {"detail": "provider down"}


def test_only_accepts_post(client):
    assert client.get(URL).status_code == 405

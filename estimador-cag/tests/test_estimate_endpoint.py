from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.llm_service import EstimationResult

URL = "/api/v1/estimate"


def test_returns_the_generated_estimation(client, monkeypatch):
    example = ESTIMATION_EXAMPLES[0]
    settings = get_settings()
    received = {}

    async def fake_generate(transcription, settings):
        received["transcription"] = transcription
        return EstimationResult(
            estimation=example["estimation"],
            total_hours=175,
            input_tokens=900,
            output_tokens=120,
        )

    monkeypatch.setattr(llm_service, "generate_estimation", fake_generate)

    response = client.post(URL, json={"transcription": example["meeting_summary"]})

    assert response.status_code == 200
    assert received["transcription"] == example["meeting_summary"]
    assert response.json() == {
        "estimation": example["estimation"],
        "model": settings.llm_model,
        "provider": settings.llm_provider,
        "total_hours": 175,
        "input_tokens": 900,
        "output_tokens": 120,
    }


def test_rejects_blank_transcription(client):
    assert client.post(URL, json={"transcription": "   "}).status_code == 422
    assert client.post(URL, json={"transcription": ""}).status_code == 422


def test_rejects_missing_or_mistyped_transcription(client):
    assert client.post(URL, json={}).status_code == 422
    assert client.post(URL, json={"transcription": 42}).status_code == 422


def test_maps_provider_failure_to_502(client, monkeypatch):
    async def failing_generate(transcription, settings):
        raise llm_service.LLMServiceError("provider down")

    monkeypatch.setattr(llm_service, "generate_estimation", failing_generate)

    response = client.post(URL, json={"transcription": "Una reunión"})

    assert response.status_code == 502
    assert response.json() == {"detail": "provider down"}


def test_only_accepts_post(client):
    assert client.get(URL).status_code == 405

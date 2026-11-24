from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service


def test_create_estimation_returns_the_generated_estimation(client, monkeypatch):
    example = ESTIMATION_EXAMPLES[0]
    received = {}

    async def fake_generate(transcription, settings):
        received["transcription"] = transcription
        return example["estimation"]

    monkeypatch.setattr(llm_service, "generate_estimation", fake_generate)

    response = client.post(
        "/estimations", json={"transcription": example["meeting_summary"]}
    )

    assert response.status_code == 200
    assert received["transcription"] == example["meeting_summary"]
    assert response.json() == {
        "estimation": example["estimation"],
        "provider": "openai",
        "model": "gpt-4o-mini",
    }


def test_create_estimation_rejects_blank_transcription(client):
    assert client.post("/estimations", json={"transcription": "   "}).status_code == 422
    assert client.post("/estimations", json={}).status_code == 422


def test_create_estimation_maps_provider_failure_to_502(client, monkeypatch):
    async def failing_generate(transcription, settings):
        raise llm_service.LLMServiceError("provider down")

    monkeypatch.setattr(llm_service, "generate_estimation", failing_generate)

    response = client.post("/estimations", json={"transcription": "Una reunión"})

    assert response.status_code == 502
    assert response.json() == {"detail": "provider down"}

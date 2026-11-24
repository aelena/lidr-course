import asyncio

import pytest

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.llm_service import EstimationDraft, EstimationResult, Task

URL = "/api/v1/estimate"


def test_create_estimation_returns_the_generated_estimation(client, monkeypatch):
    example = ESTIMATION_EXAMPLES[0]
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
        "model": "gpt-4o-mini",
        "provider": "openai",
        "total_hours": 175,
        "input_tokens": 900,
        "output_tokens": 120,
    }


def test_create_estimation_rejects_blank_transcription(client):
    assert client.post(URL, json={"transcription": "   "}).status_code == 422
    assert client.post(URL, json={}).status_code == 422


def test_create_estimation_maps_provider_failure_to_502(client, monkeypatch):
    async def failing_generate(transcription, settings):
        raise llm_service.LLMServiceError("provider down")

    monkeypatch.setattr(llm_service, "generate_estimation", failing_generate)

    response = client.post(URL, json={"transcription": "Una reunión"})

    assert response.status_code == 502
    assert response.json() == {"detail": "provider down"}


def _stub_draft(monkeypatch, draft):
    async def fake_draft(settings, system, user):
        return draft, 800, 90

    monkeypatch.setattr(llm_service, "_draft_with_openai", fake_draft)


def test_generate_estimation_computes_the_total_in_code(monkeypatch):
    _stub_draft(
        monkeypatch,
        EstimationDraft(
            title="Portal de socios",
            tasks=[Task(name="Diseño", hours=60), Task(name="Backend", hours=85)],
            team="2 desarrolladores",
            duration="5 semanas",
            assumptions=["Pasarela de pago estándar"],
        ),
    )

    result = asyncio.run(llm_service.generate_estimation("Una reunión", get_settings()))

    assert result.total_hours == 145
    assert "**Total estimado: 145 horas**" in result.estimation
    assert "### Supuestos:\n- Pasarela de pago estándar" in result.estimation
    assert (result.input_tokens, result.output_tokens) == (800, 90)


@pytest.mark.parametrize(
    "draft",
    [
        None,
        EstimationDraft(title="T", tasks=[], team="x", duration="y", assumptions=[]),
        EstimationDraft(
            title="T",
            tasks=[Task(name="A", hours=0)],
            team="x",
            duration="y",
            assumptions=[],
        ),
    ],
)
def test_generate_estimation_rejects_unusable_drafts(monkeypatch, draft):
    _stub_draft(monkeypatch, draft)

    with pytest.raises(llm_service.LLMServiceError):
        asyncio.run(llm_service.generate_estimation("Una reunión", get_settings()))

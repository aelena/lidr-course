"""Evaluates the estimations the service produces.

The offline tests feed the service a canned model answer and check the
estimation it builds. The live test calls the configured provider and is
skipped unless RUN_LLM_EVAL=1 is set, because it needs a real API key and
spends tokens.
"""

import asyncio
import re

import pytest

from app.config import get_settings
from app.services import llm_service
from app.services.llm_service import EstimationDraft, Task
from tests.conftest import RUN_LLM_EVAL

TASK_LINE = re.compile(r"^\d+\. .+: (\d+) horas$", re.M)
TOTAL_LINE = re.compile(r"^\*\*Total estimado: (\d+) horas\*\*$", re.M)


def assert_well_formed(estimation: str, total_hours: int) -> list[int]:
    """Check the structure of an estimation and return the hours of its tasks."""
    assert estimation.startswith("## Estimación: ")
    assert "### Desglose de tareas:" in estimation
    assert "**Equipo recomendado: " in estimation
    assert "**Duración estimada: " in estimation
    hours = [int(value) for value in TASK_LINE.findall(estimation)]
    assert hours, "the estimation has no tasks"
    assert all(value > 0 for value in hours)
    assert int(TOTAL_LINE.search(estimation).group(1)) == sum(hours) == total_hours
    return hours


def _stub_draft(monkeypatch, draft):
    async def fake_draft(settings, system, user):
        return draft, 800, 90

    monkeypatch.setattr(llm_service, f"_draft_with_{get_settings().llm_provider}", fake_draft)


def _generate(transcription: str):
    return asyncio.run(llm_service.generate_estimation(transcription, get_settings()))


def test_estimation_total_matches_the_task_breakdown(monkeypatch):
    _stub_draft(
        monkeypatch,
        EstimationDraft(
            title="Portal de socios",
            tasks=[Task(name="Diseño", hours=60), Task(name="Backend", hours=85)],
            team="2 desarrolladores",
            duration="5 semanas",
            assumptions=[],
        ),
    )

    result = _generate("Una reunión")

    assert assert_well_formed(result.estimation, result.total_hours) == [60, 85]
    assert result.total_hours == 145
    assert "Supuestos" not in result.estimation
    assert (result.input_tokens, result.output_tokens) == (800, 90)


def test_estimation_lists_the_assumptions(monkeypatch):
    _stub_draft(
        monkeypatch,
        EstimationDraft(
            title="Portal de socios",
            tasks=[Task(name="Diseño", hours=60)],
            team="2 desarrolladores",
            duration="5 semanas",
            assumptions=["Pasarela de pago estándar", "Sin app móvil"],
        ),
    )

    result = _generate("Una reunión")

    assert result.estimation.endswith(
        "### Supuestos:\n- Pasarela de pago estándar\n- Sin app móvil"
    )


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
        EstimationDraft(
            title="T",
            tasks=[Task(name="A", hours=10), Task(name="B", hours=-5)],
            team="x",
            duration="y",
            assumptions=[],
        ),
    ],
    ids=["no-answer", "no-tasks", "zero-hours", "negative-hours"],
)
def test_unusable_model_answers_are_rejected(monkeypatch, draft):
    _stub_draft(monkeypatch, draft)

    with pytest.raises(llm_service.LLMServiceError):
        _generate("Una reunión")


@pytest.mark.skipif(not RUN_LLM_EVAL, reason="set RUN_LLM_EVAL=1 to call the LLM")
def test_live_estimation_is_well_formed_and_calibrated_by_the_examples():
    # Close to the inventory example (175 hours), so the model should land near it.
    result = _generate(
        "El cliente tiene una distribuidora de material eléctrico con dos almacenes. "
        "Quiere una aplicación web para controlar el stock: alta de productos, "
        "entradas y salidas, usuarios con distintos permisos y un panel con "
        "indicadores para gerencia."
    )

    hours = assert_well_formed(result.estimation, result.total_hours)
    assert len(hours) >= 3
    assert 90 <= result.total_hours <= 350
    assert result.input_tokens > 0 and result.output_tokens > 0

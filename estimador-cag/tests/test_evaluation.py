"""Evaluates the estimations the service produces.

The offline tests feed the service a canned model answer and check the
estimation it builds. The live test calls the configured provider and is
skipped unless RUN_LLM_EVAL=1 is set, because it needs a real API key and
spends tokens.
"""

import asyncio

import pytest

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.evaluation import evaluate_estimation_structure
from app.services.llm_service import (
    EstimationDraft,
    GenerationOptions,
    ModelAnswer,
    Task,
    preprocess_transcription,
)
from tests.conftest import RUN_LLM_EVAL

VALID = ESTIMATION_EXAMPLES[0]["estimation"]


def _stub_answer(monkeypatch, draft, finish_reason="stop"):
    calls = {}

    async def fake_draft(settings, model, options, system, user):
        calls.update(model=model, system=system, user=user)
        return ModelAnswer(draft, finish_reason, 800, 90)

    provider = get_settings().llm_provider
    monkeypatch.setattr(llm_service, f"_draft_with_{provider}", fake_draft)
    return calls


def _generate(transcription: str, options: GenerationOptions = GenerationOptions()):
    return asyncio.run(llm_service.generate_estimation(transcription, options))


def _draft(tasks, assumptions=()):
    return EstimationDraft(
        title="Portal de socios",
        tasks=[Task(name=name, hours=hours) for name, hours in tasks],
        team="2 desarrolladores",
        duration="5 semanas",
        assumptions=list(assumptions),
    )


# --- evaluate_estimation_structure ---------------------------------------


def test_a_well_formed_estimation_is_valid():
    assert evaluate_estimation_structure(VALID, "stop") == {
        "is_valid": True,
        "task_count": 5,
        "checks": {
            "has_title": True,
            "has_task_breakdown": True,
            "has_total": True,
            "total_matches_tasks": True,
            "has_team": True,
            "has_duration": True,
            "is_complete": True,
        },
        "issues": [],
    }


def test_anthropic_finish_reason_counts_as_complete():
    assert evaluate_estimation_structure(VALID, "end_turn")["is_valid"] is True


@pytest.mark.parametrize(
    ("estimation", "finish_reason", "issue"),
    [
        (VALID.replace("## Estimación: ", "## "), "stop", "has_title"),
        (VALID.replace("175 horas", "200 horas"), "stop", "total_matches_tasks"),
        (VALID.replace("**Equipo recomendado: ", "Equipo: "), "stop", "has_team"),
        (VALID.replace("**Duración estimada: ", "Plazo: "), "stop", "has_duration"),
        (VALID, "length", "is_complete"),
    ],
)
def test_each_defect_is_reported(estimation, finish_reason, issue):
    result = evaluate_estimation_structure(estimation, finish_reason)

    assert result["is_valid"] is False
    assert result["issues"] == [issue]


def test_text_without_structure_fails_every_structural_check():
    result = evaluate_estimation_structure("No puedo estimar esto.", "stop")

    assert result["is_valid"] is False
    assert result["task_count"] == 0
    assert result["issues"] == [
        "has_title",
        "has_task_breakdown",
        "has_total",
        "total_matches_tasks",
        "has_team",
        "has_duration",
    ]


# --- generate_estimation --------------------------------------------------


def test_generated_estimation_passes_the_evaluation(monkeypatch):
    _stub_answer(monkeypatch, _draft([("Diseño", 60), ("Backend", 85)]))

    result = _generate("Una reunión")

    assert result["total_hours"] == 145
    assert "**Total estimado: 145 horas**" in result["estimation"]
    assert "Supuestos" not in result["estimation"]
    assert result["finish_reason"] == "stop"
    assert (result["input_tokens"], result["output_tokens"]) == (800, 90)
    validation = evaluate_estimation_structure(result["estimation"], "stop")
    assert validation["is_valid"] is True
    assert validation["task_count"] == 2


def test_estimation_lists_the_assumptions(monkeypatch):
    _stub_answer(
        monkeypatch,
        _draft([("Diseño", 60)], ["Pasarela de pago estándar", "Sin app móvil"]),
    )

    result = _generate("Una reunión")

    assert result["estimation"].endswith(
        "### Supuestos:\n- Pasarela de pago estándar\n- Sin app móvil"
    )


def test_model_option_overrides_the_configured_model(monkeypatch):
    calls = _stub_answer(monkeypatch, _draft([("Diseño", 60)]))

    result = _generate("Una reunión", GenerationOptions(model="otro-modelo"))

    assert calls["model"] == result["model"] == "otro-modelo"


def test_preprocessing_option_controls_transcription_cleanup(monkeypatch):
    calls = _stub_answer(monkeypatch, _draft([("Diseño", 60)]))
    messy = "  Cliente:   quiero   una web. \n\n\n\n Comercial:\tvale.  "

    _generate(messy)
    assert "Cliente: quiero una web.\n\nComercial: vale." in calls["user"]

    _generate(messy, GenerationOptions(preprocessing=False))
    assert messy in calls["user"]


def test_preprocess_transcription_keeps_the_words():
    assert preprocess_transcription(" a  b \n\n\n c\t d ") == "a b\n\nc d"


@pytest.mark.parametrize(
    "draft",
    [
        None,
        _draft([]),
        _draft([("A", 0)]),
        _draft([("A", 10), ("B", -5)]),
    ],
    ids=["no-answer", "no-tasks", "zero-hours", "negative-hours"],
)
def test_unusable_model_answers_are_rejected(monkeypatch, draft):
    _stub_answer(monkeypatch, draft)

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

    validation = evaluate_estimation_structure(
        result["estimation"], result["finish_reason"]
    )
    assert validation["is_valid"], validation["issues"]
    assert validation["task_count"] >= 3
    assert 90 <= result["total_hours"] <= 350
    assert result["input_tokens"] > 0 and result["output_tokens"] > 0

"""Evaluates the estimations the service produces.

The offline tests check the structural evaluation and feed the service canned
model answers. The live test calls the configured provider and is skipped
unless RUN_LLM_EVAL=1 is set, because it needs a real API key and spends
tokens.
"""

import asyncio

import pytest

from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.evaluation import (
    evaluate_estimation_structure,
    parse_number,
    sum_task_hours,
)
from app.services.llm_service import Completion, GenerationOptions
from tests.conftest import RUN_LLM_EVAL

BASIC = ESTIMATION_EXAMPLES[0]["estimation"]

STRUCTURED = """\
## Project summary
A stock control web application for two warehouses.

## Task breakdown
| Task | Hours | Cost (EUR) |
|---|---|---|
| UI/UX design | 40 | 2,500.00 |
| Backend API | 60 | 3,750.00 |
| Testing and QA | 20 | 1,250.00 |

## Totals
- Total hours: 120
- Total cost: 7,500.00 EUR
- Recommended team: 2 full-stack developers
- Estimated duration: 5 weeks

## Risks and assumptions
- The payment gateway is a standard one.
"""


def _stub_completions(monkeypatch, *texts, finish_reason="stop"):
    """Answer each provider call with the next text and record the prompts."""
    calls = []
    answers = iter(texts)

    async def fake_complete(
        settings, model, system, user, *, max_tokens, thinking_budget=None
    ):
        calls.append(
            {
                "model": model,
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            }
        )
        return Completion(next(answers), finish_reason, 800, 90)

    monkeypatch.setattr(llm_service, "_complete", fake_complete)
    return calls


def _generate(transcription: str, options: GenerationOptions | None = None):
    return asyncio.run(llm_service.generate_estimation(transcription, options))


# --- evaluate_estimation_structure ---------------------------------------


def test_the_examples_format_is_valid():
    assert evaluate_estimation_structure(BASIC, "stop") == {
        "is_valid": True,
        "task_count": 5,
        "checks": {
            "has_task_breakdown": True,
            "has_total_hours": True,
            "total_hours_match_tasks": True,
            "has_team": True,
            "has_duration": True,
            "is_complete": True,
        },
        "issues": [],
    }


def test_the_structured_format_is_valid_and_checks_the_costs():
    result = evaluate_estimation_structure(STRUCTURED, "end_turn")

    assert result["is_valid"] is True
    assert result["task_count"] == 3
    assert result["checks"]["has_total_cost"] is True
    assert result["checks"]["total_cost_matches_tasks"] is True


@pytest.mark.parametrize(
    ("estimation", "finish_reason", "issues"),
    [
        (BASIC.replace("175 horas", "200 horas"), "stop", ["total_hours_match_tasks"]),
        (BASIC.replace("**Equipo recomendado: ", "Equipo: "), "stop", ["has_team"]),
        (BASIC.replace("**Duración estimada: ", "Plazo: "), "stop", ["has_duration"]),
        (BASIC, "length", ["is_complete"]),
        (BASIC, "max_tokens", ["is_complete"]),
        (
            STRUCTURED.replace("Total hours: 120", "Total hours: 150"),
            "stop",
            ["total_hours_match_tasks"],
        ),
        (
            STRUCTURED.replace("Total cost: 7,500.00", "Total cost: 9,000.00"),
            "stop",
            ["total_cost_matches_tasks"],
        ),
    ],
)
def test_each_defect_is_reported(estimation, finish_reason, issues):
    result = evaluate_estimation_structure(estimation, finish_reason)

    assert result["is_valid"] is False
    assert result["issues"] == issues


def test_a_total_row_in_the_table_is_not_counted_as_a_task():
    with_total_row = STRUCTURED.replace(
        "\n\n## Totals", "\n| **Total** | **120** | **7,500.00** |\n\n## Totals"
    )

    result = evaluate_estimation_structure(with_total_row, "stop")

    assert result["task_count"] == 3
    assert result["is_valid"] is True


def test_text_without_structure_fails_every_structural_check():
    result = evaluate_estimation_structure("No puedo estimar esto.", "stop")

    assert result["is_valid"] is False
    assert result["task_count"] == 0
    assert result["issues"] == [
        "has_task_breakdown",
        "has_total_hours",
        "total_hours_match_tasks",
        "has_team",
        "has_duration",
    ]


def test_sum_task_hours_reads_both_formats():
    assert sum_task_hours(BASIC) == 175
    assert sum_task_hours(STRUCTURED) == 120
    assert sum_task_hours("No puedo estimar esto.") is None


def test_sum_task_hours_reads_bold_task_names_and_bold_hours():
    estimation = (
        "1. **Diseño UI/UX:** 40 horas  \n   - Prototipos\n"
        "2. **Backend API**: 60 horas\n"
        "3. Autenticación: **20 horas**\n"
    )

    assert sum_task_hours(estimation) == 120


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("175", 175),
        ("37.5", 37.5),
        ("37,5", 37.5),
        ("2,500", 2500),
        ("2.500", 2500),
        ("10,937.50", 10937.5),
        ("10.937,50", 10937.5),
    ],
)
def test_parse_number_understands_both_separator_styles(text, value):
    assert parse_number(text) == value


# --- generate_estimation --------------------------------------------------


def test_generation_returns_the_estimation_with_hours_totalled_in_code(monkeypatch):
    wrong_total = BASIC.replace("175 horas", "200 horas")
    calls = _stub_completions(monkeypatch, f"\n{wrong_total}\n")

    result = _generate("Una reunión")

    assert result["estimation"] == wrong_total
    assert result["total_hours"] == 175
    assert result["finish_reason"] == "stop"
    assert (result["input_tokens"], result["output_tokens"]) == (800, 90)
    assert result["latency_ms"] >= 0
    assert result["extracted_requirements"] is None
    assert [call["user"] for call in calls] == ["Una reunión"]
    assert calls[0]["system"] == llm_service.build_system_prompt()


def test_model_max_tokens_and_thinking_budget_reach_the_provider(monkeypatch):
    calls = _stub_completions(monkeypatch, BASIC)

    result = _generate(
        "Una reunión",
        GenerationOptions(model="otro-modelo", max_tokens=2000, thinking_budget=1024),
    )

    assert result["model"] == "otro-modelo"
    assert calls[0]["model"] == "otro-modelo"
    assert calls[0]["max_tokens"] == 2000
    assert calls[0]["thinking_budget"] == 1024


def test_prompt_options_shape_the_system_prompt(monkeypatch):
    calls = _stub_completions(monkeypatch, BASIC, BASIC)

    _generate("Una reunión", GenerationOptions(preprocessing="inline_cleaning"))
    _generate("Una reunión", GenerationOptions(example_format="json", num_examples=1))

    assert calls[0]["system"] == llm_service.build_system_prompt(inline_cleaning=True)
    assert calls[1]["system"] == llm_service.build_system_prompt(
        example_format="json", num_examples=1
    )


def test_two_phase_preprocessing_extracts_the_requirements_first(monkeypatch):
    calls = _stub_completions(monkeypatch, "- Control de stock\n- Roles\n", BASIC)

    result = _generate(
        "Hola, qué tal el finde... queremos control de stock y roles.",
        GenerationOptions(preprocessing="two_phase", thinking_budget=1024),
    )

    assert len(calls) == 2
    assert calls[0]["system"] == llm_service.EXTRACTION_SYSTEM_PROMPT
    assert calls[0]["user"].startswith("Hola, qué tal el finde")
    assert calls[0]["thinking_budget"] is None
    assert calls[1]["user"] == "- Control de stock\n- Roles"
    assert calls[1]["thinking_budget"] == 1024
    assert llm_service.INLINE_CLEANING_BLOCK not in calls[1]["system"]
    assert result["extracted_requirements"] == "- Control de stock\n- Roles"
    assert (result["input_tokens"], result["output_tokens"]) == (1600, 180)


def test_a_truncated_answer_is_returned_with_its_finish_reason(monkeypatch):
    _stub_completions(monkeypatch, "## Estimación: Port", finish_reason="length")

    result = _generate("Una reunión")

    assert result["finish_reason"] == "length"
    assert result["total_hours"] is None
    validation = evaluate_estimation_structure(result["estimation"], "length")
    assert "is_complete" in validation["issues"]


@pytest.mark.parametrize("text", ["", "   \n"])
def test_an_empty_answer_is_an_error(monkeypatch, text):
    _stub_completions(monkeypatch, text)

    with pytest.raises(llm_service.LLMServiceError):
        _generate("Una reunión")


def test_an_empty_extraction_is_an_error(monkeypatch):
    _stub_completions(monkeypatch, "  ", BASIC)

    with pytest.raises(llm_service.LLMServiceError):
        _generate("Una reunión", GenerationOptions(preprocessing="two_phase"))


@pytest.mark.skipif(not RUN_LLM_EVAL, reason="set RUN_LLM_EVAL=1 to call the LLM")
def test_live_estimation_is_complete_and_calibrated_by_the_examples():
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
    assert validation["checks"]["is_complete"]
    assert validation["task_count"] >= 3
    assert 90 <= result["total_hours"] <= 350
    assert result["input_tokens"] > 0 and result["output_tokens"] > 0

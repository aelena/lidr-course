import json
import re

from app.context.examples import (
    ESTIMATION_EXAMPLES,
    RAW_EXAMPLES,
    euros,
    format_examples_for_prompt,
    render_estimation,
    select_examples,
    task_cost,
    total_cost,
    total_hours,
)
from app.services import llm_service
from app.services.llm_service import build_system_prompt


def test_at_least_two_examples_with_summary_and_estimation():
    assert len(ESTIMATION_EXAMPLES) >= 2
    for example in ESTIMATION_EXAMPLES:
        assert set(example) == {"meeting_summary", "estimation"}
        assert example["meeting_summary"].strip()
        assert example["estimation"].startswith("## Estimación: ")


def test_raw_examples_have_every_field_and_positive_hours():
    for example in RAW_EXAMPLES:
        for field in ("meeting_summary", "title", "team", "duration"):
            assert example[field].strip(), field
        assert example["tasks"]
        for task in example["tasks"]:
            assert task["name"].strip()
            assert isinstance(task["hours"], int) and task["hours"] > 0


def test_raw_examples_are_priced_at_the_rates_the_prompt_gives():
    for example in RAW_EXAMPLES:
        for task in example["tasks"]:
            assert f"{task['rate']} EUR/hour" in llm_service.RATES, task["name"]


def test_rendered_examples_follow_the_estimation_format():
    for raw, example in zip(RAW_EXAMPLES, ESTIMATION_EXAMPLES):
        estimation = example["estimation"]
        assert "### Desglose de tareas:" in estimation
        assert len(
            re.findall(r"^\d+\. .+: \d+ horas · [\d.]+ EUR$", estimation, re.M)
        ) == len(raw["tasks"])
        assert f"**Total estimado: {total_hours(raw)} horas**" in estimation
        assert f"**Coste total: {euros(total_cost(raw))} (sin IVA)**" in estimation
        assert f"**Equipo recomendado: {raw['team']}**" in estimation
        assert f"**Duración estimada: {raw['duration']}**" in estimation


def test_render_estimation_computes_total_from_tasks():
    rendered = render_estimation(
        {
            "title": "Demo",
            "tasks": [
                {"name": "A", "hours": 10, "rate": 55},
                {"name": "B", "hours": 20, "rate": 45},
            ],
            "team": "1 desarrollador",
            "duration": "1 semana",
        }
    )
    assert "1. A: 10 horas · 550 EUR\n2. B: 20 horas · 900 EUR" in rendered
    assert "**Total estimado: 30 horas**" in rendered
    assert "**Coste total: 1.450 EUR (sin IVA)**" in rendered


def test_select_examples_takes_the_first_n():
    assert select_examples(1) == RAW_EXAMPLES[:1]
    assert select_examples(0) == []
    assert select_examples(-2) == []
    assert select_examples(99) == RAW_EXAMPLES


def test_format_examples_for_prompt_in_markdown():
    block = format_examples_for_prompt(RAW_EXAMPLES)

    assert block.count("<example ") == len(RAW_EXAMPLES)
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"] in block
        assert example["estimation"] in block


def test_format_examples_for_prompt_in_json():
    block = format_examples_for_prompt(RAW_EXAMPLES[:1], "json")

    payload = block.split("<estimation>\n")[1].split("\n</estimation>")[0]
    data = json.loads(payload)
    assert data["title"] == RAW_EXAMPLES[0]["title"]
    assert data["tasks"] == [
        {**task, "cost_eur": task_cost(task)} for task in RAW_EXAMPLES[0]["tasks"]
    ]
    assert data["total_hours"] == total_hours(RAW_EXAMPLES[0])
    assert data["total_cost_eur"] == total_cost(RAW_EXAMPLES[0]) == 8850


def test_format_examples_for_prompt_with_no_examples_is_empty():
    assert format_examples_for_prompt([]) == ""


def test_system_prompt_orders_role_rates_output_spec_and_examples():
    prompt = build_system_prompt()

    positions = [
        prompt.index(section)
        for section in (
            llm_service.ROLE,
            llm_service.RATES,
            llm_service.ACTIVE_OUTPUT_PROMPT,
            llm_service.EXAMPLES_INTRO,
            ESTIMATION_EXAMPLES[0]["estimation"],
            ESTIMATION_EXAMPLES[1]["estimation"],
        )
    ]
    assert positions == sorted(positions)
    assert positions[0] == 0
    assert llm_service.INLINE_CLEANING_BLOCK not in prompt


def test_system_prompt_places_the_cleaning_block_after_the_role():
    prompt = build_system_prompt(inline_cleaning=True)

    assert prompt.startswith(
        f"{llm_service.ROLE}\n\n{llm_service.INLINE_CLEANING_BLOCK}\n\n{llm_service.RATES}"
    )


def test_system_prompt_limits_and_formats_the_examples():
    prompt = build_system_prompt(example_format="json", num_examples=1)

    assert prompt.count("<example ") == 1
    assert '"total_hours": 175' in prompt


def test_system_prompt_without_examples_drops_the_whole_block():
    for prompt in (
        build_system_prompt(use_examples=False),
        build_system_prompt(num_examples=0),
    ):
        assert "<example " not in prompt
        assert llm_service.EXAMPLES_INTRO not in prompt
        assert prompt.endswith(llm_service.ACTIVE_OUTPUT_PROMPT)


def test_system_prompt_uses_the_active_output_prompt(monkeypatch):
    monkeypatch.setattr(
        llm_service, "ACTIVE_OUTPUT_PROMPT", llm_service.PROMPT_OUTPUT_STRUCTURED
    )

    prompt = build_system_prompt(use_examples=False)

    assert prompt.endswith(llm_service.PROMPT_OUTPUT_STRUCTURED)
    assert llm_service.PROMPT_OUTPUT_BASIC not in prompt

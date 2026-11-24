import json
import re

from app.context.examples import (
    ESTIMATION_EXAMPLES,
    EXAMPLES_PATH,
    render_estimation,
    total_hours,
)
from app.services.llm_service import build_system_prompt, build_user_prompt

RAW_EXAMPLES = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


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


def test_rendered_examples_follow_the_estimation_format():
    for raw, example in zip(RAW_EXAMPLES, ESTIMATION_EXAMPLES):
        estimation = example["estimation"]
        assert "### Desglose de tareas:" in estimation
        assert len(re.findall(r"^\d+\. .+: \d+ horas$", estimation, re.M)) == len(
            raw["tasks"]
        )
        assert f"**Total estimado: {total_hours(raw)} horas**" in estimation
        assert f"**Equipo recomendado: {raw['team']}**" in estimation
        assert f"**Duración estimada: {raw['duration']}**" in estimation


def test_render_estimation_computes_total_from_tasks():
    rendered = render_estimation(
        {
            "title": "Demo",
            "tasks": [{"name": "A", "hours": 10}, {"name": "B", "hours": 5}],
            "team": "1 desarrollador",
            "duration": "1 semana",
        }
    )
    assert "1. A: 10 horas\n2. B: 5 horas" in rendered
    assert "**Total estimado: 15 horas**" in rendered
    assert "Supuestos" not in rendered


def test_system_prompt_injects_every_example():
    prompt = build_system_prompt()
    assert prompt.startswith("Eres un estimador de software experto")
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"] in prompt
        assert example["estimation"] in prompt


def test_user_prompt_carries_the_transcription():
    assert "<transcripcion>\nUna reunión\n</transcripcion>" in build_user_prompt(
        "Una reunión"
    )

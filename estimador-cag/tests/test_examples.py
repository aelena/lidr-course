from app.context.examples import ESTIMATION_EXAMPLES, render_estimation
from app.services.llm_service import build_system_prompt


def test_at_least_two_examples_with_summary_and_estimation():
    assert len(ESTIMATION_EXAMPLES) >= 2
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"].strip()
        assert example["estimation"].startswith("## Estimación: ")


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


def test_system_prompt_injects_every_example():
    prompt = build_system_prompt()
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"] in prompt
        assert example["estimation"] in prompt

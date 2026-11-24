"""Past estimations injected into the LLM prompt as reference.

The data lives in examples.json. Each entry is rendered to markdown with the
total computed from the task breakdown, the same way new estimations are.
"""

import json
from pathlib import Path
from typing import Any, Literal

EXAMPLES_PATH = Path(__file__).with_name("examples.json")

ExampleFormat = Literal["markdown", "json"]


def total_hours(estimation: dict[str, Any]) -> int:
    return sum(task["hours"] for task in estimation["tasks"])


def render_estimation(estimation: dict[str, Any]) -> str:
    tasks = "\n".join(
        f"{number}. {task['name']}: {task['hours']} horas"
        for number, task in enumerate(estimation["tasks"], start=1)
    )
    rendered = (
        f"## Estimación: {estimation['title']}\n\n"
        f"### Desglose de tareas:\n{tasks}\n\n"
        f"**Total estimado: {total_hours(estimation)} horas**\n"
        f"**Equipo recomendado: {estimation['team']}**\n"
        f"**Duración estimada: {estimation['duration']}**"
    )
    assumptions = estimation.get("assumptions") or []
    if assumptions:
        items = "\n".join(f"- {assumption}" for assumption in assumptions)
        rendered += f"\n\n### Supuestos:\n{items}"
    return rendered


def render_estimation_json(estimation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "title": estimation["title"],
            "tasks": estimation["tasks"],
            "total_hours": total_hours(estimation),
            "team": estimation["team"],
            "duration": estimation["duration"],
        },
        ensure_ascii=False,
        indent=2,
    )


def load_raw_examples(path: Path = EXAMPLES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_examples(
    example_format: ExampleFormat = "markdown", limit: int | None = None
) -> list[dict[str, str]]:
    """Return the examples ready for the prompt, at most `limit` of them."""
    render = render_estimation_json if example_format == "json" else render_estimation
    selected = RAW_EXAMPLES if limit is None else RAW_EXAMPLES[:limit]
    return [
        {
            "meeting_summary": example["meeting_summary"],
            "estimation": render(example),
        }
        for example in selected
    ]


RAW_EXAMPLES = load_raw_examples()
ESTIMATION_EXAMPLES = format_examples()

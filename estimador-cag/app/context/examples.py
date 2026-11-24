"""Past estimations injected into the LLM prompt as reference.

The data lives in examples.json. Each entry is rendered to markdown with the
total computed from the task breakdown, the same way new estimations are.
"""

import json
from pathlib import Path
from typing import Any

EXAMPLES_PATH = Path(__file__).with_name("examples.json")


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


def load_examples(path: Path = EXAMPLES_PATH) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "meeting_summary": example["meeting_summary"],
            "estimation": render_estimation(example),
        }
        for example in raw
    ]


ESTIMATION_EXAMPLES = load_examples()

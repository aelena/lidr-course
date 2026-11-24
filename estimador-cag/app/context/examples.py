"""Past estimations injected into the LLM prompt as reference.

The data lives in examples.json. Each entry is rendered to the markdown
format the model is expected to reproduce, with the total computed from the
task breakdown.
"""

import json
from pathlib import Path
from typing import Any

EXAMPLES_PATH = Path(__file__).with_name("examples.json")


def render_estimation(example: dict[str, Any]) -> str:
    tasks = "\n".join(
        f"{number}. {task['name']}: {task['hours']} horas"
        for number, task in enumerate(example["tasks"], start=1)
    )
    total = sum(task["hours"] for task in example["tasks"])
    return (
        f"## Estimación: {example['title']}\n\n"
        f"### Desglose de tareas:\n{tasks}\n\n"
        f"**Total estimado: {total} horas**\n"
        f"**Equipo recomendado: {example['team']}**\n"
        f"**Duración estimada: {example['duration']}**"
    )


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

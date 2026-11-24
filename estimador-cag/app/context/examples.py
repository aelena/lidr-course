"""Past estimations injected into the LLM prompt as reference.

The data lives in examples.json; the total of each one is computed from its
task breakdown.
"""

import json
from pathlib import Path
from typing import Any, Literal

EXAMPLES_PATH = Path(__file__).with_name("examples.json")

ExampleFormat = Literal["markdown", "json"]
Example = dict[str, Any]


def total_hours(estimation: Example) -> int:
    return sum(task["hours"] for task in estimation["tasks"])


def render_estimation(estimation: Example) -> str:
    tasks = "\n".join(
        f"{number}. {task['name']}: {task['hours']} horas"
        for number, task in enumerate(estimation["tasks"], start=1)
    )
    return (
        f"## Estimación: {estimation['title']}\n\n"
        f"### Desglose de tareas:\n{tasks}\n\n"
        f"**Total estimado: {total_hours(estimation)} horas**\n"
        f"**Equipo recomendado: {estimation['team']}**\n"
        f"**Duración estimada: {estimation['duration']}**"
    )


def render_estimation_json(estimation: Example) -> str:
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


RENDERERS = {"markdown": render_estimation, "json": render_estimation_json}


def load_examples(path: Path = EXAMPLES_PATH) -> list[Example]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_examples(num_examples: int) -> list[Example]:
    """The first `num_examples` past estimations."""
    return RAW_EXAMPLES[: max(num_examples, 0)]


def format_examples_for_prompt(
    examples: list[Example], example_format: ExampleFormat = "markdown"
) -> str:
    """Render the examples as the block that goes into the system prompt."""
    render = RENDERERS[example_format]
    return "\n\n".join(
        f'<example number="{number}">\n'
        f"<meeting_summary>\n{example['meeting_summary']}\n</meeting_summary>\n"
        f"<estimation>\n{render(example)}\n</estimation>\n"
        f"</example>"
        for number, example in enumerate(examples, start=1)
    )


RAW_EXAMPLES = load_examples()

ESTIMATION_EXAMPLES = [
    {
        "meeting_summary": example["meeting_summary"],
        "estimation": render_estimation(example),
    }
    for example in RAW_EXAMPLES
]

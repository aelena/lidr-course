"""Structural checks on a generated estimation."""

import re
from typing import Any

TASK_LINE = re.compile(r"^\d+\. .+: (\d+) horas$", re.M)
TOTAL_LINE = re.compile(r"^\*\*Total estimado: (\d+) horas\*\*$", re.M)

# OpenAI reports "stop" and Anthropic "end_turn" when the model finished by itself.
COMPLETE_FINISH_REASONS = {"stop", "end_turn"}


def evaluate_estimation_structure(estimation: str, finish_reason: str) -> dict[str, Any]:
    """Check that an estimation has every section and adds up."""
    hours = [int(value) for value in TASK_LINE.findall(estimation)]
    total = TOTAL_LINE.search(estimation)
    checks = {
        "has_title": estimation.startswith("## Estimación: "),
        "has_task_breakdown": "### Desglose de tareas:" in estimation and bool(hours),
        "has_total": total is not None,
        "total_matches_tasks": bool(hours)
        and total is not None
        and int(total.group(1)) == sum(hours),
        "has_team": "**Equipo recomendado: " in estimation,
        "has_duration": "**Duración estimada: " in estimation,
        "is_complete": finish_reason in COMPLETE_FINISH_REASONS,
    }
    return {
        "is_valid": all(checks.values()),
        "task_count": len(hours),
        "checks": checks,
        "issues": [name for name, passed in checks.items() if not passed],
    }

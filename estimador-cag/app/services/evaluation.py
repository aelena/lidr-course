"""Reads a generated estimation back and checks its structure.

The model writes free markdown, in one of two shapes: the numbered list used
by the past estimations ("1. Diseño UI/UX: 40 horas") or the table asked for
by the structured output prompt ("| Task | Hours | Cost (EUR) |"). Both are
understood here.
"""

import re
from typing import Any

NUMBER = r"\d[\d.,]*"
NUMBERED_TASK = re.compile(
    rf"^\s*\d+\.\s+.+?:\s*\**\s*({NUMBER})\**\s*(?:horas|hours|h)\b", re.M | re.I
)
HOURS_CELL = re.compile(rf"^({NUMBER})\s*(?:horas|hours|h)?$", re.I)
COST_CELL = re.compile(rf"^(?:EUR|€)?\s*({NUMBER})\s*(?:EUR|€)?$", re.I)
TOTAL_HOURS = re.compile(
    rf"(?:Total hours|Total estimado|Total de horas)\**\s*:\s*\**\s*({NUMBER})", re.I
)
TOTAL_COST = re.compile(
    rf"(?:Total cost|Coste total|Costo total)\**\s*:\s*\**\s*(?:EUR|€)?\s*({NUMBER})",
    re.I,
)
TEAM = re.compile(r"Recommended team|Equipo recomendado", re.I)
DURATION = re.compile(r"Estimated duration|Duración estimada", re.I)

# OpenAI reports "stop" and Anthropic "end_turn" when the model finished by itself.
COMPLETE_FINISH_REASONS = {"stop", "end_turn"}

COST_TOLERANCE = 1.0


def parse_number(text: str) -> float:
    """Parse "175", "37.5", "10,937.50" or "10.937,50"."""
    text = text.strip().rstrip(".,")
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        text = text.replace(thousands, "").replace(decimal, ".")
    elif "," in text or "." in text:
        separator = "," if "," in text else "."
        head, _, tail = text.rpartition(separator)
        if len(tail) == 3 and head:
            text = text.replace(separator, "")
        else:
            text = text.replace(separator, ".")
    return float(text)


def extract_tasks(estimation: str) -> list[dict[str, float | None]]:
    """Return the hours, and the cost when given, of every task in the breakdown."""
    tasks: list[dict[str, float | None]] = []
    for line in estimation.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("*").strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or "total" in cells[0].lower():
            continue
        hours = HOURS_CELL.match(cells[1])
        if not hours:
            continue
        cost = COST_CELL.match(cells[2]) if len(cells) > 2 else None
        tasks.append(
            {
                "hours": parse_number(hours.group(1)),
                "cost": parse_number(cost.group(1)) if cost else None,
            }
        )
    if tasks:
        return tasks
    return [
        {"hours": parse_number(value), "cost": None}
        for value in NUMBERED_TASK.findall(estimation)
    ]


def sum_task_hours(estimation: str) -> float | None:
    """Total the hours of the breakdown; None when no task could be read."""
    tasks = extract_tasks(estimation)
    return sum(task["hours"] for task in tasks) if tasks else None


def evaluate_estimation_structure(estimation: str, finish_reason: str) -> dict[str, Any]:
    """Check that an estimation has every section, adds up and was not cut short."""
    tasks = extract_tasks(estimation)
    hours = sum(task["hours"] for task in tasks)
    costs = [task["cost"] for task in tasks if task["cost"] is not None]
    total_hours = TOTAL_HOURS.search(estimation)
    total_cost = TOTAL_COST.search(estimation)

    checks = {
        "has_task_breakdown": bool(tasks),
        "has_total_hours": total_hours is not None,
        "total_hours_match_tasks": bool(tasks)
        and total_hours is not None
        and abs(parse_number(total_hours.group(1)) - hours) < 0.01,
        "has_team": TEAM.search(estimation) is not None,
        "has_duration": DURATION.search(estimation) is not None,
        "is_complete": finish_reason in COMPLETE_FINISH_REASONS,
    }
    if costs or total_cost:
        checks["has_total_cost"] = total_cost is not None
        checks["total_cost_matches_tasks"] = (
            len(costs) == len(tasks)
            and total_cost is not None
            and abs(parse_number(total_cost.group(1)) - sum(costs)) <= COST_TOLERANCE
        )
    return {
        "is_valid": all(checks.values()),
        "task_count": len(tasks),
        "checks": checks,
        "issues": [name for name, passed in checks.items() if not passed],
    }

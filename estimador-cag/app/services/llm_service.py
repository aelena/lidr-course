"""Generates estimations by calling the LLM provider selected in the settings.

The model returns the estimation as structured data. The markdown and the
total hours are produced here, so the total always matches the breakdown.
"""

from dataclasses import dataclass
from functools import lru_cache

import anthropic
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import Settings
from app.context.examples import ESTIMATION_EXAMPLES, render_estimation, total_hours

logger = structlog.get_logger()

MAX_OUTPUT_TOKENS = 16000

SYSTEM_PROMPT = """\
Eres un estimador de software experto. Generas estimaciones de proyectos a \
partir de la transcripción de una reunión con el cliente, basándote en \
estimaciones anteriores.

A continuación tienes estimaciones anteriores de la empresa. Úsalas como \
referencia para calibrar el nivel de detalle, las horas por tipo de tarea, el \
equipo y la duración.

{examples}

Cuando recibas una transcripción:
- Identifica qué pide el cliente y desglósalo en tareas concretas con sus horas.
- Calibra las horas con las estimaciones anteriores, ajustando por alcance y complejidad.
- Devuelve el título del proyecto, las tareas con sus horas, el equipo \
recomendado y la duración estimada. No calcules el total de horas: se suma \
aparte a partir de las tareas.
- Si la transcripción deja fuera información que cambia la estimación, indica \
en los supuestos lo que has asumido. Si no has asumido nada, deja la lista vacía.
"""


class Task(BaseModel):
    name: str = Field(description="Nombre de la tarea, como en los ejemplos.")
    hours: int = Field(description="Horas estimadas para la tarea.")


class EstimationDraft(BaseModel):
    """Structured estimation returned by the model."""

    title: str = Field(description="Título del proyecto, sin el prefijo 'Estimación:'.")
    tasks: list[Task]
    team: str = Field(description="Equipo recomendado.")
    duration: str = Field(description="Duración estimada, por ejemplo '6-8 semanas'.")
    assumptions: list[str] = Field(description="Supuestos asumidos; vacía si no hay.")


@dataclass(frozen=True)
class EstimationResult:
    estimation: str
    total_hours: int
    input_tokens: int
    output_tokens: int


class LLMServiceError(Exception):
    """The LLM provider failed or returned no usable estimation."""


def build_system_prompt(examples: list[dict[str, str]] = ESTIMATION_EXAMPLES) -> str:
    rendered = "\n\n".join(
        f'<ejemplo numero="{number}">\n'
        f"<resumen_reunion>\n{example['meeting_summary']}\n</resumen_reunion>\n"
        f"<estimacion>\n{example['estimation']}\n</estimacion>\n"
        f"</ejemplo>"
        for number, example in enumerate(examples, start=1)
    )
    return SYSTEM_PROMPT.format(examples=rendered)


def build_user_prompt(transcription: str) -> str:
    return (
        "Genera la estimación para esta reunión.\n\n"
        f"<transcripcion>\n{transcription}\n</transcripcion>"
    )


@lru_cache
def _openai_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)


@lru_cache
def _anthropic_client(api_key: str) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=api_key)


async def _draft_with_openai(
    settings: Settings, system: str, user: str
) -> tuple[EstimationDraft | None, int, int]:
    client = _openai_client(settings.llm_api_key.get_secret_value())
    try:
        response = await client.chat.completions.parse(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=EstimationDraft,
        )
    except openai.APIError as error:
        raise LLMServiceError(f"OpenAI request failed: {error}") from error
    usage = response.usage
    return (
        response.choices[0].message.parsed,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


async def _draft_with_anthropic(
    settings: Settings, system: str, user: str
) -> tuple[EstimationDraft | None, int, int]:
    client = _anthropic_client(settings.llm_api_key.get_secret_value())
    try:
        response = await client.messages.parse(
            model=settings.llm_model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=EstimationDraft,
        )
    except anthropic.APIError as error:
        raise LLMServiceError(f"Anthropic request failed: {error}") from error
    if response.stop_reason == "refusal":
        raise LLMServiceError("Anthropic declined to generate the estimation")
    return (
        response.parsed_output,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


async def generate_estimation(transcription: str, settings: Settings) -> EstimationResult:
    """Return the estimation for a meeting transcription."""
    system = build_system_prompt()
    user = build_user_prompt(transcription)
    log = logger.bind(provider=settings.llm_provider, model=settings.llm_model)
    log.debug("estimation_requested", transcription_chars=len(transcription))
    if settings.llm_provider == "openai":
        draft, input_tokens, output_tokens = await _draft_with_openai(
            settings, system, user
        )
    else:
        draft, input_tokens, output_tokens = await _draft_with_anthropic(
            settings, system, user
        )
    if draft is None or not draft.tasks:
        log.warning("estimation_unusable", reason="no tasks")
        raise LLMServiceError("The model returned no usable estimation")
    if any(task.hours <= 0 for task in draft.tasks):
        log.warning("estimation_unusable", reason="non-positive hours")
        raise LLMServiceError("The model returned a task without positive hours")
    data = draft.model_dump()
    log.info(
        "estimation_generated",
        tasks=len(draft.tasks),
        total_hours=total_hours(data),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return EstimationResult(
        estimation=render_estimation(data),
        total_hours=total_hours(data),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

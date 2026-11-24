"""Generates estimations by calling the LLM provider selected in the settings.

The model returns the estimation as structured data. The markdown and the
total hours are produced here, so the total always matches the breakdown.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import anthropic
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.context.examples import (
    ExampleFormat,
    format_examples,
    render_estimation,
    total_hours,
)

logger = structlog.get_logger()

DEFAULT_ANTHROPIC_MAX_TOKENS = 16000

ROLE = """\
Eres un estimador de software experto. Generas estimaciones de proyectos a \
partir de la transcripción de una reunión con el cliente, basándote en \
estimaciones anteriores."""

EXAMPLES_INTRO = """\
A continuación tienes estimaciones anteriores de la empresa. Úsalas como \
referencia para calibrar el nivel de detalle, las horas por tipo de tarea, el \
equipo y la duración."""

INSTRUCTIONS = """\
Cuando recibas una transcripción:
- Identifica qué pide el cliente y desglósalo en tareas concretas con sus horas.
- Calibra las horas con las estimaciones anteriores si las hay, ajustando por \
alcance y complejidad.
- Devuelve el título del proyecto, las tareas con sus horas, el equipo \
recomendado y la duración estimada. No calcules el total de horas: se suma \
aparte a partir de las tareas.
- Si la transcripción deja fuera información que cambia la estimación, indica \
en los supuestos lo que has asumido. Si no has asumido nada, deja la lista vacía.
"""


class Task(BaseModel):
    name: str = Field(description="Nombre de la tarea, sin las horas.")
    hours: int = Field(description="Horas estimadas para la tarea.")


class EstimationDraft(BaseModel):
    """Structured estimation returned by the model."""

    title: str = Field(description="Título del proyecto, sin el prefijo 'Estimación:'.")
    tasks: list[Task]
    team: str = Field(description="Equipo recomendado.")
    duration: str = Field(description="Duración estimada, por ejemplo '6-8 semanas'.")
    assumptions: list[str] = Field(description="Supuestos asumidos; vacía si no hay.")


@dataclass(frozen=True)
class GenerationOptions:
    """Per-request choices on how the prompt is built and the model is called."""

    preprocessing: bool = True
    example_format: ExampleFormat = "markdown"
    num_examples: int | None = None
    use_examples: bool = True
    model: str | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None


@dataclass(frozen=True)
class ModelAnswer:
    draft: EstimationDraft | None
    finish_reason: str
    input_tokens: int
    output_tokens: int


class LLMServiceError(Exception):
    """The LLM provider failed or returned no usable estimation."""


def preprocess_transcription(transcription: str) -> str:
    """Trim every line and collapse repeated spaces and blank lines."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in transcription.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def build_system_prompt(options: GenerationOptions = GenerationOptions()) -> str:
    sections = [ROLE]
    if options.use_examples:
        examples = format_examples(options.example_format, options.num_examples)
        rendered = "\n\n".join(
            f'<ejemplo numero="{number}">\n'
            f"<resumen_reunion>\n{example['meeting_summary']}\n</resumen_reunion>\n"
            f"<estimacion>\n{example['estimation']}\n</estimacion>\n"
            f"</ejemplo>"
            for number, example in enumerate(examples, start=1)
        )
        sections += [EXAMPLES_INTRO, rendered]
    sections.append(INSTRUCTIONS)
    return "\n\n".join(sections)


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
    settings: Settings, model: str, options: GenerationOptions, system: str, user: str
) -> ModelAnswer:
    client = _openai_client(settings.llm_api_key.get_secret_value())
    params: dict[str, Any] = {}
    if options.max_tokens is not None:
        params["max_completion_tokens"] = options.max_tokens
    if options.thinking_budget is not None:
        logger.warning("thinking_budget_ignored", provider="openai", model=model)
    try:
        response = await client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=EstimationDraft,
            **params,
        )
    except openai.LengthFinishReasonError as error:
        raise LLMServiceError(
            "The model ran out of tokens before finishing the estimation; "
            "raise max_tokens"
        ) from error
    except openai.OpenAIError as error:
        raise LLMServiceError(f"OpenAI request failed: {error}") from error
    choice = response.choices[0]
    usage = response.usage
    return ModelAnswer(
        draft=choice.message.parsed,
        finish_reason=choice.finish_reason,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


async def _draft_with_anthropic(
    settings: Settings, model: str, options: GenerationOptions, system: str, user: str
) -> ModelAnswer:
    client = _anthropic_client(settings.llm_api_key.get_secret_value())
    max_tokens = options.max_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS
    params: dict[str, Any] = {}
    if options.thinking_budget is not None:
        if options.thinking_budget >= max_tokens:
            raise LLMServiceError("thinking_budget must be lower than max_tokens")
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": options.thinking_budget,
        }
    try:
        response = await client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=EstimationDraft,
            **params,
        )
    except anthropic.AnthropicError as error:
        raise LLMServiceError(f"Anthropic request failed: {error}") from error
    if response.stop_reason == "refusal":
        raise LLMServiceError("Anthropic declined to generate the estimation")
    if response.stop_reason == "max_tokens":
        raise LLMServiceError(
            "The model ran out of tokens before finishing the estimation; "
            "raise max_tokens"
        )
    return ModelAnswer(
        draft=response.parsed_output,
        finish_reason=response.stop_reason or "unknown",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


async def generate_estimation(
    transcription: str, options: GenerationOptions = GenerationOptions()
) -> dict[str, Any]:
    """Return the estimation for a meeting transcription and how it was produced."""
    settings = get_settings()
    model = options.model or settings.llm_model
    if options.preprocessing:
        transcription = preprocess_transcription(transcription)
    system = build_system_prompt(options)
    user = build_user_prompt(transcription)
    log = logger.bind(provider=settings.llm_provider, model=model)
    log.debug(
        "estimation_requested",
        transcription_chars=len(transcription),
        use_examples=options.use_examples,
        example_format=options.example_format,
        num_examples=options.num_examples,
    )
    draft_with = (
        _draft_with_openai if settings.llm_provider == "openai" else _draft_with_anthropic
    )
    answer = await draft_with(settings, model, options, system, user)
    draft = answer.draft
    if draft is None or not draft.tasks:
        log.warning("estimation_unusable", reason="no tasks")
        raise LLMServiceError("The model returned no usable estimation")
    if any(task.hours <= 0 for task in draft.tasks):
        log.warning("estimation_unusable", reason="non-positive hours")
        raise LLMServiceError("The model returned a task without positive hours")
    data = draft.model_dump()
    result = {
        "estimation": render_estimation(data),
        "model": model,
        "provider": settings.llm_provider,
        "finish_reason": answer.finish_reason,
        "total_hours": total_hours(data),
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
    }
    log.info(
        "estimation_generated",
        tasks=len(draft.tasks),
        total_hours=result["total_hours"],
        finish_reason=answer.finish_reason,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )
    return result

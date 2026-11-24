"""Generates estimations by calling the LLM provider selected in the settings."""

import logging
from functools import lru_cache

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import Settings
from app.context.examples import ESTIMATION_EXAMPLES

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 16000

SYSTEM_PROMPT = """\
Eres un consultor de software sénior que prepara estimaciones de proyectos a \
partir de transcripciones de reuniones con clientes.

A continuación tienes estimaciones anteriores de la empresa. Úsalas como \
referencia para calibrar el nivel de detalle, las horas por tipo de tarea y el \
formato de salida.

{examples}

Cuando recibas una transcripción:
- Identifica qué pide el cliente y desglósalo en tareas concretas con sus horas.
- Calibra las horas con las estimaciones anteriores, ajustando por alcance y complejidad.
- Responde solo con la estimación, en markdown y con la misma estructura que los \
ejemplos: título, desglose de tareas, total estimado, equipo recomendado y \
duración estimada.
- Si la transcripción deja fuera información que cambia la estimación, añade al \
final una sección "Supuestos" con lo que has asumido.
"""


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


async def _generate_with_openai(settings: Settings, system: str, user: str) -> str:
    client = _openai_client(settings.llm_api_key.get_secret_value())
    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except openai.APIError as error:
        raise LLMServiceError(f"OpenAI request failed: {error}") from error
    return response.choices[0].message.content or ""


async def _generate_with_anthropic(settings: Settings, system: str, user: str) -> str:
    client = _anthropic_client(settings.llm_api_key.get_secret_value())
    try:
        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIError as error:
        raise LLMServiceError(f"Anthropic request failed: {error}") from error
    if response.stop_reason == "refusal":
        raise LLMServiceError("Anthropic declined to generate the estimation")
    return "".join(block.text for block in response.content if block.type == "text")


async def generate_estimation(transcription: str, settings: Settings) -> str:
    """Return a markdown estimation for a meeting transcription."""
    system = build_system_prompt()
    user = build_user_prompt(transcription)
    logger.debug(
        "Requesting estimation from %s (%s)", settings.llm_provider, settings.llm_model
    )
    if settings.llm_provider == "openai":
        estimation = await _generate_with_openai(settings, system, user)
    else:
        estimation = await _generate_with_anthropic(settings, system, user)
    if not estimation.strip():
        raise LLMServiceError("The model returned an empty estimation")
    return estimation.strip()

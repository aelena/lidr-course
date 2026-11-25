"""Builds the CAG prompt and calls the LLM provider selected in the settings.

The model writes the estimation as markdown. `app.services.evaluation` reads
that markdown back to total the hours and to check its structure.

`generate_estimation` answers in one piece, for the HTTP endpoint.
`stream_estimation` yields the text as the model writes it, for the chat UI.
Both build the same system prompt and report the same metrics.
"""

import time
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Literal

import anthropic
import openai
import structlog
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

from app.config import Settings, get_settings
from app.context.examples import (
    ExampleFormat,
    format_examples_for_prompt,
    select_examples,
)
from app.services.evaluation import sum_task_hours

log = structlog.get_logger()

DEFAULT_MAX_TOKENS = 4096

# none: the transcription is sent as is.
# inline_cleaning: the system prompt says how to read a noisy transcription.
# two_phase: a first LLM call turns the transcription into a requirements list.
PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]

# One turn of a conversation: {"role": "user" | "assistant", "content": "..."}.
Message = dict[str, str]
# Text deltas while the model writes; the Completion is the generator's return value.
CompletionStream = Generator[str, None, "Completion"]


# ---------------------------------------------------------------------------
# Prompt building blocks
#
# The two output prompt variants live side by side so they can be compared
# in the live session (Block 3.4) by editing the ACTIVE_OUTPUT_PROMPT
# assignment below. Uvicorn `--reload` picks up the change automatically.
# ---------------------------------------------------------------------------

ROLE = (
    "You are a senior software consultant with 15+ years of experience in project "
    "estimation. Your task is to produce a detailed software project estimation based "
    "on a meeting transcription provided by the user."
)

# Estimated billing rates of a Spanish software agency; the examples use the same ones.
RATES = (
    "Price the work at Spanish market rates, in EUR and excluding VAT: approximately "
    "55 EUR/hour for development, 45 EUR/hour for UI/UX design, 40 EUR/hour for testing "
    "and QA, and 60 EUR/hour for analysis, data engineering, data migration and "
    "third-party integrations. Provide realistic, well-justified numbers."
)

EXAMPLES_INTRO = (
    "Below are reference estimations from previous projects. Use them as a guide "
    "for structure, level of detail, and realistic pricing. Adapt the content to "
    "match the specific project described in the transcription."
)

PROMPT_OUTPUT_BASIC = "Generate an estimation for the project described in the transcription."

PROMPT_OUTPUT_STRUCTURED = """\
Generate the estimation with this exact structure:

## Project summary
[2-3 sentences describing the project scope and goals]

## Task breakdown
| Task | Hours | Cost (EUR) |
[one row per task; cost = hours * the hourly rate of the profile doing it]

## Totals
- Total hours: [number]
- Total cost: [number] EUR
- Recommended team: [composition]
- Estimated duration: [weeks]

## Risks and assumptions
- [3-5 bullet points covering technical risks, scope assumptions, and external dependencies]"""

# >>> Block 3.4 live switch: change the right-hand side to PROMPT_OUTPUT_STRUCTURED
ACTIVE_OUTPUT_PROMPT = PROMPT_OUTPUT_BASIC

INLINE_CLEANING_BLOCK = """\
The transcription you receive is from a real meeting and may contain:
- Informal small talk you must ignore
- Implicit requirements you must surface explicitly
- Contradictions where you must trust the most recent statement
- Non-technical jargon you must interpret

Extract ONLY the functional and technical requirements relevant to the estimation."""

EXTRACTION_SYSTEM_PROMPT = (
    "You are an analyst. Read the meeting transcription and produce a clean, "
    "deduplicated bullet list of functional requirements, non-functional "
    "requirements, integrations, constraints and explicit deadlines. Ignore "
    "fillers, divagations and off-topic remarks. Output Markdown only."
)


@dataclass(frozen=True)
class GenerationOptions:
    """Per-request knobs that drive prompt construction and the LLM call."""

    preprocessing: PreprocessingMode = "none"
    example_format: ExampleFormat = "markdown"
    num_examples: int = 3
    use_examples: bool = True
    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_budget: int | None = None


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int


class LLMServiceError(Exception):
    """The LLM provider failed or returned no usable estimation."""


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def build_system_prompt(
    example_format: ExampleFormat = "markdown",
    num_examples: int = 3,
    use_examples: bool = True,
    inline_cleaning: bool = False,
) -> str:
    """Assemble the system prompt with role, rates, output spec and (optionally) examples."""
    examples = (
        format_examples_for_prompt(select_examples(num_examples), example_format)
        if use_examples
        else ""
    )
    sections = (
        ROLE,
        INLINE_CLEANING_BLOCK if inline_cleaning else "",
        RATES,
        ACTIVE_OUTPUT_PROMPT,
        f"{EXAMPLES_INTRO}\n\n{examples}" if examples else "",
    )
    return "\n\n".join(filter(None, sections))


def system_prompt_for(opts: GenerationOptions) -> str:
    """The system prompt a call with these options sends."""
    return build_system_prompt(
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        inline_cleaning=opts.preprocessing == "inline_cleaning",
    )


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


@lru_cache
def _openai_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)


@lru_cache
def _anthropic_client(api_key: str) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=api_key)


async def _complete_with_openai(
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    thinking_budget: int | None,
) -> Completion:
    if thinking_budget is not None:
        log.warning("thinking_budget_ignored", provider="openai", model=model)
    try:
        response = await _openai_client(api_key).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens,
        )
    except openai.OpenAIError as error:
        raise LLMServiceError(f"OpenAI request failed: {error}") from error
    choice, usage = response.choices[0], response.usage
    return Completion(
        text=choice.message.content or "",
        finish_reason=choice.finish_reason,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


async def _complete_with_anthropic(
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    thinking_budget: int | None,
) -> Completion:
    thinking = _anthropic_thinking(thinking_budget, max_tokens)
    try:
        response = await _anthropic_client(api_key).messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **thinking,
        )
    except anthropic.AnthropicError as error:
        raise LLMServiceError(f"Anthropic request failed: {error}") from error
    return _anthropic_completion(response)


def _anthropic_thinking(thinking_budget: int | None, max_tokens: int) -> dict[str, Any]:
    if thinking_budget is None:
        return {}
    if thinking_budget >= max_tokens:
        raise LLMServiceError("thinking_budget must be lower than max_tokens")
    return {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}


def _anthropic_completion(response: anthropic.types.Message) -> Completion:
    if response.stop_reason == "refusal":
        raise LLMServiceError("Anthropic declined to generate the estimation")
    return Completion(
        text="".join(block.text for block in response.content if block.type == "text"),
        finish_reason=response.stop_reason or "unknown",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


PROVIDERS = {"openai": _complete_with_openai, "anthropic": _complete_with_anthropic}


async def _complete(
    settings: Settings,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    thinking_budget: int | None = None,
) -> Completion:
    """Send one system + user exchange to the configured provider."""
    return await PROVIDERS[settings.llm_provider](
        settings.llm_api_key.get_secret_value(),
        model,
        system,
        user,
        max_tokens,
        thinking_budget,
    )


# ---------------------------------------------------------------------------
# Streaming provider calls
#
# Synchronous clients: Streamlit runs the script in a plain thread and
# `st.write_stream` consumes an ordinary generator.
# ---------------------------------------------------------------------------


@lru_cache
def _openai_stream_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


@lru_cache
def _anthropic_stream_client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)


def _stream_with_openai(
    api_key: str,
    model: str,
    system: str,
    messages: Sequence[Message],
    max_tokens: int,
    thinking_budget: int | None,
) -> CompletionStream:
    if thinking_budget is not None:
        log.warning("thinking_budget_ignored", provider="openai", model=model)
    parts: list[str] = []
    finish_reason, usage = "unknown", None
    try:
        chunks = _openai_stream_client(api_key).chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            max_completion_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in chunks:
            usage = chunk.usage or usage  # arrives in a last chunk without choices
            for choice in chunk.choices:
                finish_reason = choice.finish_reason or finish_reason
                if choice.delta.content:
                    parts.append(choice.delta.content)
                    yield choice.delta.content
    except openai.OpenAIError as error:
        raise LLMServiceError(f"OpenAI request failed: {error}") from error
    return Completion(
        text="".join(parts),
        finish_reason=finish_reason,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


def _stream_with_anthropic(
    api_key: str,
    model: str,
    system: str,
    messages: Sequence[Message],
    max_tokens: int,
    thinking_budget: int | None,
) -> CompletionStream:
    thinking = _anthropic_thinking(thinking_budget, max_tokens)
    try:
        with _anthropic_stream_client(api_key).messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=list(messages),
            **thinking,
        ) as stream:
            yield from stream.text_stream
            response = stream.get_final_message()
    except anthropic.AnthropicError as error:
        raise LLMServiceError(f"Anthropic request failed: {error}") from error
    return _anthropic_completion(response)


STREAM_PROVIDERS = {"openai": _stream_with_openai, "anthropic": _stream_with_anthropic}


def _stream(
    settings: Settings,
    model: str,
    system: str,
    messages: Sequence[Message],
    *,
    max_tokens: int,
    thinking_budget: int | None = None,
) -> CompletionStream:
    """Stream one system + conversation exchange from the configured provider."""
    return STREAM_PROVIDERS[settings.llm_provider](
        settings.llm_api_key.get_secret_value(),
        model,
        system,
        messages,
        max_tokens,
        thinking_budget,
    )


def _drain(stream: CompletionStream) -> Completion:
    """Run a stream to its end, discarding the deltas."""
    while True:
        try:
            next(stream)
        except StopIteration as done:
            return done.value


# ---------------------------------------------------------------------------
# Main entrypoints
# ---------------------------------------------------------------------------


def _log_start(settings: Settings, model: str, opts: GenerationOptions, **extra: Any) -> None:
    log.info(
        "generating_estimation",
        provider=settings.llm_provider,
        model=model,
        preprocessing=opts.preprocessing,
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        max_tokens=opts.max_tokens,
        thinking_budget=opts.thinking_budget,
        **extra,
    )


def _requirements(extraction: Completion | None) -> str | None:
    """The requirements list of a two-phase extraction, if there was one."""
    if extraction and not extraction.text.strip():
        raise LLMServiceError("The model extracted no requirements")
    return extraction.text.strip() if extraction else None


def _build_result(
    settings: Settings,
    model: str,
    extraction: Completion | None,
    completion: Completion,
    t0: float,
    **extra: Any,
) -> dict[str, Any]:
    estimation = completion.text.strip()
    if not estimation:
        raise LLMServiceError("The model returned an empty estimation")
    calls = [call for call in (extraction, completion) if call]
    result = {
        "estimation": estimation,
        "model": model,
        "provider": settings.llm_provider,
        "finish_reason": completion.finish_reason,
        "total_hours": sum_task_hours(estimation),
        "input_tokens": sum(call.input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "extracted_requirements": _requirements(extraction),
        **extra,
    }
    log.info(
        "estimation_generated",
        **{key: result[key] for key in result if key not in ("estimation", "extracted_requirements")},
    )
    return result


async def generate_estimation(
    transcription: str, opts: GenerationOptions | None = None
) -> dict[str, Any]:
    """Generate a software estimation from a meeting transcription using the configured LLM."""
    opts = opts or GenerationOptions()
    settings = get_settings()
    model = opts.model or settings.llm_model
    complete = partial(_complete, settings, model, max_tokens=opts.max_tokens)
    t0 = time.perf_counter()

    _log_start(settings, model, opts)

    extraction = (
        await complete(EXTRACTION_SYSTEM_PROMPT, transcription)
        if opts.preprocessing == "two_phase"
        else None
    )
    completion = await complete(
        system_prompt_for(opts),
        _requirements(extraction) or transcription,
        thinking_budget=opts.thinking_budget,
    )
    return _build_result(settings, model, extraction, completion, t0)


class EstimationStream:
    """The text deltas of an estimation, as the model writes them.

    Once iterated to the end, `result` holds the same dict that
    `generate_estimation` returns, plus `first_token_ms`.
    """

    def __init__(self, deltas: Generator[str, None, dict[str, Any]]) -> None:
        self._deltas = deltas
        self.result: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[str]:
        self.result = yield from self._deltas


def stream_estimation(
    messages: Sequence[Message], opts: GenerationOptions | None = None
) -> EstimationStream:
    """Stream the answer to a conversation whose last message is the user's.

    Earlier turns travel after the system prompt, so the user can follow up on
    an estimation. Nothing is sent to the provider until the stream is iterated.
    """
    return EstimationStream(_estimation_deltas(list(messages), opts or GenerationOptions()))


def _estimation_deltas(
    messages: list[Message], opts: GenerationOptions
) -> Generator[str, None, dict[str, Any]]:
    settings = get_settings()
    model = opts.model or settings.llm_model
    stream = partial(_stream, settings, model, max_tokens=opts.max_tokens)
    t0 = time.perf_counter()
    _log_start(settings, model, opts, turns=len(messages), streaming=True)

    extraction = (
        _drain(stream(EXTRACTION_SYSTEM_PROMPT, messages[-1:]))
        if opts.preprocessing == "two_phase"
        else None
    )
    if requirements := _requirements(extraction):
        messages = [*messages[:-1], {**messages[-1], "content": requirements}]

    first_token_ms = None
    deltas = stream(system_prompt_for(opts), messages, thinking_budget=opts.thinking_budget)
    while True:
        try:
            delta = next(deltas)
        except StopIteration as done:
            completion = done.value
            break
        if first_token_ms is None:
            first_token_ms = round((time.perf_counter() - t0) * 1000)
        yield delta
    return _build_result(
        settings, model, extraction, completion, t0, first_token_ms=first_token_ms
    )

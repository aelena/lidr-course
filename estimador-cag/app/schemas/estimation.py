from pydantic import BaseModel, Field, field_validator

from app.context.examples import ExampleFormat
from app.services.llm_service import DEFAULT_MAX_TOKENS, PreprocessingMode


class EstimationRequest(BaseModel):
    transcription: str = Field(
        min_length=1, description="Transcription of the meeting with the client."
    )
    preprocessing: PreprocessingMode = Field(
        default="none",
        description=(
            "How a noisy transcription is handled: 'none' sends it as is, "
            "'inline_cleaning' adds cleaning instructions to the system prompt, "
            "'two_phase' makes a first LLM call that turns it into a requirements list."
        ),
    )
    example_format: ExampleFormat = Field(
        default="markdown",
        description="How the past estimations are written in the prompt.",
    )
    num_examples: int = Field(
        default=3, ge=0, description="Maximum number of past estimations to inject."
    )
    use_examples: bool = Field(
        default=True,
        description="Inject past estimations. Set to false to compare against no context.",
    )
    model: str | None = Field(
        default=None, description="Model to use instead of LLM_MODEL."
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        description="Maximum tokens the model may generate.",
    )
    thinking_budget: int | None = Field(
        default=None,
        ge=1024,
        description=(
            "Thinking tokens for Anthropic models that accept a budget, such as "
            "claude-haiku-4-5. Ignored by OpenAI."
        ),
    )
    evaluate: bool = Field(
        default=False,
        description="Include a structural validation of the estimation in the response.",
    )

    @field_validator("transcription")
    @classmethod
    def transcription_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcription must not be blank")
        return value


class EstimationValidation(BaseModel):
    is_valid: bool
    task_count: int
    checks: dict[str, bool]
    issues: list[str] = Field(description="Names of the checks that failed.")


class EstimationResponse(BaseModel):
    estimation: str = Field(description="Estimation in markdown, as the model wrote it.")
    model: str
    provider: str
    finish_reason: str = Field(description="Why the model stopped generating.")
    total_hours: float | None = Field(
        description=(
            "Hours of the task breakdown added up in code; null when no task "
            "could be read from the estimation."
        )
    )
    input_tokens: int
    output_tokens: int
    latency_ms: int = Field(description="Time spent calling the model.")
    extracted_requirements: str | None = Field(
        default=None,
        description="Requirements list from the first call of 'two_phase' preprocessing.",
    )
    validation: EstimationValidation | None = None

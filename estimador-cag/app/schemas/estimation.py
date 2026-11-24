from pydantic import BaseModel, Field, field_validator

from app.context.examples import ExampleFormat


class EstimationRequest(BaseModel):
    transcription: str = Field(
        min_length=1, description="Transcription of the meeting with the client."
    )
    preprocessing: bool = Field(
        default=True,
        description="Normalise the whitespace of the transcription before sending it.",
    )
    example_format: ExampleFormat = Field(
        default="markdown",
        description="How the past estimations are written in the prompt.",
    )
    num_examples: int | None = Field(
        default=None, ge=1, description="Past estimations to inject; all when omitted."
    )
    use_examples: bool = Field(
        default=True,
        description="Inject past estimations. Set to false to compare against no context.",
    )
    model: str | None = Field(
        default=None, description="Model to use instead of LLM_MODEL."
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens the model may generate."
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
    estimation: str = Field(description="Estimation in markdown.")
    model: str
    provider: str
    finish_reason: str = Field(description="Why the model stopped generating.")
    total_hours: int = Field(description="Sum of the hours of every task.")
    input_tokens: int
    output_tokens: int
    validation: EstimationValidation | None = None

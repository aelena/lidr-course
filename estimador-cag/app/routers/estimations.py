from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.services import llm_service

router = APIRouter(tags=["estimations"])


class EstimationRequest(BaseModel):
    transcription: str = Field(
        min_length=1, description="Transcription of the meeting with the client."
    )


class EstimationResponse(BaseModel):
    estimation: str = Field(description="Estimation in markdown.")
    model: str
    provider: str
    total_hours: int = Field(description="Sum of the hours of every task.")
    input_tokens: int
    output_tokens: int


@router.post("/estimate", response_model=EstimationResponse)
async def create_estimation(
    request: EstimationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EstimationResponse:
    if not request.transcription.strip():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "transcription must not be blank"
        )
    try:
        result = await llm_service.generate_estimation(request.transcription, settings)
    except llm_service.LLMServiceError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    return EstimationResponse(
        estimation=result.estimation,
        model=settings.llm_model,
        provider=settings.llm_provider,
        total_hours=result.total_hours,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.services import llm_service

router = APIRouter(prefix="/estimations", tags=["estimations"])


class EstimationRequest(BaseModel):
    transcription: str = Field(
        min_length=1, description="Transcription of the meeting with the client."
    )


class EstimationResponse(BaseModel):
    estimation: str
    provider: str
    model: str


@router.post("", response_model=EstimationResponse)
async def create_estimation(
    request: EstimationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EstimationResponse:
    if not request.transcription.strip():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "transcription must not be blank"
        )
    try:
        estimation = await llm_service.generate_estimation(
            request.transcription, settings
        )
    except llm_service.LLMServiceError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    return EstimationResponse(
        estimation=estimation,
        provider=settings.llm_provider,
        model=settings.llm_model,
    )

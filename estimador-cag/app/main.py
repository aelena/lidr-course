from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.config import configure_logging, get_settings
from app.routers import estimations


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()
    log = structlog.get_logger()
    settings = get_settings()
    log.info(
        "application_started",
        environment=settings.app_env,
        provider=settings.llm_provider,
        model=settings.llm_model,
    )
    yield
    log.info("application_shutdown")


app = FastAPI(
    title="Software Estimation CAG Service",
    description="AI-powered software estimation service using Cache Augmented Generation architecture",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.include_router(estimations.router)


@app.get("/health", tags=["health"], summary="Service status")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.app_env,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }

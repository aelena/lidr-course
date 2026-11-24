import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.routers import estimations

DESCRIPTION = """\
Genera estimaciones de proyectos de software a partir de la transcripción de \
una reunión con el cliente.

Usa CAG (Context-Augmented Generation): las estimaciones anteriores viajan \
en el prompt de cada llamada al LLM como referencia.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create the LLM client and other shared resources here.
    yield
    # Shutdown: release them here.


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    app = FastAPI(
        title="Estimador CAG",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(estimations.router, prefix="/api/v1")

    @app.get("/health", tags=["health"], summary="Estado del servicio")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "environment": settings.app_env,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }

    return app


app = create_app()

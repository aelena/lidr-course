import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.routers import estimations


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create the LLM client and other shared resources here.
    yield
    # Shutdown: release them here.


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    app = FastAPI(title="estimador-cag", lifespan=lifespan)
    app.include_router(estimations.router)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

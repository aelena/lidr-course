from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import get_settings
from routers import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create the LLM client and other shared resources here.
    yield
    # Shutdown: release them here.


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    app.include_router(health.router)
    return app


app = create_app()

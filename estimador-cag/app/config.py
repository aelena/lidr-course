import logging
from functools import lru_cache
from typing import Literal, Self

import structlog
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, read from the environment and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    app_env: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    @model_validator(mode="after")
    def require_key_for_selected_provider(self) -> Self:
        key = self.llm_api_key
        if key is None or not key.get_secret_value().strip():
            raise ValueError(
                f"{self.llm_provider.upper()}_API_KEY is required when "
                f"LLM_PROVIDER={self.llm_provider}"
            )
        return self

    @property
    def llm_api_key(self) -> SecretStr | None:
        """API key of the provider selected in LLM_PROVIDER."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.anthropic_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_logging() -> None:
    """Set up structlog: JSON in production, human-readable in development."""
    settings = get_settings()
    level = logging.getLevelNamesMapping()[settings.log_level]

    if settings.app_env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        # Drops events below LOG_LEVEL before any processor runs.
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # Third-party libraries (uvicorn, httpx) still log through the standard library.
    logging.basicConfig(level=level)

"""
Application configuration using pydantic-settings.

Loads settings from environment variables or .env file.
Never stores real secrets in code — all sensitive values come from environment.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    app_debug: bool = Field(default=False, description="Enable debug mode")
    app_secret_key: str = Field(default="change-me", description="Secret key for signing")
    app_host: str = Field(default="0.0.0.0", description="Server host")
    app_port: int = Field(default=8000, description="Server port")

    # Database
    database_url: str = Field(default="sqlite:///./ciis_ai.db", description="Database connection URL")

    # CIIS Integration
    ciis_base_url: str = Field(default="http://localhost:8080", description="CIIS application URL")
    ciis_service_token: str = Field(default="", description="Service-to-service auth token")

    # Model Configuration
    model_dir: str = Field(default="./models", description="Directory for model artifacts")
    model_min_samples: int = Field(default=50, description="Minimum samples for training")
    model_acceptance_f1: float = Field(default=0.5, description="Minimum F1 score for model activation")
    model_random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # Security
    api_key: str = Field(default="", description="API key for service authentication")
    cors_origins: str = Field(default="http://localhost:8080", description="Allowed CORS origins")
    max_request_timeout: int = Field(default=30, description="Max request timeout in seconds")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="./logs/ciis_ai.log", description="Log file path")
    log_sensitive_data: bool = Field(default=False, description="Never enable in production")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @property
    def model_path(self) -> Path:
        """Get the model directory as a Path, creating it if needed."""
        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_path(self) -> Path:
        """Get the log directory as a Path, creating it if needed."""
        path = Path(self.log_file).parent
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()

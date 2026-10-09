"""Application configuration and environment settings."""

import os
from pathlib import Path
from pydantic import BaseModel

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Production settings with environment variable support."""

        PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
        DATA_DIR: Path = PROJECT_ROOT / "data"
        DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
        CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
        EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

        OLLAMA_BASE_URL: str = "http://localhost:11434"
        DEFAULT_MODEL: str = "qwen2.5:0.5b"
        FALLBACK_MODEL: str = "smollm2:360m"
        OLLAMA_TIMEOUT_SECONDS: float = 60.0

        APP_HOST: str = "0.0.0.0"
        APP_PORT: int = 8000
        CLOUDFLARE_TUNNEL_PORT: int = 8000

        LOG_LEVEL: str = "INFO"

        ANCHOR_DATA_RATIO: float = 0.70
        NEW_WEB_DATA_RATIO: float = 0.30

        MIN_CITATION_SCORE: float = 0.85
        MAX_REGRESSION_DELTA: float = 0.05

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

except ImportError:
    # Graceful fallback when pydantic-settings is not yet installed in local environment
    class Settings(BaseModel):  # type: ignore[no-redef]
        """Fallback settings reading directly from os.getenv."""

        PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
        DATA_DIR: Path = PROJECT_ROOT / "data"
        DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
        CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
        EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen2.5:0.5b")
        FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "smollm2:360m")
        OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60.0"))

        APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
        APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
        CLOUDFLARE_TUNNEL_PORT: int = int(os.getenv("CLOUDFLARE_TUNNEL_PORT", "8000"))

        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        ANCHOR_DATA_RATIO: float = float(os.getenv("ANCHOR_DATA_RATIO", "0.70"))
        NEW_WEB_DATA_RATIO: float = float(os.getenv("NEW_WEB_DATA_RATIO", "0.30"))

        MIN_CITATION_SCORE: float = float(os.getenv("MIN_CITATION_SCORE", "0.85"))
        MAX_REGRESSION_DELTA: float = float(os.getenv("MAX_REGRESSION_DELTA", "0.05"))


settings = Settings()

# Ensure directories exist
for path in [
    settings.DATA_DIR,
    settings.DATASETS_DIR,
    settings.CHECKPOINTS_DIR,
    settings.EXPERIMENTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

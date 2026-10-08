"""Configuration settings using pydantic-settings."""


from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "dev"
    INTERNAL_API_KEY: str | None = None
    MAX_FILE_SIZE_MB: int = 5
    MAX_PDF_PAGES: int = 15
    MAX_TEXT_CHARS: int = 60000

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    EMBED_BATCH_SIZE: int = 32
    EMBED_CACHE_SIZE: int = 5000

    CHROMA_MODE: str = "persistent"  # persistent or http
    CHROMA_PATH: str = "./data/chroma"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "swipex_jobs"

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    LLM_TIMEOUT_SECONDS: int = 8
    LLM_MAX_RETRIES: int = 2
    LLM_CACHE_TTL_SECONDS: int = 86400

    ATS_KEYWORD_WEIGHT: float = 0.6
    ATS_SEMANTIC_WEIGHT: float = 0.4
    ATS_THRESHOLD: float = 80.0
    SEM_SIM_FLOOR: float = 0.20
    SEM_SIM_CEIL: float = 0.75

    REC_W_SEMANTIC: float = 0.45
    REC_W_SKILL: float = 0.25
    REC_W_SWIPE: float = 0.15
    REC_W_LOCATION_EXP: float = 0.10
    REC_W_FRESH_COMP: float = 0.05

    MAX_CONCURRENT_AI_TASKS: int = 4
    HIGH_MATCH_DEFAULT_THRESHOLD: float = 80.0
    INDEX_BATCH_SIZE: int = 64          # jobs per embedding batch during bulk index
    ALERT_HIGH_MATCH_THRESHOLD: float = 88.0   # score to trigger high-match alert

    @property
    def LLM_ENABLED(self) -> bool:
        """Derived setting: True only if a non-empty Gemini API key is configured."""
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip())

    @model_validator(mode="after")
    def validate_weights_and_bounds(self) -> "Settings":
        total = (
            self.REC_W_SEMANTIC
            + self.REC_W_SKILL
            + self.REC_W_SWIPE
            + self.REC_W_LOCATION_EXP
            + self.REC_W_FRESH_COMP
        )
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Recommender weights must sum to 1.0 (current sum: {total:.4f})")

        ats_sum = self.ATS_KEYWORD_WEIGHT + self.ATS_SEMANTIC_WEIGHT
        if abs(ats_sum - 1.0) > 1e-5:
            raise ValueError(f"ATS weights must sum to 1.0 (current sum: {ats_sum:.4f})")

        return self


settings = Settings()

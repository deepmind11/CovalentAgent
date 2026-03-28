"""Configuration and settings for CovalentAgent."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    hf_token: str = Field(default="", description="HuggingFace token for ESM-2")
    pubmed_api_key: str = Field(default="", description="PubMed API key")

    chroma_persist_dir: Path = Field(
        default=Path("./chroma_data"), description="ChromaDB persistence directory"
    )
    esm_model: str = Field(
        default="facebook/esm2_t33_650M_UR50D", description="ESM-2 model identifier"
    )
    chemprop_model_dir: Path = Field(
        default=Path("./models/cached"), description="Chemprop cached models directory"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

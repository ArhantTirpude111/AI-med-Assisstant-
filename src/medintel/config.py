"""Typed, layered application configuration.

Settings are composed from three layers, **lowest to highest** precedence:

1. ``config/config.yaml``         — non-secret defaults committed to the repo
2. environment variables          — ``MEDINTEL_*`` (nested keys via ``__``)
3. ``.env`` file                  — secrets, never committed

Usage::

    from medintel.config import get_settings
    settings = get_settings()
    settings.retrieval.top_k

``get_settings`` is cached so the (potentially file-reading) construction happens
once per process.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Repository root = three levels up from this file (src/medintel/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Active YAML path, consulted by the (lowest-priority) YAML settings source.
_ACTIVE_CONFIG_PATH: Path = DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# Sub-configuration sections (plain models, populated from the merged dict)
# ---------------------------------------------------------------------------
class LLMConfig(BaseModel):
    provider: str = "openai"  # openai | local | mock
    model: str = "gpt-4o-mini"
    # Local (free, offline) HuggingFace causal-LM used when provider == "local".
    local_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Optional OpenAI-compatible base URL (e.g. Groq/OpenRouter free tiers).
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: int = 60
    max_retries: int = 3


class EmbeddingsConfig(BaseModel):
    provider: str = "openai"  # openai | huggingface | hash
    model: str = "text-embedding-3-small"
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 1536
    batch_size: int = 64


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 900
    chunk_overlap: int = 150
    respect_page_boundaries: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 5
    fetch_k: int = 20
    min_confidence: float = 0.35
    max_context_chars: int = 8000
    dedupe: bool = True


class OCRConfig(BaseModel):
    enabled: bool = True
    language: str = "eng"
    dpi: int = 300
    tesseract_cmd: str | None = None
    min_native_chars_per_page: int = 40


class NERConfig(BaseModel):
    enabled: bool = True
    backend: str = "auto"  # auto | scispacy | rules
    scispacy_model: str = "en_core_sci_sm"


class GuardrailsConfig(BaseModel):
    attach_disclaimer: bool = True
    detect_emergencies: bool = True
    refuse_below_confidence: bool = True


class PathsConfig(BaseModel):
    data_dir: str = "data"
    upload_dir: str = "data/uploads"
    index_dir: str = "data/index"
    db_url: str = "sqlite:///data/medintel.db"

    def resolve(self, root: Path) -> None:
        """Make relative directory paths absolute and ensure they exist."""
        for attr in ("data_dir", "upload_dir", "index_dir"):
            p = Path(getattr(self, attr))
            if not p.is_absolute():
                p = root / p
            p.mkdir(parents=True, exist_ok=True)
            setattr(self, attr, str(p))


# ---------------------------------------------------------------------------
# Root settings object
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Root configuration aggregating all sections."""

    model_config = SettingsConfigDict(
        env_prefix="MEDINTEL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "MedIntel AI"
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = False
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    ner: NERConfig = Field(default_factory=NERConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority high -> low: init kwargs > env > .env > YAML defaults > secrets.
        # `_YamlSource` is defined later in this module; resolved at call time.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
            file_secret_settings,
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class _YamlSource(PydanticBaseSettingsSource):
    """Lowest-priority settings source backed by ``config/config.yaml``.

    Placed *below* env and dotenv sources so that ``MEDINTEL_*`` variables and
    ``.env`` correctly override the YAML defaults (per the documented precedence).
    pydantic-settings deep-merges nested dicts across sources, so overriding a
    single nested key (e.g. ``paths.index_dir``) leaves sibling YAML keys intact.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = _load_yaml(_ACTIVE_CONFIG_PATH)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: D102
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


@functools.lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> Settings:
    """Build (and cache) the merged ``Settings`` instance.

    Precedence (low → high): ``config/config.yaml`` < env vars < ``.env``.
    The YAML defaults are supplied by :class:`_YamlSource` at the lowest priority,
    so environment variables and ``.env`` always win.
    """
    global _ACTIVE_CONFIG_PATH
    if config_path:
        _ACTIVE_CONFIG_PATH = Path(config_path)
    settings = Settings()
    settings.paths.resolve(PROJECT_ROOT)
    return settings


def reload_settings() -> Settings:
    """Clear the cache and rebuild (useful in tests / after config edits)."""
    get_settings.cache_clear()
    return get_settings()

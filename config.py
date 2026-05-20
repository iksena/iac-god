# config.py
import os
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


def _parse_csv_env(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    parts = [part.strip() for part in value.split(",")]
    return tuple(part for part in parts if part)


OPENROUTER_QUANTIZATION_ORDER: tuple[str, ...] = (
    "int4",
    "int8",
    "fp4",
    "fp6",
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
)


def quantizations_from_min(min_quantization: str | None) -> tuple[str, ...]:
    if not min_quantization:
        return ()

    q = min_quantization.strip().lower()
    if not q:
        return ()

    if q not in OPENROUTER_QUANTIZATION_ORDER:
        supported = ", ".join(OPENROUTER_QUANTIZATION_ORDER)
        raise ValueError(
            f"Unsupported min quantization '{q}'. Supported values: {supported}"
        )

    start = OPENROUTER_QUANTIZATION_ORDER.index(q)
    return OPENROUTER_QUANTIZATION_ORDER[start:]


# ---------------------------------------------------------------------------
# o-series / reasoning model detection
# ---------------------------------------------------------------------------
# OpenAI o1, o3, o4, codex-mini, and similar reasoning models have two
# API differences from standard chat-completion models:
#   1. They use `max_completion_tokens` instead of `max_tokens`.
#   2. They do not accept a `temperature` parameter (fixed at 1).
# This helper centralises that detection so llm_client.py can branch cleanly.

_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "codex")


def is_openai_reasoning_model(model: str) -> bool:
    """Return True when *model* is an OpenAI reasoning (o-series / codex) model.

    Matches: o1, o1-mini, o1-preview, o3, o3-mini, o4, o4-mini,
             codex-mini-latest, codex, o3-mini-high, etc.
    Does NOT match: gpt-4o, gpt-3.5-turbo, text-davinci-*, etc.
    """
    name = (model or "").strip().lower()
    # Strip optional "openai/" namespace prefix (in case someone copies an
    # OpenRouter-style slug by mistake).
    if name.startswith("openai/"):
        name = name[len("openai/"):]
    return any(
        name == p or name.startswith(p + "-") or name.startswith(p + "_")
        for p in _OPENAI_REASONING_PREFIXES
    )


class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    CLAUDE = "claude"
    OPENAI = "openai"


class DeployTarget(Enum):
    NONE = "none"
    LOCALSTACK = "localstack"
    AWS = "aws"


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.OPENROUTER
    # provider: LLMProvider = LLMProvider.OPENAI
    model: str = "x-ai/grok-4.1-fast"
    temperature: float = 0.0
    max_tokens: int = 8192
    reasoning_enabled: bool = True

    # OpenRouter
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_provider_only: tuple[str, ...] = field(
        default_factory=lambda: _parse_csv_env(os.getenv("OPENROUTER_PROVIDER_ONLY"))
    )
    openrouter_min_quantization: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MIN_QUANTIZATION", "").strip().lower()
    )

    # Anthropic direct
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    # OpenAI direct
    # Activate by setting LLM_PROVIDER=openai in .env (or building LLMConfig
    # with provider=LLMProvider.OPENAI explicitly in your entry-point).
    #
    # Supported models include: gpt-4o, gpt-4o-mini, o3-mini, o3, o4-mini,
    # codex-mini-latest, etc.  o-series / codex models are handled
    # automatically: temperature is omitted and max_completion_tokens is used
    # instead of max_tokens (see is_openai_reasoning_model).
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    # Optional base_url override - useful for Azure OpenAI or a local proxy.
    # Leave empty to use the default api.openai.com endpoint.
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "")
    )


@dataclass
class DeployConfig:
    target: DeployTarget = DeployTarget.NONE
    localstack_endpoint: str = field(
        default_factory=lambda: os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
    )
    localstack_reset_wait: float = 5
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    aws_profile: str = field(
        default_factory=lambda: os.getenv("AWS_PROFILE", "default")
    )
    stack_creation_timeout: int = 15 * 60
    stack_deletion_timeout: int = 1 * 60


# ---------------------------------------------------------------------------
# Staged remediation threshold
# ---------------------------------------------------------------------------
# When stage_error_counts[stage] reaches this value the stage escalates from
# simple mode (engineer self-corrects directly) to moderate mode (full
# retriever -> remediator -> engineer pipeline).
SIMPLE_MODE_THRESHOLD: int = 0


def build_openrouter_provider_preferences(config: LLMConfig) -> dict:
    provider: dict[str, object] = {}

    if config.openrouter_provider_only:
        provider["only"] = list(config.openrouter_provider_only)

    quantizations = quantizations_from_min(config.openrouter_min_quantization)
    if quantizations:
        provider["quantizations"] = list(quantizations)

    return provider


DEFAULT_CONFIG = LLMConfig()
DEFAULT_DEPLOY_CONFIG = DeployConfig()

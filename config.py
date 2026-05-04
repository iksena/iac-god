# config.py
import os
from dataclasses import dataclass
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


class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    CLAUDE = "claude"


class DeployTarget(Enum):
    NONE = "none"
    LOCALSTACK = "localstack"
    AWS = "aws"


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.OPENROUTER
    model: str = "x-ai/grok-4.1-fast"
    temperature: float = 0.0
    max_tokens: int = 8192
    reasoning_enabled: bool = True

    # OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_provider_only: tuple[str, ...] = _parse_csv_env(
        os.getenv("OPENROUTER_PROVIDER_ONLY")
    )
    openrouter_min_quantization: str = os.getenv("OPENROUTER_MIN_QUANTIZATION", "").strip().lower()

    # Anthropic direct
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class DeployConfig:
    target: DeployTarget = DeployTarget.NONE
    localstack_endpoint: str = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
    localstack_reset_wait: float = 5
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_profile: str = os.getenv("AWS_PROFILE", "default")
    stack_creation_timeout: int = 15 * 60
    stack_deletion_timeout: int = 5 * 60


# ---------------------------------------------------------------------------
# Staged remediation threshold
# ---------------------------------------------------------------------------
# When stage_error_counts[stage] reaches this value the stage escalates from
# simple mode (engineer self-corrects directly) to moderate mode (full
# retriever → remediator → engineer pipeline).
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

# config.py
import os
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

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
    model: str = "arcee-ai/trinity-large-preview:free"   # or "claude-3-5-sonnet-20241022"
    temperature: float = 0.1
    max_tokens: int = 8192

    # OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Anthropic direct
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

@dataclass
class DeployConfig:
    target: DeployTarget = DeployTarget.NONE
    # LocalStack settings
    localstack_endpoint: str = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
    localstack_reset_wait: float = 1.5      # Seconds to wait after state reset
    # AWS settings (only used when target=AWS)
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_profile: str = os.getenv("AWS_PROFILE", "default")
    # Shared
    stack_creation_timeout: int = 300       # Seconds before giving up on stack creation
    stack_deletion_timeout: int = 120       # Seconds to wait for cleanup after each iteration

DEFAULT_CONFIG = LLMConfig()
DEFAULT_DEPLOY_CONFIG = DeployConfig()
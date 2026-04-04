# config.py
import os
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    CLAUDE = "claude"

@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.OPENROUTER
    model: str = "arcee-ai/trinity-large-preview:free"   # or "claude-3-5-sonnet-20241022"
    temperature: float = 0.2
    max_tokens: int = 4096

    # OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Anthropic direct
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

DEFAULT_CONFIG = LLMConfig()
"""
LLM generation layer: wraps OpenAI / Azure OpenAI with usage tracking.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float = field(init=False)

    # GPT-4o pricing as of mid-2024 (update as needed)
    _COST_PER_1K = {"prompt": 0.005, "completion": 0.015}

    def __post_init__(self):
        self.estimated_cost_usd = (
            self.prompt_tokens / 1000 * self._COST_PER_1K["prompt"]
            + self.completion_tokens / 1000 * self._COST_PER_1K["completion"]
        )


class LLMGenerator:
    """
    Generates answers from prompt messages.
    Supports OpenAI and Azure OpenAI with usage telemetry.
    """

    def __init__(self, use_azure: bool | None = None):
        self.use_azure = use_azure if use_azure is not None else bool(os.getenv("AZURE_OPENAI_API_KEY"))
        self._llm = self._build_llm()

    def _build_llm(self):
        temperature = float(os.getenv("TEMPERATURE", 0.1))
        max_tokens = int(os.getenv("MAX_TOKENS", 1024))

        if self.use_azure:
            logger.info("Using Azure OpenAI")
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info("Using OpenAI model: %s", model)
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(self, messages: list[BaseMessage]) -> GenerationResult:
        """Generate an answer and return it with full usage metadata."""
        t0 = time.perf_counter()
        response = self._llm.invoke(messages)
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = response.response_metadata.get("token_usage", {})
        result = GenerationResult(
            answer=response.content,
            model=response.response_metadata.get("model_name", "unknown"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
        )

        logger.info(
            "Generated answer in %.0fms | tokens: %d prompt + %d completion | cost: $%.5f",
            latency_ms,
            result.prompt_tokens,
            result.completion_tokens,
            result.estimated_cost_usd,
        )
        return result

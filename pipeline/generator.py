from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

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

    _COST_PER_1K = {"prompt": 0.005, "completion": 0.015}

    def __post_init__(self):
        self.estimated_cost_usd = (
            self.prompt_tokens / 1000 * self._COST_PER_1K["prompt"]
            + self.completion_tokens / 1000 * self._COST_PER_1K["completion"]
        )


class MockLLM:
    def invoke(self, messages):
        system_content = ""
        user_content = ""
        for msg in messages:
            role = getattr(msg, "type", "")
            if role == "system":
                system_content = msg.content
            elif role == "human":
                user_content = msg.content
        chunk_count = system_content.count("[Source:")
        answer = (
            f"[MOCK MODE]\n\n"
            f"Question: {user_content}\n\n"
            f"Pipeline working — retrieved {chunk_count} chunk(s) from your documents.\n\n"
            f"Add OPENAI_API_KEY and set MOCK_LLM=false for real GPT-4 answers."
        )
        return MockResponse(answer)


class MockResponse:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {
            "model_name": "mock",
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }


class LLMGenerator:
    def __init__(self, use_azure=None):
        self.mock_mode = os.getenv("MOCK_LLM", "false").lower() == "true"
        if self.mock_mode:
            logger.info("MOCK MODE enabled")
            self._llm = MockLLM()
        else:
            from langchain_openai import AzureChatOpenAI, ChatOpenAI
            self.use_azure = use_azure if use_azure is not None else bool(os.getenv("AZURE_OPENAI_API_KEY"))
            temperature = float(os.getenv("TEMPERATURE", 0.1))
            max_tokens = int(os.getenv("MAX_TOKENS", 1024))
            if self.use_azure:
                self._llm = AzureChatOpenAI(
                    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                self._llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

    def generate(self, messages: list[BaseMessage]) -> GenerationResult:
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
        logger.info("Generated in %.0fms | tokens: %d | cost: $%.5f",
                    latency_ms, result.total_tokens, result.estimated_cost_usd)
        return result
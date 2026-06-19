from __future__ import annotations

import os
from dataclasses import dataclass

from configs.config import cfg


@dataclass
class LLMResponse:
    content: str


class OfflineLLM:
    def invoke(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            "当前未启用在线 LLM，已使用本地规则生成分析。"
            "如需更强的综合能力，请配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。"
        )


def _valid_key(value: str | None) -> bool:
    return bool(value and not value.startswith("your-") and value != "sk-")


def get_llm(
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
):
    provider = provider or cfg["MODEL"].get("PROVIDER", "offline")
    temperature = (
        temperature
        if temperature is not None
        else cfg["MODEL"].get("TEMPERATURE", 0.3)
    )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return OfflineLLM()

    if provider == "deepseek":
        resolved_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or cfg["MODEL"]["DEEPSEEK"].get("API_KEY")
        )
        if not _valid_key(resolved_key):
            return OfflineLLM()

        return ChatOpenAI(
            model=model_name or cfg["MODEL"]["DEEPSEEK"]["MODEL_NAME"],
            api_key=resolved_key,
            base_url=base_url or cfg["MODEL"]["DEEPSEEK"]["BASE_URL"],
            temperature=temperature,
        )

    if provider == "openai":
        resolved_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or cfg["MODEL"]["OPENAI"].get("API_KEY")
        )
        if not _valid_key(resolved_key):
            return OfflineLLM()

        return ChatOpenAI(
            model=model_name or cfg["MODEL"]["OPENAI"]["MODEL_NAME"],
            api_key=resolved_key,
            base_url=base_url or None,
            temperature=temperature,
        )

    return OfflineLLM()

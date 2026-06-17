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


def get_llm():
    provider = cfg["MODEL"].get("PROVIDER", "offline")
    temperature = cfg["MODEL"].get("TEMPERATURE", 0.3)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return OfflineLLM()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or cfg["MODEL"]["DEEPSEEK"].get("API_KEY")
        if not _valid_key(api_key):
            return OfflineLLM()

        return ChatOpenAI(
            model=cfg["MODEL"]["DEEPSEEK"]["MODEL_NAME"],
            api_key=api_key,
            base_url=cfg["MODEL"]["DEEPSEEK"]["BASE_URL"],
            temperature=temperature,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or cfg["MODEL"]["OPENAI"].get("API_KEY")
        if not _valid_key(api_key):
            return OfflineLLM()

        return ChatOpenAI(
            model=cfg["MODEL"]["OPENAI"]["MODEL_NAME"],
            api_key=api_key,
            temperature=temperature,
        )

    return OfflineLLM()

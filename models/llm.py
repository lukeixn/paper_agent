from __future__ import annotations

import os

from configs.config import cfg


class LLM:
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Install requirements or use models.langchain_llm.OfflineLLM."
            ) from exc

        provider = cfg["MODEL"]["PROVIDER"]

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY") or cfg["MODEL"]["OPENAI"]["API_KEY"]
            self.client = OpenAI(
                api_key=api_key
            )
            self.model = cfg["MODEL"]["OPENAI"]["MODEL_NAME"]

        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY") or cfg["MODEL"]["DEEPSEEK"]["API_KEY"]
            self.client = OpenAI(
                api_key=api_key,
                base_url=cfg["MODEL"]["DEEPSEEK"]["BASE_URL"]
            )
            self.model = cfg["MODEL"]["DEEPSEEK"]["MODEL_NAME"]
        else:
            raise ValueError(f"Unknown online provider: {provider}")

    def chat(self, messages, tools=None):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=cfg["MODEL"]["TEMPERATURE"]
        )

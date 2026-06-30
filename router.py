from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from models.langchain_llm import OfflineLLM, get_llm


AGENT_DESCRIPTIONS = {
    "survey_agent": "研究综述：研究方向、发展脉络、现状、趋势与代表性工作",
    "innovation_agent": "创新分析：核心贡献、创新价值、差异、优势与新颖性",
    "method_agent": "方法比较：模型架构、技术路线、关键机制、实现与方法对比",
    "limitation_agent": "局限与机会：不足、风险、缺点、研究空白与后续机会",
}
ALL_AGENTS = list(AGENT_DESCRIPTIONS)


@dataclass(frozen=True)
class RouteDecision:
    route: str
    agents: list[str]
    reason: str


class Router:
    def route(
        self,
        query: str,
        *,
        contextual_query: str | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> RouteDecision:
        llm_decision = self._llm_route(
            query,
            contextual_query=contextual_query,
            model_config=model_config or {},
        )
        if llm_decision is not None:
            return llm_decision
        return self._rule_route(query)

    def _llm_route(
        self,
        query: str,
        *,
        contextual_query: str | None,
        model_config: dict[str, Any],
    ) -> RouteDecision | None:
        if not model_config or model_config.get("provider") == "offline":
            return None
        llm = get_llm(
            provider=model_config.get("provider"),
            api_key=model_config.get("api_key"),
            model_name=model_config.get("model_name"),
            base_url=model_config.get("base_url"),
            temperature=0,
        )
        if isinstance(llm, OfflineLLM):
            return None

        english = str(
            model_config.get("output_language", "")
        ).lower().startswith("en")
        reason_language_rule = (
            "Write the reason in English."
            if english
            else "reason 使用中文。"
        )
        descriptions = "\n".join(
            f"- {name}: {description}"
            for name, description in AGENT_DESCRIPTIONS.items()
        )
        prompt = f"""
你是多 Agent 研究系统的任务路由器。请判断本轮问题需要调用哪些专家 Agent。

可选 Agent：
{descriptions}

本轮用户原始问题（最高优先级）：
{query}

结合会话历史改写后的独立问题：
{contextual_query or query}

强制要求：
1. 只能从给定的四个 Agent 名称中选择。
2. 选择完成本轮问题所必需的最小 Agent 集合，不要机械地全选。
3. 如果问题要求研究选题、方向建议或综合决策，应选择所有相关 Agent。
4. 原始问题与独立问题冲突时，以本轮原始问题的意图为准。
5. 只输出一个 JSON 对象，不要输出 Markdown 或其他文字。
6. {reason_language_rule}

输出格式：
{{"agents":["method_agent"],"reason":"简洁说明选择依据"}}
"""
        try:
            content = llm.invoke(prompt).content.strip()
            payload = self._parse_llm_payload(content)
        except Exception:
            return None

        raw_agents = payload.get("agents")
        if not isinstance(raw_agents, list):
            return None
        agents = list(
            dict.fromkeys(
                agent
                for agent in raw_agents
                if isinstance(agent, str) and agent in ALL_AGENTS
            )
        )
        if not agents:
            return None

        reason = str(payload.get("reason", "")).strip()
        route = "multi_agent" if len(agents) > 1 else agents[0]
        reason_prefix = "LLM route" if english else "LLM 路由"
        fallback_reason = (
            "selected required specialists for this question"
            if english
            else "根据本轮问题选择所需专家"
        )
        return RouteDecision(
            route=route,
            agents=agents,
            reason=f"{reason_prefix}: {reason or fallback_reason}",
        )

    @staticmethod
    def _parse_llm_payload(content: str) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                return {}
            try:
                payload = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return payload if isinstance(payload, dict) else {}

    def _rule_route(self, query: str) -> RouteDecision:
        text = query.lower()
        agents: list[str] = []

        decision_terms = [
            "哪个方向",
            "什么方向",
            "研究方向",
            "研究机会",
            "发论文",
            "发表论文",
            "论文选题",
            "选题",
            "投稿",
            "值得研究",
            "适合研究",
        ]
        if any(term in text for term in decision_terms):
            english = any(
                term in text
                for term in [
                    "research direction",
                    "topic",
                    "publish",
                    "publication",
                    "worth studying",
                ]
            )
            return RouteDecision(
                route="multi_agent",
                agents=ALL_AGENTS,
                reason=(
                    "Rule fallback: this question needs trends, innovation, "
                    "methods, and risks for research-direction decisions"
                    if english
                    else "规则回退：该问题需要综合趋势、创新、方法和风险进行研究方向决策"
                ),
            )

        if any(word in text for word in ["创新", "novel", "innovation", "贡献", "contribution", "优势"]):
            agents.append("innovation_agent")
        if any(word in text for word in ["方法", "模型", "架构", "method", "architecture", "怎么做", "机制", "技术"]):
            agents.append("method_agent")
        if any(word in text for word in ["不足", "局限", "问题", "limitation", "risk", "缺点"]):
            agents.append("limitation_agent")
        if any(word in text for word in ["综述", "总结", "survey", "overview", "latest", "最新", "方向"]):
            agents.append("survey_agent")

        if not agents:
            agents = ALL_AGENTS

        route = "multi_agent" if len(agents) > 1 else agents[0]
        english = any(
            word in text
            for word in [
                "novel",
                "innovation",
                "contribution",
                "method",
                "architecture",
                "limitation",
                "risk",
                "survey",
                "overview",
                "latest",
            ]
        )
        return RouteDecision(
            route=route,
            agents=agents,
            reason=(
                f"Rule fallback: selected {', '.join(agents)} from the question intent"
                if english
                else f"规则回退：根据问题意图选择 {', '.join(agents)}"
            ),
        )

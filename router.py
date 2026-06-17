from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    route: str
    agents: list[str]
    reason: str


class Router:
    def route(self, query: str) -> RouteDecision:
        text = query.lower()
        agents: list[str] = []

        if any(word in text for word in ["创新", "novel", "innovation", "贡献", "contribution", "优势"]):
            agents.append("innovation_agent")
        if any(word in text for word in ["方法", "模型", "架构", "method", "architecture", "怎么做", "机制", "技术"]):
            agents.append("method_agent")
        if any(word in text for word in ["不足", "局限", "问题", "limitation", "risk", "缺点"]):
            agents.append("limitation_agent")
        if any(word in text for word in ["综述", "总结", "survey", "overview", "latest", "最新", "方向"]):
            agents.append("survey_agent")

        if not agents:
            agents = [
                "survey_agent",
                "innovation_agent",
                "method_agent",
                "limitation_agent",
            ]

        route = "multi_agent" if len(agents) > 1 else agents[0]
        return RouteDecision(
            route=route,
            agents=agents,
            reason=f"根据问题意图选择 {', '.join(agents)}",
        )

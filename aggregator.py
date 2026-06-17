from __future__ import annotations

from state.state import MainState


class ReportAggregator:
    def build_report(self, state: MainState) -> str:
        papers = state.get("retrieved_papers", [])
        agent_results = state.get("agent_results", {})

        lines = [
            "# 论文 Agent 分析报告",
            "",
            f"## 用户问题",
            state.get("query", ""),
            "",
            "## 路由结果",
            f"- route: {state.get('route', '')}",
            f"- agents: {', '.join(agent_results.keys()) or 'none'}",
            "",
            "## 检索到的论文",
        ]

        if not papers:
            lines.append("- 未检索到相关论文。")
        else:
            for index, paper in enumerate(papers, start=1):
                lines.append(f"- {index}. {paper.title} (score={paper.score})")

        for agent_name, result in agent_results.items():
            lines.extend(["", f"## {agent_name}", result])

        errors = state.get("errors", [])
        if errors:
            lines.extend(["", "## 运行警告"])
            lines.extend(f"- {error}" for error in errors)

        return "\n".join(lines).strip() + "\n"

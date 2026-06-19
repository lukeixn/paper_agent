from __future__ import annotations

from models.langchain_llm import OfflineLLM, get_llm
from state.state import AgentOutput, MainState


class ReportAgent:
    name = "report_agent"
    title = "综合报告 Agent"

    def run(self, state: MainState) -> str:
        outputs = state.get("agent_outputs", [])
        successful = [output for output in outputs if not output["error"]]
        model_config = state.get("global_context", {}).get("model_config", {})
        llm = get_llm(
            provider=model_config.get("provider"),
            api_key=model_config.get("api_key"),
            model_name=model_config.get("model_name"),
            base_url=model_config.get("base_url"),
            temperature=model_config.get("temperature"),
        )
        if not isinstance(llm, OfflineLLM) and successful:
            try:
                return llm.invoke(self._report_prompt(state, successful)).content
            except Exception:
                pass
        return self._fallback_report(state, outputs)

    @staticmethod
    def _report_prompt(
        state: MainState,
        outputs: list[AgentOutput],
    ) -> str:
        sections = "\n\n".join(
            f"## {output['title']}\n{output['content']}"
            for output in outputs
        )
        paper_titles = "\n".join(
            f"- {paper.title}" for paper in state.get("retrieved_papers", [])
        )
        return f"""
你是最终 ReportAgent。请基于多个彼此独立的专家 Agent 输出，生成一份
结构清晰、避免重复、忠于证据的中文研究报告。

用户问题：
{state.get("query", "")}

检索论文：
{paper_titles}

独立 Agent 输出：
{sections}

要求：
1. 开头给出简洁的执行摘要。
2. 综合不同 Agent 的观点，而不是简单拼接。
3. 明确区分研究趋势、创新、方法、局限和研究机会。
4. 不要添加独立 Agent 和论文材料中没有依据的事实。
"""

    @staticmethod
    def _fallback_report(
        state: MainState,
        outputs: list[AgentOutput],
    ) -> str:
        lines = [
            "# 论文 Agent 分析报告",
            "",
            "## 用户问题",
            state.get("query", ""),
            "",
            "## 检索论文",
        ]
        papers = state.get("retrieved_papers", [])
        if papers:
            lines.extend(
                f"- {index}. {paper.title} (score={paper.score})"
                for index, paper in enumerate(papers, start=1)
            )
        else:
            lines.append("- 未检索到相关论文。")

        for output in outputs:
            lines.extend(["", f"## {output['title']}"])
            if output["error"]:
                lines.append(f"执行失败：{output['error']}")
            else:
                lines.append(output["content"])
        return "\n".join(lines).strip() + "\n"


# Compatibility alias for older imports.
ReportAggregator = ReportAgent

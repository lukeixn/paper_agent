from __future__ import annotations

from models.langchain_llm import OfflineLLM, get_llm
from state.state import AgentOutput, MainState


class ReportAgent:
    name = "report_agent"
    title = "综合报告 Agent"

    @staticmethod
    def response_mode(state: MainState) -> str:
        if not state.get("user_question_history"):
            return "report"
        query = state.get("query", "").lower()
        explicit_report_terms = [
            "完整报告",
            "综合报告",
            "生成报告",
            "重新汇总",
            "全面总结",
            "总结成报告",
            "full report",
        ]
        if any(term in query for term in explicit_report_terms):
            return "report"
        return "follow_up"

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
        history_text = "\n".join(
            f"{index}. {question}"
            for index, question in enumerate(
                state.get("user_question_history", []),
                start=1,
            )
        )
        if ReportAgent.response_mode(state) == "follow_up":
            return f"""
你是连续研究会话中的最终回答 Agent。请综合多个独立专家 Agent 的本轮分析，
直接回答用户当前追问。你的任务是延续会话，而不是每轮重新生成完整研究报告。

同一会话中的历史用户问题：
{history_text}

本轮用户当前追问（唯一回答目标，最高优先级）：
{state.get("query", "")}

用于检索和分析的独立问题：
{state.get("standalone_query", state.get("query", ""))}

本轮检索论文：
{paper_titles}

本轮独立 Agent 输出：
{sections}

强制要求：
1. 开头直接给出本轮追问的答案或判断，不要使用“执行摘要”。
2. 根据问题自然组织内容，不要强制套用趋势、创新、方法、局限等固定章节。
3. 不要重复上一轮问题的背景和已经明确的结论，除非它们是回答本轮问题所必需的。
4. 需要比较时使用紧凑的对比结构；需要解释时围绕因果关系展开；需要建议时给出可执行结论。
5. 综合各 Agent 的观点，不要逐个复述 Agent 输出。
6. 历史问题只用于保持指代、研究对象和约束连续；冲突时以当前追问为准。
7. 不要添加 Agent 输出和论文材料中没有依据的事实。
8. 除非用户明确要求重新汇总或生成完整报告，否则保持对话式回答。
"""
        return f"""
你是最终 ReportAgent。请基于多个彼此独立的专家 Agent 输出，生成一份
结构清晰、避免重复、忠于证据的中文研究报告。

同一会话中的历史用户问题（仅用于理解当前问题）：
{history_text or "无"}

本轮用户当前问题（最高优先级）：
{state.get("query", "")}

用于检索和分析的独立问题：
{state.get("standalone_query", state.get("query", ""))}

检索论文：
{paper_titles}

独立 Agent 输出：
{sections}

强制要求：
0. 最终报告必须直接回答本轮当前问题。历史问题只用于解析上下文，不得喧宾夺主。
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
        if ReportAgent.response_mode(state) == "follow_up":
            lines = [
                "# 本轮回答",
                "",
                state.get("query", ""),
            ]
            for output in outputs:
                if output["error"]:
                    continue
                lines.extend(
                    ["", f"### {output['title']}", output["content"]]
                )
            if len(lines) == 3:
                lines.extend(["", "本轮 Agent 未能生成有效分析。"])
            return "\n".join(lines).strip() + "\n"

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

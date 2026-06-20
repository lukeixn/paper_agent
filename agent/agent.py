from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from models.langchain_llm import OfflineLLM, get_llm
from schemas import Paper
from state.state import AgentOutput, AgentTaskState


class BaseAgent(ABC):
    name = "base_agent"
    title = "Base Agent"

    def __init__(self, profile_path: str | None = None):
        self.profile_path = profile_path
        self.profile = self.load_profile()

    def load_profile(self) -> str:
        if not self.profile_path:
            return ""
        path = Path(self.profile_path)
        return path.read_text(encoding="utf8") if path.exists() else ""

    def __call__(self, task: AgentTaskState) -> AgentOutput:
        try:
            return {
                "agent_name": self.name,
                "title": self.title,
                "content": self.run(task),
                "error": "",
            }
        except Exception as exc:
            return {
                "agent_name": self.name,
                "title": self.title,
                "content": "",
                "error": str(exc),
            }

    @abstractmethod
    def run(self, task: AgentTaskState) -> str:
        raise NotImplementedError

    @staticmethod
    def paper_block(papers: list[Paper]) -> str:
        blocks = []
        for index, paper in enumerate(papers, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[{index}] {paper.title}",
                        f"相关度: {paper.score}",
                        f"摘要: {paper.summary or paper.abstract}",
                        f"关键词: {', '.join(paper.keywords)}",
                        f"贡献: {'; '.join(paper.contributions)}",
                        f"局限: {'; '.join(paper.limitations)}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def maybe_llm(
        self,
        instruction: str,
        task: AgentTaskState,
    ) -> str | None:
        model_config = task["model_config"]
        llm = get_llm(
            provider=model_config.get("provider"),
            api_key=model_config.get("api_key"),
            model_name=model_config.get("model_name"),
            base_url=model_config.get("base_url"),
            temperature=model_config.get("temperature"),
        )
        if isinstance(llm, OfflineLLM):
            return None

        prompt = f"""
{self.profile}

历史对话（仅用于理解当前追问）：
{task["conversation_context"] or "无"}

用户当前问题：
{task["user_query"]}

用于检索和分析的独立问题：
{task["query"]}

论文材料：
{self.paper_block(task["papers"])}

你的独立任务：
{instruction}

只完成你的任务。不要推测其他 Agent 的结论，也不要撰写最终综合报告。
"""
        return llm.invoke(prompt).content


class SurveyAgent(BaseAgent):
    name = "survey_agent"
    title = "研究综述 Agent"

    def run(self, task: AgentTaskState) -> str:
        result = self.maybe_llm(
            "归纳论文代表的研究方向、发展脉络和整体趋势。",
            task,
        )
        if result:
            return result
        lines = ["这些论文主要体现以下研究方向："]
        for paper in task["papers"]:
            keywords = "、".join(paper.keywords[:5]) or "暂无关键词"
            lines.append(f"- {paper.title}: {keywords}")
        return "\n".join(lines)


class InnovationAgent(BaseAgent):
    name = "innovation_agent"
    title = "创新分析 Agent"

    def __init__(self):
        super().__init__("profiles/innovation.md")

    def run(self, task: AgentTaskState) -> str:
        result = self.maybe_llm(
            "分析核心创新、创新价值，以及与已有工作的差异。",
            task,
        )
        if result:
            return result
        lines = ["主要创新点："]
        for paper in task["papers"]:
            contributions = paper.contributions[:3] or [paper.summary[:160]]
            lines.append(f"- {paper.title}: {'; '.join(contributions)}")
        return "\n".join(lines)


class MethodAgent(BaseAgent):
    name = "method_agent"
    title = "方法比较 Agent"

    def run(self, task: AgentTaskState) -> str:
        result = self.maybe_llm(
            "比较论文的方法、模型结构、技术路线和关键机制。",
            task,
        )
        if result:
            return result
        lines = ["方法与技术路线："]
        for paper in task["papers"]:
            summary = paper.summary or paper.abstract
            lines.append(f"- {paper.title}: {summary[:260]}")
        return "\n".join(lines)


class LimitationAgent(BaseAgent):
    name = "limitation_agent"
    title = "局限与机会 Agent"

    def run(self, task: AgentTaskState) -> str:
        result = self.maybe_llm(
            "分析论文的局限、风险和后续研究机会。",
            task,
        )
        if result:
            return result
        lines = ["局限与后续机会："]
        for paper in task["papers"]:
            limitations = paper.limitations[:3] or ["原始数据未明确给出局限。"]
            lines.append(f"- {paper.title}: {'; '.join(limitations)}")
        return "\n".join(lines)


AGENT_REGISTRY = {
    SurveyAgent.name: SurveyAgent,
    InnovationAgent.name: InnovationAgent,
    MethodAgent.name: MethodAgent,
    LimitationAgent.name: LimitationAgent,
}

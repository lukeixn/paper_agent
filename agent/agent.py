from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from models.langchain_llm import OfflineLLM, get_llm
from schemas import Paper
from state.state import MainState


class BaseAgent(ABC):
    name = "base_agent"
    title = "Base Agent"

    def __init__(self, profile_path: str | None = None):
        self.profile_path = profile_path
        self.llm = get_llm()
        self.profile = self.load_profile()

    def load_profile(self) -> str:
        if not self.profile_path:
            return ""

        path = Path(self.profile_path)
        if not path.exists():
            return ""

        return path.read_text(encoding="utf8")

    def __call__(self, state: MainState) -> MainState:
        try:
            result = self.run(state)
            state.setdefault("agent_results", {})[self.name] = result
            state.setdefault("local_contexts", {})[self.name] = result
        except Exception as exc:
            state.setdefault("errors", []).append(f"{self.name}: {exc}")
        return state

    @abstractmethod
    def run(self, state: MainState) -> str:
        raise NotImplementedError

    def paper_block(self, papers: list[Paper]) -> str:
        blocks = []
        for index, paper in enumerate(papers, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[{index}] {paper.title}",
                        f"score: {paper.score}",
                        f"summary: {paper.summary}",
                        f"keywords: {', '.join(paper.keywords)}",
                        f"contributions: {'; '.join(paper.contributions)}",
                        f"limitations: {'; '.join(paper.limitations)}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def maybe_llm(self, instruction: str, state: MainState) -> str | None:
        if isinstance(self.llm, OfflineLLM):
            return None

        papers = state.get("retrieved_papers", [])
        prompt = f"""
{self.profile}

用户问题:
{state.get("query", "")}

论文材料:
{self.paper_block(papers)}

任务:
{instruction}
"""
        return self.llm.invoke(prompt).content


class SurveyAgent(BaseAgent):
    name = "survey_agent"
    title = "研究综述 Agent"

    def run(self, state: MainState) -> str:
        llm_result = self.maybe_llm("请归纳这些论文代表的研究方向和整体趋势。", state)
        if llm_result:
            return llm_result

        papers = state.get("retrieved_papers", [])
        lines = ["这些论文集中体现了以下方向："]
        for paper in papers:
            keywords = "、".join(paper.keywords[:5]) or "暂无关键词"
            lines.append(f"- {paper.title}: {keywords}")
        return "\n".join(lines)


class InnovationAgent(BaseAgent):
    name = "innovation_agent"
    title = "创新点 Agent"

    def __init__(self):
        super().__init__("profiles/innovation.md")

    def run(self, state: MainState) -> str:
        llm_result = self.maybe_llm("请分析核心创新、创新价值、与已有工作的区别。", state)
        if llm_result:
            return llm_result

        lines = ["可提取的主要创新点："]
        for paper in state.get("retrieved_papers", []):
            contributions = paper.contributions[:3] or [paper.summary[:160]]
            lines.append(f"- {paper.title}: {'; '.join(contributions)}")
        return "\n".join(lines)


class MethodAgent(BaseAgent):
    name = "method_agent"
    title = "技术路线 Agent"

    def run(self, state: MainState) -> str:
        llm_result = self.maybe_llm("请比较这些论文的方法、模型结构和技术路线。", state)
        if llm_result:
            return llm_result

        lines = ["方法和技术路线概览："]
        for paper in state.get("retrieved_papers", []):
            summary = paper.summary or paper.abstract
            lines.append(f"- {paper.title}: {summary[:260]}")
        return "\n".join(lines)


class LimitationAgent(BaseAgent):
    name = "limitation_agent"
    title = "局限与机会 Agent"

    def run(self, state: MainState) -> str:
        llm_result = self.maybe_llm("请分析这些论文的局限、风险和后续研究机会。", state)
        if llm_result:
            return llm_result

        lines = ["局限和后续机会："]
        for paper in state.get("retrieved_papers", []):
            limitations = paper.limitations[:3] or ["原始数据中未明确给出局限。"]
            lines.append(f"- {paper.title}: {'; '.join(limitations)}")
        return "\n".join(lines)


AGENT_REGISTRY = {
    SurveyAgent.name: SurveyAgent,
    InnovationAgent.name: InnovationAgent,
    MethodAgent.name: MethodAgent,
    LimitationAgent.name: LimitationAgent,
}

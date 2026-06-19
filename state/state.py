from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from schemas import Paper


class AgentTaskState(TypedDict):
    agent_name: str
    query: str
    papers: list[Paper]
    model_config: dict[str, Any]


class AgentOutput(TypedDict):
    agent_name: str
    title: str
    content: str
    error: str


class MainState(TypedDict, total=False):
    query: str
    route: str
    selected_agents: list[str]
    global_context: dict[str, Any]
    retrieved_papers: list[Paper]
    agent_outputs: Annotated[list[AgentOutput], operator.add]
    agent_results: dict[str, str]
    final_report: str
    errors: Annotated[list[str], operator.add]


def create_state(query: str) -> MainState:
    return {
        "query": query,
        "route": "",
        "selected_agents": [],
        "global_context": {},
        "retrieved_papers": [],
        "agent_outputs": [],
        "agent_results": {},
        "final_report": "",
        "errors": [],
    }


State = MainState

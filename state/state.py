from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from schemas import Paper


class ConversationMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentTaskState(TypedDict):
    agent_name: str
    query: str
    user_query: str
    user_question_history: list[str]
    papers: list[Paper]
    model_config: dict[str, Any]


class AgentOutput(TypedDict):
    agent_name: str
    title: str
    content: str
    error: str


class MainState(TypedDict, total=False):
    query: str
    standalone_query: str
    response_mode: Literal["report", "follow_up"]
    conversation_history: list[ConversationMessage]
    user_question_history: list[str]
    route: str
    selected_agents: list[str]
    global_context: dict[str, Any]
    retrieved_papers: list[Paper]
    agent_outputs: Annotated[list[AgentOutput], operator.add]
    agent_results: dict[str, str]
    final_report: str
    errors: Annotated[list[str], operator.add]


def create_state(
    query: str,
    conversation_history: list[ConversationMessage] | None = None,
) -> MainState:
    return {
        "query": query,
        "standalone_query": query,
        "response_mode": "report",
        "conversation_history": list(conversation_history or []),
        "user_question_history": [],
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

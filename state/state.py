from __future__ import annotations

from typing import Any, TypedDict

from schemas import Paper


class MainState(TypedDict, total=False):
    query: str
    route: str
    global_context: dict[str, Any]
    local_contexts: dict[str, Any]
    retrieved_papers: list[Paper]
    agent_results: dict[str, str]
    final_report: str
    errors: list[str]


def create_state(query: str) -> MainState:
    return {
        "query": query,
        "route": "",
        "global_context": {},
        "local_contexts": {},
        "retrieved_papers": [],
        "agent_results": {},
        "final_report": "",
        "errors": [],
    }


# Backward-compatible alias for older experiments in this project.
State = MainState

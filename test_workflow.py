from __future__ import annotations

import threading

import workflow
from agent.agent import AGENT_REGISTRY
from workflow import run_pipeline, run_pipeline_state, workflow_info


def test_workflow_info() -> None:
    info = workflow_info()
    assert info["parallel"] is True
    assert info["nodes"] == [
        "retrieve",
        "route",
        "run_agent",
        "report_agent",
    ]
    assert ("route", "run_agent[]") in info["edges"]
    assert ("run_agent[]", "report_agent") in info["edges"]


def test_pipeline_generates_report() -> None:
    report = run_pipeline(
        "Mamba innovation limitation",
        top_k=3,
        model_config={"provider": "offline"},
    )
    assert "# 论文 Agent 分析报告" in report
    assert "VAMBA" in report or "Video Mamba" in report
    assert "创新分析 Agent" in report
    assert "局限与机会 Agent" in report


def test_parallel_agents_receive_isolated_context() -> None:
    agent_names = [
        "survey_agent",
        "innovation_agent",
        "method_agent",
        "limitation_agent",
    ]
    barrier = threading.Barrier(len(agent_names), timeout=5)
    seen_keys: list[set[str]] = []
    config_ids: list[int] = []
    lock = threading.Lock()

    def make_fake_agent(agent_name: str):
        class FakeAgent:
            def __call__(self, task):
                with lock:
                    seen_keys.append(set(task.keys()))
                    config_ids.append(id(task["model_config"]))
                task["model_config"]["branch_marker"] = agent_name
                barrier.wait()
                return {
                    "agent_name": agent_name,
                    "title": agent_name,
                    "content": f"{agent_name} completed",
                    "error": "",
                }

        return FakeAgent

    original_registry = dict(AGENT_REGISTRY)
    original_retrieve = workflow.retrieve_node
    try:
        for agent_name in agent_names:
            AGENT_REGISTRY[agent_name] = make_fake_agent(agent_name)
        workflow.retrieve_node = lambda state: {"retrieved_papers": []}

        state = run_pipeline_state(
            "a broad research question",
            model_config={"provider": "offline"},
            require_langgraph=True,
        )
    finally:
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(original_registry)
        workflow.retrieve_node = original_retrieve

    expected_keys = {"agent_name", "query", "papers", "model_config"}
    assert seen_keys == [expected_keys] * len(agent_names)
    assert len(set(config_ids)) == len(agent_names)
    assert set(state["agent_results"]) == set(agent_names)
    assert "branch_marker" not in state["global_context"]["model_config"]


if __name__ == "__main__":
    test_workflow_info()
    test_pipeline_generates_report()
    test_parallel_agents_receive_isolated_context()
    print("workflow tests passed")

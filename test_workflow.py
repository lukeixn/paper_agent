from __future__ import annotations

import threading

import workflow
from agent.agent import AGENT_REGISTRY
from workflow import (
    contextualize_query_node,
    create_pipeline_state,
    run_pipeline,
    run_pipeline_state,
    stream_pipeline_state,
    workflow_info,
)


def test_workflow_info() -> None:
    info = workflow_info()
    assert info["parallel"] is True
    assert info["nodes"] == [
        "contextualize",
        "retrieve",
        "route",
        "run_agent",
        "report_agent",
    ]
    assert ("START", "contextualize") in info["edges"]
    assert ("contextualize", "retrieve") in info["edges"]
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

    expected_keys = {
        "agent_name",
        "query",
        "user_query",
        "conversation_context",
        "papers",
        "model_config",
    }
    assert seen_keys == [expected_keys] * len(agent_names)
    assert len(set(config_ids)) == len(agent_names)
    assert set(state["agent_results"]) == set(agent_names)
    assert "branch_marker" not in state["global_context"]["model_config"]


def test_follow_up_question_uses_conversation_context() -> None:
    history = [
        {
            "role": "user",
            "content": "分析 Mamba 在长视频理解中的主要方法。",
        },
        {
            "role": "assistant",
            "content": "上一轮报告讨论了状态空间模型和长序列建模。",
        },
        {
            "role": "user",
            "content": "这些方法分别有什么优势？",
        },
        {
            "role": "assistant",
            "content": "上一轮比较了效率、长程依赖和计算复杂度。",
        },
    ]
    state = create_pipeline_state(
        "它目前最大的局限是什么？",
        model_config={"provider": "offline"},
        conversation_history=history,
    )

    update = contextualize_query_node(state)

    assert "Mamba" in update["standalone_query"]
    assert "这些方法分别有什么优势" in update["standalone_query"]
    assert "当前追问" in update["standalone_query"]
    assert "它目前最大的局限是什么" in update["standalone_query"]
    assert "状态空间模型" in update["conversation_context"]


def test_stream_reports_each_parallel_agent() -> None:
    events = list(
        stream_pipeline_state(
            "Mamba innovation limitation",
            top_k=2,
            model_config={"provider": "offline"},
            require_langgraph=True,
        )
    )

    nodes = [event["node"] for event in events]
    assert nodes[:4] == [
        "contextualize",
        "contextualize",
        "retrieve",
        "route",
    ]
    agent_events = [
        event for event in events if event["node"] == "run_agent"
    ]
    assert {
        event["agent_name"] for event in agent_events
    } == {"innovation_agent", "limitation_agent"}
    assert nodes[-2:] == ["report_agent", "end"]
    assert events[-1]["state"]["final_report"]


if __name__ == "__main__":
    test_workflow_info()
    test_pipeline_generates_report()
    test_parallel_agents_receive_isolated_context()
    test_follow_up_question_uses_conversation_context()
    test_stream_reports_each_parallel_agent()
    print("workflow tests passed")

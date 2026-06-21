from __future__ import annotations

import workflow
from agent.agent import AGENT_REGISTRY
from router import Router
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
    seen_keys: list[set[str]] = []
    config_ids: list[int] = []
    question_histories: list[list[str]] = []

    def make_fake_agent(agent_name: str):
        class FakeAgent:
            def __call__(self, task):
                seen_keys.append(set(task.keys()))
                config_ids.append(id(task["model_config"]))
                question_histories.append(
                    list(task["user_question_history"])
                )
                task["model_config"]["branch_marker"] = agent_name
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
            conversation_history=[
                {"role": "user", "content": "first user question"},
                {
                    "role": "assistant",
                    "content": "full assistant report",
                },
                {"role": "user", "content": "second user question"},
            ],
        )
    finally:
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(original_registry)
        workflow.retrieve_node = original_retrieve

    expected_keys = {
        "agent_name",
        "query",
        "user_query",
        "user_question_history",
        "papers",
        "model_config",
    }
    assert seen_keys == [expected_keys] * len(agent_names), seen_keys
    assert len(set(config_ids)) == len(agent_names)
    assert question_histories == [
        ["first user question", "second user question"]
    ] * len(agent_names)
    assert set(state["agent_results"]) == set(agent_names)
    assert "branch_marker" not in state["global_context"]["model_config"]


def test_follow_up_question_uses_user_question_history() -> None:
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
    assert update["user_question_history"] == [
        "分析 Mamba 在长视频理解中的主要方法。",
        "这些方法分别有什么优势？",
    ]
    assert "状态空间模型" not in "\n".join(
        update["user_question_history"]
    )


def test_full_reports_are_excluded_from_agent_history() -> None:
    history = [
        {
            "role": "user",
            "content": "第一轮研究主题是 KV 缓存压缩。",
        },
        {
            "role": "assistant",
            "content": "第一轮摘要。" + "A" * 20000 + "第一轮结论。",
        },
        {
            "role": "user",
            "content": "第二轮请比较视觉模型中的方法。",
        },
        {
            "role": "assistant",
            "content": "第二轮摘要。" + "B" * 20000 + "第二轮结论。",
        },
    ]

    state = create_pipeline_state(
        "继续比较它们的适用场景。",
        model_config={"provider": "offline"},
        conversation_history=history,
    )
    update = contextualize_query_node(state)

    assert update["user_question_history"] == [
        "第一轮研究主题是 KV 缓存压缩。",
        "第二轮请比较视觉模型中的方法。",
    ]
    assert "第一轮摘要" not in update["standalone_query"]
    assert "第二轮摘要" not in update["standalone_query"]


def test_conversation_histories_are_isolated_between_states() -> None:
    first_state = create_pipeline_state(
        "继续分析它的方法。",
        model_config={"provider": "offline"},
        conversation_history=[
            {"role": "user", "content": "会话甲研究 Mamba。"},
            {"role": "assistant", "content": "会话甲报告。"},
        ],
    )
    second_state = create_pipeline_state(
        "继续分析它的方法。",
        model_config={"provider": "offline"},
        conversation_history=[
            {"role": "user", "content": "会话乙研究 Transformer。"},
            {"role": "assistant", "content": "会话乙报告。"},
        ],
    )

    first_state.update(contextualize_query_node(first_state))
    second_state.update(contextualize_query_node(second_state))

    assert first_state["user_question_history"] == [
        "会话甲研究 Mamba。"
    ]
    assert second_state["user_question_history"] == [
        "会话乙研究 Transformer。"
    ]
    assert "Transformer" not in first_state["standalone_query"]
    assert "Mamba" not in second_state["standalone_query"]

    first_state["user_question_history"].append("仅修改会话甲")
    assert "仅修改会话甲" not in second_state["user_question_history"]


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
    assert {
        output["agent_name"]
        for output in events[-1]["state"]["agent_outputs"]
    } == {"innovation_agent", "limitation_agent"}


def test_research_direction_question_uses_all_agents() -> None:
    decision = Router().route(
        "KV 缓存处理我该做哪个方向合适发论文？"
    )

    assert decision.route == "multi_agent"
    assert decision.agents == [
        "survey_agent",
        "innovation_agent",
        "method_agent",
        "limitation_agent",
    ]


if __name__ == "__main__":
    test_workflow_info()
    test_pipeline_generates_report()
    test_parallel_agents_receive_isolated_context()
    test_follow_up_question_uses_user_question_history()
    test_full_reports_are_excluded_from_agent_history()
    test_conversation_histories_are_isolated_between_states()
    test_stream_reports_each_parallel_agent()
    test_research_direction_question_uses_all_agents()
    print("workflow tests passed")

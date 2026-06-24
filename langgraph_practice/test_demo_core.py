from __future__ import annotations

try:
    from .demo_core import run_demo, stream_demo
except ImportError:
    from demo_core import run_demo, stream_demo


def test_demo_selects_dynamic_agents() -> None:
    state = run_demo("Compare method and limitation for KV cache compression.")

    assert state["selected_agents"] == ["method_agent", "limitation_agent"]
    assert {
        output["agent_name"] for output in state["agent_outputs"]
    } == {"method_agent", "limitation_agent"}
    assert "LangGraph Demo Result" in state["final_answer"]


def test_demo_stream_reports_parallel_agents() -> None:
    events = list(stream_demo("Find innovation and risk for long video understanding."))

    agent_events = [
        event for event in events if event["node"] == "run_agent"
    ]
    assert {
        event["agent_name"] for event in agent_events
    } == {"innovation_agent", "limitation_agent"}
    assert events[0]["node"] == "START"
    assert events[-1]["node"] == "END"
    assert events[-1]["state"]["final_answer"]


if __name__ == "__main__":
    test_demo_selects_dynamic_agents()
    test_demo_stream_reports_parallel_agents()
    print("langgraph practice tests passed")

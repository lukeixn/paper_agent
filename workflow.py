from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.agent import AGENT_REGISTRY
from aggregator import ReportAgent
from router import Router
from state.state import AgentOutput, AgentTaskState, MainState, create_state
from vector_store.search import PaperSearchEngine

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send
except ImportError:  # pragma: no cover
    END = None
    START = None
    StateGraph = None
    Send = None


def langgraph_available() -> bool:
    return StateGraph is not None and Send is not None


def retrieve_node(state: MainState) -> dict[str, Any]:
    global_context = state.get("global_context", {})
    search_engine = PaperSearchEngine(
        data_dir=global_context.get("data_dir", "data")
    )
    papers = search_engine.search(
        state.get("query", ""),
        top_k=int(global_context.get("top_k", 5)),
    )
    return {"retrieved_papers": papers}


def route_node(state: MainState) -> dict[str, Any]:
    decision = Router().route(state.get("query", ""))
    global_context = dict(state.get("global_context", {}))
    global_context["route_reason"] = decision.reason
    return {
        "route": decision.route,
        "selected_agents": decision.agents,
        "global_context": global_context,
    }


def dispatch_agents(state: MainState):
    if Send is None:
        raise RuntimeError("LangGraph Send API is unavailable.")

    model_config = dict(
        state.get("global_context", {}).get("model_config", {})
    )
    return [
        Send(
            "run_agent",
            {
                "agent_name": agent_name,
                "query": state.get("query", ""),
                "papers": list(state.get("retrieved_papers", [])),
                "model_config": dict(model_config),
            },
        )
        for agent_name in state.get("selected_agents", [])
    ]


def run_agent_node(task: AgentTaskState) -> dict[str, Any]:
    agent_name = task["agent_name"]
    agent_cls = AGENT_REGISTRY[agent_name]
    output = agent_cls()(task)
    update: dict[str, Any] = {"agent_outputs": [output]}
    if output["error"]:
        update["errors"] = [f"{agent_name}: {output['error']}"]
    return update


def report_agent_node(state: MainState) -> dict[str, Any]:
    selected_order = {
        name: index
        for index, name in enumerate(state.get("selected_agents", []))
    }
    outputs = sorted(
        state.get("agent_outputs", []),
        key=lambda output: selected_order.get(output["agent_name"], 999),
    )
    report_state = dict(state)
    report_state["agent_outputs"] = outputs
    report = ReportAgent().run(report_state)
    agent_results = {
        output["agent_name"]: output["content"]
        for output in outputs
        if not output["error"]
    }

    global_context = dict(state.get("global_context", {}))
    model_config = dict(global_context.get("model_config", {}))
    model_config["api_key"] = ""
    global_context["model_config"] = model_config
    return {
        "agent_outputs": [],
        "agent_results": agent_results,
        "final_report": report,
        "global_context": global_context,
    }


def build_workflow():
    if not langgraph_available():
        raise RuntimeError(
            "LangGraph is not installed. Install langgraph to run the graph workflow."
        )

    graph = StateGraph(MainState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("route", route_node)
    graph.add_node("run_agent", run_agent_node)
    graph.add_node("report_agent", report_agent_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "route")
    graph.add_conditional_edges(
        "route",
        dispatch_agents,
        ["run_agent"],
    )
    graph.add_edge("run_agent", "report_agent")
    graph.add_edge("report_agent", END)
    return graph.compile()


def create_pipeline_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
) -> MainState:
    state = create_state(query)
    state["global_context"] = {
        "top_k": top_k,
        "data_dir": data_dir,
        "model_config": model_config or {},
    }
    return state


def run_graph_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
) -> MainState:
    return build_workflow().invoke(
        create_pipeline_state(query, top_k, data_dir, model_config)
    )


def _task_for_agent(state: MainState, agent_name: str) -> AgentTaskState:
    return {
        "agent_name": agent_name,
        "query": state.get("query", ""),
        "papers": list(state.get("retrieved_papers", [])),
        "model_config": dict(
            state.get("global_context", {}).get("model_config", {})
        ),
    }


def run_compatible_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
) -> MainState:
    state = create_pipeline_state(query, top_k, data_dir, model_config)
    state.update(retrieve_node(state))
    state.update(route_node(state))

    selected_agents = state.get("selected_agents", [])
    with ThreadPoolExecutor(max_workers=max(len(selected_agents), 1)) as executor:
        futures = [
            executor.submit(
                run_agent_node,
                _task_for_agent(state, agent_name),
            )
            for agent_name in selected_agents
        ]
        for future in futures:
            update = future.result()
            state["agent_outputs"].extend(update.get("agent_outputs", []))
            state["errors"].extend(update.get("errors", []))

    report_update = report_agent_node(state)
    state["agent_results"] = report_update["agent_results"]
    state["final_report"] = report_update["final_report"]
    state["global_context"] = report_update["global_context"]
    return state


def run_pipeline_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
    *,
    require_langgraph: bool = False,
) -> MainState:
    if langgraph_available():
        return run_graph_state(query, top_k, data_dir, model_config)
    if require_langgraph:
        raise RuntimeError(
            "LangGraph is not installed in this environment. Install requirements.txt and rerun."
        )
    return run_compatible_state(query, top_k, data_dir, model_config)


def run_pipeline(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
    *,
    require_langgraph: bool = False,
) -> str:
    return run_pipeline_state(
        query,
        top_k=top_k,
        data_dir=data_dir,
        model_config=model_config,
        require_langgraph=require_langgraph,
    ).get("final_report", "")


def workflow_info() -> dict[str, Any]:
    return {
        "langgraph_available": langgraph_available(),
        "parallel": True,
        "nodes": ["retrieve", "route", "run_agent", "report_agent"],
        "edges": [
            ("START", "retrieve"),
            ("retrieve", "route"),
            ("route", "run_agent[]"),
            ("run_agent[]", "report_agent"),
            ("report_agent", "END"),
        ],
    }

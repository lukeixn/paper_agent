from __future__ import annotations

from typing import Any, Callable

from agent.agent import AGENT_REGISTRY
from aggregator import ReportAggregator
from router import Router
from state.state import MainState, create_state
from vector_store.search import PaperSearchEngine

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - depends on optional runtime package
    END = None
    START = None
    StateGraph = None


AgentNode = Callable[[MainState], MainState]


def langgraph_available() -> bool:
    return StateGraph is not None


def retrieve_node(state: MainState) -> MainState:
    data_dir = state.get("global_context", {}).get("data_dir", "data")
    top_k = int(state.get("global_context", {}).get("top_k", 5))
    search_engine = PaperSearchEngine(data_dir=data_dir)
    state["retrieved_papers"] = search_engine.search(state.get("query", ""), top_k=top_k)
    return state


def route_node(state: MainState) -> MainState:
    decision = Router().route(state.get("query", ""))
    state["route"] = decision.route
    state["selected_agents"] = decision.agents
    state.setdefault("global_context", {})["route_reason"] = decision.reason
    return state


def make_agent_node(agent_name: str) -> AgentNode:
    def node(state: MainState) -> MainState:
        if agent_name not in state.get("selected_agents", []):
            return state

        agent_cls = AGENT_REGISTRY[agent_name]
        return agent_cls()(state)

    node.__name__ = f"{agent_name}_node"
    return node


def aggregate_node(state: MainState) -> MainState:
    state["final_report"] = ReportAggregator().build_report(state)
    model_config = state.get("global_context", {}).get("model_config")
    if isinstance(model_config, dict):
        model_config["api_key"] = ""
    return state


def build_workflow():
    if StateGraph is None or START is None or END is None:
        raise RuntimeError(
            "LangGraph is not installed. Install langgraph to run the graph workflow."
        )

    graph = StateGraph(MainState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("route", route_node)

    agent_order = [
        "survey_agent",
        "innovation_agent",
        "method_agent",
        "limitation_agent",
    ]
    for agent_name in agent_order:
        graph.add_node(agent_name, make_agent_node(agent_name))

    graph.add_node("aggregate", aggregate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "route")
    graph.add_edge("route", "survey_agent")
    graph.add_edge("survey_agent", "innovation_agent")
    graph.add_edge("innovation_agent", "method_agent")
    graph.add_edge("method_agent", "limitation_agent")
    graph.add_edge("limitation_agent", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


def create_pipeline_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
) -> MainState:
    state = create_state(query)
    state["global_context"]["top_k"] = top_k
    state["global_context"]["data_dir"] = data_dir
    state["global_context"]["model_config"] = model_config or {}
    return state


def run_graph_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
) -> MainState:
    state = create_pipeline_state(query, top_k, data_dir, model_config)
    app = build_workflow()
    return app.invoke(state)


def run_compatible_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
) -> MainState:
    state = create_pipeline_state(query, top_k, data_dir, model_config)
    for node in [
        retrieve_node,
        route_node,
        make_agent_node("survey_agent"),
        make_agent_node("innovation_agent"),
        make_agent_node("method_agent"),
        make_agent_node("limitation_agent"),
        aggregate_node,
    ]:
        state = node(state)

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
    state = run_pipeline_state(
        query,
        top_k=top_k,
        data_dir=data_dir,
        model_config=model_config,
        require_langgraph=require_langgraph,
    )
    return state.get("final_report", "")


def workflow_info() -> dict[str, Any]:
    return {
        "langgraph_available": langgraph_available(),
        "nodes": [
            "retrieve",
            "route",
            "survey_agent",
            "innovation_agent",
            "method_agent",
            "limitation_agent",
            "aggregate",
        ],
        "edges": [
            ("START", "retrieve"),
            ("retrieve", "route"),
            ("route", "survey_agent"),
            ("survey_agent", "innovation_agent"),
            ("innovation_agent", "method_agent"),
            ("method_agent", "limitation_agent"),
            ("limitation_agent", "aggregate"),
            ("aggregate", "END"),
        ],
    }

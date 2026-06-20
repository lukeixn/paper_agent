from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.agent import AGENT_REGISTRY
from aggregator import ReportAgent
from models.langchain_llm import OfflineLLM, get_llm
from router import Router
from state.state import (
    AgentOutput,
    AgentTaskState,
    ConversationMessage,
    MainState,
    create_state,
)
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


MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARACTERS = 12000


def trim_conversation_history(
    history: list[ConversationMessage] | None,
) -> list[ConversationMessage]:
    trimmed: list[ConversationMessage] = []
    used_characters = 0
    for message in reversed(list(history or [])):
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        remaining = MAX_HISTORY_CHARACTERS - used_characters
        if remaining <= 0:
            break
        content = content[-remaining:]
        trimmed.append({"role": role, "content": content})
        used_characters += len(content)
        if len(trimmed) >= MAX_HISTORY_MESSAGES:
            break
    return list(reversed(trimmed))


def format_conversation_context(
    history: list[ConversationMessage] | None,
) -> str:
    labels = {"user": "用户", "assistant": "助手"}
    return "\n\n".join(
        f"{labels[message['role']]}：{message['content']}"
        for message in trim_conversation_history(history)
    )


def _offline_standalone_query(
    query: str,
    history: list[ConversationMessage],
) -> str:
    recent_user_messages = [
        message["content"]
        for message in history
        if message["role"] == "user"
    ]
    if not recent_user_messages:
        return query
    previous_questions = "\n".join(
        f"- {message}" for message in recent_user_messages[-4:]
    )
    return f"相关历史问题：\n{previous_questions}\n当前追问：{query}"


def contextualize_query_node(state: MainState) -> dict[str, Any]:
    query = state.get("query", "").strip()
    history = trim_conversation_history(state.get("conversation_history", []))
    context = format_conversation_context(history)
    if not context:
        return {
            "standalone_query": query,
            "conversation_history": history,
            "conversation_context": "",
        }

    model_config = state.get("global_context", {}).get("model_config", {})
    llm = get_llm(
        provider=model_config.get("provider"),
        api_key=model_config.get("api_key"),
        model_name=model_config.get("model_name"),
        base_url=model_config.get("base_url"),
        temperature=model_config.get("temperature"),
    )
    standalone_query = _offline_standalone_query(query, history)
    if not isinstance(llm, OfflineLLM):
        prompt = f"""
你负责理解连续研究对话。请结合历史对话，把“当前追问”改写为一个语义完整、
可独立用于论文检索和任务路由的问题。

历史对话：
{context}

当前追问：
{query}

只输出改写后的独立问题，不要解释，不要回答问题。
"""
        try:
            rewritten = llm.invoke(prompt).content.strip()
            if rewritten:
                standalone_query = rewritten
        except Exception:
            pass

    return {
        "standalone_query": standalone_query,
        "conversation_history": history,
        "conversation_context": context,
    }


def retrieve_node(state: MainState) -> dict[str, Any]:
    global_context = state.get("global_context", {})
    search_engine = PaperSearchEngine(
        data_dir=global_context.get("data_dir", "data")
    )
    papers = search_engine.search(
        state.get("standalone_query") or state.get("query", ""),
        top_k=int(global_context.get("top_k", 5)),
    )
    return {"retrieved_papers": papers}


def route_node(state: MainState) -> dict[str, Any]:
    decision = Router().route(
        state.get("standalone_query") or state.get("query", "")
    )
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
                "query": state.get("standalone_query")
                or state.get("query", ""),
                "user_query": state.get("query", ""),
                "conversation_context": state.get(
                    "conversation_context", ""
                ),
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
    graph.add_node("contextualize", contextualize_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("route", route_node)
    graph.add_node("run_agent", run_agent_node)
    graph.add_node("report_agent", report_agent_node)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "retrieve")
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
    conversation_history: list[ConversationMessage] | None = None,
) -> MainState:
    state = create_state(query, conversation_history)
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
    conversation_history: list[ConversationMessage] | None = None,
) -> MainState:
    return build_workflow().invoke(
        create_pipeline_state(
            query,
            top_k,
            data_dir,
            model_config,
            conversation_history,
        )
    )


def _task_for_agent(state: MainState, agent_name: str) -> AgentTaskState:
    return {
        "agent_name": agent_name,
        "query": state.get("standalone_query") or state.get("query", ""),
        "user_query": state.get("query", ""),
        "conversation_context": state.get("conversation_context", ""),
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
    conversation_history: list[ConversationMessage] | None = None,
) -> MainState:
    state = create_pipeline_state(
        query,
        top_k,
        data_dir,
        model_config,
        conversation_history,
    )
    state.update(contextualize_query_node(state))
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
    conversation_history: list[ConversationMessage] | None = None,
) -> MainState:
    if langgraph_available():
        return run_graph_state(
            query,
            top_k,
            data_dir,
            model_config,
            conversation_history,
        )
    if require_langgraph:
        raise RuntimeError(
            "LangGraph is not installed in this environment. Install requirements.txt and rerun."
        )
    return run_compatible_state(
        query,
        top_k,
        data_dir,
        model_config,
        conversation_history,
    )


def run_pipeline(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
    *,
    require_langgraph: bool = False,
    conversation_history: list[ConversationMessage] | None = None,
) -> str:
    return run_pipeline_state(
        query,
        top_k=top_k,
        data_dir=data_dir,
        model_config=model_config,
        require_langgraph=require_langgraph,
        conversation_history=conversation_history,
    ).get("final_report", "")


def workflow_info() -> dict[str, Any]:
    return {
        "langgraph_available": langgraph_available(),
        "parallel": True,
        "nodes": [
            "contextualize",
            "retrieve",
            "route",
            "run_agent",
            "report_agent",
        ],
        "edges": [
            ("START", "contextualize"),
            ("contextualize", "retrieve"),
            ("retrieve", "route"),
            ("route", "run_agent[]"),
            ("run_agent[]", "report_agent"),
            ("report_agent", "END"),
        ],
    }

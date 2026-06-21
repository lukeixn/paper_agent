from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

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


def extract_user_question_history(
    history: list[ConversationMessage] | None,
) -> list[str]:
    questions: list[str] = []
    for message in list(history or []):
        content = str(message.get("content", "")).strip()
        if message.get("role") == "user" and content:
            questions.append(content)
    return questions


def format_user_question_history(
    questions: list[str],
) -> str:
    return "\n\n".join(
        f"{index}. {question}"
        for index, question in enumerate(questions, start=1)
    )


def _offline_standalone_query(
    query: str,
    user_question_history: list[str],
) -> str:
    if not user_question_history:
        return query
    previous_questions = "\n".join(
        f"- {message}" for message in user_question_history
    )
    return f"相关历史问题：\n{previous_questions}\n当前追问：{query}"


def contextualize_query_node(state: MainState) -> dict[str, Any]:
    query = state.get("query", "").strip()
    user_question_history = extract_user_question_history(
        state.get("conversation_history", [])
    )
    history_text = format_user_question_history(user_question_history)
    if not user_question_history:
        return {
            "standalone_query": query,
            "user_question_history": [],
        }

    model_config = state.get("global_context", {}).get("model_config", {})
    llm = get_llm(
        provider=model_config.get("provider"),
        api_key=model_config.get("api_key"),
        model_name=model_config.get("model_name"),
        base_url=model_config.get("base_url"),
        temperature=model_config.get("temperature"),
    )
    standalone_query = _offline_standalone_query(
        query,
        user_question_history,
    )
    if not isinstance(llm, OfflineLLM):
        prompt = f"""
你负责理解同一会话中的连续研究问题。

以下内容只有历史用户问题，不包含也不依赖任何助手回答：
{history_text}

当前追问：
{query}

强制要求：
1. 当前追问是本轮唯一要解决的问题，优先级最高。
2. 必须利用历史用户问题解析省略的研究对象、代词、比较对象、范围和约束。
3. 不要把历史问题逐条回答，不要擅自改变当前追问的意图。
4. 将当前追问改写成语义完整、可独立用于论文检索和 Agent 路由的问题。
5. 只输出改写后的独立问题，不要解释，不要回答问题。
"""
        try:
            rewritten = llm.invoke(prompt).content.strip()
            if rewritten:
                standalone_query = rewritten
        except Exception:
            pass

    return {
        "standalone_query": standalone_query,
        "user_question_history": user_question_history,
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
    # Routing follows the current request so historical topics cannot override
    # which specialists are needed for this turn.
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
                "query": state.get("standalone_query")
                or state.get("query", ""),
                "user_query": state.get("query", ""),
                "user_question_history": list(
                    state.get("user_question_history", [])
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


def _merge_graph_update(
    state: MainState,
    update: dict[str, Any],
) -> None:
    for key, value in update.items():
        if key in {"agent_outputs", "errors"}:
            state.setdefault(key, []).extend(value)
        else:
            state[key] = value


def stream_graph_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
    conversation_history: list[ConversationMessage] | None = None,
) -> Iterator[dict[str, Any]]:
    initial_state = create_pipeline_state(
        query,
        top_k,
        data_dir,
        model_config,
        conversation_history,
    )
    accumulated_state = dict(initial_state)
    yield {
        "node": "contextualize",
        "status": "running",
        "state": dict(accumulated_state),
    }

    for graph_update in build_workflow().stream(
        initial_state,
        stream_mode="updates",
    ):
        for node, update in graph_update.items():
            _merge_graph_update(accumulated_state, update)
            event: dict[str, Any] = {
                "node": node,
                "status": "completed",
                "state": dict(accumulated_state),
            }
            if node == "run_agent":
                outputs = update.get("agent_outputs", [])
                if outputs:
                    event["agent_name"] = outputs[0]["agent_name"]
                    event["error"] = outputs[0]["error"]
            yield event

    yield {
        "node": "end",
        "status": "completed",
        "state": dict(accumulated_state),
    }


def _task_for_agent(state: MainState, agent_name: str) -> AgentTaskState:
    return {
        "agent_name": agent_name,
        "query": state.get("standalone_query") or state.get("query", ""),
        "user_query": state.get("query", ""),
        "user_question_history": list(
            state.get("user_question_history", [])
        ),
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


def stream_pipeline_state(
    query: str,
    top_k: int = 5,
    data_dir: str = "data",
    model_config: dict[str, Any] | None = None,
    *,
    require_langgraph: bool = False,
    conversation_history: list[ConversationMessage] | None = None,
) -> Iterator[dict[str, Any]]:
    if langgraph_available():
        yield from stream_graph_state(
            query,
            top_k,
            data_dir,
            model_config,
            conversation_history,
        )
        return
    if require_langgraph:
        raise RuntimeError(
            "LangGraph is not installed in this environment. Install requirements.txt and rerun."
        )
    state = run_compatible_state(
        query,
        top_k,
        data_dir,
        model_config,
        conversation_history,
    )
    yield {
        "node": "end",
        "status": "completed",
        "state": state,
    }


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

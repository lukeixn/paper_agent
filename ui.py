from __future__ import annotations

import base64
import html
import importlib
import inspect
import json
import os
from typing import Any
from urllib.parse import urlencode

import streamlit as st

import workflow
from academic_search import AcademicSearchService
from configs.config import cfg
from paper_library import PaperLibrary
from skill_library import AgentSkillLibrary


AGENT_LABELS = {
    "survey_agent": "研究综述",
    "innovation_agent": "创新分析",
    "method_agent": "方法比较",
    "limitation_agent": "局限与机会",
}

WORKFLOW_LABELS = {
    "contextualize": ("理解问题", "结合连续对话重写追问"),
    "retrieve": ("检索论文", "从本地论文库选取证据"),
    "route": ("任务路由", "选择需要参与的研究 Agent"),
    "report_agent": ("报告汇总", "综合各 Agent 结果生成回答"),
}

TOPOLOGY_SVG_STYLE = """
.topo-boundary{fill:rgba(8,25,47,.48);stroke:#52759b;stroke-width:1;stroke-dasharray:4 5}
.topo-section,.topo-foot{fill:#6689af;font-family:Consolas,monospace;font-size:9px;letter-spacing:1.5px;text-anchor:middle}
.topo-foot{fill:#42678e;font-size:8px}
.topo-empty{fill:#42678e;font-family:Consolas,monospace;font-size:9px;letter-spacing:1.2px;text-anchor:middle}
.topo-link{fill:none;stroke:#31577f;stroke-width:1.4;stroke-dasharray:4 5;marker-end:url(#topology-arrow)}
.topo-link.active{stroke:#5ab3ff;stroke-width:1.8;stroke-dasharray:7 6;filter:url(#topology-glow);animation:flow 1.1s linear infinite}
.topo-link.branch,.topo-link.merge{marker-end:none}
.topo-link.disabled{opacity:.15}
.topo-link.error{stroke:#e16d91;opacity:.8}
.topo-disc{fill:#102947;stroke:#426f9f;stroke-width:1.3}
.topo-halo{fill:none;stroke:transparent;stroke-width:2}
.topo-code,.topo-router-title,.topo-router-sub,.topo-label,.topo-agent-label{text-anchor:middle;dominant-baseline:middle;font-family:Consolas,"Microsoft YaHei",sans-serif}
.topo-code{fill:#8ab4df;font-size:11px;font-weight:700}
.topo-label,.topo-agent-label{fill:#83a4c5;font-size:10px}
.topo-router-title{fill:#b9d9f7;font-size:11px;font-weight:700;letter-spacing:.8px}
.topo-router-sub{fill:#6f96bd;font-size:7px;letter-spacing:.6px}
.completed .topo-disc{fill:#123a65;stroke:#55a8f5}
.running .topo-disc{fill:#154775;stroke:#7cc5ff;stroke-width:2;filter:url(#topology-glow)}
.running .topo-halo{stroke:#4da9f7;opacity:.7;animation:pulse 1.25s ease-out infinite}
.running text,.completed text{fill:#d9efff}
.topo-agent.disabled{opacity:.25}
.topo-agent.clickable{cursor:pointer}
.topo-agent.clickable:hover .topo-disc{stroke:#a8d8ff;stroke-width:2.4;filter:url(#topology-glow)}
.error .topo-disc{fill:#55243a;stroke:#e16d91}
@keyframes flow{to{stroke-dashoffset:-26}}
@keyframes pulse{0%{opacity:.75}100%{opacity:0}}
"""


def current_workflow():
    global workflow
    parameters = inspect.signature(
        workflow.run_pipeline_state
    ).parameters
    if (
        "conversation_history" not in parameters
        or not hasattr(workflow, "stream_pipeline_state")
    ):
        workflow = importlib.reload(workflow)
    return workflow


def conversation_title(query: str, max_length: int = 24) -> str:
    title = " ".join(query.strip().split())
    if len(title) <= max_length:
        return title
    return title[: max_length - 1] + "…"


def initialize_conversations() -> None:
    if "conversation_sessions" in st.session_state:
        return

    legacy_messages = list(st.session_state.pop("chat_messages", []))
    legacy_state = st.session_state.pop("analysis_state", None)
    title = (
        conversation_title(legacy_messages[0]["content"])
        if legacy_messages
        else "新会话"
    )
    st.session_state["conversation_sessions"] = {
        "conversation-1": {
            "title": title,
            "messages": legacy_messages,
            "analysis_state": legacy_state,
        }
    }
    st.session_state["active_conversation_id"] = "conversation-1"
    st.session_state["next_conversation_number"] = 2


def active_conversation() -> dict[str, Any]:
    initialize_conversations()
    sessions = st.session_state["conversation_sessions"]
    active_id = st.session_state["active_conversation_id"]
    return sessions[active_id]


def create_conversation() -> str:
    initialize_conversations()
    current = active_conversation()
    if not current["messages"]:
        return st.session_state["active_conversation_id"]

    number = st.session_state["next_conversation_number"]
    conversation_id = f"conversation-{number}"
    st.session_state["next_conversation_number"] = number + 1
    st.session_state["conversation_sessions"][conversation_id] = {
        "title": "新会话",
        "messages": [],
        "analysis_state": None,
    }
    st.session_state["active_conversation_id"] = conversation_id
    return conversation_id


def render_conversation_sidebar() -> None:
    initialize_conversations()
    st.sidebar.subheader("会话")
    if st.sidebar.button(
        "新建会话",
        key="new_conversation",
        width="stretch",
    ):
        create_conversation()
        st.rerun()

    sessions = st.session_state["conversation_sessions"]
    active_id = st.session_state["active_conversation_id"]
    for conversation_id, conversation in reversed(list(sessions.items())):
        is_active = conversation_id == active_id
        label = conversation["title"]
        if st.sidebar.button(
            label,
            key=f"open_{conversation_id}",
            type="primary" if is_active else "secondary",
            width="stretch",
        ):
            if not is_active:
                st.session_state[
                    "active_conversation_id"
                ] = conversation_id
                st.rerun()


def _topology_status(
    status: str,
) -> str:
    return status if status in {
        "pending",
        "running",
        "completed",
        "error",
        "disabled",
    } else "pending"


def topology_agent_layout(
    selected_agents: list[str],
) -> list[tuple[str, int, int]]:
    count = len(selected_agents)
    x_positions = {
        1: [210],
        2: [140, 280],
        3: [100, 210, 320],
        4: [72, 164, 256, 348],
    }.get(count, [])
    return [
        (agent_name, x_positions[index], 390)
        for index, agent_name in enumerate(selected_agents)
    ]


def agent_output_payload(
    agent_outputs: list[dict[str, Any]] | None,
) -> dict[str, dict[str, str]]:
    return {
        output["agent_name"]: {
            "title": str(
                output.get(
                    "title",
                    AGENT_LABELS.get(
                        output["agent_name"],
                        output["agent_name"],
                    ),
                )
            ),
            "content": str(output.get("content", "")),
            "error": str(output.get("error", "")),
        }
        for output in (agent_outputs or [])
    }


def render_workflow_diagram(
    placeholder,
    *,
    node_status: dict[str, str] | None = None,
    agent_status: dict[str, str] | None = None,
    selected_agents: list[str] | None = None,
    paper_count: int = 0,
    agent_outputs: list[dict[str, Any]] | None = None,
) -> None:
    node_status = node_status or {}
    agent_status = agent_status or {}
    selected_agents = selected_agents or []
    output_map = agent_output_payload(agent_outputs)
    selected_count = len(selected_agents)
    context_status = _topology_status(
        node_status.get("contextualize", "pending")
    )
    retrieve_status = _topology_status(
        node_status.get("retrieve", "pending")
    )
    route_status = _topology_status(node_status.get("route", "pending"))
    report_status = _topology_status(
        node_status.get("report_agent", "pending")
    )
    context_line = (
        "active" if context_status == "completed" else "pending"
    )
    retrieve_line = (
        "active" if retrieve_status == "completed" else "pending"
    )
    route_line = "active" if route_status == "completed" else "pending"

    agent_nodes = []
    branch_links = []
    merge_links = []
    for index, (agent_name, x, y) in enumerate(
        topology_agent_layout(selected_agents),
        start=1,
    ):
        label = AGENT_LABELS[agent_name]
        status = agent_status.get(agent_name, "pending")
        merge_status = (
            "error"
            if status == "error"
            else "active"
            if status == "completed"
            else "pending"
        )
        clickable = (
            "clickable"
            if agent_name in output_map
            else ""
        )
        click_handler = (
            f'onclick="openAgent({html.escape(json.dumps(agent_name))})"'
            if clickable
            else ""
        )
        agent_nodes.append(
            f"""
            <g class="topo-agent {_topology_status(status)} {clickable}"
               {click_handler}>
                <circle class="topo-halo" cx="{x}" cy="{y}" r="31"></circle>
                <circle class="topo-disc" cx="{x}" cy="{y}" r="23"></circle>
                <text class="topo-code" x="{x}" y="{y + 3}">A{index}</text>
                <text class="topo-agent-label" x="{x}" y="{y + 43}">
                    {html.escape(label)}
                </text>
            </g>
            """
        )
        branch_links.append(
            f"""
            <path class="topo-link branch {route_line}"
                  d="M210 296 C210 325 {x} 330 {x} {y - 28}"></path>
            """
        )
        merge_links.append(
            f"""
            <path class="topo-link merge {merge_status}"
                  d="M{x} {y + 28} C{x} 470 210 475 210 505"></path>
            """
        )

    running_names = [
        WORKFLOW_LABELS[name][0]
        for name, status in node_status.items()
        if status == "running" and name in WORKFLOW_LABELS
    ]
    running_agents = [
        AGENT_LABELS[name]
        for name, status in agent_status.items()
        if status == "running" and name in AGENT_LABELS
    ]
    if running_agents:
        live_text = f"{len(running_agents)} AGENTS ACTIVE"
    elif running_names:
        live_text = running_names[0]
    elif node_status.get("report_agent") == "completed":
        live_text = "TRACE COMPLETE"
    else:
        live_text = "SYSTEM READY"

    empty_agent_state = (
        ""
        if selected_agents
        else """
        <text class="topo-empty" x="210" y="395">
            WAITING FOR ROUTER
        </text>
        """
    )

    panel_markup = f"""
        <div class="workflow-panel">
            <div class="topology-header">
                <div>
                    <span class="topology-eyebrow">LANGGRAPH TOPOLOGY</span>
                    <div class="workflow-heading">实时执行图</div>
                    <div class="topology-hint">
                        点击已完成 Agent 查看原始输出
                    </div>
                </div>
                <div class="topology-live"><i></i>{html.escape(live_text)}</div>
            </div>
            <svg xmlns="http://www.w3.org/2000/svg"
                 class="topology-map" viewBox="0 0 420 610"
                 role="img" aria-label="LangGraph 多 Agent 实时执行拓扑">
                <style>{TOPOLOGY_SVG_STYLE}</style>
                <defs>
                    <pattern id="topology-grid" width="24" height="24"
                             patternUnits="userSpaceOnUse">
                        <path d="M 24 0 L 0 0 0 24"
                              fill="none" stroke="#173254"
                              stroke-width=".7"></path>
                    </pattern>
                    <filter id="topology-glow" x="-80%" y="-80%"
                            width="260%" height="260%">
                        <feGaussianBlur stdDeviation="4"
                                        result="blur"></feGaussianBlur>
                        <feMerge>
                            <feMergeNode in="blur"></feMergeNode>
                            <feMergeNode in="SourceGraphic"></feMergeNode>
                        </feMerge>
                    </filter>
                    <marker id="topology-arrow" markerWidth="7"
                            markerHeight="7" refX="5" refY="3.5"
                            orient="auto">
                        <path d="M0,0 L0,7 L6,3.5 z"
                              fill="#3f70a6"></path>
                    </marker>
                </defs>
                <rect width="420" height="610"
                      fill="url(#topology-grid)"></rect>

                <text class="topo-section" x="210" y="25">QUERY PIPELINE</text>
                <rect class="topo-boundary" x="132" y="38"
                      width="156" height="155" rx="18"></rect>

                <path class="topo-link {context_line}"
                      d="M210 94 L210 125"></path>
                <path class="topo-link {retrieve_line}"
                      d="M210 168 L210 220"></path>

                <g class="topo-core {context_status}">
                    <circle class="topo-halo" cx="210" cy="72" r="28"></circle>
                    <circle class="topo-disc" cx="210" cy="72" r="22"></circle>
                    <text class="topo-code" x="210" y="76">Q</text>
                    <text class="topo-label" x="210" y="108">理解问题</text>
                </g>

                <g class="topo-core {retrieve_status}">
                    <circle class="topo-halo" cx="210" cy="147" r="27"></circle>
                    <circle class="topo-disc" cx="210" cy="147" r="21"></circle>
                    <text class="topo-code" x="210" y="151">R</text>
                    <text class="topo-label" x="210" y="184">
                        检索 {paper_count:02d} 篇
                    </text>
                </g>

                <g class="topo-router {route_status}">
                    <circle class="topo-halo" cx="210" cy="258" r="49"></circle>
                    <circle class="topo-disc" cx="210" cy="258" r="39"></circle>
                    <text class="topo-router-title" x="210" y="256">ROUTER</text>
                    <text class="topo-router-sub" x="210" y="274">
                        {selected_count:02d} AGENTS
                    </text>
                </g>

                <rect class="topo-boundary" x="35" y="317"
                      width="350" height="150" rx="20"></rect>
                <text class="topo-section" x="210" y="307">
                    PARALLEL AGENTS
                </text>

                {''.join(branch_links)}
                {''.join(agent_nodes)}
                {empty_agent_state}
                {''.join(merge_links)}

                <g class="topo-report {report_status}">
                    <circle class="topo-halo" cx="210" cy="545" r="42"></circle>
                    <circle class="topo-disc" cx="210" cy="545" r="33"></circle>
                    <text class="topo-router-title" x="210" y="543">REPORT</text>
                    <text class="topo-router-sub" x="210" y="561">SYNTHESIS</text>
                </g>

                <text class="topo-foot" x="210" y="600">
                    STATE STREAM / LANGGRAPH CONNECTED
                </text>
            </svg>
            <div class="agent-modal" id="agent-modal" aria-hidden="true">
                <div class="agent-modal-backdrop" onclick="closeAgent()"></div>
                <section class="agent-modal-dialog">
                    <header>
                        <div>
                            <span>AGENT RAW OUTPUT</span>
                            <h3 id="agent-modal-title"></h3>
                        </div>
                        <button type="button" onclick="closeAgent()"
                                aria-label="关闭">×</button>
                    </header>
                    <pre id="agent-modal-content"></pre>
                </section>
            </div>
        </div>
        """
    outputs_json = json.dumps(
        output_map,
        ensure_ascii=False,
    ).replace("</", "<\\/")
    component_markup = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
        *{{box-sizing:border-box}}
        body{{margin:0;background:transparent;font-family:"Microsoft YaHei",sans-serif}}
        .workflow-panel{{position:relative;overflow:hidden;border:1px solid #17375d;border-radius:8px;padding:.9rem .85rem .45rem;background:#071426;color:#dcecff;box-shadow:0 18px 38px rgba(7,25,48,.18),inset 0 1px 0 rgba(129,190,255,.05)}}
        .topology-header{{display:flex;align-items:flex-start;justify-content:space-between;gap:.75rem;padding:0 .15rem .55rem;border-bottom:1px solid #15345a}}
        .topology-eyebrow{{display:block;color:#5c83ad;font-family:Consolas,monospace;font-size:.62rem;letter-spacing:.08rem;margin-bottom:.2rem}}
        .workflow-heading{{color:#edf6ff;font-size:1.05rem;font-weight:680}}
        .topology-hint{{color:#567ca2;font-size:.64rem;margin-top:.2rem}}
        .topology-live{{display:flex;align-items:center;gap:.35rem;color:#7295b9;font-family:Consolas,monospace;font-size:.6rem;white-space:nowrap}}
        .topology-live i{{width:.42rem;height:.42rem;border-radius:50%;background:#54a8ff;box-shadow:0 0 10px rgba(84,168,255,.85)}}
        .topology-map{{display:block;width:100%;height:auto;aspect-ratio:420/610;object-fit:contain;margin-top:.35rem}}
        .agent-modal{{position:absolute;inset:0;display:none;z-index:20}}
        .agent-modal.open{{display:block}}
        .agent-modal-backdrop{{position:absolute;inset:0;background:rgba(2,10,23,.76);backdrop-filter:blur(3px)}}
        .agent-modal-dialog{{position:absolute;inset:8% 6%;display:flex;flex-direction:column;overflow:hidden;border:1px solid #2d6091;border-radius:8px;background:#091a30;box-shadow:0 20px 50px rgba(0,0,0,.45)}}
        .agent-modal-dialog header{{display:flex;align-items:flex-start;justify-content:space-between;padding:.9rem 1rem;border-bottom:1px solid #183b62}}
        .agent-modal-dialog header span{{color:#6293bd;font-family:Consolas,monospace;font-size:.62rem;letter-spacing:.08rem}}
        .agent-modal-dialog h3{{margin:.18rem 0 0;color:#e3f2ff;font-size:1rem}}
        .agent-modal-dialog button{{width:2rem;height:2rem;border:0;background:transparent;color:#8eb7db;font-size:1.5rem;cursor:pointer}}
        .agent-modal-dialog pre{{flex:1;overflow:auto;margin:0;padding:1rem;color:#c9dff2;font-family:"Microsoft YaHei",sans-serif;font-size:.76rem;line-height:1.7;white-space:pre-wrap;word-break:break-word}}
        </style>
    </head>
    <body>
        {panel_markup}
        <script>
        const agentOutputs = {outputs_json};
        function openAgent(name) {{
            const output = agentOutputs[name];
            if (!output) return;
            document.getElementById("agent-modal-title").textContent =
                output.title || name;
            document.getElementById("agent-modal-content").textContent =
                output.error ? "执行失败：\\n" + output.error : output.content;
            const modal = document.getElementById("agent-modal");
            modal.classList.add("open");
            modal.setAttribute("aria-hidden", "false");
        }}
        function closeAgent() {{
            const modal = document.getElementById("agent-modal");
            modal.classList.remove("open");
            modal.setAttribute("aria-hidden", "true");
        }}
        document.addEventListener("keydown", (event) => {{
            if (event.key === "Escape") closeAgent();
        }});
        </script>
    </body>
    </html>
    """
    with placeholder.container():
        st.markdown(
            '<div class="workflow-sticky-marker"></div>',
            unsafe_allow_html=True,
        )
        st.iframe(
            component_markup,
            height=720,
            width="stretch",
        )


def completed_workflow_progress(
    state: dict[str, Any] | None,
) -> tuple[
    dict[str, str],
    dict[str, str],
    list[str],
    int,
    list[dict[str, Any]],
]:
    if not state or not state.get("final_report"):
        return {}, {}, [], 0, []
    selected_agents = state.get("selected_agents", [])
    node_status = {
        name: "completed" for name in WORKFLOW_LABELS
    }
    agent_status = {
        name: "completed"
        for name in selected_agents
        if name in AGENT_LABELS
    }
    return (
        node_status,
        agent_status,
        selected_agents,
        len(state.get("retrieved_papers", [])),
        list(state.get("agent_outputs", [])),
    )


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { color: #1e2926; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            border-right: 1px solid #dde3df;
        }
        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 720;
            line-height: 1.15;
            margin: 0;
        }
        .app-subtitle {
            color: #60706a;
            margin: .4rem 0 1.6rem;
        }
        .status-line {
            display: flex;
            gap: .65rem;
            align-items: center;
            color: #53635d;
            font-size: .9rem;
            margin-bottom: 1rem;
        }
        .status-dot {
            width: .55rem;
            height: .55rem;
            border-radius: 50%;
            background: #147d64;
            display: inline-block;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #dde3df;
            border-radius: 6px;
            padding: .8rem 1rem;
            background: #fff;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #dde3df;
            border-radius: 6px;
            background: #fff;
        }
        .paper-score {
            color: #147d64;
            font-weight: 650;
        }
        div[data-testid="stHorizontalBlock"]:has(.workflow-sticky-marker) {
            align-items: flex-start;
            overflow: visible;
        }
        div[data-testid="stColumn"]:has(.workflow-sticky-marker) {
            position: sticky;
            top: 4rem;
            z-index: 3;
            align-self: flex-start;
            height: fit-content;
            overflow: visible;
        }
        div[data-testid="stColumn"]:has(.workflow-sticky-marker)
        > div[data-testid="stVerticalBlock"] {
            overflow: visible;
        }
        .workflow-panel {
            position: relative;
            overflow: hidden;
            border: 1px solid #17375d;
            border-radius: 8px;
            padding: .9rem .85rem .45rem;
            background: #071426;
            color: #dcecff;
            box-shadow:
                0 18px 38px rgba(7, 25, 48, .18),
                inset 0 1px 0 rgba(129, 190, 255, .05);
        }
        .topology-header {
            position: relative;
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: .75rem;
            padding: 0 .15rem .55rem;
            border-bottom: 1px solid #15345a;
        }
        .topology-eyebrow {
            display: block;
            color: #5c83ad;
            font-family: "Consolas", monospace;
            font-size: .62rem;
            letter-spacing: .08rem;
            margin-bottom: .2rem;
        }
        .workflow-heading {
            color: #edf6ff;
            font-size: 1.05rem;
            font-weight: 680;
        }
        .topology-live {
            display: flex;
            align-items: center;
            gap: .35rem;
            color: #7295b9;
            font-family: "Consolas", monospace;
            font-size: .6rem;
            white-space: nowrap;
        }
        .topology-live i {
            width: .42rem;
            height: .42rem;
            display: inline-block;
            border-radius: 50%;
            background: #54a8ff;
            box-shadow: 0 0 10px rgba(84, 168, 255, .85);
        }
        .topology-image {
            display: block;
            width: 100%;
            height: auto;
            aspect-ratio: 420 / 610;
            object-fit: contain;
            max-height: calc(100vh - 12rem);
            margin-top: .35rem;
        }
        .topo-boundary {
            fill: rgba(8, 25, 47, .48);
            stroke: #52759b;
            stroke-width: 1;
            stroke-dasharray: 4 5;
        }
        .topo-section,
        .topo-foot {
            fill: #6689af;
            font-family: "Consolas", monospace;
            font-size: 9px;
            letter-spacing: 1.5px;
            text-anchor: middle;
        }
        .topo-foot {
            fill: #42678e;
            font-size: 8px;
        }
        .topo-link {
            fill: none;
            stroke: #31577f;
            stroke-width: 1.4;
            stroke-dasharray: 4 5;
            marker-end: url(#topology-arrow);
        }
        .topo-link.active {
            stroke: #5ab3ff;
            stroke-width: 1.8;
            stroke-dasharray: 7 6;
            filter: url(#topology-glow);
            animation: topology-flow 1.1s linear infinite;
        }
        .topo-link.branch,
        .topo-link.merge {
            marker-end: none;
        }
        .topo-link.disabled {
            opacity: .15;
        }
        .topo-disc {
            fill: #102947;
            stroke: #426f9f;
            stroke-width: 1.3;
        }
        .topo-halo {
            fill: none;
            stroke: transparent;
            stroke-width: 2;
        }
        .topo-code,
        .topo-router-title,
        .topo-router-sub,
        .topo-label,
        .topo-agent-label {
            text-anchor: middle;
            dominant-baseline: middle;
            font-family: "Consolas", "Microsoft YaHei", sans-serif;
        }
        .topo-code {
            fill: #8ab4df;
            font-size: 11px;
            font-weight: 700;
        }
        .topo-label,
        .topo-agent-label {
            fill: #83a4c5;
            font-size: 10px;
        }
        .topo-router-title {
            fill: #b9d9f7;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .8px;
        }
        .topo-router-sub {
            fill: #6f96bd;
            font-size: 7px;
            letter-spacing: .6px;
        }
        .topo-core.completed .topo-disc,
        .topo-router.completed .topo-disc,
        .topo-agent.completed .topo-disc,
        .topo-report.completed .topo-disc {
            fill: #123a65;
            stroke: #55a8f5;
        }
        .topo-core.running .topo-disc,
        .topo-router.running .topo-disc,
        .topo-agent.running .topo-disc,
        .topo-report.running .topo-disc {
            fill: #154775;
            stroke: #7cc5ff;
            stroke-width: 2;
            filter: url(#topology-glow);
        }
        .topo-core.running .topo-halo,
        .topo-router.running .topo-halo,
        .topo-agent.running .topo-halo,
        .topo-report.running .topo-halo {
            stroke: #4da9f7;
            opacity: .7;
            animation: topology-pulse 1.25s ease-out infinite;
        }
        .topo-core.running text,
        .topo-router.running text,
        .topo-agent.running text,
        .topo-report.running text,
        .topo-core.completed text,
        .topo-router.completed text,
        .topo-agent.completed text,
        .topo-report.completed text {
            fill: #d9efff;
        }
        .topo-agent.disabled {
            opacity: .25;
        }
        .topo-core.error .topo-disc,
        .topo-router.error .topo-disc,
        .topo-agent.error .topo-disc,
        .topo-report.error .topo-disc {
            fill: #55243a;
            stroke: #e16d91;
        }
        @keyframes topology-flow {
            to { stroke-dashoffset: -26; }
        }
        @keyframes topology-pulse {
            0% { opacity: .75; transform: scale(.88); transform-origin: center; }
            100% { opacity: 0; transform: scale(1.18); transform-origin: center; }
        }
        @media (max-width: 900px) {
            div[data-testid="stColumn"]:has(.workflow-sticky-marker) {
                position: static;
                top: auto;
            }
            .topology-image {
                max-height: 34rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_model(provider: str) -> str:
    provider_key = provider.upper()
    return cfg["MODEL"].get(provider_key, {}).get("MODEL_NAME", "")


def model_settings() -> dict[str, Any]:
    st.sidebar.subheader("模型连接")
    provider = st.sidebar.selectbox(
        "服务商",
        ["deepseek", "openai", "offline"],
        format_func=lambda value: {
            "deepseek": "DeepSeek",
            "openai": "OpenAI",
            "offline": "离线模式",
        }[value],
    )

    api_key = ""
    model_name = ""
    base_url = ""
    if provider != "offline":
        api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            placeholder="仅在当前会话中使用",
        )
        model_name = st.sidebar.text_input(
            "模型",
            value=default_model(provider),
        )
        if provider == "deepseek":
            base_url = st.sidebar.text_input(
                "API 地址",
                value=cfg["MODEL"]["DEEPSEEK"]["BASE_URL"],
            )

    temperature = st.sidebar.slider(
        "温度",
        min_value=0.0,
        max_value=1.5,
        value=float(cfg["MODEL"].get("TEMPERATURE", 0.3)),
        step=0.1,
    )
    top_k = st.sidebar.slider("检索论文数", 1, 30, 10)

    return {
        "provider": provider,
        "api_key": api_key.strip(),
        "model_name": model_name.strip(),
        "base_url": base_url.strip(),
        "temperature": temperature,
        "top_k": top_k,
    }


def render_papers(state: dict[str, Any]) -> None:
    papers = state.get("retrieved_papers", [])
    if not papers:
        st.info("没有检索到相关论文。")
        return

    library = PaperLibrary()
    for index, paper in enumerate(papers, start=1):
        with st.expander(f"{index}. {paper.title}", expanded=index <= 2):
            st.markdown(
                f'<span class="paper-score">相关度 {paper.score:.4f}</span>',
                unsafe_allow_html=True,
            )
            if paper.authors:
                st.caption("作者：" + "、".join(paper.authors))
            st.write(paper.summary or paper.abstract or "暂无摘要。")
            if paper.keywords:
                st.caption("关键词：" + " · ".join(paper.keywords[:8]))
            render_paper_original_access(
                paper,
                library,
                key_prefix=f"retrieved_{index}",
            )


def render_paper_original_access(
    paper: Any,
    library: PaperLibrary,
    *,
    key_prefix: str,
) -> None:
    source_url = getattr(paper, "source_url", "")
    pdf_path = library.pdf_path_for(paper)

    links = []
    if source_url:
        links.append(f"[原始页面]({source_url})")
    if pdf_path:
        links.append(f"本地 PDF：`{pdf_path.name}`")

    if links:
        st.markdown(" / ".join(links))
    else:
        st.caption("未记录原始页面或本地 PDF。")

    if pdf_path:
        preview_key = f"{key_prefix}_pdf_preview_open"
        if st.button(
            "关闭本地 PDF 预览"
            if st.session_state.get(preview_key)
            else "打开本地 PDF",
            key=f"{key_prefix}_open_pdf",
            width="stretch",
        ):
            st.session_state[preview_key] = not st.session_state.get(
                preview_key,
                False,
            )
        if st.session_state.get(preview_key):
            st.iframe(
                pdf_preview_html(pdf_path),
                height=760,
                width="stretch",
            )
        st.caption(
            "如果浏览器无法内嵌预览，请使用下面的下载按钮。"
        )
        st.download_button(
            "下载原文 PDF",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            key=f"{key_prefix}_download_pdf",
            width="stretch",
        )


def pdf_preview_html(path: Any) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"""
    <html>
      <body style="margin:0;background:#101820;">
        <object
          data="data:application/pdf;base64,{data}"
          type="application/pdf"
          width="100%"
          height="740"
        >
          <p style="font-family:sans-serif;color:#f5f7fb;padding:1rem;">
            当前浏览器无法内嵌预览 PDF，请使用下载按钮打开原文。
          </p>
        </object>
      </body>
    </html>
    """


def render_agents(state: dict[str, Any]) -> None:
    results = state.get("agent_results", {})
    if not results:
        st.info("本次问题没有触发分析 Agent。")
        return

    for agent_name, result in results.items():
        st.subheader(AGENT_LABELS.get(agent_name, agent_name))
        st.markdown(result)


def render_result(state: dict[str, Any]) -> None:
    papers = state.get("retrieved_papers", [])
    results = state.get("agent_results", {})
    selected_agents = state.get("selected_agents", [])

    metric_columns = st.columns(3)
    metric_columns[0].metric("检索论文", len(papers))
    metric_columns[1].metric("执行 Agent", len(results))
    metric_columns[2].metric("路由", state.get("route", ""))

    final_tab_label = (
        "本轮回答"
        if state.get("response_mode") == "follow_up"
        else "完整报告"
    )
    tabs = st.tabs(["概览", "论文", "Agent 分析", final_tab_label])
    with tabs[0]:
        st.subheader("执行路径")
        st.write("检索 → 路由 → 并行 Agent → ReportAgent")
        if selected_agents:
            st.markdown(
                "\n".join(
                    f"- {AGENT_LABELS.get(name, name)}"
                    for name in selected_agents
                )
            )
        reason = state.get("global_context", {}).get("route_reason")
        if reason:
            st.caption(reason)
        errors = state.get("errors", [])
        if errors:
            for error in errors:
                st.warning(error)

    with tabs[1]:
        render_papers(state)

    with tabs[2]:
        render_agents(state)

    with tabs[3]:
        report = state.get("final_report", "")
        st.markdown(report)
        st.download_button(
            "下载 Markdown 报告",
            data=report.encode("utf-8"),
            file_name="paper-agent-report.md",
            mime="text/markdown",
            width="stretch",
        )


def runtime_model_config(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: settings[key]
        for key in [
            "provider",
            "api_key",
            "model_name",
            "base_url",
            "temperature",
        ]
    }


def has_online_credentials(settings: dict[str, Any]) -> bool:
    if settings["provider"] == "offline":
        return False
    environment_key = (
        "DEEPSEEK_API_KEY"
        if settings["provider"] == "deepseek"
        else "OPENAI_API_KEY"
    )
    return bool(settings["api_key"] or os.getenv(environment_key))


def render_online_academic_search(
    settings: dict[str, Any],
    library: PaperLibrary,
) -> None:
    st.subheader("在线搜索")
    st.caption(
        "从开放学术数据源搜索最多 100 篇候选，经 AI 排名后由你手动选择"
        "最多 10 篇加入论文库。"
    )
    scholar_query = st.text_input(
        "检索内容",
        placeholder="例如：long video understanding memory",
        key="academic_search_query",
    )
    search_clicked = st.button(
        "搜索论文",
        key="academic_search_button",
        type="primary",
        width="stretch",
    )
    if search_clicked:
        if not scholar_query.strip():
            st.warning("请输入论文检索内容。")
        elif not has_online_credentials(settings):
            st.warning("AI 排名需要在线模型和 API Key。")
        else:
            try:
                with st.status(
                    "正在浏览候选论文并进行 AI 排名…",
                    expanded=True,
                ) as status:
                    existing_titles = {
                        paper.title for paper in library.list_papers()
                    }
                    candidates = AcademicSearchService().search(
                        scholar_query,
                        max_candidates=100,
                        existing_titles=existing_titles,
                        model_config=runtime_model_config(settings),
                    )
                    st.session_state["academic_candidates"] = candidates
                    status.update(
                        label=f"排名完成，共获得 {len(candidates)} 篇候选",
                        state="complete",
                        expanded=False,
                    )
            except Exception as exc:
                st.error(f"论文检索失败：{exc}")

    candidates = st.session_state.get("academic_candidates", [])
    if not candidates:
        st.info("输入检索内容后开始搜索。搜索结果不会自动加入论文库。")
        return

    visible_candidates = candidates[:30]
    rows = [
        {
            "选择": False,
            "排名": candidate.rank,
            "标题": candidate.title,
            "年份": candidate.year,
            "作者": "、".join(candidate.authors[:3]),
            "AI 契合度": candidate.final_score,
            "引用": candidate.cited_by_count,
            "公开 PDF": bool(candidate.pdf_url),
            "已在库中": candidate.already_exists,
            "论文页面": candidate.landing_page_url,
        }
        for candidate in visible_candidates
    ]
    edited_rows = st.data_editor(
        rows,
        key="academic_result_editor",
        width="stretch",
        hide_index=True,
        disabled=[
            "排名",
            "标题",
            "年份",
            "作者",
            "AI 契合度",
            "引用",
            "公开 PDF",
            "已在库中",
            "论文页面",
        ],
        column_config={
            "选择": st.column_config.CheckboxColumn(width="small"),
            "排名": st.column_config.NumberColumn(width="small"),
            "标题": st.column_config.TextColumn(width="large"),
            "年份": st.column_config.NumberColumn(width="small"),
            "AI 契合度": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                width="medium",
            ),
            "公开 PDF": st.column_config.CheckboxColumn(width="small"),
            "已在库中": st.column_config.CheckboxColumn(width="small"),
            "论文页面": st.column_config.LinkColumn(width="small"),
        },
    )
    selected_ranks = {
        int(row["排名"]) for row in edited_rows if row["选择"]
    }
    selected_candidates = [
        candidate
        for candidate in candidates
        if candidate.rank in selected_ranks
    ]
    invalid_candidates = [
        candidate
        for candidate in selected_candidates
        if not candidate.importable
    ]
    st.caption(
        f"已手动选择 {len(selected_candidates)} 篇；"
        "页面展示 AI 排名前 30 篇，单次最多导入 10 篇。"
    )

    import_clicked = st.button(
        "将所选论文加入数据库",
        key="academic_import_button",
        type="primary",
        width="stretch",
        disabled=not selected_candidates,
    )
    if not import_clicked:
        return

    if len(selected_candidates) > 10:
        st.warning("单次最多导入 10 篇，请减少选择。")
    elif invalid_candidates:
        st.warning("选择中包含无公开 PDF 或已存在的论文。")
    elif not has_online_credentials(settings):
        st.warning("导入论文需要在线模型和 API Key。")
    else:
        try:
            with st.status(
                "正在下载、解析并加入论文数据库…",
                expanded=True,
            ) as status:
                results, index_result = library.import_search_candidates(
                    selected_candidates,
                    model_config=runtime_model_config(settings),
                )
                for result in results:
                    prefix = "完成" if result.success else "失败"
                    st.write(f"{prefix}：{result.title or result.filename}")
                status.update(
                    label="所选论文处理完成",
                    state="complete",
                    expanded=True,
                )

            successful = [result for result in results if result.success]
            failed = [result for result in results if not result.success]
            if successful:
                st.success(f"成功加入 {len(successful)} 篇论文。")
            for result in failed:
                st.error(f"{result.filename}：{result.message}")
            if index_result:
                st.caption(
                    "FAISS 已重建："
                    f"{index_result['paper_count']} 篇，"
                    f"{index_result['dimension']} 维。"
                )
        except Exception as exc:
            st.error(f"所选论文导入失败：{exc}")


def render_local_pdf_import(
    settings: dict[str, Any],
    library: PaperLibrary,
) -> None:
    st.subheader("本地 PDF 导入")
    st.caption(
        "适用于需要登录后手动下载的论文。上传后会解析整篇 PDF 的全部页面，"
        "生成结构化信息与向量，并重建 FAISS 索引。"
    )
    uploaded_files = st.file_uploader(
        "选择本地 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="local_pdf_uploader",
        help="支持一次上传多篇论文。全文会分块解析，不再只读取开头部分。",
    )
    overwrite = st.checkbox(
        "覆盖同名论文",
        value=False,
        key="local_pdf_overwrite",
    )

    if settings["provider"] == "offline":
        st.info("全文解析需要在线 LLM，请选择 DeepSeek 或 OpenAI。")
    elif not has_online_credentials(settings):
        st.info("请在侧边栏输入 API Key 后再导入。")

    import_clicked = st.button(
        "解析全文并加入论文库",
        key="local_pdf_import_button",
        type="primary",
        disabled=not uploaded_files,
        width="stretch",
    )
    if not import_clicked:
        return
    if not has_online_credentials(settings):
        st.warning("导入论文需要可用的在线模型和 API Key。")
        return

    try:
        with st.status("正在分块解析 PDF 全文…", expanded=True) as status:
            st.write(f"准备解析 {len(uploaded_files)} 篇论文")
            results, index_result = library.import_many(
                uploaded_files,
                model_config=runtime_model_config(settings),
                overwrite=overwrite,
            )
            for result in results:
                prefix = "完成" if result.success else "失败"
                st.write(f"{prefix}：{result.title or result.filename}")
            status.update(
                label="本地论文处理完成",
                state="complete",
                expanded=True,
            )

        successful = [result for result in results if result.success]
        failed = [result for result in results if not result.success]
        if successful:
            st.success(f"成功加入 {len(successful)} 篇论文。")
        for result in failed:
            st.error(f"{result.filename}：{result.message}")
        if index_result:
            st.caption(
                "FAISS 已重建："
                f"{index_result['paper_count']} 篇，"
                f"{index_result['dimension']} 维。"
            )
    except Exception as exc:
        st.error(f"本地论文导入失败：{exc}")


def render_academic_search(settings: dict[str, Any]) -> None:
    library = PaperLibrary()
    st.header("论文检索")
    st.caption("在线寻找论文，或将已经下载到本地的 PDF 手动加入论文库。")
    online_tab, local_tab = st.tabs(["在线搜索", "本地 PDF 导入"])
    with online_tab:
        render_online_academic_search(settings, library)
    with local_tab:
        render_local_pdf_import(settings, library)


def render_library(settings: dict[str, Any]) -> None:
    del settings
    library = PaperLibrary()
    stats = library.stats()

    st.header("论文数据库")
    st.caption("查看当前论文库、论文来源和 FAISS 索引状态。")

    columns = st.columns(3)
    columns[0].metric("论文数量", stats["paper_count"])
    columns[1].metric(
        "Embedding 维度",
        stats["embedding_dimension"] or "不一致",
    )
    columns[2].metric(
        "FAISS 索引",
        "可用" if stats["index_exists"] and stats["mapping_exists"] else "未建立",
    )

    papers = library.list_papers()
    search_text = st.text_input(
        "筛选论文",
        placeholder="输入标题、作者或关键词",
    ).strip().lower()
    if search_text:
        papers = [
            paper
            for paper in papers
            if search_text
            in " ".join(
                [
                    paper.title,
                    " ".join(paper.authors),
                    " ".join(paper.keywords),
                ]
            ).lower()
        ]

    rows = [
        {
            "删除": False,
            "标题": paper.title,
            "作者": "、".join(paper.authors[:4]),
            "关键词": "、".join(paper.keywords[:6]),
            "向量维度": len(paper.embedding),
            "来源": paper.discovery_source or "本地导入",
            "原始页面": paper.source_url,
            "本地 PDF": (
                library.pdf_path_for(paper).name
                if library.pdf_path_for(paper)
                else ""
            ),
            "_source_file": paper.source_file,
        }
        for paper in papers
    ]
    edited_rows = st.data_editor(
        rows,
        key="paper_library_delete_editor",
        width="stretch",
        hide_index=True,
        disabled=[
            "标题",
            "作者",
            "关键词",
            "向量维度",
            "来源",
            "原始页面",
            "本地 PDF",
            "_source_file",
        ],
        column_config={
            "删除": st.column_config.CheckboxColumn(width="small"),
            "标题": st.column_config.TextColumn(width="large"),
            "作者": st.column_config.TextColumn(width="medium"),
            "关键词": st.column_config.TextColumn(width="medium"),
            "向量维度": st.column_config.NumberColumn(width="small"),
            "来源": st.column_config.TextColumn(width="small"),
            "原始页面": st.column_config.LinkColumn(width="small"),
            "本地 PDF": st.column_config.TextColumn(width="medium"),
            "_source_file": None,
        },
    )

    selected_for_delete = [row for row in edited_rows if row.get("删除")]
    if selected_for_delete:
        st.warning(
            f"已选择 {len(selected_for_delete)} 篇论文。删除会移除论文 JSON、可匹配的 PDF，并重建 FAISS 索引。"
        )
    confirm_delete = st.checkbox(
        "我确认删除所选论文",
        value=False,
        key="confirm_delete_papers",
        disabled=not selected_for_delete,
    )
    delete_clicked = st.button(
        "删除所选论文",
        key="delete_selected_papers",
        type="secondary",
        width="stretch",
        disabled=not selected_for_delete or not confirm_delete,
    )
    if delete_clicked:
        try:
            results, index_result = library.delete_many(
                [row["_source_file"] for row in selected_for_delete]
            )
            successful = [result for result in results if result.success]
            failed = [result for result in results if not result.success]
            if successful:
                deleted_file_count = sum(
                    len(result.deleted_files or [])
                    for result in successful
                )
                st.success(
                    f"已删除 {len(successful)} 篇论文，移除 {deleted_file_count} 个文件。"
                )
            for result in failed:
                st.error(f"{result.title or result.json_file}：{result.message}")
            if index_result:
                st.caption(
                    "FAISS 已更新："
                    f"{index_result['paper_count']} 篇，"
                    f"{index_result['dimension']} 维。"
                )
            st.rerun()
        except Exception as exc:
            st.error(f"删除论文失败：{exc}")

    with st.expander("论文详情"):
        if papers:
            selected_title = st.selectbox(
                "选择论文",
                [paper.title for paper in papers],
            )
            selected = next(
                paper for paper in papers if paper.title == selected_title
            )
            st.subheader(selected.title)
            if selected.authors:
                st.caption("作者：" + "、".join(selected.authors))
            render_paper_original_access(
                selected,
                library,
                key_prefix="library_detail",
            )
            st.write(selected.summary or selected.abstract or "暂无摘要。")
            if selected.contributions:
                st.markdown("**主要贡献**")
                for contribution in selected.contributions:
                    st.markdown(f"- {contribution}")
        else:
            st.info("当前筛选条件下没有论文。")


def render_agent_skills() -> None:
    library = AgentSkillLibrary()
    st.header("Agent Skills")
    st.caption(
        "为指定 Agent 安装 Markdown Skill。每个 Agent 只会读取自己的 Skill 文件。"
    )

    agent_name = st.selectbox(
        "目标 Agent",
        list(AGENT_LABELS),
        format_func=lambda value: AGENT_LABELS[value],
        key="skill_target_agent",
    )
    uploaded_skill = st.file_uploader(
        "选择 Markdown Skill",
        type=["md", "markdown"],
        accept_multiple_files=False,
        key="agent_skill_uploader",
        help="文件会保存到 agent_skills/目标 Agent/ 目录，下一次分析立即生效。",
    )
    overwrite = st.checkbox(
        "覆盖同名 Skill",
        value=False,
        key="agent_skill_overwrite",
    )
    install_clicked = st.button(
        "安装 Skill",
        type="primary",
        width="stretch",
        disabled=uploaded_skill is None,
        key="install_agent_skill",
    )
    if install_clicked and uploaded_skill is not None:
        try:
            skill = library.save(
                agent_name,
                uploaded_skill.name,
                uploaded_skill.getvalue(),
                overwrite=overwrite,
            )
            st.success(
                f"已为 {AGENT_LABELS[agent_name]} 安装 {skill.filename}。"
            )
        except (ValueError, FileExistsError) as exc:
            st.error(str(exc))

    st.subheader("已安装 Skills")
    skills = library.list_installed()
    if not skills:
        st.info("尚未安装 Skill。")
        return

    st.dataframe(
        [
            {
                "Agent": AGENT_LABELS[skill.agent_name],
                "文件": skill.filename,
                "来源": "内置" if skill.source == "builtin" else "外部",
                "状态": (
                    "已被外部同名 Skill 覆盖"
                    if skill.source == "builtin"
                    and any(
                        external.agent_name == skill.agent_name
                        and external.filename == skill.filename
                        for external in library.list(skill.agent_name)
                    )
                    else "生效中"
                ),
                "编辑": skill_view_url(skill),
                "字符数": len(skill.content),
                "保存位置": str(skill.path),
            }
            for skill in skills
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Agent": st.column_config.TextColumn(width="medium"),
            "文件": st.column_config.TextColumn(width="medium"),
            "来源": st.column_config.TextColumn(width="small"),
            "状态": st.column_config.TextColumn(width="medium"),
            "编辑": st.column_config.LinkColumn(
                display_text="编辑",
                width="small",
            ),
            "字符数": st.column_config.NumberColumn(width="small"),
            "保存位置": st.column_config.TextColumn(width="large"),
        },
    )

    selected_agent_skills = library.list_effective(agent_name)
    if selected_agent_skills:
        with st.expander(f"预览 {AGENT_LABELS[agent_name]} 的 Skills"):
            selected_filename = st.selectbox(
                "Skill 文件",
                [skill.filename for skill in selected_agent_skills],
                key="skill_preview_file",
            )
            selected_skill = next(
                skill
                for skill in selected_agent_skills
                if skill.filename == selected_filename
            )
            st.markdown(selected_skill.content)


def skill_view_url(skill) -> str:
    return "?" + urlencode(
        {
            "skill_agent": skill.agent_name,
            "skill_file": skill.filename,
            "skill_source": skill.source,
        }
    )


def requested_skill(library: AgentSkillLibrary):
    agent_name = st.query_params.get("skill_agent")
    filename = st.query_params.get("skill_file")
    source = st.query_params.get("skill_source")
    if not all([agent_name, filename, source]):
        return None
    try:
        return library.get_installed(
            str(agent_name),
            str(filename),
            str(source),
        )
    except ValueError:
        return None


def render_skill_viewer(skill) -> None:
    library = AgentSkillLibrary()
    source_label = "内置 Skill" if skill.source == "builtin" else "外部 Skill"
    st.caption(
        f"{AGENT_LABELS[skill.agent_name]} · {source_label} · {skill.filename}"
    )
    st.header(skill.filename)
    edited_content = st.text_area(
        "Markdown 内容",
        value=skill.content,
        height=520,
        key=(
            f"edit_skill_{skill.agent_name}_"
            f"{skill.source}_{skill.filename}"
        ),
    )
    save_clicked = st.button(
        "保存修改",
        type="primary",
        width="stretch",
        key="save_skill_changes",
    )
    if save_clicked:
        try:
            library.update(
                skill.agent_name,
                skill.filename,
                skill.source,
                edited_content,
            )
            st.success("Skill 已保存，下一次 Agent 执行时生效。")
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))

    with st.expander("Markdown 预览"):
        st.markdown(edited_content)
    st.divider()
    st.link_button(
        "返回 Agent Skills",
        "?",
        width="stretch",
    )


def main() -> None:
    st.set_page_config(
        page_title="Paper Agent",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    settings = model_settings()
    st.sidebar.divider()
    workspace = st.sidebar.radio(
        "工作区",
        ["研究分析", "论文检索", "论文数据库", "Agent Skills"],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "API Key 只保存在当前浏览器会话中，不会写入配置文件或报告。"
    )

    st.markdown('<h1 class="app-title">Paper Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">检索论文库，通过 LangGraph 调度多个研究 Agent，并生成结构化报告。</p>',
        unsafe_allow_html=True,
    )
    graph_status = (
        "LangGraph 已连接"
        if current_workflow().langgraph_available()
        else "兼容模式"
    )
    st.markdown(
        f'<div class="status-line"><span class="status-dot"></span>{graph_status}</div>',
        unsafe_allow_html=True,
    )

    skill = requested_skill(AgentSkillLibrary())
    if skill is not None:
        render_skill_viewer(skill)
        return
    if any(
        key in st.query_params
        for key in ["skill_agent", "skill_file", "skill_source"]
    ):
        st.error("Skill 链接无效或文件已不存在。")
        st.link_button("返回 Agent Skills", "?", width="stretch")
        return

    if workspace == "论文数据库":
        render_library(settings)
        return
    if workspace == "论文检索":
        render_academic_search(settings)
        return
    if workspace == "Agent Skills":
        render_agent_skills()
        return

    render_conversation_sidebar()
    conversation = active_conversation()

    chat_column, graph_column = st.columns(
        [1.7, 1],
        gap="large",
    )
    graph_placeholder = graph_column.empty()
    progress = completed_workflow_progress(
        conversation.get("analysis_state")
    )
    render_workflow_diagram(
        graph_placeholder,
        node_status=progress[0],
        agent_status=progress[1],
        selected_agents=progress[2],
        paper_count=progress[3],
        agent_outputs=progress[4],
    )

    messages = conversation["messages"]
    with chat_column:
        if not messages:
            st.info(
                "这是一个新会话。输入研究主题后，本会话中的后续问题都会作为连续追问。"
            )

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input(
            "在当前会话中继续提问",
        )

    if query:
        if settings["provider"] != "offline" and not (
            settings["api_key"]
            or os.getenv(
                "DEEPSEEK_API_KEY"
                if settings["provider"] == "deepseek"
                else "OPENAI_API_KEY"
            )
        ):
            st.warning("请输入 API Key，或选择离线模式。")
        else:
            history = list(messages)
            if not messages:
                conversation["title"] = conversation_title(query)
            messages.append({"role": "user", "content": query.strip()})
            node_status = {
                "contextualize": "running",
                "retrieve": "pending",
                "route": "pending",
                "report_agent": "pending",
            }
            agent_status: dict[str, str] = {}
            selected_agents: list[str] = []
            completed_agents: set[str] = set()
            paper_count = 0
            render_workflow_diagram(
                graph_placeholder,
                node_status=node_status,
                agent_status=agent_status,
            )

            with chat_column:
                with st.chat_message("user"):
                    st.markdown(query.strip())
                with st.status(
                    "正在理解研究问题…",
                    expanded=True,
                ) as status:
                    state: dict[str, Any] = {}
                    event_line = st.empty()
                    for event in current_workflow().stream_pipeline_state(
                        query.strip(),
                        top_k=settings["top_k"],
                        model_config=runtime_model_config(settings),
                        require_langgraph=True,
                        conversation_history=history,
                    ):
                        state = event["state"]
                        node = event["node"]
                        if node == "contextualize" and event["status"] == "completed":
                            node_status["contextualize"] = "completed"
                            node_status["retrieve"] = "running"
                            event_line.write("正在检索相关论文")
                            status.update(label="正在检索相关论文…")
                        elif node == "retrieve":
                            node_status["retrieve"] = "completed"
                            node_status["route"] = "running"
                            paper_count = len(
                                state.get("retrieved_papers", [])
                            )
                            event_line.write(
                                f"已检索 {paper_count} 篇论文，正在路由任务"
                            )
                            status.update(label="正在选择研究 Agent…")
                        elif node == "route":
                            node_status["route"] = "completed"
                            selected_agents = state.get(
                                "selected_agents", []
                            )
                            agent_status = {
                                name: "running"
                                for name in selected_agents
                            }
                            event_line.write(
                                f"{len(selected_agents)} 个 Agent 正在并行分析"
                            )
                            status.update(label="研究 Agent 正在并行分析…")
                        elif node == "run_agent":
                            agent_name = event.get("agent_name", "")
                            if agent_name:
                                completed_agents.add(agent_name)
                                agent_status[agent_name] = (
                                    "error"
                                    if event.get("error")
                                    else "completed"
                                )
                                event_line.write(
                                    f"{AGENT_LABELS.get(agent_name, agent_name)}"
                                    " 已完成"
                                )
                            if set(selected_agents) <= completed_agents:
                                node_status["report_agent"] = "running"
                                status.update(label="正在汇总最终报告…")
                        elif node == "report_agent":
                            node_status["report_agent"] = "completed"
                            event_line.write("最终报告已生成")

                        render_workflow_diagram(
                            graph_placeholder,
                            node_status=node_status,
                            agent_status=agent_status,
                            selected_agents=selected_agents,
                            paper_count=paper_count,
                            agent_outputs=list(
                                state.get("agent_outputs", [])
                            ),
                        )

                    status.update(
                        label="分析完成",
                        state="complete",
                        expanded=False,
                    )
            report = state.get("final_report", "")
            messages.append({"role": "assistant", "content": report})
            conversation["analysis_state"] = state
            active_id = st.session_state["active_conversation_id"]
            st.session_state["conversation_sessions"][
                active_id
            ] = conversation
            st.rerun()

    if conversation.get("analysis_state"):
        with chat_column:
            st.divider()
            render_result(conversation["analysis_state"])


if __name__ == "__main__":
    main()

from __future__ import annotations

import html
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import fitz
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

TRANSLATIONS = {
    "zh": {
        "english_ui": "English UI",
        "new_session": "新建会话",
        "session": "会话",
        "workspace": "工作区",
        "workspace_analysis": "研究分析",
        "workspace_search": "论文检索",
        "workspace_library": "论文数据库",
        "provider": "服务商",
        "offline_mode": "离线模式",
        "api_key_placeholder": "仅在当前会话中使用",
        "model": "模型",
        "api_base": "API 地址",
        "temperature": "温度",
        "top_k": "检索论文数",
        "api_notice": "API Key 只保存在当前浏览器会话中，不会写入配置文件或报告。",
        "subtitle": "检索论文库，通过 LangGraph 调度多个研究 Agent，并生成结构化报告。",
        "langgraph_connected": "LangGraph 已连接",
        "compatible_mode": "兼容模式",
        "no_papers": "没有检索到相关论文。",
        "score": "相关度",
        "authors": "作者：",
        "no_abstract": "暂无摘要。",
        "keywords": "关键词：",
        "source_page": "原始页面",
        "local_pdf": "本地 PDF",
        "no_original": "未记录原始页面或本地 PDF。",
        "open_pdf": "打开本地 PDF",
        "close_pdf": "关闭本地 PDF 预览",
        "pdf_preview_hint": "预览以图片方式渲染，不依赖浏览器 PDF 插件。",
        "pdf_page": "PDF 页码",
        "pdf_page_caption": "{name} 第 {page}/{total} 页",
        "download_pdf": "下载原文 PDF",
        "pdf_fallback": "当前浏览器无法内嵌预览 PDF，请使用下载按钮打开原文。",
        "no_agents": "本次问题没有触发分析 Agent。",
        "metric_papers": "检索论文",
        "metric_agents": "执行 Agent",
        "metric_route": "路由",
        "overview": "概览",
        "papers": "论文",
        "agent_analysis": "Agent 分析",
        "followup_answer": "本轮回答",
        "full_report": "完整报告",
        "execution_path": "执行路径",
        "execution_path_value": "检索 → 路由 → 并行 Agent → ReportAgent",
        "download_report": "下载 Markdown 报告",
        "search_header": "论文检索",
        "search_caption": "在线寻找论文，或将已经下载到本地的 PDF 手动加入论文库。",
        "online_search": "在线搜索",
        "local_pdf_import": "本地 PDF 导入",
        "library_header": "论文数据库",
        "library_caption": "查看当前论文库、论文来源和 FAISS 索引状态。",
        "paper_count": "论文数量",
        "embedding_dim": "Embedding 维度",
        "faiss_index": "FAISS 索引",
        "available": "可用",
        "not_built": "未建立",
        "filter_papers": "筛选论文",
        "filter_placeholder": "输入标题、作者或关键词",
        "delete": "删除",
        "title": "标题",
        "source": "来源",
        "local_import": "本地导入",
        "select_pdf_preview": "从数据库列表预览本地 PDF",
        "select_pdf_placeholder": "选择有本地 PDF 的论文",
        "no_pdf_in_filter": "当前筛选结果中没有可预览的本地 PDF。",
        "paper_details": "论文详情",
        "select_paper": "选择论文",
        "contributions": "主要贡献",
        "no_filtered_paper": "当前筛选条件下没有论文。",
        "confirm_delete": "我确认删除所选论文",
        "delete_selected": "删除所选论文",
        "delete_warning": "已选择 {count} 篇论文。删除会移除论文 JSON、可匹配的 PDF，并重建 FAISS 索引。",
        "delete_success": "已删除 {count} 篇论文，移除 {files} 个文件。",
        "faiss_updated": "FAISS 已更新：{papers} 篇，{dim} 维。",
        "delete_failed": "删除论文失败：{error}",
        "new_conversation_info": "这是一个新会话。输入研究主题后，本会话中的后续问题都会作为连续追问。",
        "chat_placeholder": "在当前会话中继续提问",
        "missing_api_key": "请输入 API Key，或选择离线模式。",
        "understand_question": "正在理解研究问题…",
        "retrieving_papers": "正在检索相关论文…",
        "routing_agents": "正在选择研究 Agent…",
        "agents_running": "研究 Agent 正在并行分析…",
        "summarizing_report": "正在汇总最终报告…",
        "analysis_complete": "分析完成",
        "retrieved_count_routing": "已检索 {count} 篇论文，正在路由任务",
        "running_agent_count": "{count} 个 Agent 正在并行分析",
        "agent_finished": "{agent} 已完成",
        "final_report_ready": "最终报告已生成",
    },
    "en": {
        "english_ui": "English UI",
        "new_session": "New session",
        "session": "Sessions",
        "workspace": "Workspace",
        "workspace_analysis": "Research analysis",
        "workspace_search": "Paper search",
        "workspace_library": "Paper library",
        "provider": "Provider",
        "offline_mode": "Offline mode",
        "api_key_placeholder": "Only used in this browser session",
        "model": "Model",
        "api_base": "API base URL",
        "temperature": "Temperature",
        "top_k": "Papers to retrieve",
        "api_notice": "API Key is kept only in the current browser session. It is not written to config files or reports.",
        "subtitle": "Search the paper library, orchestrate research agents with LangGraph, and generate structured answers.",
        "langgraph_connected": "LangGraph connected",
        "compatible_mode": "Compatibility mode",
        "no_papers": "No relevant papers were retrieved.",
        "score": "Score",
        "authors": "Authors: ",
        "no_abstract": "No abstract available.",
        "keywords": "Keywords: ",
        "source_page": "Source page",
        "local_pdf": "Local PDF",
        "no_original": "No source page or local PDF is recorded.",
        "open_pdf": "Open local PDF",
        "close_pdf": "Close PDF preview",
        "pdf_preview_hint": "The preview is rendered as images and does not depend on the browser PDF plugin.",
        "pdf_page": "PDF page",
        "pdf_page_caption": "{name} page {page}/{total}",
        "download_pdf": "Download original PDF",
        "pdf_fallback": "This browser cannot preview the PDF inline. Please use the download button.",
        "no_agents": "This question did not trigger any analysis agent.",
        "metric_papers": "Retrieved papers",
        "metric_agents": "Executed agents",
        "metric_route": "Route",
        "overview": "Overview",
        "papers": "Papers",
        "agent_analysis": "Agent analysis",
        "followup_answer": "This turn",
        "full_report": "Full report",
        "execution_path": "Execution path",
        "execution_path_value": "Retrieve -> Route -> Parallel agents -> ReportAgent",
        "download_report": "Download Markdown report",
        "search_header": "Paper search",
        "search_caption": "Search papers online, or manually add local PDFs that you have already downloaded.",
        "online_search": "Online search",
        "local_pdf_import": "Local PDF import",
        "library_header": "Paper library",
        "library_caption": "View papers, source links, local PDFs, and FAISS index status.",
        "paper_count": "Papers",
        "embedding_dim": "Embedding dimension",
        "faiss_index": "FAISS index",
        "available": "Available",
        "not_built": "Not built",
        "filter_papers": "Filter papers",
        "filter_placeholder": "Enter title, author, or keyword",
        "delete": "Delete",
        "title": "Title",
        "source": "Source",
        "local_import": "Local import",
        "select_pdf_preview": "Preview a local PDF from the library",
        "select_pdf_placeholder": "Select a paper with a local PDF",
        "no_pdf_in_filter": "No local PDF is available under the current filter.",
        "paper_details": "Paper details",
        "select_paper": "Select paper",
        "contributions": "Main contributions",
        "no_filtered_paper": "No paper matches the current filter.",
        "confirm_delete": "I confirm deleting the selected papers",
        "delete_selected": "Delete selected papers",
        "delete_warning": "Selected {count} paper(s). Deletion removes paper JSON, matching PDFs, and rebuilds the FAISS index.",
        "delete_success": "Deleted {count} paper(s), removed {files} file(s).",
        "faiss_updated": "FAISS updated: {papers} papers, {dim} dimensions.",
        "delete_failed": "Failed to delete papers: {error}",
        "new_conversation_info": "This is a new session. Enter a research topic; later questions in this session will be treated as follow-ups.",
        "chat_placeholder": "Ask in the current session",
        "missing_api_key": "Enter an API Key, or choose offline mode.",
        "understand_question": "Understanding the research question...",
        "retrieving_papers": "Retrieving related papers...",
        "routing_agents": "Selecting research agents...",
        "agents_running": "Research agents are running in parallel...",
        "summarizing_report": "Synthesizing final answer...",
        "analysis_complete": "Analysis complete",
        "retrieved_count_routing": "Retrieved {count} paper(s); routing tasks...",
        "running_agent_count": "{count} agent(s) are running in parallel",
        "agent_finished": "{agent} finished",
        "final_report_ready": "Final report generated",
    }
}


def ui_language() -> str:
    return "en" if st.session_state.get("ui_language") == "en" else "zh"


def tr(key: str, **kwargs) -> str:
    value = TRANSLATIONS.get(ui_language(), {}).get(key, key)
    return value.format(**kwargs) if kwargs else value


def agent_label(agent_name: str) -> str:
    if ui_language() == "en":
        return {
            "survey_agent": "Survey",
            "innovation_agent": "Innovation",
            "method_agent": "Method comparison",
            "limitation_agent": "Limitations & opportunities",
        }.get(agent_name, agent_name)
    return AGENT_LABELS.get(agent_name, agent_name)


def render_language_switch() -> None:
    english_enabled = st.sidebar.checkbox(
        tr("english_ui"),
        value=ui_language() == "en",
        key="ui_language_english",
    )
    st.session_state["ui_language"] = "en" if english_enabled else "zh"


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
    st.sidebar.subheader(tr("session"))
    if st.sidebar.button(
        tr("new_session"),
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
    st.sidebar.subheader("Model connection" if ui_language() == "en" else "模型连接")
    provider = st.sidebar.selectbox(
        tr("provider"),
        ["deepseek", "openai", "offline"],
        format_func=lambda value: {
            "deepseek": "DeepSeek",
            "openai": "OpenAI",
            "offline": tr("offline_mode"),
        }[value],
    )

    api_key = ""
    model_name = ""
    base_url = ""
    if provider != "offline":
        api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            placeholder=tr("api_key_placeholder"),
        )
        model_name = st.sidebar.text_input(
            tr("model"),
            value=default_model(provider),
        )
        if provider == "deepseek":
            base_url = st.sidebar.text_input(
                tr("api_base"),
                value=cfg["MODEL"]["DEEPSEEK"]["BASE_URL"],
            )

    temperature = st.sidebar.slider(
        tr("temperature"),
        min_value=0.0,
        max_value=1.5,
        value=float(cfg["MODEL"].get("TEMPERATURE", 0.3)),
        step=0.1,
    )
    top_k = st.sidebar.slider(tr("top_k"), 1, 30, 10)

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
        st.info(tr("no_papers"))
        return

    library = PaperLibrary()
    for index, paper in enumerate(papers, start=1):
        with st.expander(f"{index}. {paper.title}", expanded=index <= 2):
            st.markdown(
                f'<span class="paper-score">{tr("score")} {paper.score:.4f}</span>',
                unsafe_allow_html=True,
            )
            if paper.authors:
                st.caption(tr("authors") + "、".join(paper.authors))
            st.write(paper.summary or paper.abstract or tr("no_abstract"))
            if paper.keywords:
                st.caption(tr("keywords") + " · ".join(paper.keywords[:8]))
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
        links.append(f"[{tr('source_page')}]({source_url})")
    if pdf_path:
        links.append(f"{tr('local_pdf')}: `{pdf_path.name}`")

    if links:
        st.markdown(" / ".join(links))
    else:
        st.caption(tr("no_original"))

    if pdf_path:
        preview_key = f"{key_prefix}_pdf_preview_open"
        if st.button(
            tr("close_pdf")
            if st.session_state.get(preview_key)
            else tr("open_pdf"),
            key=f"{key_prefix}_open_pdf",
            width="stretch",
        ):
            st.session_state[preview_key] = not st.session_state.get(
                preview_key,
                False,
            )
        if st.session_state.get(preview_key):
            render_pdf_page_preview(pdf_path, key_prefix=key_prefix)
        st.caption(
            tr("pdf_preview_hint")
        )
        st.download_button(
            tr("download_pdf"),
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            key=f"{key_prefix}_download_pdf",
            width="stretch",
        )


@st.cache_data(show_spinner=False)
def pdf_page_count(path: str, modified_time: float) -> int:
    del modified_time
    with fitz.open(path) as document:
        return document.page_count


@st.cache_data(show_spinner=False)
def pdf_page_png_bytes(
    path: str,
    modified_time: float,
    page_index: int,
    zoom: float = 1.6,
) -> bytes:
    del modified_time
    with fitz.open(path) as document:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            alpha=False,
        )
        return pixmap.tobytes("png")


def render_pdf_page_preview(path: Any, *, key_prefix: str) -> None:
    modified_time = Path(path).stat().st_mtime
    page_total = pdf_page_count(str(path), modified_time)
    if page_total <= 0:
        st.warning(tr("pdf_fallback"))
        return

    if page_total == 1:
        page_number = 1
    else:
        page_number = st.slider(
            tr("pdf_page"),
            min_value=1,
            max_value=page_total,
            value=1,
            key=f"{key_prefix}_pdf_page",
        )

    image_bytes = pdf_page_png_bytes(
        str(path),
        modified_time,
        page_number - 1,
    )
    st.image(
        image_bytes,
        caption=tr(
            "pdf_page_caption",
            name=Path(path).name,
            page=page_number,
            total=page_total,
        ),
        width="stretch",
    )


def render_agents(state: dict[str, Any]) -> None:
    results = state.get("agent_results", {})
    if not results:
        st.info(tr("no_agents"))
        return

    for agent_name, result in results.items():
        st.subheader(agent_label(agent_name))
        st.markdown(result)


def render_result(state: dict[str, Any]) -> None:
    papers = state.get("retrieved_papers", [])
    results = state.get("agent_results", {})
    selected_agents = state.get("selected_agents", [])

    metric_columns = st.columns(3)
    metric_columns[0].metric(tr("metric_papers"), len(papers))
    metric_columns[1].metric(tr("metric_agents"), len(results))
    metric_columns[2].metric(tr("metric_route"), state.get("route", ""))

    final_tab_label = (
        tr("followup_answer")
        if state.get("response_mode") == "follow_up"
        else tr("full_report")
    )
    tabs = st.tabs([tr("overview"), tr("papers"), tr("agent_analysis"), final_tab_label])
    with tabs[0]:
        st.subheader(tr("execution_path"))
        st.write(tr("execution_path_value"))
        if selected_agents:
            st.markdown(
                "\n".join(
                    f"- {agent_label(name)}"
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
            tr("download_report"),
            data=report.encode("utf-8"),
            file_name="paper-agent-report.md",
            mime="text/markdown",
            width="stretch",
        )


def runtime_model_config(settings: dict[str, Any]) -> dict[str, Any]:
    model_config = {
        key: settings[key]
        for key in [
            "provider",
            "api_key",
            "model_name",
            "base_url",
            "temperature",
        ]
    }
    model_config["output_language"] = (
        "English" if ui_language() == "en" else "Chinese"
    )
    return model_config


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
    st.header(tr("search_header"))
    st.caption(tr("search_caption"))
    online_tab, local_tab = st.tabs([tr("online_search"), tr("local_pdf_import")])
    with online_tab:
        render_online_academic_search(settings, library)
    with local_tab:
        render_local_pdf_import(settings, library)


def render_library(settings: dict[str, Any]) -> None:
    del settings
    library = PaperLibrary()
    stats = library.stats()
    delete_col = tr("delete")
    title_col = tr("title")
    authors_col = tr("authors").rstrip("：: ")
    keywords_col = tr("keywords").rstrip("：: ")
    embedding_col = tr("embedding_dim")
    source_col = tr("source")
    source_page_col = tr("source_page")
    local_pdf_col = tr("local_pdf")

    st.header(tr("library_header"))
    st.caption(tr("library_caption"))

    columns = st.columns(3)
    columns[0].metric(tr("paper_count"), stats["paper_count"])
    columns[1].metric(
        tr("embedding_dim"),
        stats["embedding_dimension"] or ("Inconsistent" if ui_language() == "en" else "不一致"),
    )
    columns[2].metric(
        tr("faiss_index"),
        tr("available") if stats["index_exists"] and stats["mapping_exists"] else tr("not_built"),
    )

    papers = library.list_papers()
    search_text = st.text_input(
        tr("filter_papers"),
        placeholder=tr("filter_placeholder"),
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
            delete_col: False,
            title_col: paper.title,
            authors_col: "、".join(paper.authors[:4]),
            keywords_col: "、".join(paper.keywords[:6]),
            embedding_col: len(paper.embedding),
            source_col: paper.discovery_source or tr("local_import"),
            source_page_col: paper.source_url,
            local_pdf_col: (
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
            title_col,
            authors_col,
            keywords_col,
            embedding_col,
            source_col,
            source_page_col,
            local_pdf_col,
            "_source_file",
        ],
        column_config={
            delete_col: st.column_config.CheckboxColumn(width="small"),
            title_col: st.column_config.TextColumn(width="large"),
            authors_col: st.column_config.TextColumn(width="medium"),
            keywords_col: st.column_config.TextColumn(width="medium"),
            embedding_col: st.column_config.NumberColumn(width="small"),
            source_col: st.column_config.TextColumn(width="small"),
            source_page_col: st.column_config.LinkColumn(width="small"),
            local_pdf_col: st.column_config.TextColumn(width="medium"),
            "_source_file": None,
        },
    )

    pdf_papers = [
        paper
        for paper in papers
        if library.pdf_path_for(paper)
    ]
    if pdf_papers:
        selected_pdf_title = st.selectbox(
            tr("select_pdf_preview"),
            [paper.title for paper in pdf_papers],
            key="library_pdf_preview_select",
        )
        selected_pdf = next(
            paper for paper in pdf_papers if paper.title == selected_pdf_title
        )
        render_paper_original_access(
            selected_pdf,
            library,
            key_prefix="library_table_preview",
        )
    else:
        st.caption(tr("no_pdf_in_filter"))

    selected_for_delete = [row for row in edited_rows if row.get(delete_col)]
    if selected_for_delete:
        st.warning(
            tr("delete_warning", count=len(selected_for_delete))
        )
    confirm_delete = st.checkbox(
        tr("confirm_delete"),
        value=False,
        key="confirm_delete_papers",
        disabled=not selected_for_delete,
    )
    delete_clicked = st.button(
        tr("delete_selected"),
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
                    tr(
                        "delete_success",
                        count=len(successful),
                        files=deleted_file_count,
                    )
                )
            for result in failed:
                st.error(f"{result.title or result.json_file}：{result.message}")
            if index_result:
                st.caption(
                    tr(
                        "faiss_updated",
                        papers=index_result["paper_count"],
                        dim=index_result["dimension"],
                    )
                )
            st.rerun()
        except Exception as exc:
            st.error(tr("delete_failed", error=exc))

    with st.expander(tr("paper_details")):
        if papers:
            selected_title = st.selectbox(
                tr("select_paper"),
                [paper.title for paper in papers],
            )
            selected = next(
                paper for paper in papers if paper.title == selected_title
            )
            st.subheader(selected.title)
            if selected.authors:
                st.caption(tr("authors") + "、".join(selected.authors))
            render_paper_original_access(
                selected,
                library,
                key_prefix="library_detail",
            )
            st.write(selected.summary or selected.abstract or tr("no_abstract"))
            if selected.contributions:
                st.markdown(f"**{tr('contributions')}**")
                for contribution in selected.contributions:
                    st.markdown(f"- {contribution}")
        else:
            st.info(tr("no_filtered_paper"))


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

    render_language_switch()
    settings = model_settings()
    st.sidebar.divider()
    workspace = st.sidebar.radio(
        tr("workspace"),
        ["研究分析", "论文检索", "论文数据库", "Agent Skills"],
        format_func=lambda value: {
            "研究分析": tr("workspace_analysis"),
            "论文检索": tr("workspace_search"),
            "论文数据库": tr("workspace_library"),
            "Agent Skills": "Agent Skills",
        }[value],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        tr("api_notice")
    )

    st.markdown('<h1 class="app-title">Paper Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="app-subtitle">{tr("subtitle")}</p>',
        unsafe_allow_html=True,
    )
    graph_status = (
        tr("langgraph_connected")
        if current_workflow().langgraph_available()
        else tr("compatible_mode")
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
                tr("new_conversation_info")
            )

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input(
            tr("chat_placeholder"),
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
            st.warning(tr("missing_api_key"))
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
                    tr("understand_question"),
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
                            event_line.write(tr("retrieving_papers"))
                            status.update(label=tr("retrieving_papers"))
                        elif node == "retrieve":
                            node_status["retrieve"] = "completed"
                            node_status["route"] = "running"
                            paper_count = len(
                                state.get("retrieved_papers", [])
                            )
                            event_line.write(
                                tr("retrieved_count_routing", count=paper_count)
                            )
                            status.update(label=tr("routing_agents"))
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
                                tr(
                                    "running_agent_count",
                                    count=len(selected_agents),
                                )
                            )
                            status.update(label=tr("agents_running"))
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
                                    tr(
                                        "agent_finished",
                                        agent=agent_label(agent_name),
                                    )
                                )
                            if set(selected_agents) <= completed_agents:
                                node_status["report_agent"] = "running"
                                status.update(label=tr("summarizing_report"))
                        elif node == "report_agent":
                            node_status["report_agent"] = "completed"
                            event_line.write(tr("final_report_ready"))

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
                        label=tr("analysis_complete"),
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

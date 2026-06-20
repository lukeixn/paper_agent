from __future__ import annotations

import html
import importlib
import inspect
import os
from typing import Any

import streamlit as st

import workflow
from academic_search import AcademicSearchService
from configs.config import cfg
from paper_library import PaperLibrary


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


def render_workflow_diagram(
    placeholder,
    *,
    node_status: dict[str, str] | None = None,
    agent_status: dict[str, str] | None = None,
    selected_agents: list[str] | None = None,
    paper_count: int = 0,
) -> None:
    node_status = node_status or {}
    agent_status = agent_status or {}
    selected_agents = selected_agents or []
    selected_set = set(selected_agents)

    agent_layout = [
        ("survey_agent", 90, 365),
        ("innovation_agent", 170, 415),
        ("method_agent", 250, 365),
        ("limitation_agent", 330, 415),
    ]
    agent_nodes = []
    for index, (agent_name, x, y) in enumerate(agent_layout, start=1):
        label = AGENT_LABELS[agent_name]
        status = agent_status.get(agent_name, "pending")
        if selected_set and agent_name not in selected_set:
            status = "disabled"
        agent_nodes.append(
            f"""
            <g class="topo-agent {_topology_status(status)}">
                <circle class="topo-halo" cx="{x}" cy="{y}" r="31"></circle>
                <circle class="topo-disc" cx="{x}" cy="{y}" r="23"></circle>
                <text class="topo-code" x="{x}" y="{y + 3}">A{index}</text>
                <text class="topo-agent-label" x="{x}" y="{y + 43}">
                    {html.escape(label)}
                </text>
            </g>
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
    agents_finished = bool(selected_agents) and all(
        agent_status.get(name) in {"completed", "error"}
        for name in selected_agents
    )
    report_line = "active" if agents_finished else "pending"
    agent_link_status = {
        name: (
            "disabled"
            if selected_set and name not in selected_set
            else route_line
        )
        for name in AGENT_LABELS
    }

    placeholder.markdown(
        f"""
        <div class="workflow-panel">
            <div class="topology-header">
                <div>
                    <span class="topology-eyebrow">LANGGRAPH TOPOLOGY</span>
                    <div class="workflow-heading">实时执行图</div>
                </div>
                <div class="topology-live"><i></i>{html.escape(live_text)}</div>
            </div>
            <svg class="topology-map" viewBox="0 0 420 610"
                 role="img" aria-label="LangGraph 多 Agent 实时执行拓扑">
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

                <path class="topo-link branch {agent_link_status["survey_agent"]}"
                      d="M184 286 C165 314 120 320 94 338"></path>
                <path class="topo-link branch {agent_link_status["innovation_agent"]}"
                      d="M198 296 C190 330 178 353 172 387"></path>
                <path class="topo-link branch {agent_link_status["method_agent"]}"
                      d="M222 296 C230 330 242 328 248 338"></path>
                <path class="topo-link branch {agent_link_status["limitation_agent"]}"
                      d="M236 286 C260 316 304 354 326 388"></path>

                {''.join(agent_nodes)}

                <path class="topo-link merge {report_line}"
                      d="M90 389 C110 472 168 478 190 505"></path>
                <path class="topo-link merge {report_line}"
                      d="M170 439 C178 470 188 486 198 505"></path>
                <path class="topo-link merge {report_line}"
                      d="M250 389 C245 454 232 483 220 505"></path>
                <path class="topo-link merge {report_line}"
                      d="M330 439 C300 474 254 486 230 510"></path>

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
        </div>
        """,
        unsafe_allow_html=True,
    )


def completed_workflow_progress(
    state: dict[str, Any] | None,
) -> tuple[dict[str, str], dict[str, str], list[str], int]:
    if not state or not state.get("final_report"):
        return {}, {}, [], 0
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
        .workflow-panel {
            position: sticky;
            top: 1rem;
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
        .topology-map {
            display: block;
            width: 100%;
            height: auto;
            max-height: calc(100vh - 13rem);
            min-height: 31rem;
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

    tabs = st.tabs(["概览", "论文", "Agent 分析", "完整报告"])
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
            "标题": paper.title,
            "作者": "、".join(paper.authors[:4]),
            "关键词": "、".join(paper.keywords[:6]),
            "向量维度": len(paper.embedding),
            "来源": paper.discovery_source or "本地导入",
            "原始页面": paper.source_url,
        }
        for paper in papers
    ]
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "标题": st.column_config.TextColumn(width="large"),
            "作者": st.column_config.TextColumn(width="medium"),
            "关键词": st.column_config.TextColumn(width="medium"),
            "向量维度": st.column_config.NumberColumn(width="small"),
            "来源": st.column_config.TextColumn(width="small"),
            "原始页面": st.column_config.LinkColumn(width="small"),
        },
    )

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
            st.write(selected.summary or selected.abstract or "暂无摘要。")
            if selected.contributions:
                st.markdown("**主要贡献**")
                for contribution in selected.contributions:
                    st.markdown(f"- {contribution}")
        else:
            st.info("当前筛选条件下没有论文。")


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
        ["研究分析", "论文检索", "论文数据库"],
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

    if workspace == "论文数据库":
        render_library(settings)
        return
    if workspace == "论文检索":
        render_academic_search(settings)
        return

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if st.sidebar.button("新建会话", width="stretch"):
        st.session_state["chat_messages"] = []
        st.session_state.pop("analysis_state", None)
        st.rerun()

    chat_column, graph_column = st.columns(
        [2.15, 1],
        gap="large",
    )
    graph_placeholder = graph_column.empty()
    progress = completed_workflow_progress(
        st.session_state.get("analysis_state")
    )
    render_workflow_diagram(
        graph_placeholder,
        node_status=progress[0],
        agent_status=progress[1],
        selected_agents=progress[2],
        paper_count=progress[3],
    )

    messages = st.session_state["chat_messages"]
    with chat_column:
        if not messages:
            st.info(
                "输入第一个研究问题。之后可以直接追问，Agent 会结合当前会话继续分析。"
            )

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input(
            "输入研究问题，或继续追问上一轮结果",
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
                        )

                    status.update(
                        label="分析完成",
                        state="complete",
                        expanded=False,
                    )
            report = state.get("final_report", "")
            messages.append({"role": "assistant", "content": report})
            st.session_state["analysis_state"] = state
            with chat_column:
                with st.chat_message("assistant"):
                    st.markdown(report)

    if "analysis_state" in st.session_state:
        st.divider()
        render_result(st.session_state["analysis_state"])


if __name__ == "__main__":
    main()
